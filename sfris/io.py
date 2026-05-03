#!/usr/bin/env python3
"""Data structures, input parser, output manager, logger, and checkpoint."""

import os
import pickle
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from datetime import datetime
from typing import List, Tuple, Optional

# ============================================================
# Internal Constants
# ============================================================
# Placement
DEFAULT_PLACEMENTS_PER_CUT = 30
DEFAULT_PLACEMENT_ATTEMPTS = 5
DEFAULT_GAP_FRACTIONS = [1.0, 0.75, 0.5, 0.25, 0.0]
MAX_ITERATIONS_RATIO = 1.43

# MD
MD_TIMESTEP = 1.0             # fs
MD_THERMOSTAT = "Berendsen"

# Optimization
OPT_MAX_CYCLE = 200

# Bond detection
BOND_THRESHOLD = 1.3
SINGLE_BOND_MIN_RATIO = 0.95
DOUBLE_BOND_MIN_RATIO = 0.80

# Covalent radii (Angstrom)
COVALENT_RADII = {
    'H': 0.31, 'He': 0.28,
    'Li': 1.28, 'Be': 0.96,
    'B': 0.84, 'C': 0.76, 'N': 0.71, 'O': 0.66, 'F': 0.57, 'Ne': 0.58,
    'Na': 1.66, 'Mg': 1.41,
    'Al': 1.21, 'Si': 1.11, 'P': 1.07, 'S': 1.05, 'Cl': 1.02, 'Ar': 1.06,
    'K': 2.03, 'Ca': 1.76,
    'Ga': 1.22, 'Ge': 1.20, 'As': 1.19, 'Se': 1.20, 'Br': 1.20, 'Kr': 1.16,
    'Rb': 2.20, 'Sr': 1.95,
    'In': 1.42, 'Sn': 1.39, 'Sb': 1.39, 'Te': 1.38, 'I': 1.39, 'Xe': 1.40,
    'Cs': 2.44, 'Ba': 2.15,
    'Tl': 1.45, 'Pb': 1.46, 'Bi': 1.48, 'Po': 1.40, 'At': 1.50, 'Rn': 1.50,
}


# ============================================================
# Data Structures
# ============================================================
@dataclass
class SFRISParams:
    """SFRIS parameters."""
    # ==================== SYSTEM ====================
    nproc: int = 8
    memory: int = 4000  # MB
    xtb_path: str = "xtb"
    orca_path: str = ""
    psi4_path: str = ""
    gaussian_path: str = ""
    scratch_dir: str = "./sfris_scratch"

    # ==================== CUT ====================
    max_cuts: int = 3
    min_priority: float = 0.4
    cut_functional_group: bool = True
    cut_multiple_bond: bool = True
    cut_ring: bool = False

    # ==================== MD (optional, test/debug) ====================
    run_md: bool = False
    md_temp: float = 400.0
    md_steps: int = 100
    md_method: str = "GFN2-xTB"

    # ==================== SEARCH ====================
    placements_per_cut: int = 30
    max_iterations: int = 0
    placement_attempts: int = 5

    # ==================== OPTIMIZATION ====================
    opt_method: str = "GFN2-xTB"
    opt_convergence: str = "normal"
    advopt_method: str = "B3LYP-D3(BJ)/def2-TZVP"
    advopt_program: str = ""

    # ==================== CLUSTER ====================
    cluster_rmsd: float = 0.5
    cluster_energy: float = 1.0
    cluster_heavy_only: bool = True
    mirror_check: bool = False

    # ==================== FILTER ====================
    energy_cutoff: float = 100.0  # kcal/mol
    advopt_filter: float = 50.0  # kcal/mol, skip AdvOpt above this
    scan_limit: int = 0  # 0=disabled, early stop after N consecutive misses
    scan_parallel: int = 0  # concurrent xTB jobs in Stage 1
    scan_rotations: int = 3  # rotations per direction (0=random legacy)

    # ==================== FLAGS ====================
    advopt: bool = False  # Enable Block 2 (DFT refinement)
    nma: bool = False     # Normal-mode analysis
    debug: bool = False   # Debug mode: save pouredMIN etc.

    # ==================== ORCA BLOCKS ====================
    orca_opt_block: str = ""
    orca_sp_block: str = ""


@dataclass
class MINEntry:
    """Entry for a MIN structure with multi-level energies."""
    mol: 'Molecule' = None   # forward ref; actual type from sfris.engine
    preopt_mol: 'Molecule' = None  # pre-optimization (placed fragments)
    xtb_mol: 'Molecule' = None     # xTB-optimized geometry
    xtb_energy: Optional[float] = None
    dft_energy: Optional[float] = None
    zpve: Optional[float] = None
    frequencies: Optional[List[float]] = None
    n_imaginary: int = 0
    is_reference: bool = False
    source: str = ""


# ============================================================
# Input Parser
# ============================================================
class InputParser:
    """
    Parse SFRIS input file.

    Format:
        # Comment line
        0 1
        C  x  y  z
        ...
        Parameters
        [SYSTEM]
        NProc = 8
        ...
        [NMA]
        [DEBUG]
        []
    """

    PARAM_MAP = {
        'nproc': ('nproc', int),
        'memory': ('memory', int),
        'xtbpath': ('xtb_path', str),
        'orcapath': ('orca_path', str),
        'psi4path': ('psi4_path', str),
        'gaussianpath': ('gaussian_path', str),
        'scratchdir': ('scratch_dir', str),
        'cutcontrol': ('_cut_control', 'tuple_int_float'),
        'maxcuts': ('max_cuts', int),
        'minpriority': ('min_priority', float),
        'cutfunctionalgroup': ('cut_functional_group', bool),
        'cutmultiplebond': ('cut_multiple_bond', bool),
        'cutring': ('cut_ring', bool),
        'runmd': ('run_md', bool),
        'mdparams': ('_md_params', 'tuple_float_int'),
        'mdtemp': ('md_temp', float),
        'mdsteps': ('md_steps', int),
        'mdmethod': ('md_method', str),
        'convergencerounds': ('_convergence', 'tuple_int_int_auto'),
        'placementspercut': ('placements_per_cut', int),
        'maxiterations': ('max_iterations', 'int_or_zero'),
        'placementattempts': ('placement_attempts', 'int_or_auto'),
        'optmethod': ('opt_method', str),
        'optconvergence': ('opt_convergence', str),
        'advoptmethod': ('advopt_method', str),
        'advoptprogram': ('advopt_program', str),
        'combothreshold': ('_combo_threshold', 'tuple_float_float'),
        'clusterrmsd': ('cluster_rmsd', float),
        'clusterenergy': ('cluster_energy', float),
        'comboheavyonly': ('cluster_heavy_only', bool),
        'clusterheavyonly': ('cluster_heavy_only', bool),
        'mirrorcheck': ('mirror_check', bool),
        'energycutoff': ('energy_cutoff', float),
        'energywindow': ('energy_cutoff', float),
        'advoptfilter': ('advopt_filter', float),
        'scanlimit': ('scan_limit', 'int_or_zero'),
        'scanparallel': ('scan_parallel', int),
        'scanrotations': ('scan_rotations', int),
    }

    FLAG_SECTIONS = {'ADVOPT': 'advopt', 'NMA': 'nma', 'DEBUG': 'debug'}

    @staticmethod
    def parse_bool(value: str) -> bool:
        val_lower = value.lower().strip()
        if val_lower in ('true', 'yes', '1', 'on'):
            return True
        elif val_lower in ('false', 'no', '0', 'off'):
            return False
        else:
            raise ValueError(f"Invalid boolean value: {value}")

    @staticmethod
    def parse_int_or_auto(value: str, default: int) -> int:
        val_stripped = value.strip().lower()
        if val_stripped in ('auto', ''):
            return default
        try:
            val_int = int(val_stripped)
            if val_int <= 0:
                return default
            return val_int
        except ValueError:
            return default

    @staticmethod
    def parse_tuple(value: str, types: List[type]) -> tuple:
        parts = [p.strip() for p in value.split(',')]
        if len(parts) != len(types):
            raise ValueError(
                f"Expected {len(types)} values, got {len(parts)}"
            )
        return tuple(t(p) for t, p in zip(types, parts))

    @classmethod
    def parse_convergence_rounds(cls, value: str) -> Tuple[int, int, int]:
        if value.strip().lower() == 'auto':
            return (DEFAULT_PLACEMENTS_PER_CUT, 0, DEFAULT_PLACEMENT_ATTEMPTS)
        parts = [p.strip() for p in value.split(',')]
        if len(parts) < 2:
            raise ValueError(
                f"ConvergenceRounds requires at least 2 values, got {len(parts)}"
            )
        placements = (int(parts[0]) if parts[0].lower() != 'auto'
                      else DEFAULT_PLACEMENTS_PER_CUT)
        if placements <= 0:
            placements = DEFAULT_PLACEMENTS_PER_CUT
        max_iter = cls.parse_int_or_auto(parts[1], 0)
        if len(parts) >= 3:
            placement_attempts = cls.parse_int_or_auto(
                parts[2], DEFAULT_PLACEMENT_ATTEMPTS
            )
        else:
            placement_attempts = DEFAULT_PLACEMENT_ATTEMPTS
        return (placements, max_iter, placement_attempts)

    @classmethod
    def parse(cls, filepath: str) -> Tuple['Molecule', SFRISParams]:
        # Deferred import to avoid circular dependency
        from sfris.engine import Atom, Molecule

        with open(filepath, 'r') as f:
            content = f.read()

        lines = content.split('\n')
        mol = Molecule()
        params = SFRISParams()

        # Apply global config (overridden by input file values below)
        from sfris.config import apply_config
        apply_config(params)

        state = 'header'
        current_section = None
        current_block = None
        block_content = []
        found_end = False

        for line in lines:
            stripped = line.strip()

            if state == 'header':
                if stripped.startswith('#') or stripped == '':
                    continue
                parts = stripped.split()
                if len(parts) == 2 and parts[0].lstrip('-').isdigit():
                    mol.charge = int(parts[0])
                    mol.multiplicity = int(parts[1])
                    state = 'coords'
                continue

            if state == 'coords':
                if stripped.lower() in ('options', 'parameters'):
                    state = 'options'
                    continue
                if stripped == '' or stripped.startswith('#'):
                    continue
                parts = stripped.split()
                if len(parts) >= 4:
                    elem = parts[0]
                    x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
                    idx = len(mol.atoms)
                    atom = Atom(elem, x, y, z, idx, idx)
                    mol.atoms.append(atom)
                continue

            if state == 'options':
                if stripped == '[]':
                    found_end = True
                    break

                # Section header
                if stripped.startswith('[') and stripped.endswith(']'):
                    section_name = stripped[1:-1].upper()
                    if section_name in cls.FLAG_SECTIONS:
                        attr_name = cls.FLAG_SECTIONS[section_name]
                        setattr(params, attr_name, True)
                    current_section = section_name
                    continue

                # ORCA block start
                if stripped.startswith('%') and not stripped.endswith('%'):
                    block_name = stripped[1:].lower()
                    if block_name == 'orcaopt':
                        current_block = 'orca_opt'
                    elif block_name == 'orcasp':
                        current_block = 'orca_sp'
                    block_content = []
                    continue

                if stripped == '%%':
                    if current_block == 'orca_opt':
                        params.orca_opt_block = '\n'.join(block_content)
                    elif current_block == 'orca_sp':
                        params.orca_sp_block = '\n'.join(block_content)
                    current_block = None
                    continue

                if current_block is not None:
                    block_content.append(line)
                    continue

                if stripped.upper() == 'END':
                    found_end = True
                    break

                if stripped == '' or stripped.startswith('#'):
                    continue

                # Parse key = value
                if '=' in stripped:
                    key, value = stripped.split('=', 1)
                    key = key.strip().lower().replace('_', '')
                    value = value.strip()
                    if '#' in value:
                        value = value.split('#')[0].strip()

                    if key in cls.PARAM_MAP:
                        attr_name, attr_type = cls.PARAM_MAP[key]
                        try:
                            if attr_type == bool:
                                setattr(params, attr_name,
                                        cls.parse_bool(value))
                            elif attr_type == 'int_or_auto':
                                setattr(params, attr_name,
                                        cls.parse_int_or_auto(
                                            value, DEFAULT_PLACEMENT_ATTEMPTS))
                            elif attr_type == 'int_or_zero':
                                setattr(params, attr_name,
                                        cls.parse_int_or_auto(value, 0))
                            elif attr_type == 'tuple_int_float':
                                vals = cls.parse_tuple(value, [int, float])
                                if attr_name == '_cut_control':
                                    params.max_cuts = (vals[0] if vals[0] > 0
                                                       else 3)
                                    params.min_priority = (vals[1] if vals[1] > 0
                                                           else 0.4)
                            elif attr_type == 'tuple_float_int':
                                vals = cls.parse_tuple(value, [float, int])
                                if attr_name == '_md_params':
                                    params.md_temp = (vals[0] if vals[0] > 0
                                                      else 400.0)
                                    params.md_steps = (vals[1] if vals[1] > 0
                                                       else 100)
                            elif attr_type == 'tuple_int_int_auto':
                                vals = cls.parse_convergence_rounds(value)
                                if attr_name == '_convergence':
                                    params.placements_per_cut = vals[0]
                                    params.max_iterations = vals[1]
                                    params.placement_attempts = vals[2]
                            elif attr_type == 'tuple_float_float':
                                if value.strip().lower() == 'auto':
                                    pass  # keep defaults
                                else:
                                    vals = cls.parse_tuple(value, [float, float])
                                    if attr_name == '_combo_threshold':
                                        params.cluster_rmsd = (
                                            vals[0] if vals[0] > 0 else 0.5)
                                        params.cluster_energy = (
                                            vals[1] if vals[1] > 0 else 1.0)
                            else:
                                setattr(params, attr_name, attr_type(value))
                        except (ValueError, TypeError) as e:
                            print(f"Warning: Failed to parse {key}={value}: {e}")

        if state == 'options' and not found_end:
            raise ValueError(
                "Input file missing [] end marker. "
                "Add [] at the end of the Parameters section."
            )

        return mol, params


# ============================================================
# Output Manager
# ============================================================
class OutputManager:
    """Manage output files and directories.

    Directory layout (flat, alongside .inp):
        base_dir/              <- same directory as .inp
        +-- SFRIS_MIN.lis
        +-- SFRIS.run
        +-- SFRIS_param.log
        +-- optMIN/            <- working structures (kept in debug, removed otherwise)
        +-- optMIN_scan/       <- Stage 1 snapshot (debug only)
        +-- optMIN_dft/        <- Stage 2 snapshot (debug only)
        +-- optMIN_nma/        <- NMA snapshot (debug only)
        +-- MIN/               <- final results (written at pipeline end)
        +-- pouredMIN/         <- discarded structures (debug only)
    """

    def __init__(self, base_dir: str, debug: bool = False):
        self.base_dir = Path(base_dir)
        self.min_dir = self.base_dir / "MIN"
        self.opt_min_dir = self.base_dir / "optMIN"
        self.debug = debug
        self.poured_min_dir = self.base_dir / "pouredMIN" if debug else None
        self._setup_directories()

    def _setup_directories(self):
        for d in [self.base_dir, self.opt_min_dir]:
            d.mkdir(parents=True, exist_ok=True)
        if self.poured_min_dir:
            self.poured_min_dir.mkdir(parents=True, exist_ok=True)

    def save_opt_min(self, mol, index: int, comment: str = "") -> str:
        """Save intermediate structure to optMIN/ (working directory).
        Always available, used throughout the pipeline."""
        filename = f"optMIN_{index:03d}.xyz"
        filepath = self.opt_min_dir / filename
        mol.save_xyz(str(filepath), comment)
        return str(filepath)

    def save_final_min(self, mol, index: int, comment: str = "") -> str:
        """Save final structure to MIN/. Called only at pipeline end."""
        self.min_dir.mkdir(parents=True, exist_ok=True)
        filename = f"MIN_{index:03d}.xyz"
        filepath = self.min_dir / filename
        mol.save_xyz(str(filepath), comment)
        return str(filepath)

    def finalize_min(self, entries: list):
        """Write final MIN/ directory from finished entries.
        Clears and recreates MIN/ to ensure exact match with entries.
        Removes optMIN/ since MIN/ is now the authoritative copy."""
        if self.min_dir.exists():
            shutil.rmtree(self.min_dir)
        self.min_dir.mkdir(parents=True, exist_ok=True)
        for i, entry in enumerate(entries):
            best_e = (entry.dft_energy if entry.dft_energy is not None
                      else entry.xtb_energy)
            comment = f"Energy={best_e:.10f}" if best_e else ""
            filename = f"MIN_{i:03d}.xyz"
            filepath = self.min_dir / filename
            entry.mol.save_xyz(str(filepath), comment)
        # Remove working directory in normal mode - MIN/ is the final product
        # In debug mode, keep optMIN/ for cross-referencing with snapshots
        if not self.debug and self.opt_min_dir.exists():
            shutil.rmtree(self.opt_min_dir)

    def save_poured(self, mol, index: int, source: str = "",
                    comment: str = "") -> Optional[str]:
        """Save poured (discarded) structure. Only in debug mode."""
        if not self.debug or self.poured_min_dir is None:
            return None
        filename = f"pouredMIN_{index:03d}_from_{source}.xyz"
        filepath = self.poured_min_dir / filename
        full_comment = f"Source={source} {comment}".strip()
        mol.save_xyz(str(filepath), full_comment)
        return str(filepath)

    def clean_opt_min_dir(self, keep_count: int):
        """Remove optMIN files with index >= keep_count."""
        for f in self.opt_min_dir.glob("optMIN_*.xyz"):
            try:
                idx = int(f.stem.split('_')[1])
                if idx >= keep_count:
                    f.unlink()
            except (ValueError, IndexError):
                pass

    def snapshot_opt_min(self, stage: str):
        """Copy current optMIN/ to optMIN_{stage}/ for debug traceability.

        Creates e.g. optMIN_scan/, optMIN_dft/, optMIN_nma/ so that
        each pipeline stage's structures are preserved even after
        subsequent stages overwrite optMIN/.
        """
        if not self.debug:
            return
        snapshot_dir = self.base_dir / f"optMIN_{stage}"
        if snapshot_dir.exists():
            shutil.rmtree(snapshot_dir)
        snapshot_dir.mkdir(parents=True, exist_ok=True)
        for f in sorted(self.opt_min_dir.glob("optMIN_*.xyz")):
            shutil.copy2(str(f), str(snapshot_dir / f.name))

    def clean_path_files(self, keep_count: int):
        """Remove SEARCH_Conf log files with index >= keep_count.
        Also cleans legacy PATH_MIN files from older versions."""
        for pattern in ["SEARCH_Conf_*.log", "PATH_MIN_*.log"]:
            for f in self.base_dir.glob(pattern):
                try:
                    idx = int(f.stem.split('_')[2])
                    if pattern.startswith("PATH") or idx >= keep_count:
                        f.unlink()
                except (ValueError, IndexError):
                    pass

    def finalize_isomer(self, entries: list, isomer_groups: list):
        """Write ISOMER/ directory with one representative per isomer group.

        Args:
            entries: list of MINEntry (current min_entries)
            isomer_groups: list of dicts with 'group', 'representative',
                          'is_representative' keys
        """
        isomer_dir = self.base_dir / "ISOMER"
        if isomer_dir.exists():
            shutil.rmtree(isomer_dir)
        isomer_dir.mkdir(parents=True, exist_ok=True)
        seen_groups = set()
        rank = 0
        # Collect representatives sorted by energy
        reps = []
        for i, entry in enumerate(entries):
            g = isomer_groups[i]
            if g['is_representative'] and g['group'] not in seen_groups:
                seen_groups.add(g['group'])
                best_e = (entry.dft_energy if entry.dft_energy is not None
                          else entry.xtb_energy)
                reps.append((i, entry, g, best_e))
        reps.sort(
            key=lambda x: x[3] if x[3] is not None else float('inf')
        )
        for rank, (min_idx, entry, g, best_e) in enumerate(reps):
            comment = (
                f"Isomer {g['group']} (MIN {min_idx}) "
                f"Energy={best_e:.10f}" if best_e else
                f"Isomer {g['group']} (MIN {min_idx})"
            )
            filename = f"ISOMER_{rank:03d}_{g['group']}.xyz"
            filepath = isomer_dir / filename
            entry.mol.save_xyz(str(filepath), comment)

    def write_log(self, filename: str, content: str, append: bool = False):
        filepath = self.base_dir / filename
        mode = 'a' if append else 'w'
        with open(filepath, mode, encoding='utf-8') as f:
            f.write(content)

    def append_log(self, filename: str, line: str):
        filepath = self.base_dir / filename
        with open(filepath, 'a', encoding='utf-8') as f:
            f.write(line + '\n')


# ============================================================
# Logger
# ============================================================
class Logger:
    """Dual-level logging: console (key events) + file (full detail)."""

    def __init__(self, output_manager: OutputManager,
                 log_filename: str = "SFRIS.run",
                 quiet: bool = False):
        self.output = output_manager
        self.log_filename = log_filename
        self.quiet = quiet

    def _timestamp(self) -> str:
        return datetime.now().strftime('%H:%M:%S')

    def info(self, message: str, console: bool = True):
        line = f"[{self._timestamp()}] {message}"
        self.output.append_log(self.log_filename, line)
        if console and not self.quiet:
            print(line)

    def detail(self, message: str):
        line = f"[{self._timestamp()}] {message}"
        self.output.append_log(self.log_filename, line)

    def iteration(self, iter_num: int, status: str, result: str = None):
        if result:
            line = f"[{self._timestamp()}] Iter {iter_num}: {status} -> {result}"
        else:
            line = f"[{self._timestamp()}] Iter {iter_num}: {status}"
        self.output.append_log(self.log_filename, line)
        if not self.quiet and result and "MIN" in result and "duplicate" not in result:
            print(line)

    def section(self, title: str):
        line = "=" * 60
        header = f"{line}\n{title}\n{line}"
        self.output.append_log(self.log_filename, f"\n{header}")
        if not self.quiet:
            print(f"\n{header}")


# ============================================================
# Checkpoint Manager
# ============================================================
class CheckpointManager:
    CHECKPOINT_VERSION = "4.0"

    def __init__(self, output_dir: Path):
        self.checkpoint_path = output_dir / "SFRIS.ck"
        self.new_checkpoint_path = output_dir / "SFRIS.ck2"

    def save(self, state: dict):
        state['_version'] = self.CHECKPOINT_VERSION
        state['_timestamp'] = datetime.now().isoformat()
        try:
            with open(self.new_checkpoint_path, 'wb') as f:
                pickle.dump(state, f, protocol=pickle.HIGHEST_PROTOCOL)
            shutil.copy2(self.new_checkpoint_path, self.checkpoint_path)
        except Exception as e:
            if self.new_checkpoint_path.exists():
                self.new_checkpoint_path.unlink()
            raise e

    def load(self) -> Optional[dict]:
        for path in [self.checkpoint_path, self.new_checkpoint_path]:
            if not path.exists():
                continue
            try:
                with open(path, 'rb') as f:
                    state = pickle.load(f)
                if state.get('_version') != self.CHECKPOINT_VERSION:
                    continue
                return state
            except Exception:
                continue
        return None

    def exists(self) -> bool:
        return (self.checkpoint_path.exists()
                or self.new_checkpoint_path.exists())

    def remove(self):
        for p in [self.checkpoint_path, self.new_checkpoint_path]:
            if p.exists():
                p.unlink()

    @staticmethod
    def compute_input_hash(filepath: str) -> str:
        import hashlib
        with open(filepath, 'rb') as f:
            return hashlib.md5(f.read()).hexdigest()