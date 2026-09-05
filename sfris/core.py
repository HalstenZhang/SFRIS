#!/usr/bin/env python3
"""Main controller module for SFRIS isomer search."""

import re
import time
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import List, Tuple, Optional
from sfris.symmetry_rmsd import compute_rmsd_with_symmetry_fallback, check_graph_isomorphic

from sfris.io import (
    SFRISParams, MINEntry, InputParser, OutputManager, Logger,
    CheckpointManager,
    DEFAULT_PLACEMENTS_PER_CUT, DEFAULT_GAP_FRACTIONS,
    MAX_ITERATIONS_RATIO, OPT_MAX_CYCLE,
)
from sfris.backends import XTBInterface, AdvOptInterface
from sfris.engine import (
    Atom, Molecule, Bond, TopologyAnalyzer, Fragmenter,
    StabilityChecker, StructureClusterer,
)


from sfris import __version__ as SFRIS_VERSION
SFRIS_BANNER = r"""
============================================================
       ~~o--o~~  ~~o  o~~  ~~o--o~~o--o~~
          ____  _____ ____  ___ ____
         / ___||  ___|  _ \|_ _/ ___|
         \___ \| |_  | |_) || |\___ \
          ___) |  _| |  _ < | | ___) |
         |____/|_|   |_| \_\___|____/
       ~~o--o~~o--o~~  ~~o  o~~  ~~o--o~~

    Systematic Fragmentation-Recombination
               Isomer Search
          {version}, 2026

    Author: Dapeng Zhang
    ORCID:  0000-0002-2879-5613
    Web:    https://halsten.pcmq.net/sfris

    Released under the MIT License.

    If you use SFRIS in your research, please cite:
      D. Zhang, ChemRxiv 2026,
      doi:10.26434/chemrxiv.15007216/v1
============================================================
""".format(version=SFRIS_VERSION)


class SFRIS:
    """Main SFRIS controller."""

    def __init__(self, input_file: str, quiet: bool = False):
        self.input_file = input_file
        self.input_hash = CheckpointManager.compute_input_hash(input_file)
        self.mol_name = Path(input_file).stem
        self.mol, self.params = InputParser.parse(input_file)
        self._stop_file = Path(input_file).resolve().parent / f"{self.mol_name}.STOP"
        self._stop_requested = False
        self.xtb = XTBInterface(
            xtb_path=self.params.xtb_path,
            scratch_dir=self.params.scratch_dir,
            nproc=self.params.nproc,
            memory=self.params.memory,
        )
        self.output_dir = str(Path(input_file).resolve().parent)
        self.output = OutputManager(self.output_dir, debug=self.params.debug)
        self.logger = Logger(self.output, quiet=quiet)
        self.checkpoint = CheckpointManager(self.output.base_dir)
        self.fragmenter = Fragmenter(self.params)
        self.stability_checker = None
        self.clusterer = StructureClusterer(
            rmsd_threshold=self.params.cluster_rmsd,
            energy_threshold=self.params.cluster_energy,
            heavy_only=self.params.cluster_heavy_only,
            mirror_check=self.params.mirror_check,
        )
        self.advopt_interface: Optional[AdvOptInterface] = None
        if self.params.advopt:
            advopt_prog, advopt_path = AdvOptInterface.detect_program(self.params)
            if advopt_prog and advopt_path:
                self.advopt_interface = AdvOptInterface(
                    program=advopt_prog, program_path=advopt_path,
                    method=self.params.advopt_method,
                    nproc=self.params.nproc, memory=self.params.memory,
                    scratch_dir=self.params.scratch_dir,
                    debug=self.params.debug,
                )
                test_ok, test_msg = self.advopt_interface.test_program()
                if test_ok:
                    self.logger.info(f"Advanced optimization: {advopt_prog} ({advopt_path})")
                    self.logger.info(f"  Method: {self.params.advopt_method}")
                    self.logger.info(f"  Program test: {test_msg}")
                else:
                    self.logger.info(f"WARNING: AdvOpt program not accessible: {test_msg}")
                    self.advopt_interface = None
            else:
                self.logger.info("WARNING: [ADVOPT] enabled but no program path configured.")
        else:
            self.logger.info("Block 2 (DFT refinement) not enabled.")
        if self.params.nma:
            self.logger.info("NMA (normal-mode analysis) enabled.")
        if self.params.debug:
            self.logger.info("DEBUG mode enabled (pouredMIN will be saved).")
        if self.params.run_md:
            self.logger.info(
                f"MD exploration enabled: {self.params.md_steps} steps, "
                f"{self.params.md_temp} K"
            )
        self.min_entries: List[MINEntry] = []
        self.poured_count: int = 0
        self.reference_energy: Optional[float] = None
        self.advopt_ref_energy: Optional[float] = None
        self.cut_history: List[List[Tuple[int, int]]] = []
        self.placement_log: List[dict] = []
        self.timing = {
            'start_time': None, 'end_time': None,
            'setup_cpu': 0.0, 'setup_clock': 0.0,
            'scan_cpu': 0.0, 'scan_clock': 0.0,
            'scan_opt_cpu': 0.0, 'scan_opt_clock': 0.0,
            'cluster1_cpu': 0.0, 'cluster1_clock': 0.0,
            'md_cpu': 0.0, 'md_clock': 0.0,
            'advopt_cpu': 0.0, 'advopt_clock': 0.0,
            'cluster2_cpu': 0.0, 'cluster2_clock': 0.0,
            'nma_cpu': 0.0, 'nma_clock': 0.0,
            'iterations': 0,
        }
        # Index map tracking for scan poured log
        self._scan_poured_log: List[dict] = []
        self._scan_old_to_new: dict = {}
        self._remap_history: List[tuple] = []
        self._isomer_groups: List[dict] = []

    # ================================================================
    # Initialization
    # ================================================================
    def initialize(self):
        """Initialize output and write parameter log."""
        param_content = (
            "# ============================================================\n"
            "# SFRIS Parameters\n"
            "# ============================================================\n"
            f"# Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
            f"# Input file: {self.input_file}\n"
            "# ============================================================\n\n"
            "[INPUT]\n"
            f"formula           = {self.mol.formula}\n"
            f"atoms             = {self.mol.n_atoms}\n"
            f"charge            = {self.mol.charge}\n"
            f"multiplicity      = {self.mol.multiplicity}\n\n"
            "[SYSTEM]\n"
            f"nproc             = {self.params.nproc}\n"
            f"memory            = {self.params.memory} MB\n"
            f"xtb_path          = {self.params.xtb_path}\n"
            f"orca_path         = {self.params.orca_path if self.params.orca_path else '(none)'}\n"
            f"psi4_path         = {self.params.psi4_path if self.params.psi4_path else '(none)'}\n"
            f"gaussian_path     = {self.params.gaussian_path if self.params.gaussian_path else '(none)'}\n"
            f"scratch_dir       = {self.params.scratch_dir}\n\n"
            "[CUT]\n"
            f"CutControl        = {self.params.max_cuts}, {self.params.min_priority}\n\n"
            "[SEARCH]\n"
            f"ConvergenceRounds = {self.params.placements_per_cut}, "
            f"{'auto' if self.params.max_iterations <= 0 else self.params.max_iterations}, "
            f"{self.params.placement_attempts}\n"
            f"ScanRotations     = {self.params.scan_rotations}\n\n"
            "[OPTIMIZATION]\n"
            f"opt_method        = {self.params.opt_method}\n"
            f"advopt_method     = {self.params.advopt_method}\n"
            f"advopt_program    = {self.params.advopt_program if self.params.advopt_program else '(auto-detect)'}\n\n"
            "[CLUSTER]\n"
            f"ComboThreshold    = {self.params.cluster_rmsd}, {self.params.cluster_energy}\n"
            f"heavy_only        = {self.params.cluster_heavy_only}\n\n"
            "[FILTER]\n"
            f"energy_cutoff     = {self.params.energy_cutoff} kcal/mol\n"
            f"AdvOptFilter      = {self.params.advopt_filter} kcal/mol\n"
            f"ScanLimit         = {self.params.scan_limit} (0=disabled)\n"
            f"ScanParallel      = {self.params.scan_parallel}\n\n"
            "[FLAGS]\n"
            f"NMA               = {self.params.nma}\n"
            f"DEBUG             = {self.params.debug}\n\n"
            "# Pipeline: SCAN -> MIN(xTB) -> [ADVOPT -> MIN(DFT)] -> [NMA -> ZPVE]\n"
            "# ============================================================\n"
        )
        self.output.write_log("SFRIS_param.log", param_content)
        run_header = (
            "# ============================================================\n"
            "# SFRIS Run Log\n"
            "# ============================================================\n"
            f"# Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
            f"# Input: {self.input_file} ({self.mol.formula}, {self.mol.n_atoms} atoms)\n"
            "# Pipeline: SCAN -> MIN(xTB) -> [ADVOPT -> MIN(DFT)] -> [NMA]\n"
            "# ============================================================\n"
            + SFRIS_BANNER
        )
        self.output.write_log("SFRIS.run", run_header)
        self.timing['start_time'] = datetime.now()

    def prepare_molecule(self) -> bool:
        """Pre-optimize input geometry, analyze topology, and get reference energy."""
        # --- Pre-optimization of input geometry ---
        self.logger.info("Pre-optimizing input geometry...")
        input_coords = self.mol.coords.copy()
        preopt_mol, preopt_energy, preopt_ok = self.xtb.optimize(
            self.mol, self.params.opt_method, "normal"
        )
        if not preopt_ok:
            self.logger.info(
                "WARNING: Pre-optimization failed. "
                "Using original input geometry."
            )
        else:
            # Compute RMSD between input and optimized
            diff = input_coords - preopt_mol.coords
            rmsd = np.sqrt(np.mean(np.sum(diff ** 2, axis=1)))
            self.logger.info(
                f"  Pre-optimization RMSD: {rmsd:.4f} A "
                f"(E = {preopt_energy:.6f} Eh)"
            )
            if rmsd > 0.5:
                self.logger.info(
                    f"  WARNING: Input geometry changed significantly "
                    f"(RMSD = {rmsd:.4f} A > 0.5 A)."
                )
                self.logger.info(
                    "  Please ensure the input is a reasonable "
                    "equilibrium structure."
                )
            # Update molecule with optimized geometry
            for i, atom in enumerate(preopt_mol.atoms):
                self.mol.atoms[i].x = atom.x
                self.mol.atoms[i].y = atom.y
                self.mol.atoms[i].z = atom.z
            self.logger.info("  Using pre-optimized geometry for bond analysis.")

        # --- Topology analysis ---
        self.logger.info("Analyzing molecular topology...")
        analyzer = TopologyAnalyzer(self.mol, self.params)
        self.mol = analyzer.analyze()
        cuttable = [b for b in self.mol.bonds
                    if b.cuttable and b.priority >= self.params.min_priority]
        self.logger.info(
            f"  Bonds: {len(self.mol.bonds)} total, {len(cuttable)} cuttable "
            f"(priority >= {self.params.min_priority})"
        )
        self.logger.info("  Getting reference energy...")
        energy, success = self.xtb.single_point(self.mol, self.params.md_method)
        if not success:
            self.logger.info("  Failed to get reference energy")
            return False
        self.reference_energy = energy
        self.logger.info(f"  Reference energy: {energy:.6f} Eh")
        self.stability_checker = StabilityChecker(self.params, self.reference_energy)
        ref_entry = MINEntry(
            mol=self.mol.copy(), xtb_energy=energy,
            is_reference=True, source="reference",
        )
        ref_entry.mol.energy = energy
        self.min_entries.append(ref_entry)
        self.output.save_opt_min(self.mol, 0, f"Energy={energy:.10f} reference")
        self._write_min_lis()
        self.logger.info(f"  MIN 0: reference structure, E={energy:.6f} Eh")
        return True

    # ================================================================
    # Duplicate check
    # ================================================================
    def is_duplicate(self, mol, existing_mols, rmsd_threshold=None, energy_threshold=None):
        if rmsd_threshold is None:
            rmsd_threshold = self.params.cluster_rmsd
        if energy_threshold is None:
            energy_threshold = self.params.cluster_energy
        ENERGY_IDENTICAL = 0.1 / 627.509
        energy_threshold_eh = energy_threshold / 627.509
        for i, ref in enumerate(existing_mols):
            if mol.n_atoms != ref.n_atoms:
                continue
            if mol.energy is not None and ref.energy is not None:
                energy_diff = abs(mol.energy - ref.energy)
                if energy_diff < ENERGY_IDENTICAL:
                    return True, i
                if energy_diff > energy_threshold_eh:
                    continue
            rmsd = self.clusterer._calculate_rmsd_kabsch(
                mol, ref, heavy_only=self.params.cluster_heavy_only
            )
            if rmsd < rmsd_threshold:
                return True, i
        return False, -1

    # ================================================================
    # MIN list output
    # ================================================================
    def _write_min_lis(self):
        content = f"# Generated by SFRIS {SFRIS_VERSION} (https://halsten.pcmq.net/sfris)\n"
        content += "# SFRIS Minimum Structure List\n"
        content += f"# Input: {self.input_file}\n"
        content += f"# Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        content += f"# Structures: {len(self.min_entries)}\n"
        if self._isomer_groups:
            n_groups = len(set(g['group'] for g in self._isomer_groups))
            content += f"# Constitutional isomers: {n_groups}\n"
        content += "\n"
        for i, entry in enumerate(self.min_entries):
            tag = " (reference structure)" if entry.is_reference else ""
            grp_str = ""
            if self._isomer_groups and i < len(self._isomer_groups):
                g = self._isomer_groups[i]
                if g['is_representative']:
                    grp_str = f" [Isomer {g['group']}]"
                else:
                    grp_str = f" [Isomer {g['group']}, conformer of MIN {g['representative']}]"
            content += f"# Geometry of MIN {i}{tag}{grp_str}\n"
            for atom in entry.mol.atoms:
                content += (f"{atom.element:2s}  {atom.x:15.10f}  "
                            f"{atom.y:15.10f}  {atom.z:15.10f}\n")
            if entry.xtb_energy is not None:
                content += f"Energy(xTB)  = {entry.xtb_energy:.10f} Eh\n"
                if self.reference_energy is not None:
                    dE_xtb = (entry.xtb_energy - self.reference_energy) * 627.509
                    content += f"dE(xTB)      = {dE_xtb:10.2f} kcal/mol\n"
            if entry.dft_energy is not None:
                content += f"Energy(DFT)  = {entry.dft_energy:.10f} Eh\n"
                if self.advopt_ref_energy is not None:
                    dE_dft = (entry.dft_energy - self.advopt_ref_energy) * 627.509
                    content += f"dE(DFT)      = {dE_dft:10.2f} kcal/mol\n"
            if entry.zpve is not None:
                content += f"ZPVE         = {entry.zpve:.10f} Eh\n"
                best_e = (entry.dft_energy if entry.dft_energy is not None
                          else entry.xtb_energy)
                if best_e is not None:
                    e_zpve = best_e + entry.zpve
                    content += f"E+ZPVE       = {e_zpve:.10f} Eh\n"
                    ref_e = (self.advopt_ref_energy
                             if self.advopt_ref_energy is not None
                             else self.reference_energy)
                    if (ref_e is not None and len(self.min_entries) > 0
                            and self.min_entries[0].zpve is not None):
                        ref_best = (self.min_entries[0].dft_energy
                                    if self.min_entries[0].dft_energy is not None
                                    else self.min_entries[0].xtb_energy)
                        if ref_best is not None:
                            ref_ezpve = ref_best + self.min_entries[0].zpve
                            dE_zpve = (e_zpve - ref_ezpve) * 627.509
                            content += f"dE+ZPVE      = {dE_zpve:10.2f} kcal/mol\n"
            if entry.frequencies is not None:
                content += f"Nmode        : {len(entry.frequencies)}\n"
                for j in range(0, len(entry.frequencies), 6):
                    chunk = entry.frequencies[j:j + 6]
                    content += "  " + "  ".join(f"{f:.6f}" for f in chunk) + "\n"
                if entry.n_imaginary > 0:
                    content += (f"WARNING: {entry.n_imaginary} imaginary "
                                f"frequency detected\n")
            content += "\n"
        content += f"# Total MIN structures: {len(self.min_entries)}\n"
        if self._isomer_groups:
            n_groups = len(set(g['group'] for g in self._isomer_groups))
            content += f"# Constitutional isomers: {n_groups}\n"
        self.output.write_log("SFRIS_MIN.lis", content)

    def _delta_e(self, entry, eq_ref, eq_ref_label):
        """Relative energy on the reference's own potential energy surface.

        Returns None when the entry has no energy at that level, e.g. a
        structure excluded from Block 2 by AdvOptFilter. Mixing an xTB
        absolute energy with a DFT reference is meaningless.
        """
        if eq_ref is None:
            return None
        e = (entry.dft_energy if eq_ref_label == "DFT"
             else entry.xtb_energy)
        if e is None:
            return None
        return (e - eq_ref) * 627.509

    def _write_isomer_lis(self):
        """Write isomer-only list file (one representative per isomer group)."""
        if not self._isomer_groups or not self.min_entries:
            return
        # Determine energy reference
        eq_ref = (
            self.advopt_ref_energy
            if self.advopt_ref_energy is not None
            else self.reference_energy
        )
        eq_ref_label = (
            "DFT"
            if any(e.dft_energy is not None for e in self.min_entries)
            else "xTB"
        )
        # Collect unique isomer representatives
        seen_groups = set()
        isomer_entries = []
        for i, entry in enumerate(self.min_entries):
            g = self._isomer_groups[i]
            if g['group'] not in seen_groups and g['is_representative']:
                seen_groups.add(g['group'])
                isomer_entries.append((i, entry, g))
        # Sort by best energy
        isomer_entries.sort(
            key=lambda x: (
                x[1].dft_energy if x[1].dft_energy is not None
                else x[1].xtb_energy
                if x[1].xtb_energy is not None else float('inf')
            )
        )
        n_isomers = len(isomer_entries)
        content = (
            f"# Generated by SFRIS {SFRIS_VERSION}\n"
            f"# SFRIS Isomer List (representatives only)\n"
            f"# Input: {self.input_file}\n"
            f"# Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
            f"# Isomers: {n_isomers}\n\n"
        )
        # Energy summary table
        content += (
            f"# Energy summary (relative to {eq_ref_label} reference, "
            f"kcal/mol):\n"
        )
        content += "# " + "-" * 53 + "\n"
        for rank, (min_idx, entry, g) in enumerate(isomer_entries):
            delta_e = self._delta_e(entry, eq_ref, eq_ref_label)
            ref_tag = " (ref)" if entry.is_reference else ""
            if delta_e is not None:
                content += (
                    f"#   {rank:3d}  Isomer {g['group']:>5s}  "
                    f"MIN {min_idx:3d}  "
                    f"{delta_e:10.2f} kcal/mol{ref_tag}\n"
                )
            else:
                content += (
                    f"#   {rank:3d}  Isomer {g['group']:>5s}  "
                    f"MIN {min_idx:3d}  "
                    f"{'not refined':>10s}{ref_tag}\n"
                )
        content += "# " + "-" * 53 + "\n\n"
        # Geometry section
        for rank, (min_idx, entry, g) in enumerate(isomer_entries):
            delta_e = self._delta_e(entry, eq_ref, eq_ref_label)
            content += (
                f"# Isomer {g['group']} (MIN {min_idx})"
            )
            if delta_e is not None:
                content += f"  dE({eq_ref_label})={delta_e:.2f} kcal/mol"
            content += "\n"
            for atom in entry.mol.atoms:
                content += (
                    f"{atom.element:2s}  {atom.x:15.10f}  "
                    f"{atom.y:15.10f}  {atom.z:15.10f}\n"
                )
            if entry.dft_energy is not None:
                content += f"Energy(DFT)  = {entry.dft_energy:.10f} Eh\n"
            if entry.xtb_energy is not None:
                content += f"Energy(xTB)  = {entry.xtb_energy:.10f} Eh\n"
            if entry.zpve is not None:
                content += f"ZPVE         = {entry.zpve:.10f} Eh\n"
            content += "\n"
        content += f"# Total isomers: {n_isomers}\n"
        self.output.write_log("SFRIS_ISOMER.lis", content)

    # ================================================================
    # Checkpoint helpers
    # ================================================================
    def _save_checkpoint(self, phase, iteration=0):
        state = {
            'phase': phase, 'iteration': iteration,
            'min_count': len(self.min_entries),
            'poured_count': self.poured_count,
            'rng_state': self.fragmenter.rng.bit_generator.state,
            'cut_history': self.cut_history,
            'reference_energy': self.reference_energy,
            'advopt_ref_energy': self.advopt_ref_energy,
            'input_hash': self.input_hash,
        }
        self.checkpoint.save(state)

    def _check_stop(self):
        """Check for stop file. Returns True if stop requested."""
        if self._stop_file.exists():
            self._stop_requested = True
            self._stop_file.unlink()
            self.logger.info(
                f"Stop file detected ({self._stop_file.name}). "
                f"Saving checkpoint and stopping."
            )
            return True
        return False

    def _restore_checkpoint(self):
        state = self.checkpoint.load()
        if state is None:
            return None
        if state.get('input_hash') != self.input_hash:
            self.logger.info("Checkpoint found but input file changed, starting fresh")
            return None
        return state

    def _reload_min_entries(self, min_count):
        """Reload MIN entries from optMIN/ working directory."""
        self.min_entries = []
        for i in range(min_count):
            xyz_path = self.output.opt_min_dir / f"optMIN_{i:03d}.xyz"
            if xyz_path.exists():
                mol = self._load_xyz(xyz_path)
                entry = MINEntry(
                    mol=mol, xtb_energy=mol.energy,
                    is_reference=(i == 0), source="checkpoint",
                )
                self.min_entries.append(entry)

    def _load_xyz(self, filepath):
        mol = Molecule()
        with open(filepath, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        n_atoms = int(lines[0].strip())
        comment = lines[1].strip()
        energy_match = re.search(r'Energy[=\s]*([-\d.]+)', comment)
        if energy_match:
            mol.energy = float(energy_match.group(1))
        for i, line in enumerate(lines[2:2 + n_atoms]):
            parts = line.split()
            elem = parts[0]
            x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
            atom = Atom(elem, x, y, z, i, i)
            mol.atoms.append(atom)
        mol.charge = self.mol.charge
        mol.multiplicity = self.mol.multiplicity
        return mol

    # ================================================================
    # Main pipeline
    # ================================================================
    def run_search(self):
        checkpoint_state = self._restore_checkpoint()
        if checkpoint_state:
            self.logger.section("Resuming from checkpoint")
            self.logger.info(
                f"  Phase: {checkpoint_state['phase']}, "
                f"Iteration: {checkpoint_state['iteration']}, "
                f"MIN: {checkpoint_state['min_count']}"
            )
            self.reference_energy = checkpoint_state['reference_energy']
            self.advopt_ref_energy = checkpoint_state.get('advopt_ref_energy')
            self.cut_history = checkpoint_state['cut_history']
            self.poured_count = checkpoint_state.get('poured_count', 0)
            self.fragmenter.rng.bit_generator.state = checkpoint_state['rng_state']
            self.stability_checker = StabilityChecker(self.params, self.reference_energy)
            # Pre-optimize before topology analysis (same as fresh start)
            preopt_mol, preopt_energy, preopt_ok = self.xtb.optimize(
                self.mol, self.params.opt_method, "normal"
            )
            if preopt_ok:
                for i, atom in enumerate(preopt_mol.atoms):
                    self.mol.atoms[i].x = atom.x
                    self.mol.atoms[i].y = atom.y
                    self.mol.atoms[i].z = atom.z
            analyzer = TopologyAnalyzer(self.mol, self.params)
            self.mol = analyzer.analyze()
            self._reload_min_entries(checkpoint_state['min_count'])
            start_phase = checkpoint_state['phase']
            start_iteration = checkpoint_state['iteration']
        else:
            start_phase = 'SCAN'
            start_iteration = 0
            self.logger.section("Starting SFRIS pipeline")
            setup_cpu_start = time.process_time()
            setup_clock_start = time.time()
            if not self.prepare_molecule():
                self.logger.info("Failed to prepare molecule")
                return False
            self.timing['setup_cpu'] = time.process_time() - setup_cpu_start
            self.timing['setup_clock'] = time.time() - setup_clock_start
            self._write_bonds_file()
            self._write_cuts_file()
            self._init_placement_file()
            self._init_index_map()
            self._save_checkpoint('SCAN', 0)

        if start_phase == 'SCAN':
            self._run_stage_scan(start_iteration)
            if self._stop_requested:
                self.logger.section("SFRIS stopped by user")
                return False
            self._save_checkpoint('ADVOPT', 0)
            start_phase = 'ADVOPT'
        if start_phase == 'ADVOPT':
            self._run_stage_advopt()
            if self._stop_requested:
                self.logger.section("SFRIS stopped by user")
                return False
            self._save_checkpoint('NMA', 0)
            start_phase = 'NMA'
        if start_phase == 'NMA' and self.params.nma:
            self._run_stage_nma()
            if self._stop_requested:
                self.logger.section("SFRIS stopped by user")
                return False

        # Finalize: write MIN/ directory with final results
        self.output.finalize_min(self.min_entries)
        self.logger.info(
            f"Final structures written to MIN/ ({len(self.min_entries)} files)"
        )
        # Write scan poured log with final fate info (after full pipeline)
        self._write_scan_poured_log()

        self.timing['end_time'] = datetime.now()
        self._group_constitutional_isomers()

        # Safety fallback: if advopt_ref_energy is missing but reference
        # has DFT energy, use it to avoid xTB vs DFT mismatch
        if self.advopt_ref_energy is None and self.min_entries:
            for entry in self.min_entries:
                if entry.is_reference and entry.dft_energy is not None:
                    self.advopt_ref_energy = entry.dft_energy
                    break

        self._write_min_lis()
        self._write_isomer_lis()
        if self._isomer_groups:
            self.output.finalize_isomer(self.min_entries, self._isomer_groups)
        self._write_summary()
        self._write_timing_file()
        self.checkpoint.remove()
        self.logger.section("SFRIS completed successfully")
        final_msg = f"Final result: {len(self.min_entries)} MIN structures"
        if self._isomer_groups:
            n_groups = len(set(g['group'] for g in self._isomer_groups))
            final_msg += f" ({n_groups} constitutional isomers)"
        self.logger.info(final_msg)
        return True

    # ----------------------------------------------------------------
    # Stage 1: SCAN helpers
    # ----------------------------------------------------------------
    def _execute_scan_plan(self, iter_plan, existing_mols, scan_poured_log,
                           batch_size, scan_limit, start_iteration=0,
                           iter_offset=0):
        """Execute a scan iteration plan (shared by Phase 1 and Phase 2).

        Args:
            iter_plan: list of (cut_idx, bonds_to_cut, dir_idx, direction, rotation)
                       rotation is np.ndarray or None (None = random)
            existing_mols: list of known molecules (modified in-place)
            scan_poured_log: list to append poured entries (modified in-place)
            batch_size: parallel workers
            scan_limit: early stop after N consecutive misses per cut (0=disabled)
            start_iteration: skip iterations <= this (for checkpoint resume)
            iter_offset: added to plan index for global iteration numbering

        Returns:
            (last_iteration, productive_keys, early_stopped)
            productive_keys: set of (cut_idx, dir_idx) that found new structures
        """
        from concurrent.futures import ThreadPoolExecutor
        from collections import defaultdict

        opt_fail_logged = 0
        early_stopped = False
        plan_idx = 0
        iteration = iter_offset
        productive_keys = set()

        # Per-cut tracking instead of global counter
        no_new_per_cut = defaultdict(int)
        exhausted_cuts = set()

        with ThreadPoolExecutor(max_workers=batch_size) as executor:
            while plan_idx < len(iter_plan):
                # Check for stop file between batches
                if self._check_stop():
                    self._save_checkpoint('SCAN', iteration)
                    break
                batch_tasks = []
                while len(batch_tasks) < batch_size and plan_idx < len(iter_plan):
                    cut_idx, bonds_to_cut, dir_idx, direction, rot = iter_plan[plan_idx]
                    iter_num = plan_idx + 1 + iter_offset
                    plan_idx += 1
                    if iter_num <= start_iteration:
                        continue
                    # Skip iterations for exhausted cuts
                    if cut_idx in exhausted_cuts:
                        continue
                    fragments = self.fragmenter.cut_molecule(self.mol, bonds_to_cut)
                    combined, placement_info = self.fragmenter.place_fragments_with_direction(
                        fragments, direction, rotation=rot
                    )
                    combined.charge = self.mol.charge
                    combined.multiplicity = self.mol.multiplicity
                    preopt_mol = combined.copy()
                    batch_tasks.append({
                        'iter_num': iter_num, 'bonds_to_cut': bonds_to_cut,
                        'dir_idx': dir_idx, 'cut_idx': cut_idx,
                        'combined': combined, 'preopt_mol': preopt_mol,
                        'placement_info': placement_info,
                    })
                if not batch_tasks:
                    # Check if all remaining iterations are exhausted
                    remaining = any(
                        iter_plan[i][0] not in exhausted_cuts
                        for i in range(plan_idx, len(iter_plan))
                    )
                    if not remaining:
                        early_stopped = True
                        break
                    continue

                def _optimize_worker(task):
                    opt_mol, energy, success = self.xtb.optimize(
                        task['combined'], method=self.params.opt_method,
                        convergence=self.params.opt_convergence,
                        max_cycle=OPT_MAX_CYCLE,
                    )
                    xtb_err = getattr(self.xtb, '_last_error', '')
                    return opt_mol, energy, success, xtb_err

                futures = {
                    executor.submit(_optimize_worker, task): task
                    for task in batch_tasks
                }
                results = []
                for future in futures:
                    task = futures[future]
                    try:
                        opt_mol, energy, success, xtb_err = future.result(timeout=7200)
                    except Exception as e:
                        opt_mol, energy, success, xtb_err = None, None, False, str(e)
                    results.append((task, opt_mol, energy, success, xtb_err))
                results.sort(key=lambda x: x[0]['iter_num'])

                for task, opt_mol, energy, success, xtb_err in results:
                    iter_num = task['iter_num']
                    bonds_to_cut = task['bonds_to_cut']
                    dir_idx = task['dir_idx']
                    cut_idx = task['cut_idx']
                    preopt_mol = task['preopt_mol']
                    combined = task['combined']
                    placement_info = task['placement_info']
                    iteration = iter_num
                    self.timing['iterations'] += 1
                    cut_info = [(b.atom1, b.atom2) for b in bonds_to_cut]
                    cut_str = ",".join(
                        [f"{self.mol.atoms[a].element}{a}-"
                         f"{self.mol.atoms[b].element}{b}"
                         for a, b in cut_info]
                    )
                    if not success or opt_mol is None:
                        no_new_per_cut[cut_idx] += 1
                        self.logger.iteration(iter_num, f"OPT_FAILED [{cut_str}]")
                        if xtb_err and opt_fail_logged < 3:
                            self.logger.info(f"  xTB error: {xtb_err}")
                            opt_fail_logged += 1
                        if (scan_limit > 0
                                and no_new_per_cut[cut_idx] >= scan_limit):
                            exhausted_cuts.add(cut_idx)
                            self.logger.info(
                                f"  Cut [{cut_str}] exhausted "
                                f"({scan_limit} consecutive misses)"
                            )
                        continue
                    if opt_mol.n_atoms == combined.n_atoms:
                        for j, atom in enumerate(opt_mol.atoms):
                            atom.original_index = combined.atoms[j].original_index
                    is_stable, reason = self.stability_checker.is_stable(
                        opt_mol, self.mol, energy
                    )
                    if not is_stable:
                        no_new_per_cut[cut_idx] += 1
                        self.logger.iteration(iter_num, f"{reason} [{cut_str}]")
                        if (scan_limit > 0
                                and no_new_per_cut[cut_idx] >= scan_limit):
                            exhausted_cuts.add(cut_idx)
                            self.logger.info(
                                f"  Cut [{cut_str}] exhausted "
                                f"({scan_limit} consecutive misses)"
                            )
                        continue
                    opt_mol.energy = energy
                    opt_mol.charge = self.mol.charge
                    opt_mol.multiplicity = self.mol.multiplicity
                    opt_mol.sort_by_original_index()
                    self.cut_history.append(cut_info)
                    placement_entry = {
                        'iteration': iter_num, 'status': 'CONNECTED',
                        'cuts': cut_str,
                        'n_fragments': len(placement_info.get('fragments', [])),
                        'fragments': placement_info.get('fragments', []),
                        'placement': placement_info.get('placement', {}),
                        'direction': placement_info.get('direction', []),
                    }
                    self.placement_log.append(placement_entry)
                    self._append_placement_entry(placement_entry)
                    status = f"CONNECTED [{cut_str}]"
                    is_dup, dup_idx = self.is_duplicate(opt_mol, existing_mols)
                    if is_dup:
                        no_new_per_cut[cut_idx] += 1
                        self.logger.iteration(
                            iter_num, status, f"duplicate of MIN {dup_idx}"
                        )
                        poured_file = ""
                        if self.params.debug:
                            self.output.save_poured(
                                opt_mol, self.poured_count, "scan",
                                f"Energy={opt_mol.energy:.10f} "
                                f"Iter={iter_num} DupOf={dup_idx}"
                                if opt_mol.energy
                                else f"Iter={iter_num} DupOf={dup_idx}",
                            )
                            poured_file = (
                                f"pouredMIN_{self.poured_count:03d}_from_scan.xyz"
                            )
                            self.poured_count += 1
                        scan_poured_log.append({
                            'iter': iter_num, 'dup_of': dup_idx,
                            'energy': opt_mol.energy, 'cut': cut_str,
                            'poured_file': poured_file,
                        })
                        if (scan_limit > 0
                                and no_new_per_cut[cut_idx] >= scan_limit):
                            exhausted_cuts.add(cut_idx)
                            self.logger.info(
                                f"  Cut [{cut_str}] exhausted "
                                f"({scan_limit} consecutive misses)"
                            )
                    else:
                        no_new_per_cut[cut_idx] = 0
                        min_idx = len(self.min_entries)
                        entry = MINEntry(
                            mol=opt_mol, preopt_mol=preopt_mol,
                            xtb_mol=opt_mol.copy(),
                            xtb_energy=opt_mol.energy,
                            source=f"scan_cut[{cut_str}]_dir{dir_idx}",
                        )
                        self.min_entries.append(entry)
                        existing_mols.append(opt_mol)
                        self.output.save_opt_min(
                            opt_mol, min_idx,
                            f"Energy={opt_mol.energy:.10f} Cut=[{cut_str}]",
                        )
                        self._write_min_lis()
                        self._write_path_file(min_idx, entry)
                        self.logger.iteration(
                            iter_num, status, f"MIN {min_idx}"
                        )
                        self._save_checkpoint('SCAN', iter_num)
                        productive_keys.add((cut_idx, dir_idx))

        return iteration, productive_keys, early_stopped

    # ----------------------------------------------------------------
    # Stage 1: SCAN (two-phase adaptive or legacy random)
    # ----------------------------------------------------------------
    def _run_stage_scan(self, start_iteration=0):
        self.logger.section("Stage 1: SCAN (Placement + xTB Optimization)")
        all_cut_combinations = self.fragmenter.enumerate_cut_combinations(self.mol)
        n_cuts = len(all_cut_combinations)
        n_directions = len(self.fragmenter.fibonacci_directions)
        total_base = n_cuts * n_directions
        all_cut_combinations.sort(
            key=lambda bonds: max(b.priority for b in bonds), reverse=True
        )
        scan_limit = self.params.scan_limit
        batch_size = max(
            1,
            self.params.scan_parallel
            if self.params.scan_parallel > 0
            else self.params.nproc
        )
        scan_rotations = self.params.scan_rotations
        existing_mols = [e.mol for e in self.min_entries]
        scan_cpu_start = time.process_time()
        scan_clock_start = time.time()
        scan_poured_log = []

        if scan_rotations > 0:
            # ==== Systematic two-phase adaptive scan ====
            coarse_angles = [
                2 * np.pi * k / scan_rotations
                for k in range(scan_rotations)
            ]
            fine_angles = [
                2 * np.pi * (k + 0.5) / scan_rotations
                for k in range(scan_rotations)
            ]
            total_coarse = total_base * scan_rotations
            total_max = total_base * scan_rotations * 2

            auto_max = max(200, int(total_max * MAX_ITERATIONS_RATIO))
            if self.params.max_iterations <= 0:
                self.params.max_iterations = auto_max
            elif self.params.max_iterations < total_coarse:
                self.params.max_iterations = auto_max

            self.logger.info(
                f"Cut combinations: {n_cuts}, Directions: {n_directions}, "
                f"Rotations: {scan_rotations} coarse + "
                f"{scan_rotations} fine (adaptive)"
            )
            self.logger.info(
                f"Phase 1: {total_coarse} iterations, "
                f"Max: {self.params.max_iterations}"
            )
            self.logger.info(
                f"Priority-sorted cuts, ScanLimit: "
                f"{'disabled' if scan_limit <= 0 else f'{scan_limit}/cut'}, "
                f"Parallel: {batch_size} workers"
            )

            # Phase 1: Coarse scan -- all (cut, direction, coarse_angle)
            phase1_plan = []
            for cut_idx, bonds_to_cut in enumerate(all_cut_combinations):
                for dir_idx, direction in enumerate(
                    self.fragmenter.fibonacci_directions
                ):
                    for angle in coarse_angles:
                        rot = Fragmenter._rotation_around_axis(
                            direction, angle
                        )
                        phase1_plan.append(
                            (cut_idx, bonds_to_cut, dir_idx, direction, rot)
                        )

            self.logger.info(
                f"[Phase 1: coarse scan ({len(phase1_plan)} iterations)]"
            )
            iteration, productive_keys, early_stopped = (
                self._execute_scan_plan(
                    phase1_plan, existing_mols, scan_poured_log,
                    batch_size, scan_limit, start_iteration,
                )
            )
            n_min_after_p1 = len(self.min_entries)
            self.logger.info(
                f"Phase 1 complete: {n_min_after_p1} MIN, "
                f"{len(productive_keys)} productive directions"
            )

            # Phase 2: Refine only productive directions
            if not early_stopped and productive_keys:
                phase2_plan = []
                for cut_idx, bonds_to_cut in enumerate(
                    all_cut_combinations
                ):
                    for dir_idx, direction in enumerate(
                        self.fragmenter.fibonacci_directions
                    ):
                        if (cut_idx, dir_idx) in productive_keys:
                            for angle in fine_angles:
                                rot = Fragmenter._rotation_around_axis(
                                    direction, angle
                                )
                                phase2_plan.append(
                                    (cut_idx, bonds_to_cut,
                                     dir_idx, direction, rot)
                                )

                if phase2_plan:
                    self.logger.info(
                        f"[Phase 2: refine ({len(phase2_plan)} iterations, "
                        f"{len(productive_keys)} directions)]"
                    )
                    iteration2, _, early_stopped2 = (
                        self._execute_scan_plan(
                            phase2_plan, existing_mols, scan_poured_log,
                            batch_size, scan_limit,
                            iter_offset=iteration,
                        )
                    )
                    iteration = iteration2
                    if early_stopped2:
                        self.logger.info(
                            f"Phase 2 early stopped at iteration {iteration}"
                        )
            elif early_stopped:
                self.logger.info(
                    f"Phase 1 early stopped at iteration {iteration}: "
                    f"all cut combinations exhausted "
                    f"(ScanLimit={scan_limit} per cut)"
                )

        else:
            # ==== Legacy random mode (ScanRotations = 0) ====
            total_iterations = total_base
            auto_max = max(200, int(total_iterations * MAX_ITERATIONS_RATIO))
            if self.params.max_iterations <= 0:
                self.params.max_iterations = auto_max
            elif self.params.max_iterations < total_iterations:
                self.params.max_iterations = auto_max

            self.logger.info(
                f"Cut combinations: {n_cuts}, Directions: {n_directions}, "
                f"Total: {total_iterations}, "
                f"Max: {self.params.max_iterations}"
            )
            self.logger.info(
                f"Priority-sorted cuts, ScanLimit: "
                f"{'disabled' if scan_limit <= 0 else f'{scan_limit}/cut'}, "
                f"Parallel: {batch_size} workers (random mode)"
            )

            legacy_plan = []
            for cut_idx, bonds_to_cut in enumerate(all_cut_combinations):
                for dir_idx, direction in enumerate(
                    self.fragmenter.fibonacci_directions
                ):
                    legacy_plan.append(
                        (cut_idx, bonds_to_cut, dir_idx, direction, None)
                    )

            iteration, _, early_stopped = self._execute_scan_plan(
                legacy_plan, existing_mols, scan_poured_log,
                batch_size, scan_limit, start_iteration,
            )
            if early_stopped:
                self.logger.info(
                    f"Early stopped at iteration {iteration}: "
                    f"all cut combinations exhausted "
                    f"(ScanLimit={scan_limit} per cut)"
                )

        self.timing['scan_cpu'] = time.process_time() - scan_cpu_start
        self.timing['scan_clock'] = time.time() - scan_clock_start
        self.logger.info(
            f"Stage 1 complete: {len(self.min_entries)} MIN structures "
            f"(including reference)"
        )

        # Stage 1 clustering
        self.logger.section("Stage 1: Final clustering")
        self._log_index_snapshot("Stage 1: SCAN result", self.min_entries)
        cluster1_cpu_start = time.process_time()
        cluster1_clock_start = time.time()
        old_to_new_map = {i: i for i in range(len(self.min_entries))}
        if len(self.min_entries) > 1:
            mols = [e.mol for e in self.min_entries]
            clusters, dist_matrix = self.clusterer.cluster(mols, self.logger)
            representatives = self.clusterer.select_representatives(
                mols, clusters
            )
            rep_indices = set(idx for idx, _ in representatives)
            ref_entry = None
            other_entries = []
            for idx, mol in representatives:
                entry = self.min_entries[idx]
                if entry.is_reference:
                    ref_entry = entry
                else:
                    other_entries.append(entry)
            new_entries = []
            if ref_entry:
                new_entries.append(ref_entry)
            other_entries.sort(
                key=lambda e: e.xtb_energy
                if e.xtb_energy is not None else float('inf')
            )
            new_entries.extend(other_entries)
            if self.params.debug:
                for i, entry in enumerate(self.min_entries):
                    if i not in rep_indices:
                        self.output.save_poured(
                            entry.mol, self.poured_count, "cluster1",
                            f"Energy={entry.xtb_energy:.10f}"
                            if entry.xtb_energy else "",
                        )
                        self.poured_count += 1
            removed_reasons = {
                i: "cluster1" for i in range(len(self.min_entries))
                if i not in rep_indices
            }
            old_entries = self.min_entries
            self.min_entries = new_entries
            self._log_index_remap(
                "Stage 1: Clustering",
                old_entries, self.min_entries, removed_reasons
            )
            new_id_map = {id(e): i for i, e in enumerate(self.min_entries)}
            old_to_new_map = {}
            for old_i, entry in enumerate(old_entries):
                if id(entry) in new_id_map:
                    old_to_new_map[old_i] = new_id_map[id(entry)]
            for i, entry in enumerate(self.min_entries):
                self.output.save_opt_min(
                    entry.mol, i,
                    f"Energy={entry.xtb_energy:.10f}"
                    if entry.xtb_energy else "",
                )
                self._write_path_file(i, entry)
            self.output.clean_opt_min_dir(len(self.min_entries))
            self.output.clean_path_files(len(self.min_entries))
            self._write_min_lis()
            self._write_rmsd_matrix(dist_matrix, "stage1")
        self.timing['cluster1_cpu'] = time.process_time() - cluster1_cpu_start
        self.timing['cluster1_clock'] = time.time() - cluster1_clock_start
        self.logger.info(
            f"After clustering: {len(self.min_entries)} MIN structures"
        )
        self.output.snapshot_opt_min("scan")
        self._scan_poured_log = scan_poured_log
        self._scan_old_to_new = old_to_new_map

    # ----------------------------------------------------------------
    # Stage 2: Advanced Optimization
    # ----------------------------------------------------------------
    def _pre_filter_for_advopt(self, max_conformers=3):
        """Group structures by graph isomorphism and select representatives.

        For each constitutional isomer group, keep only the lowest-energy
        conformers (up to max_conformers) to avoid redundant DFT calculations.

        Returns:
            set of indices to process (reference always included)
        """
        n = len(self.min_entries)
        if n <= 1:
            return set(range(n))

        # Build graph isomorphism groups
        groups = []  # list of lists of indices
        group_of = {}  # index -> group_id

        for i, entry in enumerate(self.min_entries):
            if entry.is_reference:
                # Reference always gets its own initial group
                gid = len(groups)
                groups.append([i])
                group_of[i] = gid
                continue
            matched = False
            for gid, members in enumerate(groups):
                rep = members[0]
                rep_mol = self.min_entries[rep].mol
                if check_graph_isomorphic(
                    entry.mol.elements, entry.mol.coords,
                    rep_mol.elements, rep_mol.coords,
                    heavy_only=False
                ):
                    groups[gid].append(i)
                    group_of[i] = gid
                    matched = True
                    break
            if not matched:
                gid = len(groups)
                groups.append([i])
                group_of[i] = gid

        # Select representatives: lowest xTB energy per group
        selected = set()
        for gid, members in enumerate(groups):
            members_sorted = sorted(
                members,
                key=lambda idx: (
                    self.min_entries[idx].xtb_energy
                    if self.min_entries[idx].xtb_energy is not None
                    else float('inf')
                )
            )
            for idx in members_sorted[:max_conformers]:
                selected.add(idx)

        # Always include reference (required for DFT reference energy)
        for i, entry in enumerate(self.min_entries):
            if entry.is_reference:
                selected.add(i)
                break

        n_groups = len(groups)
        n_skipped = n - len(selected)
        self.logger.info(
            f"Pre-DFT filter: {n} structures -> {n_groups} isomer groups, "
            f"{len(selected)} selected ({max_conformers} conformers/group), "
            f"{n_skipped} skipped"
        )
        return selected

    def _run_stage_advopt(self):
        if self.advopt_interface is None or len(self.min_entries) == 0:
            self.logger.section("Stage 2: Skipped (no program configured)")
            self.advopt_ref_energy = self.reference_energy
            return
        self.logger.section(
            f"Stage 2: Advanced Optimization "
            f"({self.advopt_interface.program})"
        )
        self.logger.info(f"Method: {self.params.advopt_method}")
        n_within = sum(
            1 for e in self.min_entries
            if e.is_reference or not e.xtb_energy
            or (e.xtb_energy - self.reference_energy) * 627.509
            <= self.params.advopt_filter
        )
        self.logger.info(
            f"Input: {len(self.min_entries)} MIN structures, "
            f"{n_within} within AdvOptFilter "
            f"({self.params.advopt_filter:.0f} kcal/mol)"
        )
        # Pre-filter: group by graph isomorphism, keep top conformers
        selected_indices = self._pre_filter_for_advopt(max_conformers=3)
        # Count actual DFT jobs (selected + within energy window)
        n_dft_jobs = sum(
            1 for i, e in enumerate(self.min_entries)
            if i in selected_indices and (
                e.is_reference or not e.xtb_energy
                or (e.xtb_energy - self.reference_energy) * 627.509
                <= self.params.advopt_filter
            )
        )
        self.logger.info(f"DFT jobs to run: {n_dft_jobs}")
        advopt_cpu_start = time.process_time()
        advopt_clock_start = time.time()
        advopt_results = []
        advopt_aborted = False
        consecutive_failures = 0
        MAX_CONSECUTIVE_FAILURES = 3
        n_skipped = 0
        n_prefiltered = 0
        dft_job_num = 0
        window = self.params.advopt_filter

        for i, entry in enumerate(self.min_entries):
            if self._check_stop():
                self._save_checkpoint('ADVOPT', 0)
                break
            dE_xtb = (
                (entry.xtb_energy - self.reference_energy) * 627.509
                if entry.xtb_energy else 0.0
            )
            tag = " (reference)" if entry.is_reference else ""
            if not entry.is_reference and dE_xtb > window:
                n_skipped += 1
                self.logger.info(
                    f"  AdvOpt MIN {i}{tag} "
                    f"(xTB dE={dE_xtb:.1f} kcal/mol) -> SKIPPED "
                    f"(above {window:.0f} kcal/mol window)"
                )
                continue
            if i not in selected_indices:
                n_prefiltered += 1
                continue
            dft_job_num += 1
            self.logger.info(
                f"  AdvOpt MIN {i}{tag} [{dft_job_num}/{n_dft_jobs}] "
                f"(xTB dE={dE_xtb:.1f} kcal/mol)..."
            )
            opt_mol, energy, success = self.advopt_interface.optimize(
                entry.mol, label=f"MIN_{i:03d}"
            )
            if success and opt_mol is not None and energy is not None:
                consecutive_failures = 0
                if opt_mol.n_atoms == entry.mol.n_atoms:
                    for j, atom in enumerate(opt_mol.atoms):
                        atom.original_index = entry.mol.atoms[j].original_index
                    opt_mol.sort_by_original_index()
                opt_mol.energy = energy
                opt_mol.charge = self.mol.charge
                opt_mol.multiplicity = self.mol.multiplicity
                advopt_results.append((i, opt_mol, energy))
                if entry.is_reference:
                    self.advopt_ref_energy = energy
                    self.logger.info(
                        f"    -> E={energy:.10f} Eh (DFT reference)"
                    )
                else:
                    if self.advopt_ref_energy is not None:
                        dE_dft = (energy - self.advopt_ref_energy) * 627.509
                        self.logger.info(
                            f"    -> E={energy:.10f} Eh, "
                            f"dE(DFT)={dE_dft:.1f} kcal/mol"
                        )
                    else:
                        self.logger.info(f"    -> E={energy:.10f} Eh")
            else:
                consecutive_failures += 1
                self.logger.info("    -> AdvOpt failed")
                if entry.is_reference:
                    self.logger.info(
                        "  ABORT: Reference structure optimization failed."
                    )
                    advopt_aborted = True
                    break
                if (consecutive_failures >= MAX_CONSECUTIVE_FAILURES
                        and len(advopt_results) <= 1):
                    self.logger.info(
                        f"  ABORT: {MAX_CONSECUTIVE_FAILURES} "
                        f"consecutive failures."
                    )
                    advopt_aborted = True
                    break

        if n_skipped > 0:
            self.logger.info(
                f"  Energy window filter: {n_skipped} structures skipped "
                f"(>{window:.0f} kcal/mol above reference)"
            )
        if n_prefiltered > 0:
            self.logger.info(
                f"  Pre-DFT isomer filter: {n_prefiltered} conformers skipped "
                f"(excess conformers within same isomer group)"
            )
        self.timing['advopt_cpu'] = time.process_time() - advopt_cpu_start
        self.timing['advopt_clock'] = time.time() - advopt_clock_start

        # Re-cluster DFT results
        old_s2_entries = list(self.min_entries)
        if not advopt_aborted and len(advopt_results) > 1:
            self.logger.section("Stage 2: Clustering DFT results")
            cluster2_cpu_start = time.process_time()
            cluster2_clock_start = time.time()
            advopt_mols = [mol for _, mol, _ in advopt_results]
            clusters, dist_matrix = self.clusterer.cluster(
                advopt_mols, self.logger
            )
            representatives = self.clusterer.select_representatives(
                advopt_mols, clusters
            )
            rep_set = set(idx for idx, _ in representatives)
            ref_entry = None
            other_entries = []
            for rep_idx, rep_mol in representatives:
                orig_idx, _, dft_energy = advopt_results[rep_idx]
                old_entry = (
                    self.min_entries[orig_idx]
                    if orig_idx < len(self.min_entries) else None
                )
                new_entry = MINEntry(
                    mol=rep_mol,
                    preopt_mol=old_entry.preopt_mol if old_entry else None,
                    xtb_mol=old_entry.xtb_mol if old_entry else None,
                    xtb_energy=old_entry.xtb_energy if old_entry else None,
                    dft_energy=dft_energy,
                    is_reference=(
                        old_entry.is_reference if old_entry else False
                    ),
                    source=old_entry.source if old_entry else "",
                )
                if new_entry.is_reference:
                    ref_entry = new_entry
                else:
                    other_entries.append(new_entry)
            new_entries = []
            if ref_entry:
                new_entries.append(ref_entry)
            other_entries.sort(
                key=lambda e: e.dft_energy
                if e.dft_energy is not None else float('inf')
            )
            new_entries.extend(other_entries)
            if self.params.debug:
                for k in range(len(advopt_results)):
                    if k not in rep_set:
                        _, mol, _ = advopt_results[k]
                        self.output.save_poured(
                            mol, self.poured_count, "advopt",
                            f"Energy={mol.energy:.10f}"
                            if mol.energy else "",
                        )
                        self.poured_count += 1
            self.min_entries = new_entries
            advopt_orig_set = {
                orig_idx for orig_idx, _, _ in advopt_results
            }
            survived_orig = {}
            for rep_idx, _ in representatives:
                survived_orig[advopt_results[rep_idx][0]] = (
                    advopt_results[rep_idx][2]
                )
            orig_to_new = {}
            for new_i, ne in enumerate(new_entries):
                for o_idx, dft_e in survived_orig.items():
                    if dft_e == ne.dft_energy and o_idx not in orig_to_new:
                        orig_to_new[o_idx] = new_i
                        break
            map_lines = ["== Stage 2: Clustering =="]
            for old_idx in range(len(old_s2_entries)):
                old_e = old_s2_entries[old_idx].xtb_energy
                e_str = f"E(xTB)={old_e:.10f}" if old_e else "E=N/A"
                if old_idx in orig_to_new:
                    new_i = orig_to_new[old_idx]
                    dft_e = new_entries[new_i].dft_energy
                    map_lines.append(
                        f"  old {old_idx:3d} -> new {new_i:3d}  "
                        f"({e_str}, E(DFT)={dft_e:.10f})"
                    )
                elif old_idx in advopt_orig_set:
                    map_lines.append(
                        f"  old {old_idx:3d} -> REMOVED  "
                        f"({e_str}, cluster2)"
                    )
                else:
                    map_lines.append(
                        f"  old {old_idx:3d} -> REMOVED  "
                        f"({e_str}, skipped/failed)"
                    )
            map_lines.append("")
            self.output.write_log(
                "SFRIS_index.map",
                "\n".join(map_lines) + "\n", append=True
            )
            s2_remap = {}
            s2_reasons = {}
            for old_idx in range(len(old_s2_entries)):
                if old_idx in orig_to_new:
                    s2_remap[old_idx] = orig_to_new[old_idx]
                else:
                    s2_remap[old_idx] = None
                    if old_idx in advopt_orig_set:
                        s2_reasons[old_idx] = "cluster2"
                    else:
                        s2_reasons[old_idx] = "skipped/failed"
            self._remap_history.append(
                ("Stage 2: Clustering", s2_remap, s2_reasons)
            )
            self._write_rmsd_matrix(dist_matrix, "stage2")
            self.timing['cluster2_cpu'] = (
                time.process_time() - cluster2_cpu_start
            )
            self.timing['cluster2_clock'] = (
                time.time() - cluster2_clock_start
            )
        elif not advopt_aborted and advopt_results:
            _, opt_mol, dft_energy = advopt_results[0]
            self.min_entries[0].mol = opt_mol
            self.min_entries[0].dft_energy = dft_energy

        for i, entry in enumerate(self.min_entries):
            best_e = (
                entry.dft_energy if entry.dft_energy is not None
                else entry.xtb_energy
            )
            self.output.save_opt_min(
                entry.mol, i, f"Energy={best_e:.10f}" if best_e else ""
            )
            self._write_path_file(i, entry)
        self.output.clean_opt_min_dir(len(self.min_entries))
        self.output.clean_path_files(len(self.min_entries))
        self._write_min_lis()
        self.logger.info(
            f"Stage 2 result: {len(self.min_entries)} MIN structures"
        )
        self.output.snapshot_opt_min("dft")

    # ----------------------------------------------------------------
    # NMA
    # ----------------------------------------------------------------
    def _run_stage_nma(self):
        if self.advopt_interface is None:
            self.logger.info(
                "NMA requested but no AdvOpt program available. Skipping."
            )
            return
        self.logger.section("NMA: Normal-mode Analysis")
        nma_cpu_start = time.process_time()
        nma_clock_start = time.time()

        for i, entry in enumerate(self.min_entries):
            if self._check_stop():
                self._save_checkpoint('NMA', 0)
                break
            has_cache = id(entry.mol) in self.advopt_interface._freq_cache
            cache_tag = " (from TightOpt cache)" if has_cache else ""
            self.logger.info(f"  NMA on MIN {i}{cache_tag}...")
            zpve, freqs, n_imag, success = self.advopt_interface.run_nma(
                entry.mol, label=f"MIN_{i:03d}"
            )
            if success and zpve is not None:
                entry.zpve = zpve
                entry.frequencies = freqs
                if freqs:
                    entry.n_imaginary = sum(1 for f in freqs if f < 0)
                else:
                    entry.n_imaginary = n_imag
                lowest = min(freqs) if freqs else 0.0
                self.logger.info(
                    f"    ZPVE={zpve:.6f} Eh, "
                    f"modes={len(freqs) if freqs else 0}, "
                    f"lowest={lowest:.1f} cm-1, "
                    f"n_imag={entry.n_imaginary}"
                    f"{' *** TS ***' if entry.n_imaginary > 0 else ''}"
                )
            else:
                self.logger.info(f"    NMA failed for MIN {i}")

        # Filter out structures with imaginary frequencies
        filtered = []
        removed = []
        for i, entry in enumerate(self.min_entries):
            if entry.n_imaginary > 0:
                imag_freqs = [
                    f for f in (entry.frequencies or []) if f < 0
                ]
                imag_str = ", ".join(f"{f:.1f}" for f in imag_freqs)
                removed.append((i, entry))
                self.logger.info(
                    f"  MIN {i}: REMOVED ({entry.n_imaginary} imaginary "
                    f"freq [{imag_str}] cm-1 -> not a true minimum)"
                )
            else:
                filtered.append(entry)

        if removed:
            if self.params.debug:
                for orig_idx, entry in removed:
                    best_e = (
                        entry.dft_energy if entry.dft_energy is not None
                        else entry.xtb_energy
                    )
                    imag_freqs = [
                        f for f in (entry.frequencies or []) if f < 0
                    ]
                    imag_str = ",".join(f"{f:.1f}" for f in imag_freqs)
                    comment = (
                        f"Energy={best_e:.10f} n_imag={entry.n_imaginary} "
                        f"imag_freq=[{imag_str}]"
                        if best_e else
                        f"n_imag={entry.n_imaginary} "
                        f"imag_freq=[{imag_str}]"
                    )
                    self.output.save_poured(
                        entry.mol, self.poured_count,
                        f"nma_MIN_{orig_idx:03d}", comment,
                    )
                    self.poured_count += 1
                self.logger.info(
                    f"  DEBUG: {len(removed)} removed structures "
                    f"saved to pouredMIN/"
                )

            old_nma_entries = list(self.min_entries)
            self.min_entries = filtered
            removed_reasons = {
                orig_idx: f"n_imag={entry.n_imaginary}"
                for orig_idx, entry in removed
            }
            self._log_index_remap(
                "NMA: Imaginary freq filter",
                old_nma_entries, self.min_entries, removed_reasons
            )
            self.logger.info(
                f"  Imaginary freq filter: {len(removed)} structures "
                f"removed, {len(self.min_entries)} remain"
            )
            for i, entry in enumerate(self.min_entries):
                best_e = (
                    entry.dft_energy if entry.dft_energy is not None
                    else entry.xtb_energy
                )
                self.output.save_opt_min(
                    entry.mol, i,
                    f"E={best_e:.10f}" if best_e else ""
                )
            self.output.clean_opt_min_dir(len(self.min_entries))
            self.output.clean_path_files(len(self.min_entries))

        # Frequency fingerprint deduplication
        old_freq_entries = list(self.min_entries)
        freq_dedup_removed = self._freq_dedup(
            freq_tol=5.0, energy_tol=0.5
        )
        if freq_dedup_removed:
            removed_reasons = {
                orig_idx: (
                    f"freq_dup_of_"
                    f"{self._freq_dup_pairs.get(orig_idx, '?')}"
                )
                for orig_idx, _ in freq_dedup_removed
            }
            self._log_index_remap(
                "NMA: Frequency dedup",
                old_freq_entries, self.min_entries, removed_reasons
            )
            self.logger.info(
                f"  Frequency dedup: {len(freq_dedup_removed)} duplicates "
                f"removed, {len(self.min_entries)} remain"
            )
            for i, entry in enumerate(self.min_entries):
                best_e = (
                    entry.dft_energy if entry.dft_energy is not None
                    else entry.xtb_energy
                )
                self.output.save_opt_min(
                    entry.mol, i,
                    f"E={best_e:.10f}" if best_e else ""
                )
            self.output.clean_opt_min_dir(len(self.min_entries))
            self.output.clean_path_files(len(self.min_entries))

        for i, entry in enumerate(self.min_entries):
            self._write_path_file(i, entry)
        self.timing['nma_cpu'] = time.process_time() - nma_cpu_start
        self.timing['nma_clock'] = time.time() - nma_clock_start
        self._write_min_lis()
        self.logger.info("NMA complete.")
        self.output.snapshot_opt_min("nma")

    def _freq_dedup(self, freq_tol: float = 5.0,
                    energy_tol: float = 0.5,
                    rmsd_tol: float = 0.3) -> list:
        """Remove duplicate structures using three-gate filter.

        Gate order:
          Gate 1: dE < energy_tol (kcal/mol)
          Gate 2: RMSD(Hungarian+Kabsch) < rmsd_tol
          Gate 3: Frequency fingerprint + Symmetry RMSD

        Energy safety net (Gate 3a) triggers graph isomorphism check
        when dE < TIGHT_ENERGY_TOL.
        """
        STRICT_FREQ_TOL = 3.0
        STRICT_ENERGY_TOL = 0.2
        TIGHT_ENERGY_TOL = 0.2
        energy_tol_eh = energy_tol / 627.509
        n = len(self.min_entries)
        if n < 2:
            self._freq_dup_pairs = {}
            return []

        to_remove = set()
        dup_pairs = {}
        for i in range(n):
            if i in to_remove:
                continue
            ei = self.min_entries[i]
            fi = ei.frequencies
            e_i = (
                ei.dft_energy if ei.dft_energy is not None
                else ei.xtb_energy
            )

            for j in range(i + 1, n):
                if j in to_remove:
                    continue
                ej = self.min_entries[j]
                fj = ej.frequencies
                e_j = (
                    ej.dft_energy if ej.dft_energy is not None
                    else ej.xtb_energy
                )

                # Gate 1: Energy pre-filter
                if e_i is not None and e_j is not None:
                    de_eh = abs(e_i - e_j)
                    if de_eh > energy_tol_eh:
                        continue
                else:
                    de_eh = 0.0
                de_kcal = de_eh * 627.509

                # Gate 2: RMSD (Hungarian + Kabsch)
                hung_rmsd = self.clusterer._calculate_rmsd_kabsch(
                    ei.mol, ej.mol, heavy_only=True
                )
                if hung_rmsd < rmsd_tol:
                    tier = (
                        "strict"
                        if de_kcal < STRICT_ENERGY_TOL
                        else "moderate"
                    )
                    to_remove.add(j)
                    dup_pairs[j] = i
                    self.logger.info(
                        f"  MIN {j}: DUPLICATE of MIN {i} "
                        f"({tier}: RMSD={hung_rmsd:.3f} A [hungarian], "
                        f"dE={de_kcal:.3f} kcal/mol)"
                    )
                    continue

                # Gate 3a: Energy safety net
                if de_kcal < TIGHT_ENERGY_TOL:
                    is_iso = check_graph_isomorphic(
                        ei.mol.elements, ei.mol.coords,
                        ej.mol.elements, ej.mol.coords,
                        heavy_only=False
                    )
                    if is_iso:
                        to_remove.add(j)
                        dup_pairs[j] = i
                        self.logger.info(
                            f"  MIN {j}: DUPLICATE of MIN {i} "
                            f"(energy_safety: graph_isomorphic, "
                            f"dE={de_kcal:.3f} kcal/mol)"
                        )
                        continue
                    else:
                        self.logger.info(
                            f"  MIN {i} vs {j}: "
                            f"dE={de_kcal:.3f} kcal/mol "
                            f"but different graph -> different structures"
                        )
                        continue

                # Gate 3b: Frequency fingerprint + Symmetry RMSD
                if fi is None or fj is None or len(fi) != len(fj):
                    continue
                fi_sorted = sorted(fi)
                fj_sorted = sorted(fj)
                max_fdiff = max(
                    abs(a - b) for a, b in zip(fi_sorted, fj_sorted)
                )
                if max_fdiff >= freq_tol:
                    continue

                final_rmsd, method = compute_rmsd_with_symmetry_fallback(
                    ei.mol.elements, ei.mol.coords,
                    ej.mol.elements, ej.mol.coords,
                    hungarian_rmsd=hung_rmsd, threshold=rmsd_tol,
                    heavy_only=True, mirror=True
                )
                if final_rmsd < rmsd_tol:
                    tier = (
                        "strict"
                        if max_fdiff < STRICT_FREQ_TOL
                        and de_kcal < STRICT_ENERGY_TOL
                        else "moderate"
                    )
                    to_remove.add(j)
                    dup_pairs[j] = i
                    self.logger.info(
                        f"  MIN {j}: DUPLICATE of MIN {i} "
                        f"({tier}: freq={max_fdiff:.1f} cm-1, "
                        f"RMSD={final_rmsd:.3f} A [{method}], "
                        f"dE={de_kcal:.3f} kcal/mol)"
                    )
                else:
                    self.logger.info(
                        f"  MIN {i} vs {j}: freq match "
                        f"({max_fdiff:.1f} cm-1, "
                        f"dE={de_kcal:.3f}) "
                        f"but RMSD={final_rmsd:.3f} A [{method}] "
                        f"-> different structures"
                    )

        if not to_remove:
            self._freq_dup_pairs = dup_pairs
            return []

        removed = [
            (idx, self.min_entries[idx]) for idx in sorted(to_remove)
        ]

        if self.params.debug:
            for orig_idx, entry in removed:
                best_e = (
                    entry.dft_energy if entry.dft_energy is not None
                    else entry.xtb_energy
                )
                dup_of = dup_pairs.get(orig_idx, "?")
                comment = (
                    f"Energy={best_e:.10f} FREQ_DUP_OF_MIN_{dup_of}"
                    if best_e
                    else f"FREQ_DUP_OF_MIN_{dup_of}"
                )
                self.output.save_poured(
                    entry.mol, self.poured_count,
                    f"freqdup_MIN_{orig_idx:03d}", comment,
                )
                self.poured_count += 1
            self.logger.info(
                f"  DEBUG: {len(removed)} freq-duplicates "
                f"saved to pouredMIN/"
            )

        self.min_entries = [
            e for i, e in enumerate(self.min_entries)
            if i not in to_remove
        ]
        self._freq_dup_pairs = dup_pairs
        return removed

    # ================================================================
    # Index mapping
    # ================================================================
    def _init_index_map(self):
        content = (
            "# SFRIS Index Mapping\n"
            "# Tracks MIN re-indexing throughout the pipeline\n"
            f"# Input: {self.input_file}\n"
            f"# Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        )
        self.output.write_log("SFRIS_index.map", content)
        self._scan_poured_log = []
        self._scan_old_to_new = {}
        self._remap_history = []

    def _log_index_snapshot(self, phase: str, entries: list):
        lines = [f"== {phase} =="]
        for i, entry in enumerate(entries):
            best_e = (
                entry.dft_energy if entry.dft_energy is not None
                else entry.xtb_energy
            )
            e_str = f"E={best_e:.10f}" if best_e else "E=N/A"
            src = entry.source or ""
            tag = " (reference)" if entry.is_reference else ""
            lines.append(f"  MIN {i:3d}: {e_str}  {src}{tag}")
        lines.append("")
        self.output.write_log(
            "SFRIS_index.map", "\n".join(lines) + "\n", append=True
        )

    def _log_index_remap(self, phase: str, old_entries: list,
                         new_entries: list, removed_reasons: dict = None):
        if removed_reasons is None:
            removed_reasons = {}
        new_id_map = {id(e): i for i, e in enumerate(new_entries)}
        remap_dict = {}
        lines = [f"== {phase} =="]
        for old_idx, entry in enumerate(old_entries):
            best_e = (
                entry.dft_energy if entry.dft_energy is not None
                else entry.xtb_energy
            )
            e_str = f"E={best_e:.10f}" if best_e else "E=N/A"
            new_idx = new_id_map.get(id(entry))
            remap_dict[old_idx] = new_idx
            if new_idx is not None:
                arrow = f"-> new {new_idx:3d}"
            else:
                reason = removed_reasons.get(old_idx, "removed")
                arrow = f"-> REMOVED ({reason})"
            lines.append(f"  old {old_idx:3d} {arrow}  {e_str}")
        lines.append("")
        self.output.write_log(
            "SFRIS_index.map", "\n".join(lines) + "\n", append=True
        )
        self._remap_history.append(
            (phase, remap_dict, dict(removed_reasons))
        )

    def _write_scan_poured_log(self):
        if not self._scan_poured_log:
            return
        post_c1_remaps = [
            (phase, rmap, reasons)
            for phase, rmap, reasons in self._remap_history
            if "Stage 1" not in phase
        ]
        n_optmin = max(self._scan_old_to_new.values(), default=-1) + 1
        optmin_fate = {}
        for opt_idx in range(n_optmin):
            current = opt_idx
            removed_phase = None
            for phase, rmap, reasons in post_c1_remaps:
                if current not in rmap or rmap[current] is None:
                    reason = reasons.get(current, "removed")
                    phase_short = {
                        "Stage 2: Clustering": "s2",
                        "NMA: Imaginary freq filter": "nma_imag",
                        "NMA: Frequency dedup": "nma_freq",
                    }.get(phase, phase)
                    removed_phase = f"{phase_short}:{reason}"
                    current = None
                    break
                current = rmap[current]
            optmin_fate[opt_idx] = (current, removed_phase)
        lines = [
            f"== Stage 1: SCAN duplicates "
            f"({len(self._scan_poured_log)} total) =="
        ]
        for p in self._scan_poured_log:
            old_dup = p['dup_of']
            e_str = (
                f"E={p['energy']:.10f}" if p['energy'] else "E=N/A"
            )
            f_str = (
                f"  file={p['poured_file']}" if p['poured_file'] else ""
            )
            opt_idx = self._scan_old_to_new.get(old_dup)
            if opt_idx is None:
                ref_str = f"DupOf old {old_dup} (REMOVED by clustering)"
                fate_str = ""
            else:
                final_idx, removed_phase = optmin_fate.get(
                    opt_idx, (None, None)
                )
                if final_idx is not None:
                    fate_str = f" -> final MIN_{final_idx:03d}"
                elif removed_phase:
                    fate_str = f" -> REMOVED({removed_phase})"
                else:
                    fate_str = " -> REMOVED"
                ref_str = (
                    f"DupOf old {old_dup} -> "
                    f"optMIN_{opt_idx:03d}{fate_str}"
                )
            lines.append(
                f"  Iter {p['iter']:3d}: {ref_str}  "
                f"{e_str}  cut=[{p['cut']}]{f_str}"
            )
        lines.append("")
        self.output.write_log(
            "SFRIS_index.map", "\n".join(lines) + "\n", append=True
        )

    # ================================================================
    # Constitutional isomer grouping
    # ================================================================
    def _group_constitutional_isomers(self):
        n = len(self.min_entries)
        if n == 0:
            return []
        parent = list(range(n))

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(a, b):
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[ra] = rb

        for i in range(n):
            ei = self.min_entries[i]
            for j in range(i + 1, n):
                ej = self.min_entries[j]
                if check_graph_isomorphic(
                    ei.mol.elements, ei.mol.coords,
                    ej.mol.elements, ej.mol.coords,
                    heavy_only=False
                ):
                    union(i, j)
        from collections import defaultdict
        groups = defaultdict(list)
        for i in range(n):
            groups[find(i)].append(i)
        sorted_groups = sorted(
            groups.values(), key=lambda members: min(members)
        )
        # Generate isomer labels: digit+letter+digit (e.g. 3A7)
        labels = self._generate_isomer_labels(len(sorted_groups))
        result = [None] * n
        for g_idx, members in enumerate(sorted_groups):
            label = labels[g_idx]
            # Representative = lowest energy in group
            rep = min(
                members,
                key=lambda idx: (
                    self.min_entries[idx].dft_energy
                    if self.min_entries[idx].dft_energy is not None
                    else self.min_entries[idx].xtb_energy
                    if self.min_entries[idx].xtb_energy is not None
                    else float('inf')
                )
            )
            for m in members:
                result[m] = {
                    'group': label,
                    'representative': rep,
                    'is_representative': (m == rep),
                }
        self._isomer_groups = result
        n_groups = len(sorted_groups)
        n_conformers = n - n_groups
        self.logger.info(
            f"Constitutional isomer grouping: {n} MIN -> "
            f"{n_groups} isomers, {n_conformers} conformers"
        )
        lines = [
            f"== Constitutional isomer grouping "
            f"({n_groups} isomers) =="
        ]
        for g_idx, members in enumerate(sorted_groups):
            label = labels[g_idx]
            member_str = ", ".join(f"MIN {m}" for m in members)
            lines.append(f"  Isomer {label}: {member_str}")
        lines.append("")
        self.output.write_log(
            "SFRIS_index.map", "\n".join(lines) + "\n", append=True
        )
        return result

    @staticmethod
    def _generate_isomer_labels(n_groups):
        """Generate unique isomer labels in digit+letter+digit format.

        Uses deterministic pseudo-random assignment for the first digit
        and letter, with a sequential third digit to ensure uniqueness.
        Example labels: 3A0, 7K1, 2B2, ...
        """
        import hashlib
        labels = []
        used = set()
        seq = 0
        # Deterministic seed from group count for reproducibility
        seed = int(hashlib.md5(str(n_groups).encode()).hexdigest()[:8], 16)
        rng = np.random.RandomState(seed)
        for _ in range(n_groups):
            # Try to generate a unique label
            for _attempt in range(1000):
                d1 = rng.randint(0, 10)
                letter = chr(ord('A') + rng.randint(0, 26))
                label = f"{d1}{letter}{seq}"
                if label not in used:
                    used.add(label)
                    labels.append(label)
                    seq += 1
                    break
            else:
                # Fallback: guaranteed unique
                labels.append(f"X{seq:03d}")
                seq += 1
        return labels

    # ================================================================
    # Output file writers
    # ================================================================
    def _write_summary(self):
        ref_e_str = (
            f"{self.reference_energy:.10f}"
            if self.reference_energy is not None else "N/A"
        )
        advopt_ref_str = (
            f"{self.advopt_ref_energy:.10f}"
            if self.advopt_ref_energy is not None else "N/A"
        )
        advopt_ref_line = (
            f"Reference (DFT)   : {advopt_ref_str} Eh\n"
            if any(e.dft_energy is not None for e in self.min_entries)
            else ""
        )
        content = (
            f"# Generated by SFRIS {SFRIS_VERSION} (https://halsten.pcmq.net/sfris)\n"
            "# ============================================================\n"
            "# SFRIS Summary\n"
            "# ============================================================\n"
            f"# Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
            "# ============================================================\n\n"
            f"Input             : {self.input_file}\n"
            f"Formula           : {self.mol.formula}\n"
            f"Atoms             : {self.mol.n_atoms}\n"
            f"Reference (xTB)   : {ref_e_str} Eh\n"
            f"{advopt_ref_line}\n"
            "Pipeline Results:\n"
            f"  MIN structures  : {len(self.min_entries)}\n"
        )
        if self._isomer_groups:
            n_groups = len(set(g['group'] for g in self._isomer_groups))
            n_conformers = len(self.min_entries) - n_groups
            content += (
                f"  Isomers         : {n_groups}\n"
                f"  Conformers      : {n_conformers}\n"
            )
        content += (
            f"  NMA             : "
            f"{'enabled' if self.params.nma else 'disabled'}\n"
            f"  DEBUG           : "
            f"{'enabled' if self.params.debug else 'disabled'}\n"
            f"  pouredMIN       : {self.poured_count}\n\n"
        )
        eq_ref = (
            self.advopt_ref_energy
            if self.advopt_ref_energy is not None
            else self.reference_energy
        )
        eq_ref_label = (
            "DFT"
            if any(e.dft_energy is not None for e in self.min_entries)
            else "xTB"
        )
        if self.min_entries:
            content += (
                f"Energy summary (relative to {eq_ref_label} reference, "
                f"kcal/mol):\n"
            )
            content += "-" * 55 + "\n"
            for i, entry in enumerate(self.min_entries):
                delta_e = self._delta_e(entry, eq_ref, eq_ref_label)
                if delta_e is not None:
                    tag = " (ref)" if entry.is_reference else ""
                    grp_str = ""
                    if (self._isomer_groups
                            and i < len(self._isomer_groups)):
                        g = self._isomer_groups[i]
                        if g['is_representative']:
                            grp_str = f"  Isomer {g['group']}"
                        else:
                            grp_str = (
                                f"  Isomer {g['group']} "
                                f"(=MIN {g['representative']})"
                            )
                    content += (
                        f"  MIN {i:3d}: {delta_e:10.2f} kcal/mol"
                        f"{tag}{grp_str}\n"
                    )
        content += (
            "\n# ============================================"
            "================\n"
        )
        self.output.write_log("SFRIS.sum", content)

    def _write_rmsd_matrix(self, dist_matrix, stage=""):
        n = dist_matrix.shape[0] if dist_matrix.ndim == 2 else 0
        if n == 0:
            return
        suffix = f"_{stage}" if stage else ""
        heavy_str = (
            "heavy-atom" if self.params.cluster_heavy_only else "all-atom"
        )
        content = f"# SFRIS RMSD Distance Matrix ({stage})\n"
        content += f"# Structures: {n}, Type: {heavy_str}\n\n"
        content += "        "
        for j in range(n):
            content += f"  {j:<8d}"
        content += "\n"
        for i in range(n):
            content += f"{i:<8d}"
            for j in range(n):
                if i == j:
                    content += "      -   "
                elif np.isinf(dist_matrix[i, j]):
                    content += "     inf  "
                else:
                    content += f"  {dist_matrix[i, j]:8.4f}"
            content += "\n"
        self.output.write_log(
            f"{self.mol_name}_rmsd{suffix}.mrx", content
        )

    def _write_bonds_file(self):
        content = (
            f"# SFRIS Bond Analysis\n"
            f"# Input: {self.input_file}, "
            f"Formula: {self.mol.formula}\n\n"
        )
        content += "# ID    Atom1   Atom2   Type      Priority   Status\n"
        for i, bond in enumerate(self.mol.bonds):
            atom1 = self.mol.atoms[bond.atom1]
            atom2 = self.mol.atoms[bond.atom2]
            bond_type = {
                1: "single", 2: "double", 3: "triple"
            }.get(bond.order, "?")
            status = "cuttable" if bond.cuttable else "forbidden"
            content += (
                f"{i+1:4d}    {atom1.element}{bond.atom1:<4d} "
                f"{atom2.element}{bond.atom2:<4d} {bond_type:8s}  "
                f"{bond.priority:.2f}       {status}\n"
            )
        self.output.write_log(f"{self.mol_name}_bonds.bnd", content)

    def _write_cuts_file(self):
        from itertools import combinations as comb
        content = (
            f"# SFRIS Cut Combination Index\n"
            f"# Max cuts: {self.params.max_cuts}\n\n"
        )
        cuttable_bonds = [
            (b.atom1, b.atom2) for b in self.mol.bonds if b.cuttable
        ]
        combo_id = 0
        max_cuts = min(self.params.max_cuts, len(cuttable_bonds))
        for n_cuts in range(1, max_cuts + 1):
            for cut_combo in comb(cuttable_bonds, n_cuts):
                combo_id += 1
                cut_str = ", ".join(
                    [f"{self.mol.atoms[a].element}{a}-"
                     f"{self.mol.atoms[b].element}{b}"
                     for a, b in cut_combo]
                )
                content += f"# {combo_id:3d} | {cut_str}\n"
        content += f"\n# Total combinations: {combo_id}\n"
        self.output.write_log(f"{self.mol_name}_cuts.frag", content)

    def _write_timing_file(self):
        if self.timing['start_time'] and self.timing['end_time']:
            total_clock = (
                self.timing['end_time'] - self.timing['start_time']
            ).total_seconds()
        else:
            total_clock = 0.0
        total_cpu = sum(
            self.timing[k] for k in self.timing if k.endswith('_cpu')
        )
        n_iter = max(self.timing['iterations'], 1)
        content = (
            "# SFRIS Timing Statistics\n"
            "# Phase                    CPU Time (s)    Clock Time (s)\n"
            f"Setup                      "
            f"{self.timing['setup_cpu']:12.2f}    "
            f"{self.timing['setup_clock']:12.2f}\n"
            f"Stage 1 (SCAN)             "
            f"{self.timing['scan_cpu']:12.2f}    "
            f"{self.timing['scan_clock']:12.2f}\n"
            f"  - xTB Optimization       "
            f"{self.timing['scan_opt_cpu']:12.2f}    "
            f"{self.timing['scan_opt_clock']:12.2f}\n"
            f"Stage 1 Clustering         "
            f"{self.timing['cluster1_cpu']:12.2f}    "
            f"{self.timing['cluster1_clock']:12.2f}\n"
            f"Stage 2 (ADVOPT)           "
            f"{self.timing['advopt_cpu']:12.2f}    "
            f"{self.timing['advopt_clock']:12.2f}\n"
            f"Stage 2 Clustering         "
            f"{self.timing['cluster2_cpu']:12.2f}    "
            f"{self.timing['cluster2_clock']:12.2f}\n"
            f"NMA                        "
            f"{self.timing['nma_cpu']:12.2f}    "
            f"{self.timing['nma_clock']:12.2f}\n"
            f"Total                      "
            f"{total_cpu:12.2f}    "
            f"{total_clock:12.2f}\n\n"
            f"# Iterations: {self.timing['iterations']}, "
            f"Avg: {self.timing['scan_cpu']/n_iter:.3f} s/iter\n"
            f"# Start: "
            f"{self.timing['start_time'].strftime('%Y-%m-%d %H:%M:%S') if self.timing['start_time'] else 'N/A'}\n"
            f"# End:   "
            f"{self.timing['end_time'].strftime('%Y-%m-%d %H:%M:%S') if self.timing['end_time'] else 'N/A'}\n"
        )
        self.output.write_log(f"{self.mol_name}_timing.tim", content)

    def _init_placement_file(self):
        content = (
            f"# SFRIS Placement Log\n"
            f"# Gap fractions: {DEFAULT_GAP_FRACTIONS}\n\n"
        )
        self.output.write_log(f"{self.mol_name}_place.plc", content)

    def _append_placement_entry(self, entry):
        line = (
            f"# Iter {entry.get('iteration', '?')}: "
            f"{entry.get('status', '?')}"
        )
        line += (
            f" | Cut: {entry.get('cuts', '?')} "
            f"| Frags: {entry.get('n_fragments', '?')}"
        )
        if 'placement' in entry:
            p = entry['placement']
            line += (
                f" | gap={p.get('gap', 0):.3f} "
                f"dist={p.get('dist', 0):.3f}"
            )
            if not p.get('overlap_resolved', True):
                line += " [OVERLAP]"
        self.output.append_log(f"{self.mol_name}_place.plc", line)

    def _write_path_file(self, index, entry):
        frames = []
        if entry.preopt_mol is not None:
            frames.append(
                entry.preopt_mol.to_xyz("Step 1: Placed fragments")
            )
        if entry.xtb_mol is not None:
            e_str = (
                f" E={entry.xtb_energy:.10f} Eh"
                if entry.xtb_energy else ""
            )
            frames.append(
                entry.xtb_mol.to_xyz(f"Step 2: xTB optimized{e_str}")
            )
        if entry.dft_energy is not None and entry.mol is not None:
            zpve_str = (
                f" ZPVE={entry.zpve:.6f}"
                if entry.zpve is not None else ""
            )
            frames.append(entry.mol.to_xyz(
                f"Step 3: DFT optimized "
                f"E={entry.dft_energy:.10f} Eh{zpve_str}"
            ))
        if not frames:
            return
        content = "\n".join(frames) + "\n"
        self.output.write_log(f"SEARCH_Conf_{index:03d}.log", content)