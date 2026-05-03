#!/usr/bin/env python3
"""Quantum chemistry backend interfaces: xTB, ORCA, Psi4, Gaussian."""

import os
import re
import subprocess
import tempfile
import shutil
from pathlib import Path
from typing import List, Dict, Tuple, Optional

from sfris.io import SFRISParams, OPT_MAX_CYCLE, MD_TIMESTEP
from sfris.engine import Atom, Molecule


# ============================================================
# xTB Interface
# ============================================================
class XTBInterface:
    """Interface to xTB program."""

    def __init__(self, xtb_path: str = "xtb", scratch_dir: str = "/tmp/sfris",
                 nproc: int = 1, memory: int = 2000):
        self.xtb_path = xtb_path
        self.scratch_dir = Path(scratch_dir).resolve()
        self.nproc = nproc
        self.memory = memory
        self.scratch_dir.mkdir(parents=True, exist_ok=True)
        self._last_error = ""

    def _run_xtb(self, mol: Molecule, args: List[str],
                 work_dir: Path) -> Tuple[bool, str]:
        """Run xTB with given arguments."""
        input_xyz = work_dir / "input.xyz"
        mol.save_xyz(str(input_xyz))

        cmd = [self.xtb_path, str(input_xyz)]
        cmd.extend(args)
        cmd.extend(["--chrg", str(mol.charge)])
        cmd.extend(["--uhf", str(mol.multiplicity - 1)])

        # Force C locale and single-thread for parallel safety
        env = os.environ.copy()
        env['LC_ALL'] = 'C'
        env['LC_NUMERIC'] = 'C'
        env['OMP_NUM_THREADS'] = '1'
        env['MKL_NUM_THREADS'] = '1'
        env['OMP_STACKSIZE'] = '1G'

        try:
            result = subprocess.run(
                cmd, cwd=str(work_dir),
                capture_output=True, text=True,
                encoding='utf-8', errors='ignore',
                timeout=3600, env=env,
            )
            success = result.returncode == 0
            output = result.stdout + result.stderr
        except subprocess.TimeoutExpired:
            success = False
            output = "Timeout"
        except Exception as e:
            success = False
            output = str(e)

        return success, output

    def optimize(self, mol: Molecule, method: str = "GFN2-xTB",
                 convergence: str = "normal",
                 max_cycle: int = OPT_MAX_CYCLE
                 ) -> Tuple[Optional[Molecule], Optional[float], bool]:
        """Optimize geometry. Returns (optimized_mol, energy, success)."""
        work_dir = Path(tempfile.mkdtemp(dir=self.scratch_dir, prefix="opt_"))
        self._last_error = ""

        try:
            method_arg = "--gfn2" if "GFN2" in method.upper() else "--gfn1"
            conv_map = {
                "loose": "loose", "normal": "normal",
                "tight": "tight", "vtight": "vtight",
            }
            conv_arg = conv_map.get(convergence.lower(), "normal")
            args = [method_arg, "--opt", conv_arg,
                    "--cycles", str(max_cycle)]

            success, output = self._run_xtb(mol, args, work_dir)

            if not success:
                self._last_error = self._extract_error(output)
                return None, None, False

            opt_xyz = work_dir / "xtbopt.xyz"
            if not opt_xyz.exists():
                self._last_error = "xtbopt.xyz not found"
                return None, None, False

            opt_mol = self._parse_xyz(opt_xyz)
            opt_mol.charge = mol.charge
            opt_mol.multiplicity = mol.multiplicity

            energy = self._parse_energy(output)
            if energy is None:
                self._last_error = "Could not parse energy from xTB output"
                return None, None, False

            opt_mol.energy = energy
            return opt_mol, energy, True
        finally:
            shutil.rmtree(work_dir, ignore_errors=True)

    def run_md(self, mol: Molecule, temp: float = 400.0, steps: int = 100,
               timestep: float = MD_TIMESTEP, method: str = "GFN2-xTB",
               use_omd: bool = False
               ) -> Tuple[Optional[Molecule], List[Molecule], bool]:
        """Run MD simulation. Returns (final_mol, trajectory, success)."""
        work_dir = Path(tempfile.mkdtemp(dir=self.scratch_dir, prefix="md_"))
        try:
            method_arg = "--gfn2" if "GFN2" in method.upper() else "--gfn1"
            md_flag = "--omd" if use_omd else "--md"
            args = [
                method_arg, md_flag,
                "--temp", str(temp),
                "--time", str(steps * timestep / 1000.0),
                "--step", str(timestep),
                "--dump", str(max(1, steps // 100)),
            ]
            success, output = self._run_xtb(mol, args, work_dir)
            if not success:
                return None, [], False

            final_mol = None
            for fname in ["xtbopt.xyz", "xtb.xyz", "xtbmdok.xyz"]:
                final_xyz = work_dir / fname
                if final_xyz.exists():
                    final_mol = self._parse_xyz(final_xyz)
                    final_mol.charge = mol.charge
                    final_mol.multiplicity = mol.multiplicity
                    break
            if final_mol is None:
                return None, [], False

            traj = []
            traj_xyz = work_dir / "xtb.trj"
            if traj_xyz.exists():
                traj = self._parse_trajectory(traj_xyz)
                for m in traj:
                    m.charge = mol.charge
                    m.multiplicity = mol.multiplicity
            return final_mol, traj, True
        finally:
            shutil.rmtree(work_dir, ignore_errors=True)

    def single_point(self, mol: Molecule,
                     method: str = "GFN2-xTB") -> Tuple[Optional[float], bool]:
        """Single point energy. Returns (energy, success)."""
        work_dir = Path(tempfile.mkdtemp(dir=self.scratch_dir, prefix="sp_"))
        try:
            method_arg = "--gfn2" if "GFN2" in method.upper() else "--gfn1"
            args = [method_arg]
            success, output = self._run_xtb(mol, args, work_dir)
            if not success:
                return None, False
            energy = self._parse_energy(output)
            return (energy, True) if energy is not None else (None, False)
        finally:
            shutil.rmtree(work_dir, ignore_errors=True)

    # ---- parsing helpers ----

    @staticmethod
    def _extract_error(output: str) -> str:
        if not output or not output.strip():
            return "Empty output (xTB may not have started)"
        error_lines = []
        for line in output.split('\n'):
            low = line.lower()
            if any(k in low for k in [
                'error', 'abnormal', 'failed', 'cannot',
                'not converged', 'runtime exception',
                'could not', 'impossible', 'invalid',
            ]):
                stripped = line.strip()
                if stripped and stripped not in error_lines:
                    error_lines.append(stripped)
        if error_lines:
            return " | ".join(error_lines[:3])
        last_lines = [l.strip() for l in output.strip().split('\n')
                      if l.strip()]
        return " | ".join(last_lines[-3:])

    def _parse_xyz(self, filepath: Path) -> Molecule:
        mol = Molecule()
        with open(filepath, 'r') as f:
            lines = f.readlines()
        n_atoms = int(lines[0].strip())
        for i, line in enumerate(lines[2:2 + n_atoms]):
            parts = line.split()
            elem = parts[0]
            x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
            atom = Atom(elem, x, y, z, i)
            mol.atoms.append(atom)
        return mol

    def _parse_trajectory(self, filepath: Path) -> List[Molecule]:
        trajectory = []
        with open(filepath, 'r') as f:
            content = f.read()
        lines = content.strip().split('\n')
        i = 0
        while i < len(lines):
            if lines[i].strip().isdigit():
                n_atoms = int(lines[i].strip())
                mol = Molecule()
                for j in range(n_atoms):
                    parts = lines[i + 2 + j].split()
                    elem = parts[0]
                    x = float(parts[1])
                    y = float(parts[2])
                    z = float(parts[3])
                    atom = Atom(elem, x, y, z, j)
                    mol.atoms.append(atom)
                trajectory.append(mol)
                i += n_atoms + 2
            else:
                i += 1
        return trajectory

    def _parse_energy(self, output: str) -> Optional[float]:
        for line in output.split('\n'):
            if 'TOTAL ENERGY' in line:
                parts = line.split()
                for i, p in enumerate(parts):
                    if p == 'TOTAL' and i + 2 < len(parts):
                        try:
                            return float(parts[i + 2])
                        except ValueError:
                            pass
            if 'total E' in line.lower():
                parts = line.split()
                for p in parts:
                    try:
                        e = float(p)
                        if abs(e) > 1:
                            return e
                    except ValueError:
                        continue
        return None


# ============================================================
# Advanced Optimization Interface (ORCA / Psi4 / Gaussian)
# ============================================================
class AdvOptInterface:
    """Unified interface for high-level optimization + NMA."""

    ATOMIC_NUMBER_TO_ELEMENT = {
        1: 'H', 2: 'He', 3: 'Li', 4: 'Be', 5: 'B', 6: 'C', 7: 'N',
        8: 'O', 9: 'F', 10: 'Ne', 11: 'Na', 12: 'Mg', 13: 'Al', 14: 'Si',
        15: 'P', 16: 'S', 17: 'Cl', 18: 'Ar', 19: 'K', 20: 'Ca',
        31: 'Ga', 32: 'Ge', 33: 'As', 34: 'Se', 35: 'Br', 36: 'Kr',
        53: 'I',
    }

    DISPERSION_MAP = {
        'D3(BJ)': {'orca': 'D3BJ',   'gaussian': 'EmpiricalDispersion=GD3BJ', 'psi4': '-d3bj'},
        'D3BJ':   {'orca': 'D3BJ',   'gaussian': 'EmpiricalDispersion=GD3BJ', 'psi4': '-d3bj'},
        'D3(0)':  {'orca': 'D3ZERO', 'gaussian': 'EmpiricalDispersion=GD3',   'psi4': '-d3'},
        'D3':     {'orca': 'D3ZERO', 'gaussian': 'EmpiricalDispersion=GD3',   'psi4': '-d3'},
        'D4':     {'orca': 'D4',     'gaussian': None,                        'psi4': '-d4'},
    }

    # Functionals with built-in dispersion (D is part of parametrization).
    # These must NOT be split into functional + separate dispersion.
    # Key: input name -> {program: keyword}
    BUILTIN_DISPERSION_FUNCTIONALS = {
        'wB97X-D':    {'orca': 'wB97X-D',    'psi4': 'wb97x-d',    'gaussian': 'wB97XD'},
        'wB97X-D3':   {'orca': 'wB97X-D3',   'psi4': 'wb97x-d3',   'gaussian': None},
        'wB97X-D3BJ': {'orca': 'wB97X-D3BJ', 'psi4': 'wb97x-d3bj', 'gaussian': None},
        'wB97X-D4':   {'orca': 'wB97X-D4',   'psi4': 'wb97x-d4',   'gaussian': None},
        'wB97M-D3BJ': {'orca': 'wB97M-D3BJ', 'psi4': 'wb97m-d3bj', 'gaussian': None},
        'B97-D3':     {'orca': 'B97-D3',     'psi4': 'b97-d3',     'gaussian': None},
        'B97-D3BJ':   {'orca': 'B97-D3BJ',   'psi4': 'b97-d3bj',   'gaussian': None},
        'B97M-D3BJ':  {'orca': 'B97M-D3BJ',  'psi4': 'b97m-d3bj',  'gaussian': None},
    }

    # Gaussian-specific functional name mapping
    GAUSSIAN_FUNCTIONAL_MAP = {
        'PBE':     'PBEPBE',
        'BP86':    'BP86',
        'M06-2X':  'M062X',
        'M06-L':   'M06L',
        'M06-HF':  'M06HF',
        'wB97X-D': 'wB97XD',
    }

    GAUSSIAN_BASIS_MAP = {
        # Karlsruhe (def2) - Gaussian requires no hyphen
        'def2-SV(P)':  'def2SV(P)',
        'def2-SVP':    'def2SVP',
        'def2-SVPD':   'def2SVPD',
        'def2-TZVP':   'def2TZVP',
        'def2-TZVPD':  'def2TZVPD',
        'def2-TZVPP':  'def2TZVPP',
        'def2-TZVPPD': 'def2TZVPPD',
        'def2-QZVP':   'def2QZVP',
        'def2-QZVPP':  'def2QZVPP',
        'def2-QZVPPD': 'def2QZVPPD',
        # Pople and Dunning basis sets use identical names
        # across ORCA, Psi4, and Gaussian - no mapping needed.
    }

    def __init__(self, program: str, program_path: str,
                 method: str = "B3LYP-D3(BJ)/def2-TZVP",
                 nproc: int = 8, memory: int = 4000,
                 scratch_dir: str = "/tmp/sfris",
                 debug: bool = False):
        self.program = program.lower()
        self.program_path = self._normalize_path(program_path)
        self.method = method
        self.nproc = nproc
        self.memory = memory
        self.debug = debug
        self.scratch_dir = Path(scratch_dir) / "advopt"
        self.scratch_dir.mkdir(parents=True, exist_ok=True)
        if self.program not in ('orca', 'psi4', 'gaussian'):
            raise ValueError(f"Unsupported program: {self.program}")
        self.functional, self.dispersion, self.basis = \
            self._parse_method(method)
        # Cache freq results from TightOpt Freq
        self._freq_cache: Dict[int, Tuple[
            Optional[float], Optional[List[float]], int
        ]] = {}

    @staticmethod
    def _normalize_path(path: str) -> str:
        import platform
        if not path:
            return path
        path = path.replace('\\', '/')
        if re.match(r'^/([a-zA-Z])/', path):
            drive_letter = path[1].upper()
            path = f"{drive_letter}:{path[2:]}"
        if platform.system() == 'Windows':
            if not path.lower().endswith('.exe'):
                path += '.exe'
        return path

    @staticmethod
    @staticmethod
    def _parse_method(method: str) -> Tuple[str, str, str]:
        if '/' not in method:
            raise ValueError(
                f"Invalid method format: '{method}'. Expected FUNCTIONAL/BASIS."
            )
        func_part, basis = method.rsplit('/', 1)
        # Check for built-in dispersion functionals first
        # (dispersion is part of parametrization, not a separate correction)
        if func_part in AdvOptInterface.BUILTIN_DISPERSION_FUNCTIONALS:
            return func_part, '', basis
        dispersion = ""
        functional = func_part
        disp_match = re.search(r'-D\d+(?:\([^)]*\)|BJ|ZERO)?$', func_part)
        if disp_match:
            dispersion = func_part[disp_match.start() + 1:]
            functional = func_part[:disp_match.start()]
        return functional, dispersion, basis

    def test_program(self) -> Tuple[bool, str]:
        prog_path = Path(self.program_path)
        if not prog_path.exists():
            return False, f"Program not found: {self.program_path}"
        if not os.access(str(prog_path), os.X_OK):
            return False, f"Program not executable: {self.program_path}"
        try:
            version_flags = {
                'orca': [], 'psi4': ['--version'], 'gaussian': [],
            }
            cmd = [self.program_path] + version_flags.get(self.program, [])
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=30,
                encoding='utf-8', errors='ignore',
            )
            output = result.stdout + result.stderr
            if self.program == 'orca' and (
                    'ORCA' in output or 'An Ab Initio' in output):
                return True, f"ORCA detected: {self.program_path}"
            elif self.program == 'psi4' and result.returncode == 0:
                ver = (output.strip().split('\n')[0]
                       if output.strip() else "unknown")
                return True, f"Psi4 detected: {ver}"
            elif self.program == 'gaussian':
                return True, f"Gaussian detected: {self.program_path}"
            else:
                return True, f"Program accessible: {self.program_path}"
        except FileNotFoundError:
            return False, f"Program not found: {self.program_path}"
        except subprocess.TimeoutExpired:
            return True, (f"Program started (timeout on version check): "
                          f"{self.program_path}")
        except Exception as e:
            return False, f"Error testing program: {e}"

    # ================================================================
    # Dispatch methods
    # ================================================================
    def optimize(self, mol: Molecule, label: str = ""
                 ) -> Tuple[Optional[Molecule], Optional[float], bool]:
        if self.program == 'orca':
            return self._run_orca_opt(mol, label)
        elif self.program == 'psi4':
            return self._run_psi4_opt(mol, label)
        elif self.program == 'gaussian':
            return self._run_gaussian_opt(mol, label)
        return None, None, False

    def single_point(self, mol: Molecule, label: str = ""
                     ) -> Tuple[Optional[float], bool]:
        if self.program == 'orca':
            return self._run_orca_sp(mol, label)
        elif self.program == 'psi4':
            return self._run_psi4_sp(mol, label)
        elif self.program == 'gaussian':
            return self._run_gaussian_sp(mol, label)
        return None, False

    def run_nma(self, mol: Molecule, label: str = ""
                ) -> Tuple[Optional[float], Optional[List[float]], int, bool]:
        """
        Run normal-mode analysis (frequency calculation).
        Returns (zpve, frequencies, n_imaginary, success).
        Checks cache first (ORCA TightOpt Freq already computed freqs).
        """
        cached = self._freq_cache.pop(id(mol), None)
        if cached is not None:
            zpve, freqs, n_imag = cached
            if zpve is not None:
                return zpve, freqs, n_imag, True

        if self.program == 'orca':
            return self._run_orca_freq(mol, label)
        elif self.program == 'psi4':
            return self._run_psi4_freq(mol, label)
        elif self.program == 'gaussian':
            return self._run_gaussian_freq(mol, label)
        return None, None, 0, False

    @classmethod
    def detect_program(cls, params: SFRISParams) -> Tuple[str, str]:
        if params.advopt_program:
            prog = params.advopt_program.lower()
            path_map = {
                'orca': params.orca_path,
                'psi4': params.psi4_path,
                'gaussian': params.gaussian_path,
            }
            path = path_map.get(prog, "")
            return (prog, path) if path else ("", "")
        if params.orca_path:
            return "orca", params.orca_path
        if params.psi4_path:
            return "psi4", params.psi4_path
        if params.gaussian_path:
            return "gaussian", params.gaussian_path
        return "", ""

    # ================================================================
    # Work directory naming helper
    # ================================================================
    def _make_work_dir(self, prog_prefix: str, label: str = "",
                       suffix: str = "") -> Path:
        """Create a named temporary work directory.
        With label: advopt/orca_MIN_003_opt/
        Without:    advopt/orca_xxxxxxxx/
        """
        if label:
            dir_name = f"{prog_prefix}_{label}"
            if suffix:
                dir_name += f"_{suffix}"
            work_dir = self.scratch_dir / dir_name
            # Avoid collision: append counter if exists
            if work_dir.exists():
                counter = 1
                while (self.scratch_dir / f"{dir_name}_{counter}").exists():
                    counter += 1
                work_dir = self.scratch_dir / f"{dir_name}_{counter}"
            work_dir.mkdir(parents=True, exist_ok=True)
            return work_dir
        else:
            return Path(tempfile.mkdtemp(
                dir=self.scratch_dir, prefix=f"{prog_prefix}_"
            ))

    # ================================================================
    # ORCA
    # ================================================================
    def _build_orca_keywords(self, extra: str = "") -> str:
        # Check for built-in dispersion functional
        builtin = self.BUILTIN_DISPERSION_FUNCTIONALS.get(self.functional)
        if builtin:
            orca_func = builtin['orca']
            parts = [f"! {orca_func}"]
        else:
            parts = [f"! {self.functional}"]
            if self.dispersion:
                disp_key = self.DISPERSION_MAP.get(
                    self.dispersion, {}
                ).get('orca', self.dispersion)
                parts.append(disp_key)
        parts.append(self.basis)
        if extra:
            parts.append(extra)
        return " ".join(parts)

    def _run_orca_job(self, mol: Molecule,
                      extra_keywords: str = "",
                      work_dir: Optional[Path] = None
                      ) -> Tuple[str, Path, bool]:
        """Core ORCA runner. Returns (output_text, work_dir, success)."""
        if work_dir is None:
            work_dir = Path(tempfile.mkdtemp(
                dir=self.scratch_dir, prefix="orca_"
            ))
        keyword_line = self._build_orca_keywords(extra_keywords)

        input_path = work_dir / "input.inp"
        with open(input_path, 'w') as f:
            f.write(f"{keyword_line}\n")
            f.write(f"%pal nprocs {self.nproc} end\n")
            f.write(f"%maxcore {self.memory}\n\n")
            f.write(f"* xyz {mol.charge} {mol.multiplicity}\n")
            for atom in mol.atoms:
                f.write(f"  {atom.element:<2s}  {atom.x:16.10f}  "
                        f"{atom.y:16.10f}  {atom.z:16.10f}\n")
            f.write("*\n")

        cmd = [self.program_path, str(input_path.resolve())]
        try:
            result = subprocess.run(
                cmd, cwd=str(work_dir), capture_output=True,
                text=True, encoding='utf-8', errors='ignore',
                timeout=86400,
            )
        except (FileNotFoundError, PermissionError,
                subprocess.TimeoutExpired) as e:
            return str(e), work_dir, False

        output_text = ""
        out_file = work_dir / "input.out"
        if out_file.exists():
            with open(out_file, 'r', encoding='utf-8', errors='ignore') as f:
                output_text = f.read()
        if not output_text:
            output_text = result.stdout + "\n" + result.stderr

        return output_text, work_dir, True

    def _run_orca_opt(self, mol: Molecule, label: str = ""
                      ) -> Tuple[Optional[Molecule], Optional[float], bool]:
        work_dir = None
        try:
            work_dir = self._make_work_dir("orca", label, "opt")
            output_text, work_dir, launched = self._run_orca_job(
                mol, "TightOpt Freq", work_dir
            )
            if not launched:
                return None, None, False
            if ("HURRAY" not in output_text
                    and "OPTIMIZATION RUN DONE" not in output_text):
                return None, None, False

            energy = self._parse_orca_energy(output_text)
            if energy is None:
                return None, None, False

            xyz_path = work_dir / "input.xyz"
            if not xyz_path.exists():
                return None, None, False

            opt_mol = self._parse_xyz_file(xyz_path)
            if opt_mol is None:
                return None, None, False
            opt_mol.charge = mol.charge
            opt_mol.multiplicity = mol.multiplicity
            opt_mol.energy = energy

            # Cache freq results from TightOpt Freq
            freqs = self._parse_orca_frequencies(output_text)
            zpve = self._parse_orca_zpve(output_text)
            n_imag = 0
            if freqs:
                n_imag = sum(1 for f in freqs if f < -50.0)
                if n_imag > 0:
                    print(f"[AdvOpt/ORCA] WARNING: {n_imag} imaginary "
                          f"freq detected after TightOpt")
                self._freq_cache[id(opt_mol)] = (zpve, freqs, n_imag)

            return opt_mol, energy, True
        except Exception as e:
            print(f"[AdvOpt/ORCA] Error: {e}")
            return None, None, False
        finally:
            if work_dir and work_dir.exists():
                if not self.debug:
                    shutil.rmtree(work_dir, ignore_errors=True)

    def _run_orca_sp(self, mol: Molecule, label: str = ""
                     ) -> Tuple[Optional[float], bool]:
        work_dir = None
        try:
            work_dir = self._make_work_dir("orca", label, "sp")
            output_text, work_dir, launched = self._run_orca_job(
                mol, "", work_dir
            )
            if (not launched or "ABORTING THE RUN" in output_text
                    or not output_text.strip()):
                return None, False
            energy = self._parse_orca_energy(output_text)
            return (energy, True) if energy is not None else (None, False)
        except Exception as e:
            print(f"[AdvOpt/ORCA] SP error: {e}")
            return None, False
        finally:
            if work_dir and work_dir.exists():
                if not self.debug:
                    shutil.rmtree(work_dir, ignore_errors=True)

    def _run_orca_freq(self, mol: Molecule, label: str = ""
                       ) -> Tuple[Optional[float], Optional[List[float]],
                                  int, bool]:
        work_dir = None
        try:
            work_dir = self._make_work_dir("orca", label, "freq")
            output_text, work_dir, launched = self._run_orca_job(
                mol, "Freq", work_dir
            )
            if not launched:
                return None, None, 0, False
            if "ABORTING THE RUN" in output_text:
                return None, None, 0, False

            zpve = self._parse_orca_zpve(output_text)
            freqs = self._parse_orca_frequencies(output_text)
            n_imag = sum(1 for f in freqs if f < 0) if freqs else 0

            if zpve is None:
                return None, freqs, n_imag, False
            return zpve, freqs, n_imag, True
        except Exception as e:
            print(f"[AdvOpt/ORCA] Freq error: {e}")
            return None, None, 0, False
        finally:
            if work_dir and work_dir.exists():
                if not self.debug:
                    shutil.rmtree(work_dir, ignore_errors=True)

    @staticmethod
    def _parse_orca_energy(output: str) -> Optional[float]:
        energy = None
        for line in output.split('\n'):
            if 'FINAL SINGLE POINT ENERGY' in line:
                parts = line.split()
                try:
                    energy = float(parts[-1])
                except (ValueError, IndexError):
                    pass
        return energy

    @staticmethod
    def _parse_orca_zpve(output: str) -> Optional[float]:
        for line in output.split('\n'):
            if 'Zero point energy' in line:
                parts = line.split()
                for i, p in enumerate(parts):
                    if p == 'Eh' and i > 0:
                        try:
                            return float(parts[i - 1])
                        except ValueError:
                            pass
                try:
                    return float(parts[-2])
                except (ValueError, IndexError):
                    pass
        return None

    @staticmethod
    def _parse_orca_frequencies(output: str) -> Optional[List[float]]:
        freqs = []
        in_freq_section = False
        for line in output.split('\n'):
            if 'VIBRATIONAL FREQUENCIES' in line:
                in_freq_section = True
                continue
            if in_freq_section:
                if line.strip() == '' and freqs:
                    break
                if 'cm**-1' in line:
                    parts = line.split()
                    try:
                        freq = float(parts[1])
                        if abs(freq) > 0.01:
                            freqs.append(freq)
                    except (ValueError, IndexError):
                        pass
        return freqs if freqs else None

    # ================================================================
    # Psi4
    # ================================================================
    def _psi4_func_string(self) -> str:
        builtin = self.BUILTIN_DISPERSION_FUNCTIONALS.get(self.functional)
        if builtin:
            return builtin['psi4']
        psi4_func = self.functional.lower()
        if self.dispersion:
            disp_suffix = self.DISPERSION_MAP.get(
                self.dispersion, {}
            ).get('psi4', '')
            if disp_suffix:
                psi4_func += disp_suffix
        return psi4_func

    def _write_psi4_molecule_block(self, f, mol: Molecule):
        f.write(f"memory {self.memory} mb\n\n")
        f.write("molecule mol {\n")
        f.write(f"  {mol.charge} {mol.multiplicity}\n")
        for atom in mol.atoms:
            f.write(f"  {atom.element:<2s}  {atom.x:16.10f}  "
                    f"{atom.y:16.10f}  {atom.z:16.10f}\n")
        f.write("}\n\n")

    def _run_psi4_opt(self, mol: Molecule, label: str = ""
                      ) -> Tuple[Optional[Molecule], Optional[float], bool]:
        work_dir = self._make_work_dir("psi4", label, "opt")
        try:
            psi4_func = self._psi4_func_string()
            psi4_basis = self.basis.lower()
            input_path = work_dir / "input.dat"
            output_path = work_dir / "output.dat"
            xyz_output = work_dir / "optimized.xyz"

            with open(input_path, 'w') as f:
                self._write_psi4_molecule_block(f, mol)
                f.write(f"set {{\n  basis {psi4_basis}\n"
                        f"  g_convergence gau_tight\n}}\n\n")
                f.write(f"set_num_threads({self.nproc})\n\n")
                f.write(f"E, wfn = optimize('{psi4_func}', "
                        f"return_wfn=True)\n")
                f.write(f"wfn.molecule().save_xyz_file("
                        f"'{xyz_output}', True)\n")
                f.write("print(f'SFRIS_FINAL_ENERGY {E}')\n")

            cmd = [self.program_path, str(input_path),
                   "-o", str(output_path)]
            result = subprocess.run(
                cmd, cwd=str(work_dir), capture_output=True,
                text=True, encoding='utf-8', errors='ignore',
                timeout=86400,
            )
            output_text = ""
            if output_path.exists():
                with open(output_path, 'r', encoding='utf-8',
                          errors='ignore') as f:
                    output_text = f.read()
            output_text += "\n" + result.stdout + "\n" + result.stderr

            if "SFRIS_FINAL_ENERGY" not in output_text:
                return None, None, False
            energy = self._parse_psi4_energy(output_text)
            if energy is None or not xyz_output.exists():
                return None, None, False

            opt_mol = self._parse_xyz_file(xyz_output)
            if opt_mol is None:
                return None, None, False
            opt_mol.charge = mol.charge
            opt_mol.multiplicity = mol.multiplicity
            opt_mol.energy = energy
            return opt_mol, energy, True
        except subprocess.TimeoutExpired:
            return None, None, False
        except Exception as e:
            print(f"[AdvOpt/Psi4] Error: {e}")
            return None, None, False
        finally:
            if not self.debug:
                shutil.rmtree(work_dir, ignore_errors=True)

    def _run_psi4_sp(self, mol: Molecule, label: str = ""
                     ) -> Tuple[Optional[float], bool]:
        work_dir = self._make_work_dir("psi4", label, "sp")
        try:
            psi4_func = self._psi4_func_string()
            psi4_basis = self.basis.lower()
            input_path = work_dir / "input.dat"
            output_path = work_dir / "output.dat"

            with open(input_path, 'w') as f:
                self._write_psi4_molecule_block(f, mol)
                f.write(f"set {{\n  basis {psi4_basis}\n}}\n\n")
                f.write(f"set_num_threads({self.nproc})\n\n")
                f.write(f"E = energy('{psi4_func}')\n")
                f.write("print(f'SFRIS_FINAL_ENERGY {E}')\n")

            cmd = [self.program_path, str(input_path),
                   "-o", str(output_path)]
            result = subprocess.run(
                cmd, cwd=str(work_dir), capture_output=True,
                text=True, encoding='utf-8', errors='ignore',
                timeout=86400,
            )
            output_text = ""
            if output_path.exists():
                with open(output_path, 'r', encoding='utf-8',
                          errors='ignore') as f:
                    output_text = f.read()
            output_text += "\n" + result.stdout + "\n" + result.stderr
            energy = self._parse_psi4_energy(output_text)
            return (energy, True) if energy is not None else (None, False)
        except Exception as e:
            print(f"[AdvOpt/Psi4] SP error: {e}")
            return None, False
        finally:
            if not self.debug:
                shutil.rmtree(work_dir, ignore_errors=True)

    def _run_psi4_freq(self, mol: Molecule, label: str = ""
                       ) -> Tuple[Optional[float], Optional[List[float]],
                                  int, bool]:
        work_dir = self._make_work_dir("psi4", label, "freq")
        try:
            psi4_func = self._psi4_func_string()
            psi4_basis = self.basis.lower()
            input_path = work_dir / "input.dat"
            output_path = work_dir / "output.dat"

            with open(input_path, 'w') as f:
                self._write_psi4_molecule_block(f, mol)
                f.write(f"set {{\n  basis {psi4_basis}\n}}\n\n")
                f.write(f"set_num_threads({self.nproc})\n\n")
                f.write(f"E, wfn = frequency('{psi4_func}', "
                        f"return_wfn=True)\n")
                f.write("print(f'SFRIS_FINAL_ENERGY {E}')\n")
                f.write("print(f'SFRIS_ZPVE {wfn.variable(\"ZPVE\")}')\n")

            cmd = [self.program_path, str(input_path),
                   "-o", str(output_path)]
            result = subprocess.run(
                cmd, cwd=str(work_dir), capture_output=True,
                text=True, encoding='utf-8', errors='ignore',
                timeout=86400,
            )
            output_text = ""
            if output_path.exists():
                with open(output_path, 'r', encoding='utf-8',
                          errors='ignore') as f:
                    output_text = f.read()
            output_text += "\n" + result.stdout + "\n" + result.stderr

            zpve = None
            for line in output_text.split('\n'):
                if 'SFRIS_ZPVE' in line:
                    try:
                        zpve = float(line.split()[-1])
                    except (ValueError, IndexError):
                        pass

            freqs = self._parse_psi4_frequencies(output_text)
            n_imag = sum(1 for f in freqs if f < 0) if freqs else 0
            return ((zpve, freqs, n_imag, True) if zpve is not None
                    else (None, freqs, n_imag, False))
        except Exception as e:
            print(f"[AdvOpt/Psi4] Freq error: {e}")
            return None, None, 0, False
        finally:
            if not self.debug:
                shutil.rmtree(work_dir, ignore_errors=True)

    @staticmethod
    def _parse_psi4_energy(output: str) -> Optional[float]:
        energy = None
        for line in output.split('\n'):
            if 'SFRIS_FINAL_ENERGY' in line:
                try:
                    return float(line.split()[-1])
                except (ValueError, IndexError):
                    pass
        for line in output.split('\n'):
            if 'Final Energy:' in line:
                try:
                    energy = float(line.split()[-1])
                except (ValueError, IndexError):
                    pass
        return energy

    @staticmethod
    def _parse_psi4_frequencies(output: str) -> Optional[List[float]]:
        freqs = []
        for line in output.split('\n'):
            if 'Freq [cm^-1]' in line:
                parts = line.split()
                for p in parts[2:]:
                    try:
                        f = float(p)
                        if abs(f) > 0.01:
                            freqs.append(f)
                    except ValueError:
                        pass
        return freqs if freqs else None

    # ================================================================
    # Gaussian
    # ================================================================
    def _gaussian_route(self, extra: str = "") -> str:
        gauss_basis = self.GAUSSIAN_BASIS_MAP.get(self.basis, self.basis)
        # Check for built-in dispersion functional
        builtin = self.BUILTIN_DISPERSION_FUNCTIONALS.get(self.functional)
        if builtin:
            gauss_func = builtin['gaussian']
            if gauss_func is None:
                raise ValueError(
                    f"Functional '{self.functional}' is not available "
                    f"in Gaussian. Use ORCA or Psi4."
                )
            route_parts = [f"#p {gauss_func}/{gauss_basis}"]
        else:
            gauss_func = self.GAUSSIAN_FUNCTIONAL_MAP.get(
                self.functional, self.functional
            )
            route_parts = [f"#p {gauss_func}/{gauss_basis}"]
            if self.dispersion:
                disp_entry = self.DISPERSION_MAP.get(self.dispersion, {})
                disp_key = disp_entry.get('gaussian', '')
                if disp_key is None:
                    raise ValueError(
                        f"Dispersion correction '{self.dispersion}' is not "
                        f"supported in Gaussian. Use ORCA or Psi4, "
                        f"or switch to D3(BJ)."
                    )
                if disp_key:
                    route_parts.append(disp_key)
        if extra:
            route_parts.append(extra)
        return " ".join(route_parts)

    def _write_gaussian_input(self, f, mol: Molecule,
                               route: str, title: str):
        f.write(f"%nproc={self.nproc}\n%mem={self.memory}MB\n")
        f.write(route + f"\n\n{title}\n\n")
        f.write(f"{mol.charge} {mol.multiplicity}\n")
        for atom in mol.atoms:
            f.write(f" {atom.element:<2s}  {atom.x:16.10f}  "
                    f"{atom.y:16.10f}  {atom.z:16.10f}\n")
        f.write("\n")

    def _run_gaussian_job(self, mol: Molecule, extra: str,
                           title: str,
                           work_dir: Optional[Path] = None
                           ) -> Tuple[str, Path]:
        """Run a Gaussian job. Returns (output_text, work_dir)."""
        if work_dir is None:
            work_dir = Path(tempfile.mkdtemp(
                dir=self.scratch_dir, prefix="gauss_"
            ))
        route = self._gaussian_route(extra)
        input_path = work_dir / "input.gjf"
        log_path = work_dir / "input.log"

        with open(input_path, 'w') as f:
            self._write_gaussian_input(f, mol, route, title)

        cmd = [self.program_path, str(input_path)]
        subprocess.run(
            cmd, cwd=str(work_dir), capture_output=True,
            text=True, encoding='utf-8', errors='ignore',
            timeout=86400,
        )

        output_text = ""
        if log_path.exists():
            with open(log_path, 'r', encoding='utf-8',
                      errors='ignore') as f:
                output_text = f.read()
        return output_text, work_dir

    def _run_gaussian_opt(self, mol: Molecule, label: str = ""
                          ) -> Tuple[Optional[Molecule], Optional[float],
                                     bool]:
        work_dir = None
        try:
            work_dir = self._make_work_dir("gauss", label, "opt")
            output_text, work_dir = self._run_gaussian_job(
                mol, "Opt=Tight", "SFRIS optimization", work_dir
            )
            if "Normal termination" not in output_text:
                return None, None, False
            energy = self._parse_gaussian_energy(output_text)
            opt_mol = self._parse_gaussian_geometry(output_text)
            if energy is None or opt_mol is None:
                return None, None, False
            opt_mol.charge = mol.charge
            opt_mol.multiplicity = mol.multiplicity
            opt_mol.energy = energy
            return opt_mol, energy, True
        except subprocess.TimeoutExpired:
            return None, None, False
        except Exception as e:
            print(f"[AdvOpt/Gaussian] Error: {e}")
            return None, None, False
        finally:
            if work_dir and work_dir.exists():
                if not self.debug:
                    shutil.rmtree(work_dir, ignore_errors=True)

    def _run_gaussian_sp(self, mol: Molecule, label: str = ""
                         ) -> Tuple[Optional[float], bool]:
        work_dir = None
        try:
            work_dir = self._make_work_dir("gauss", label, "sp")
            output_text, work_dir = self._run_gaussian_job(
                mol, "", "SFRIS single point", work_dir
            )
            energy = self._parse_gaussian_energy(output_text)
            return (energy, True) if energy is not None else (None, False)
        except Exception as e:
            print(f"[AdvOpt/Gaussian] SP error: {e}")
            return None, False
        finally:
            if work_dir and work_dir.exists():
                if not self.debug:
                    shutil.rmtree(work_dir, ignore_errors=True)

    def _run_gaussian_freq(self, mol: Molecule, label: str = ""
                           ) -> Tuple[Optional[float], Optional[List[float]],
                                      int, bool]:
        work_dir = None
        try:
            work_dir = self._make_work_dir("gauss", label, "freq")
            output_text, work_dir = self._run_gaussian_job(
                mol, "Freq", "SFRIS frequency", work_dir
            )
            if "Normal termination" not in output_text:
                return None, None, 0, False

            zpve = self._parse_gaussian_zpve(output_text)
            freqs = self._parse_gaussian_frequencies(output_text)
            n_imag = sum(1 for f in freqs if f < 0) if freqs else 0
            return ((zpve, freqs, n_imag, True) if zpve is not None
                    else (None, freqs, n_imag, False))
        except Exception as e:
            print(f"[AdvOpt/Gaussian] Freq error: {e}")
            return None, None, 0, False
        finally:
            if work_dir and work_dir.exists():
                if not self.debug:
                    shutil.rmtree(work_dir, ignore_errors=True)

    @staticmethod
    def _parse_gaussian_energy(output: str) -> Optional[float]:
        energy = None
        for line in output.split('\n'):
            if 'SCF Done' in line:
                parts = line.split('=')
                if len(parts) >= 2:
                    try:
                        energy = float(parts[1].split()[0])
                    except (ValueError, IndexError):
                        pass
        return energy

    @classmethod
    def _parse_gaussian_geometry(cls, output: str) -> Optional[Molecule]:
        lines = output.split('\n')
        last_block_start = -1
        for i, line in enumerate(lines):
            if 'Standard orientation' in line:
                last_block_start = i
        if last_block_start < 0:
            for i, line in enumerate(lines):
                if 'Input orientation' in line:
                    last_block_start = i
        if last_block_start < 0:
            return None

        data_start = last_block_start + 5
        mol = Molecule()
        for i in range(data_start, len(lines)):
            line = lines[i].strip()
            if line.startswith('---'):
                break
            parts = line.split()
            if len(parts) < 6:
                continue
            try:
                atomic_num = int(parts[1])
                x, y, z = float(parts[3]), float(parts[4]), float(parts[5])
                element = cls.ATOMIC_NUMBER_TO_ELEMENT.get(atomic_num, 'X')
                atom = Atom(element, x, y, z, len(mol.atoms))
                mol.atoms.append(atom)
            except (ValueError, IndexError):
                continue
        return mol if mol.n_atoms > 0 else None

    @staticmethod
    def _parse_gaussian_zpve(output: str) -> Optional[float]:
        for line in output.split('\n'):
            if 'Zero-point correction' in line:
                parts = line.split('=')
                if len(parts) >= 2:
                    try:
                        return float(parts[1].split()[0])
                    except (ValueError, IndexError):
                        pass
        return None

    @staticmethod
    def _parse_gaussian_frequencies(output: str) -> Optional[List[float]]:
        freqs = []
        for line in output.split('\n'):
            if 'Frequencies --' in line:
                parts = line.split('--')
                if len(parts) >= 2:
                    for val in parts[1].split():
                        try:
                            f = float(val)
                            if abs(f) > 0.01:
                                freqs.append(f)
                        except ValueError:
                            pass
        return freqs if freqs else None

    # ================================================================
    # Shared utilities
    # ================================================================
    @staticmethod
    def _parse_xyz_file(filepath: Path) -> Optional[Molecule]:
        try:
            with open(filepath, 'r') as f:
                content = f.read()
        except Exception:
            return None

        lines = content.strip().split('\n')
        # Find last frame in multi-frame xyz
        last_frame_start = -1
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            if line.isdigit():
                last_frame_start = i
                n_atoms = int(line)
                i += n_atoms + 2
            else:
                i += 1

        if last_frame_start < 0:
            return None
        n_atoms = int(lines[last_frame_start].strip())
        mol = Molecule()
        for j in range(n_atoms):
            line_idx = last_frame_start + 2 + j
            if line_idx >= len(lines):
                return None
            parts = lines[line_idx].split()
            if len(parts) < 4:
                return None
            elem = parts[0]
            x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
            atom = Atom(elem, x, y, z, j)
            mol.atoms.append(atom)
        return mol if mol.n_atoms == n_atoms else None