> **Note:** The version described in the [preprint](https://doi.org/10.26434/chemrxiv.15007216/v1) is **v1.0.1**
> ([release](https://github.com/HalstenZhang/SFRIS/releases/tag/v1.0.1)).
> Later versions add bug fixes and features that do not change the results
> reported in the preprint. See [CHANGELOG.md](CHANGELOG.md).

# SFRIS

**Systematic Fragmentation-Recombination Isomer Search**

SFRIS is a computational chemistry tool for automated constitutional isomer generation. It systematically dissects molecular bonds according to priority rules, places fragments in 3D space via Fibonacci sphere sampling, and refines candidates through a multi-level optimization pipeline.

- **Author:** Dapeng Zhang ([ORCID](https://orcid.org/0000-0002-2879-5613))
- **License:** MIT
- **Homepage:** <https://halsten.pcmq.net/sfris>

## Installation

```bash
pip install sfris
```

### External dependencies

SFRIS requires at least one of the following quantum chemistry programs (not included):

| Program  | Role | Notes |
|----------|------|-------|
| [xTB](https://github.com/grimme-lab/xtb) | Block 1 (semi-empirical screening) | Required |
| [ORCA](https://www.faccts.de/orca/) | Block 2 (DFT refinement) | Optional |
| [Psi4](https://psicode.org/) | Block 2 (DFT refinement) | Alternative to ORCA |
| [Gaussian](https://gaussian.com/) | Block 2 (DFT refinement) | Alternative to ORCA |

## Quick start

### 1. Configure software paths (once per machine)

```bash
sfris config init
```

Edit `~/.config/sfris/config` with your paths:

```ini
[settings]
XtbPath = /path/to/xtb
OrcaPath = /path/to/orca
# Psi4Path = /path/to/psi4
# GaussianPath = /path/to/gaussian
NProc = 8
ScratchDir = /tmp/sfris_scratch
```

### 2. Get an example and run

```bash
sfris example CH4O
sfris run CH4O.inp &
```

Results are written to the `ISOMER/` directory as individual `.xyz` files, sorted by energy. Progress is logged in `SFRIS.run`.

## Usage

```
sfris run <input_file>      # Run full search (background-safe)
sfris test <input_file>     # Test backend connectivity
sfris <input_file>          # Initialize and validate input
sfris config                # Show current config
sfris config init           # Create default config file
sfris example               # List example molecules
sfris example <name>        # Copy example to current directory
```

## Input file format

```
# Comment
0 1
C   0.000   0.000   0.000
O   1.200   0.000   0.000
H  -0.540   0.940   0.000
H  -0.540  -0.940   0.000
Parameters
[CUT]
CutControl = 3, 0.4
[SEARCH]
ConvergenceRounds = 30, auto, auto
[CLUSTER]
MirrorCheck = true
[FILTER]
ScanLimit = 150
AdvOptFilter = 50
[ADVOPT]
[NMA]
[]
```

The `[]` end marker is required. Machine-specific settings (paths, nproc) are read from the config file and do not need to be specified in the input.

### Block control

- **Block 1 only (xTB screening):** omit `[ADVOPT]` from the input file.
- **Block 1 + Block 2 (DFT refinement):** add `[ADVOPT]`. The DFT program is auto-detected from config paths, or set explicitly with `AdvOptProgram = orca`.
- **Frequency analysis:** add `[NMA]` (requires `[ADVOPT]`).

### Method specification

Methods are written as `Functional-Dispersion/Basis`:

```
AdvOptMethod = B3LYP-D3(BJ)/def2-TZVP
AdvOptMethod = PBE0-D3BJ/6-311+G(d,p)
AdvOptMethod = wB97X-D3/cc-pVTZ
AdvOptMethod = M06-2X/aug-cc-pVDZ
```

SFRIS automatically translates method strings into the correct syntax for each backend (ORCA, Psi4, Gaussian). Pople, Dunning, and Karlsruhe basis sets are all supported.

## How it works

SFRIS operates in two blocks:

- **Block 1:** Molecular bonds are systematically cut according to priority rules. Fragments are recombined by placement on a Fibonacci sphere, then screened and optimized at the GFN2-xTB level with triple-gate deduplication (energy, RMSD, graph isomorphism).
- **Block 2 (optional):** Unique isomers from Block 1 are re-optimized at the DFT level with a second round of deduplication.

## Stopping and resuming

SFRIS supports graceful interruption. To stop a running calculation:

```bash
touch CH4O.STOP
```

SFRIS will save a checkpoint and exit at the next safe point. To resume:

```bash
sfris run CH4O.inp &
```

The calculation continues from where it stopped.

## Supported elements

C, H, N, O, S, P, F, Cl, Br

## Citation

If you use SFRIS in your research, please cite the preprint:

> Zhang, D. SFRIS: A Systematic Fragmentation-Recombination Approach for
> Isomer Search. *ChemRxiv* **2026**. doi:10.26434/chemrxiv.15007216/v1

```bibtex
@misc{Zhang2026SFRIS,
  author       = {Zhang, Dapeng},
  title        = {{SFRIS}: A Systematic Fragmentation-Recombination Approach for Isomer Search},
  year         = {2026},
  howpublished = {ChemRxiv preprint},
  doi          = {10.26434/chemrxiv.15007216/v1},
  url          = {https://doi.org/10.26434/chemrxiv.15007216/v1}
}
```

To cite a specific release of the software itself, use the Zenodo archive:
<https://doi.org/10.5281/zenodo.20007904>

`CITATION.cff` in the repository root carries the same information in
machine-readable form; GitHub's "Cite this repository" button reads it.
