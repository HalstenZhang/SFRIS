"""Command-line interface for SFRIS."""

import sys
import shutil
from pathlib import Path
from sfris.core import SFRIS, SFRIS_BANNER

def _check_input(filepath):
    """Check if input file exists, exit with message if not."""
    if not Path(filepath).is_file():
        print(f"Error: Input file not found: {filepath}")
        sys.exit(1)

def _get_examples_dir():
    """Return the path to bundled example files."""
    return Path(__file__).parent / "examples"

def _handle_example(args):
    """Handle 'sfris example' subcommand."""
    examples_dir = _get_examples_dir()
    available = sorted(examples_dir.glob("*.inp"))
    if not available:
        print("No example files found.")
        return

    if len(args) == 0:
        print("Available examples:")
        for f in available:
            with open(f, "r") as fh:
                desc = fh.readline().strip().lstrip("# ")
            print(f"  {f.stem:12s}  {desc}")
        print()
        print("Usage:")
        print("  sfris example <name>    # Copy to current directory")
        print("  sfris example all       # Copy all examples")
        return

    target = args[0]
    if target == "all":
        for f in available:
            shutil.copy2(f, f.name)
            print(f"  {f.name}")
        print(f"Copied {len(available)} example files.")
    else:
        inp_file = examples_dir / f"{target}.inp"
        if not inp_file.exists():
            print(f"Error: Unknown example '{target}'.")
            print(f"Available: {', '.join(f.stem for f in available)}")
            return
        shutil.copy2(inp_file, inp_file.name)
        print(f"Copied {inp_file.name} to current directory.")
        print(f"Run with: sfris run {inp_file.name} &")

def main():
    from setproctitle import setproctitle
    setproctitle("sfris")

    if len(sys.argv) < 2:
        print(SFRIS_BANNER)
        print("Pipeline: SCAN -> MIN(xTB) -> [ADVOPT -> MIN(DFT)] -> [NMA]")
        print()
        print("Usage:")
        print("  sfris <input_file>          # Initialize only")
        print("  sfris test <input_file>     # Test backend connectivity")
        print("  sfris run <input_file>      # Run full search (background-safe)")
        print("  sfris config                # Show current config")
        print("  sfris config init           # Create default config file")
        print("  sfris example               # List example molecules")
        print("  sfris example <name>        # Copy example to current directory")
        print("  sfris lib                   # Starting structure library")
        print("  sfris --version             # Show version")
        sys.exit(1)

    if sys.argv[1] in ("--version", "-V", "version"):
        from sfris import __version__
        print("sfris %s" % __version__)
        return

    if sys.argv[1] == "lib":
        from sfris import lib
        lib.handle_lib(sys.argv[2:])
        return

    if sys.argv[1] == "config":
        print(SFRIS_BANNER)
        from sfris.config import init_config, show_config
        if len(sys.argv) >= 3 and sys.argv[2] == "init":
            path = init_config()
            print(f"Config file created: {path}")
            print("Edit this file to set default paths for your machine.")
        else:
            print(show_config())
        return

    if sys.argv[1] == "example":
        print(SFRIS_BANNER)
        _handle_example(sys.argv[2:])
        return

    try:
        if sys.argv[1] == "test" and len(sys.argv) >= 3:
            _check_input(sys.argv[2])
            print(SFRIS_BANNER)
            sfris = SFRIS(sys.argv[2])
            sfris.initialize()
            print("Testing xTB interface...")
            energy, success = sfris.xtb.single_point(
                sfris.mol, sfris.params.md_method
            )
            if success:
                print(f"  Single point: {energy:.6f} Eh")
            else:
                print("  Single point FAILED")
                return
            opt_mol, opt_energy, success = sfris.xtb.optimize(
                sfris.mol, sfris.params.opt_method,
                sfris.params.opt_convergence
            )
            if success:
                print(f"  Optimization: {opt_energy:.6f} Eh")
            else:
                print("  Optimization FAILED")
                return
            print("xTB test PASSED")

        elif sys.argv[1] == "run" and len(sys.argv) >= 3:
            _check_input(sys.argv[2])
            sfris = SFRIS(sys.argv[2], quiet=True)
            sfris.initialize()
            sfris.run_search()

        else:
            _check_input(sys.argv[1])
            print(SFRIS_BANNER)
            sfris = SFRIS(sys.argv[1])
            sfris.initialize()
            print("Initialization complete. Use 'run' to start search.")

    except KeyboardInterrupt:
        print("\nInterrupted by user.")
        sys.exit(130)
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)
    except PermissionError as e:
        print(f"Error: Permission denied: {e.filename}")
        sys.exit(1)
    except OSError as e:
        print(f"Error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Error: Unexpected error: {type(e).__name__}: {e}")
        print("Please report this issue.")
        sys.exit(1)

if __name__ == "__main__":
    main()