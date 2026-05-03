"""Global configuration file handling for machine-specific settings."""

import os
from pathlib import Path
from configparser import ConfigParser

CONFIG_DIR = Path.home() / ".config" / "sfris"
CONFIG_FILE = CONFIG_DIR / "config"

# Keys that can appear in the config file, mapped to SFRISParams attribute names.
# Uses the same key names as the input file for consistency.
CONFIG_KEYS = {
    'xtbpath':     ('xtb_path', str),
    'orcapath':    ('orca_path', str),
    'psi4path':    ('psi4_path', str),
    'gaussianpath':('gaussian_path', str),
    'scratchdir':  ('scratch_dir', str),
    'nproc':       ('nproc', int),
    'memory':      ('memory', int),
}

_DEFAULT_CONFIG_CONTENT = """\
[settings]
# Machine-specific defaults for SFRIS.
# These are overridden by values in each input file.
#
# XtbPath = /path/to/xtb
# OrcaPath = /path/to/orca
# Psi4Path = /path/to/psi4
# GaussianPath = /path/to/gaussian
# ScratchDir = /tmp/sfris_scratch
# NProc = 8
# Memory = 4000
"""

def get_config_path() -> Path:
    """Return the config file path."""
    return CONFIG_FILE

def init_config(overwrite: bool = False) -> Path:
    """Create a default config file if it does not exist.

    Args:
        overwrite: If True, overwrite existing config file.

    Returns:
        Path to the config file.
    """
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    if not CONFIG_FILE.exists() or overwrite:
        CONFIG_FILE.write_text(_DEFAULT_CONFIG_CONTENT, encoding="utf-8")
    return CONFIG_FILE

def load_config() -> dict:
    """Load config file and return a dict of {attr_name: value}.

    Returns:
        Dict mapping SFRISParams attribute names to their values.
        Empty dict if config file does not exist.
    """
    if not CONFIG_FILE.exists():
        return {}

    cp = ConfigParser()
    cp.read(CONFIG_FILE, encoding="utf-8")

    result = {}
    if not cp.has_section("settings"):
        return result

    for raw_key, raw_value in cp.items("settings"):
        # configparser lowercases keys, strip spaces from value
        normalized = raw_key.replace(" ", "").replace("_", "").lower()
        if normalized in CONFIG_KEYS:
            attr_name, attr_type = CONFIG_KEYS[normalized]
            try:
                result[attr_name] = attr_type(raw_value.strip())
            except (ValueError, TypeError):
                pass  # skip malformed values silently

    return result

def apply_config(params) -> None:
    """Apply config file values to a SFRISParams instance.

    Called before input file parsing, so input file values
    will override these.

    Args:
        params: SFRISParams instance to update.
    """
    config = load_config()
    for attr_name, value in config.items():
        setattr(params, attr_name, value)

def show_config() -> str:
    """Return a human-readable summary of the current config."""
    if not CONFIG_FILE.exists():
        return f"No config file found at {CONFIG_FILE}"

    config = load_config()
    if not config:
        return f"Config file exists at {CONFIG_FILE} but no active settings."

    lines = [f"Config file: {CONFIG_FILE}", ""]
    for attr_name, value in config.items():
        lines.append(f"  {attr_name} = {value}")
    return "\n".join(lines)
