"""
src/config.py
=============
Central configuration loader for the Ghana groundwater-subsidence project.

Loads all four YAML config files (paths, study_area, variables, thresholds)
and exposes them as module-level dictionaries and typed helper accessors.

Usage
-----
    from src.config import paths, study_area, variables, thresholds, project_root, rp

    raw_grace = rp(paths["raw"]["grace"])   # -> Path object
    epsg      = study_area["crs"]["epsg"]   # -> 32630
"""

from __future__ import annotations

from pathlib import Path
import yaml

# ---------------------------------------------------------------------------
# Locate the project root (one level up from src/)
# ---------------------------------------------------------------------------
_THIS_FILE   = Path(__file__).resolve()
_SRC_DIR     = _THIS_FILE.parent
PROJECT_ROOT = _SRC_DIR.parent


def rp(relative_path: str) -> Path:
    """
    Resolve a config-supplied relative path to an absolute Path.

    Parameters
    ----------
    relative_path : str
        A path string relative to the project root
        (as stored in config/paths.yaml).

    Returns
    -------
    Path
        Absolute Path object.
    """
    return PROJECT_ROOT / relative_path


# ---------------------------------------------------------------------------
# Load YAML config files
# ---------------------------------------------------------------------------
_CONFIG_DIR = PROJECT_ROOT / "config"


def _load_yaml(filename: str) -> dict:
    config_path = _CONFIG_DIR / filename
    if not config_path.exists():
        raise FileNotFoundError(
            f"Config file not found: {config_path}\n"
            "Run the project from the repository root or check config/."
        )
    with open(config_path, encoding="utf-8") as fh:
        return yaml.safe_load(fh)


paths      = _load_yaml("paths.yaml")
study_area = _load_yaml("study_area.yaml")
variables  = _load_yaml("variables.yaml")
thresholds = _load_yaml("thresholds.yaml")

# ---------------------------------------------------------------------------
# Convenience accessors
# ---------------------------------------------------------------------------
UTM_CRS: str = f"EPSG:{study_area['crs']['epsg']}"
GEO_CRS: str = f"EPSG:{study_area['crs']['geographic_epsg']}"

ANOMALY_BASELINE_START: str = study_area["time_periods"]["anomaly_baseline_start"]
ANOMALY_BASELINE_END:   str = study_area["time_periods"]["anomaly_baseline_end"]
