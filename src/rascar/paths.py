"""Central place for every filesystem location used by the project.

Nothing else in the package hardcodes a user-specific absolute path. The project
root is discovered from this file's location; it can be overridden with the
``RASCAR_ROOT`` environment variable and the raw data directory with
``RASCAR_DATA_DIR`` (useful for tests and for running on another machine).
"""

from __future__ import annotations

import os
from pathlib import Path

DATASET_DIRNAME = "rascar-hack-case-1"


def project_root() -> Path:
    env = os.environ.get("RASCAR_ROOT")
    if env:
        return Path(env).resolve()
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "pyproject.toml").is_file() and (parent / "src" / "rascar").is_dir():
            return parent
    return Path.cwd().resolve()


def raw_dir() -> Path:
    env = os.environ.get("RASCAR_DATA_DIR")
    if env:
        return Path(env).resolve()
    return project_root() / "data" / "raw" / DATASET_DIRNAME


def _sub(*parts: str, create: bool = True) -> Path:
    path = project_root().joinpath(*parts)
    if create:
        path.mkdir(parents=True, exist_ok=True)
    return path


def interim_dir() -> Path:
    return _sub("data", "interim")


def processed_dir() -> Path:
    return _sub("data", "processed")


def models_dir() -> Path:
    return _sub("models")


def submissions_dir() -> Path:
    return _sub("submissions")


def results_dir() -> Path:
    return _sub("results")


def reports_dir() -> Path:
    return _sub("reports")


def docs_dir() -> Path:
    return _sub("docs")


def handoff_dir() -> Path:
    return _sub("presentation_handoff")


def figures_dir() -> Path:
    return _sub("presentation_handoff", "figures")
