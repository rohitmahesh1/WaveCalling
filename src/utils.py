from __future__ import annotations
import logging
from pathlib import Path
from typing import Iterable, List, Tuple
import yaml
import pandas as pd


# ---------------------------
# Logging helpers
# ---------------------------

_LOG_CONFIGURED = False

def setup_logging(level: str = "INFO") -> None:
    """Configure root logging once.

    Args:
        level: One of 'DEBUG', 'INFO', 'WARNING', 'ERROR'.
    """
    global _LOG_CONFIGURED
    if _LOG_CONFIGURED:
        return
    logging.basicConfig(
        format="[%(asctime)s] %(levelname)s - %(message)s",
        datefmt="%H:%M:%S",
        level=getattr(logging, level.upper(), logging.INFO),
    )
    _LOG_CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)


# ---------------------------
# Config helpers
# ---------------------------

def load_config(path: Path | str) -> dict:
    """Load a YAML config file into a dict."""
    with open(path, "r") as f:
        return yaml.safe_load(f)


# ---------------------------
# Filesystem helpers
# ---------------------------

def ensure_dir(p: Path | str) -> Path:
    p = Path(p)
    p.mkdir(parents=True, exist_ok=True)
    return p


def list_files(root: Path | str, patterns: Iterable[str]) -> List[Path]:
    """Recursively list files under root that match any glob in patterns."""
    root = Path(root)
    files: List[Path] = []
    for pat in patterns:
        files.extend(root.rglob(pat))
    return sorted(files)


# ---------------------------
# Data I/O helpers
# ---------------------------

def save_dataframe(df: pd.DataFrame, out_path: Path | str) -> Path:
    out_path = Path(out_path)
    ensure_dir(out_path.parent)
    df.to_csv(out_path, index=False, na_rep="NA")
    return out_path
