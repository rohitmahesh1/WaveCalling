import os
os.environ.setdefault("MPLBACKEND", "Agg")  # headless by default in server/threads
import matplotlib
matplotlib.use("Agg", force=True)

from pathlib import Path
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Tuple


def _is_xlsx_magic(header: bytes) -> bool:
    # XLSX files are ZIP archives; ZIP magic is PK\x03\x04
    return header.startswith(b"PK\x03\x04")


def _is_xls_magic(header: bytes) -> bool:
    # Legacy Excel (.xls OLE) magic: D0 CF 11 E0 A1 B1 1A E1
    return header.startswith(b"\xD0\xCF\x11\xE0\xA1\xB1\x1A\xE1")


def _load_table(path: Path) -> pd.DataFrame:
    """
    Load a tabular file robustly:
    - If true XLSX (ZIP magic): use openpyxl engine.
    - If true XLS (OLE magic): try xlrd engine.
    - Otherwise: treat as text and let pandas sniff the delimiter.
    This handles mislabeled files (e.g., CSV named .xls).
    """
    # Read a small header to sniff actual format
    with open(path, "rb") as f:
        header = f.read(8)

    # True XLSX
    if _is_xlsx_magic(header):
        try:
            return pd.read_excel(path, header=None, engine="openpyxl")
        except ImportError as e:
            raise RuntimeError(
                "Tried to read an .xlsx Excel file but 'openpyxl' is not installed. "
                "Install it with: pip install openpyxl"
            ) from e

    # True XLS
    if _is_xls_magic(header):
        try:
            return pd.read_excel(path, header=None, engine="xlrd")
        except ImportError as e:
            raise RuntimeError(
                "Tried to read a legacy .xls Excel file but 'xlrd' is not installed. "
                "Install it with: pip install xlrd"
            ) from e
        except ValueError:
            # Some corrupt/odd .xls — fall back to text
            pass

    # Text table (CSV/TSV or mislabeled Excel)
    # Use sep=None to let pandas sniff the delimiter (requires python engine)
    try:
        return pd.read_csv(path, sep=None, engine="python", header=None)
    except Exception:
        # One more attempt: assume TSV
        return pd.read_csv(path, sep="\t", header=None)


def _keep_extremes_zero_middle(arr: np.ndarray, lower: float, upper: float) -> np.ndarray:
    """
    Zero out values within [lower, upper]; keep extreme values as-is.
    Mirrors your previous 'keep_extreme_values' logic.
    """
    out = arr.copy()
    mask = (out >= lower) & (out <= upper)
    out[mask] = 0
    return out


def table_to_heatmap(
    table_path: str | Path,
    *,
    out_dir: str | Path,
    lower: float = -1e20,
    upper: float = 1e16,
    binarize: bool = True,
    origin: str = "lower",  # "lower" to match your old behavior
    cmap: str = "hot",
    dpi: int | None = 180,
) -> Tuple[Path, int, int]:
    """
    Convert a numeric table (csv/tsv/xls/xlsx) into a heatmap PNG suitable for KymoButler.

    Returns:
        (output_image_path, num_rows, num_cols)
    """
    table_path = Path(table_path)
    out_dir = Path(out_dir)
    os.makedirs(out_dir, exist_ok=True)

    df = _load_table(table_path)

    # Convert to float and sanitize NaN/Inf
    data = df.to_numpy(dtype=float)
    if not np.isfinite(data).all():
        data = np.nan_to_num(data, copy=False, nan=0.0, posinf=0.0, neginf=0.0)

    # keep extremes and optionally binarize
    filtered = _keep_extremes_zero_middle(data, lower, upper)
    filtered = np.abs(filtered)
    if binarize:
        filtered = (filtered > 0).astype(int)

    nrows, ncols = filtered.shape

    # output path
    out_name = f"{table_path.stem}_heatmap.png"
    out_path = out_dir / out_name

    # plot
    plt.figure(figsize=(8, 6), dpi=(dpi if dpi else None))
    vmax = float(np.max(filtered)) if filtered.size else 1.0
    plt.imshow(filtered, cmap=cmap, interpolation="nearest", vmin=0, vmax=vmax, origin=origin)
    plt.axis("off")
    plt.savefig(out_path, bbox_inches="tight", pad_inches=0, dpi=(dpi if dpi else None))
    plt.close()

    return out_path, nrows, ncols
