from pathlib import Path
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Tuple


def _load_table(path: Path) -> pd.DataFrame:
    ext = path.suffix.lower()
    if ext in (".csv",):
        return pd.read_csv(path, header=None)
    if ext in (".tsv",):
        return pd.read_csv(path, sep="\t", header=None)
    if ext in (".xls", ".xlsx"):
        # read the first sheet by default; no header
        return pd.read_excel(path, header=None)
    # fallback: try CSV
    return pd.read_csv(path, header=None)


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
    data = df.to_numpy(dtype=float)

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
    plt.figure(figsize=(8, 6), dpi=dpi if dpi else None)
    vmax = np.max(filtered) if filtered.size else 1.0
    plt.imshow(filtered, cmap=cmap, interpolation="nearest", vmin=0, vmax=vmax, origin=origin)
    plt.axis("off")
    plt.savefig(out_path, bbox_inches="tight", pad_inches=0, dpi=dpi if dpi else None)
    plt.close()

    return out_path, nrows, ncols
