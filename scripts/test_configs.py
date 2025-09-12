#!/usr/bin/env python3
"""
Quick config tester:
- Runs pipeline once per config on a data directory (CSV or images).
- Builds visual QC overlays per input: base heatmap vs. base+ skeleton.
"""

from __future__ import annotations
import argparse
import sys
import time
from pathlib import Path
from typing import Iterable, List, Tuple

import cv2
import numpy as np

# --- Make sure we can import the project modules (assumes repo has a top-level `src/`) ---
HERE = Path(__file__).resolve().parent
CANDIDATES = [HERE, HERE.parent, HERE.parent.parent, Path.cwd()]
for c in CANDIDATES:
    if (c / "src").is_dir():
        sys.path.insert(0, str(c))
        break

from src.service.pipeline import iter_run_project, JobEvent  # type: ignore


def _find_base_image(base_dir: Path) -> Path | None:
    """
    Given a KymoButler base_dir (e.g., .../generated_heatmaps/<stem>),
    try to find the input heatmap next to it (same <stem>, common extensions).
    """
    parent = base_dir.parent
    stem = base_dir.name
    for ext in (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"):
        p = parent / f"{stem}{ext}"
        if p.exists():
            return p
    # Fallback: look for any image with that stem anywhere below parent
    for p in parent.rglob(f"{stem}.*"):
        if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}:
            return p
    return None


def _color_overlay(base_gray: np.ndarray, skel: np.ndarray,
                   base_weight: float = 0.85, skel_weight: float = 0.65,
                   color: Tuple[int,int,int] = (0, 255, 0)) -> np.ndarray:
    """
    Alpha-blend a colored skeleton on top of the grayscale base.
    """
    if base_gray.ndim == 2:
        base_rgb = cv2.cvtColor(base_gray, cv2.COLOR_GRAY2BGR)
    else:
        base_rgb = base_gray.copy()
    h0, w0 = base_rgb.shape[:2]
    if skel.shape[:2] != (h0, w0):
        skel = cv2.resize(skel, (w0, h0), interpolation=cv2.INTER_NEAREST)

    skel_bin = (skel > 0).astype(np.uint8)
    skel_layer = np.zeros_like(base_rgb)
    skel_layer[skel_bin == 1] = color
    out = cv2.addWeighted(base_rgb, base_weight, skel_layer, skel_weight, 0)
    return out


def _side_by_side(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Put two images side-by-side, resizing B to A's height if necessary.
    """
    ha, wa = a.shape[:2]
    hb, wb = b.shape[:2]
    if hb != ha:
        scale = ha / hb
        b = cv2.resize(b, (int(wb * scale), ha), interpolation=cv2.INTER_AREA)
    return np.hstack([a, b])


def _collect_debug_skeletons(run_out_dir: Path) -> List[Tuple[Path, Path]]:
    """
    Find (base_image, skeleton_png) pairs inside a run output folder.
    Looks for **/debug/skeleton.png and matches to the sibling heatmap file.
    """
    pairs: List[Tuple[Path, Path]] = []
    for skel_png in run_out_dir.rglob("debug/skeleton.png"):
        base_dir = skel_png.parent.parent         # .../<stem>/debug -> base_dir = .../<stem>
        base = _find_base_image(base_dir)
        if base and base.exists():
            pairs.append((base, skel_png))
    return pairs


def _read_gray(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(path)
    return img


def run_one_config(data_dir: Path, cfg_path: Path, out_root: Path,
                   limit: int | None = None, verbose: bool = True) -> Path:
    """
    Run the project pipeline for a single config and build quick QC overlays.
    Returns the run output directory for this config.
    """
    run_name = cfg_path.stem
    run_out = out_root / run_name
    run_out.mkdir(parents=True, exist_ok=True)

    def _print_evt(evt: JobEvent):
        if verbose:
            msg = f"[{evt.phase:<12}] {evt.message}"
            if evt.extra and "total_tracks" in (evt.extra or {}):
                msg += f" (total_tracks={evt.extra['total_tracks']})"
            print(msg)

    # Allow limiting by temporarily copying a few CSVs to a temp folder if requested.
    # Otherwise, we just point the pipeline at the full data_dir.
    data_root = data_dir
    if limit is not None and limit > 0:
        # Soft limit by prioritizing CSV first
        candidates = sorted(
            [*data_dir.glob("*.csv"), *data_dir.glob("*.xls"), *data_dir.glob("*.xlsx")]
        )[:limit]
        tmp = run_out / "_subset"
        tmp.mkdir(exist_ok=True, parents=True)
        for p in candidates:
            # symlink for speed (fallback to copy on platforms without symlink perms)
            dst = tmp / p.name
            try:
                if dst.exists():
                    dst.unlink()
                dst.symlink_to(p.resolve())
            except Exception:
                import shutil
                shutil.copy2(p, dst)
        data_root = tmp

    # Kick off pipeline
    for evt in iter_run_project(
        input_dir=str(data_root),
        config_path=str(cfg_path),
        output_dir=str(run_out),
        plots_out=None,
        progress_cb=_print_evt,
        config_overrides=None,
        verbose=verbose,
        cancel_cb=None,
    ):
        # just exhaust the generator; evt printing handled by callback
        pass

    # Build quick-check overlays
    qc_dir = run_out / "quickcheck"
    qc_dir.mkdir(exist_ok=True, parents=True)

    pairs = _collect_debug_skeletons(run_out)
    if verbose:
        print(f"[QC] Found {len(pairs)} skeleton(s) for overlays in {run_out}")

    for base_img_path, skel_path in pairs:
        try:
            base = _read_gray(base_img_path)
            skel = _read_gray(skel_path)

            # Normalize base (nice viewing)
            b = base.astype(np.float32)
            if b.max() > 0:
                b = (255.0 * (b / b.max())).astype(np.uint8)
            base_u8 = b.astype(np.uint8)

            overlay = _color_overlay(base_u8, skel, base_weight=0.85, skel_weight=0.65, color=(0, 255, 0))
            side = _side_by_side(cv2.cvtColor(base_u8, cv2.COLOR_GRAY2BGR), overlay)

            stem = base_img_path.stem
            out_overlay = qc_dir / f"{stem}__overlay.png"
            out_side = qc_dir / f"{stem}__base_vs_overlay.png"
            cv2.imwrite(str(out_overlay), overlay)
            cv2.imwrite(str(out_side), side)
        except Exception as e:
            print(f"[QC] Failed overlay for {base_img_path.name}: {e}")

    return run_out


def main():
    ap = argparse.ArgumentParser(description="Quickly test KymoButler configs and visualize skeleton alignment.")
    ap.add_argument("--data-dir", type=Path, required=True, help="Folder with input CSVs/images (e.g., ./data)")
    ap.add_argument("--configs", type=Path, nargs="+", required=True, help="One or more YAML configs to test")
    ap.add_argument("--out", type=Path, default=Path("./runs"), help="Output root directory")
    ap.add_argument("--limit", type=int, default=None, help="Limit number of tables to convert (quick test)")
    ap.add_argument("--quiet", action="store_true", help="Less verbose logging")
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    verbose = not args.quiet

    for cfg in args.configs:
        print(f"\n=== Running config: {cfg.name} ===")
        t0 = time.time()
        out_dir = run_one_config(args.data_dir, cfg, args.out, limit=args.limit, verbose=verbose)
        dt = time.time() - t0
        print(f"Done: {cfg.name} → {out_dir}  ({dt:.1f}s)\n"
              f"Quick-check PNGs in: {out_dir / 'quickcheck'}")


if __name__ == "__main__":
    main()
