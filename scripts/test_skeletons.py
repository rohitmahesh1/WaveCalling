#!/usr/bin/env python3
"""
Skeleton-only config tester:
- For each YAML config, convert tables -> heatmaps (optional limit),
- Run KymoButler (adapter) to produce debug/skeleton.png,
- Build visual overlays: base vs. base+skeleton.
No waves or metrics are computed.
"""

from __future__ import annotations
import argparse
import sys
import time
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np

# --- allow "src" imports when run from repo root or scripts/ ---
HERE = Path(__file__).resolve().parent
for candidate in (HERE, HERE.parent, Path.cwd()):
    if (candidate / "src").is_dir():
        sys.path.insert(0, str(candidate))
        break

from src.utils import load_config, ensure_dir, list_files, get_logger, setup_logging  # type: ignore
from src.io.table_to_heatmap import table_to_heatmap  # type: ignore
from src.extract import _build_kymo_runner as _select_kymo_backend  # type: ignore


def _is_image(p: Path) -> bool:
    return p.suffix.lower() in {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


def _normalize_gray(g: np.ndarray) -> np.ndarray:
    g = g.astype(np.float32)
    m = g.max()
    if m > 0:
        g = (255.0 * (g / m)).astype(np.uint8)
    return g.astype(np.uint8)


def _color_overlay(base_gray: np.ndarray, skel: np.ndarray,
                   base_w: float = 0.85, skel_w: float = 0.65,
                   color=(0, 255, 0)) -> np.ndarray:
    if base_gray.ndim == 2:
        base_rgb = cv2.cvtColor(base_gray, cv2.COLOR_GRAY2BGR)
    else:
        base_rgb = base_gray.copy()
    H, W = base_rgb.shape[:2]
    if skel.shape[:2] != (H, W):
        skel = cv2.resize(skel, (W, H), interpolation=cv2.INTER_NEAREST)
    skel_bin = (skel > 0).astype(np.uint8)
    layer = np.zeros_like(base_rgb)
    layer[skel_bin == 1] = color
    return cv2.addWeighted(base_rgb, base_w, layer, skel_w, 0)


def _side_by_side(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    ha, wa = a.shape[:2]
    hb, wb = b.shape[:2]
    if hb != ha:
        scale = ha / hb
        b = cv2.resize(b, (int(wb * scale), ha), interpolation=cv2.INTER_AREA)
    return np.hstack([a, b])


def _find_skeleton_png(base_dir: Path) -> Path | None:
    """kb_adapter writes <heatmap_dir>/<stem>/debug/skeleton.png"""
    p = base_dir / "debug" / "skeleton.png"
    return p if p.exists() else None


def _base_dir_for_heatmap(heatmap_path: Path) -> Path:
    return heatmap_path.parent / heatmap_path.stem


def _gather_inputs(data_dir: Path, cfg: dict, limit: int | None, dpi: int) -> List[Path]:
    """Return a list of heatmap images to run through KymoButler."""
    # pick up any images already present
    img_globs = (cfg.get("io", {}) or {}).get("image_globs", ["*.png", "*.jpg", "*.jpeg"])
    images = list_files(data_dir, img_globs)

    # convert tables -> heatmaps (respect limit if set)
    table_globs = (cfg.get("io", {}) or {}).get("table_globs", ["*.csv", "*.tsv", "*.xls", "*.xlsx"])
    tables = list_files(data_dir, table_globs)
    if limit is not None and limit > 0:
        # prioritize CSVs for your dataset
        csvs = [p for p in tables if p.suffix.lower() == ".csv"][:limit]
        others = [p for p in tables if p.suffix.lower() != ".csv"][: max(0, limit - len(csvs))]
        tables = csvs + others

    out_dir = ensure_dir(data_dir / "generated_heatmaps")
    hm_cfg = cfg.get("heatmap", {}) or {}
    lower = float(hm_cfg.get("lower", -1e20))
    upper = float(hm_cfg.get("upper", 1e16))
    binarize = bool(hm_cfg.get("binarize", True))
    origin = str(hm_cfg.get("origin", "lower"))
    cmap = str(hm_cfg.get("cmap", "hot"))

    generated: List[Path] = []
    for i, tbl in enumerate(tables, start=1):
        try:
            out_img, _, _ = table_to_heatmap(
                tbl, out_dir=out_dir, lower=lower, upper=upper,
                binarize=binarize, origin=origin, cmap=cmap, dpi=dpi
            )
            generated.append(out_img)
        except Exception as e:
            get_logger("skeleton_tester").exception(f"Heatmap generation failed for {tbl}: {e}")

    # final list, images first to keep paths stable
    all_imgs = [p for p in map(Path, images) if _is_image(p)] + generated
    all_imgs = [p for p in all_imgs
            if "/debug/" not in str(p).replace("\\", "/")
            and "/kymobutler_output/" not in str(p).replace("\\", "/")]
    # filter out pipeline artifacts if any slipped in
    def _is_overlay(p: Path) -> bool:
        s = str(p)
        return s.endswith("_overlay.png") or s.endswith("_overlay.jpg") or "_output/" in s or s.endswith("_output")
    return [p for p in all_imgs if not _is_overlay(p)]


def run_config_skeletons(data_dir: Path, cfg_path: Path, out_root: Path,
                         limit: int | None, verbose: bool) -> Path:
    cfg = load_config(cfg_path)
    loglevel = (cfg.get("logging", {}) or {}).get("level", "INFO")
    setup_logging(loglevel)
    log = get_logger(f"skeleton_tester.{cfg_path.stem}")
    if verbose:
        log.setLevel("DEBUG")

    run_kymo, run_kwargs, backend = _select_kymo_backend(cfg, log, verbose)

    # force debug images on, since we only need skeletons
    run_kwargs = dict(run_kwargs)
    run_kwargs["debug_save_images"] = True

    dpi = int((cfg.get("viz", {}) or {}).get("dpi", 180))
    inputs = _gather_inputs(data_dir, cfg, limit, dpi)
    if not inputs:
        print(f"[{cfg_path.name}] No inputs found in {data_dir}")
        return out_root / cfg_path.stem

    run_out = ensure_dir(out_root / cfg_path.stem)
    qc_dir = ensure_dir(run_out / "quickcheck")
    print(f"[{cfg_path.name}] Backend={backend}  Inputs={len(inputs)}  Out={run_out}")

    for idx, img in enumerate(inputs, start=1):
        print(f"  [{idx:>3}/{len(inputs)}] {img.name} … ", end="", flush=True)
        try:
            base_dir = run_kymo(str(img), **run_kwargs)
            skel_png = _find_skeleton_png(base_dir)
            if not skel_png:
                print("no skeleton.png (skipped)")
                continue

            base_gray = cv2.imread(str(img), cv2.IMREAD_GRAYSCALE)
            if base_gray is None:
                print("could not read base image (skipped)")
                continue
            base_u8 = _normalize_gray(base_gray)

            skel = cv2.imread(str(skel_png), cv2.IMREAD_GRAYSCALE)
            if skel is None:
                print("could not read skeleton.png (skipped)")
                continue

            overlay = _color_overlay(base_u8, skel, base_w=0.85, skel_w=0.65, color=(0, 255, 0))
            side = _side_by_side(cv2.cvtColor(base_u8, cv2.COLOR_GRAY2BGR), overlay)

            stem = img.stem
            cv2.imwrite(str(qc_dir / f"{stem}__overlay.png"), overlay)
            cv2.imwrite(str(qc_dir / f"{stem}__base_vs_overlay.png"), side)
            print("ok")
        except Exception as e:
            print(f"error: {e}")

    return run_out


def main():
    ap = argparse.ArgumentParser(description="A/B test YAML configs: produce skeleton overlays only (no waves).")
    ap.add_argument("--data-dir", type=Path, required=True, help="Folder with CSV/XLS/PNG inputs (e.g., ./data)")
    ap.add_argument("--configs", type=Path, nargs="+", required=True, help="One or more YAML config files to test")
    ap.add_argument("--out", type=Path, default=Path("./runs_skeletons"), help="Output root directory")
    ap.add_argument("--limit", type=int, default=None, help="Limit number of tables converted per config (quick pass)")
    ap.add_argument("--quiet", action="store_true", help="Reduce logging verbosity")
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    verbose = not args.quiet

    t0 = time.time()
    for cfg in args.configs:
        t1 = time.time()
        out_dir = run_config_skeletons(args.data_dir, cfg, args.out, args.limit, verbose)
        print(f"-> {cfg.name} done: {out_dir} (Δ {time.time()-t1:.1f}s)")
    print(f"ALL DONE in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
