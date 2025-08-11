from __future__ import annotations
import os
import csv
import subprocess
from pathlib import Path
from typing import Union, List, Optional, Tuple
import numpy as np
import cv2

# Our PyTorch/ONNX wrappers
from kymobutler_pt import KymoButlerPT, prob_to_mask, skeletonize
from tracker import CrossingTracker, Track

# ---------------------------
# Python-only preprocessing
# ---------------------------
def _prep_gray(img_path: Union[str, Path]) -> np.ndarray:
    gray = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
    if gray is None:
        raise FileNotFoundError(img_path)
    return gray

def _clahe(img: np.ndarray, clip=2.0, tile=(8,8)) -> np.ndarray:
    clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=tile)
    return clahe.apply(img)

def _denoise(img: np.ndarray, ksize: int = 3) -> np.ndarray:
    if ksize <= 1:
        return img
    return cv2.GaussianBlur(img, (ksize|1, ksize|1), 0)

def preprocess_image(
    image_path: Union[str, Path],
    verbose: bool = False,
    script_name: str = "Run_Kymobutler_Save_Image.wls",  # kept for API compatibility (ignored)
    output_suffix: str = "_processed_2.png",
    use_clahe: bool = True,
    denoise_ksize: int = 0,
) -> Path:
    """
    Python-only replacement for the Mathematica preprocessor.
    Produces the same output path pattern:
      <image_dir>/<stem>/proc/<stem>_processed_2.png
    """
    image_path = Path(image_path)
    proc_dir = image_path.parent / image_path.stem / "proc"
    proc_dir.mkdir(parents=True, exist_ok=True)

    gray = _prep_gray(image_path)
    proc = gray
    if use_clahe:
        proc = _clahe(proc, clip=2.0, tile=(8,8))
    if denoise_ksize > 1:
        proc = _denoise(proc, denoise_ksize)

    out_path = proc_dir / f"{image_path.stem}{output_suffix}"
    cv2.imwrite(str(out_path), proc)
    if verbose:
        print(f"[preprocess] wrote {out_path} (H,W={proc.shape})")
    return out_path

# ---------------------------
# KymoButler ONNX pipeline
# ---------------------------
def _scale_tracks_to_original(
    tracks: List[Track], seg_hw: Tuple[int,int], orig_hw: Tuple[int,int]
) -> List[Track]:
    """Map track (y,x) from segmentation size -> original size."""
    segH, segW = seg_hw
    H, W = orig_hw
    sy, sx = H / segH, W / segW
    scaled: List[Track] = []
    for t in tracks:
        pts = [(int(round(y*sy)), int(round(x*sx))) for (y, x) in t.points]
        scaled.append(Track(points=pts, id=t.id))
    return scaled

def _save_npy_tracks(tracks: List[Track], out_dir: Path, min_length: int) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    saved = 0
    for i, t in enumerate(tracks):
        arr = np.asarray(t.points, dtype=float)
        if arr.shape[0] < min_length:
            continue
        np.save(out_dir / f"{i}.npy", arr)
        saved += 1
    return saved

def run_kymobutler(
    heatmap_path: Union[str, Path],
    min_length: int = 30,
    verbose: bool = False,
    script_name: str = "Run_Kymobutler.wls",  # kept for API compatibility (ignored)
    export_dir: Union[str, Path] = "export",
    seg_size: int = 256,
    thr: float = 0.20,
    min_component_px: int = 5,
) -> Path:
    """
    Python-only replacement for the Mathematica runner.
    Does: classify -> segment -> threshold -> skeletonize -> crossing-aware tracking.
    Saves .npy tracks in:
        <heatmap_dir>/<heatmap_stem>/kymobutler_output/
    Returns the base_dir (same as your current function).
    """
    heatmap_path = Path(heatmap_path)
    base_dir = heatmap_path.parent / heatmap_path.stem
    out_dir = base_dir / "kymobutler_output"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load original grayscale (we’ll rescale tracks back to this size)
    gray_orig = cv2.imread(str(heatmap_path), cv2.IMREAD_GRAYSCALE)
    if gray_orig is None:
        raise FileNotFoundError(heatmap_path)
    H, W = gray_orig.shape

    # Load models
    kb = KymoButlerPT(export_dir=str(export_dir), seg_size=seg_size)

    # 1) classify
    cls = kb.classify(gray_orig)
    # Adjust label mapping if needed; here assume label 1 => bidirectional
    mode = "bi" if cls["label"] == 1 else "uni"
    if verbose:
        print(f"[classify] probs={cls['probs']} -> mode={mode}")

    # 2) segment at model size (seg_size x seg_size)
    if mode == "uni":
        out = kb.segment_uni(gray_orig)
        prob = np.maximum(out["ant"], out["ret"])
    else:
        prob = kb.segment_bi(gray_orig)

    # 3) threshold + skeletonize (still in seg space)
    mask = prob_to_mask(prob, thr=thr)
    skel = skeletonize(mask, min_component_px=min_component_px)

    # 4) crossing-aware tracking in seg space
    tracker = CrossingTracker(kb)
    tracks_seg = tracker.extract_tracks(gray_orig, skel)  # raw is used for decision crops; size differences don't matter here

    # 5) scale tracks back to original resolution
    tracks = _scale_tracks_to_original(tracks_seg, seg_hw=(seg_size, seg_size), orig_hw=(H, W))

    # 6) save .npy like before
    saved = _save_npy_tracks(tracks, out_dir, min_length=min_length)
    if verbose:
        print(f"[kymobutler_pt] extracted {len(tracks)} track(s); saved {saved} ≥ {min_length} to {out_dir}")

    # Optional: quick overlay for debugging
    if verbose:
        overlay = cv2.cvtColor((gray_orig if gray_orig.max()<=1 else gray_orig)/255.0, cv2.COLOR_GRAY2BGR)
        for t in tracks:
            for y, x in t.points:
                cv2.circle(overlay, (int(x), int(y)), 1, (0,255,0), -1)
        cv2.imwrite(str(base_dir / "overlay_tracks.png"), (overlay*255).astype(np.uint8))

    return base_dir
