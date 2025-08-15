from __future__ import annotations
from pathlib import Path
from typing import Union, List, Tuple, Optional
from skimage.filters import apply_hysteresis_threshold
from skimage.morphology import thin as _thin
import numpy as np
import cv2

# ORT-backed wrappers + helpers
from .kymobutler_pt import (
    KymoButlerPT,
    prob_to_mask,
    thin_and_prune,        # new: WL-like skeleton op
    filter_components,     # new: WL-like component filter
    skeletonize,
    prune_endpoints,
)
from .tracker import CrossingTracker, Track, enforce_one_point_per_row

# ---------------------------
# Python-only preprocessing
# ---------------------------

def _track_len_rows(t):
    if not t.points: return 0
    ys = [p[0] for p in t.points]
    return max(ys) - min(ys) + 1

def _extend_one_end(prob, start_y, start_x, step, max_rows=8, dx_win=3, prob_min=0.12):
    """
    Greedy 1-pixel-per-row extension along +step in y. Keeps to a small lateral window.
    Returns list of (y,x) appended in the extension direction.
    """
    H, W = prob.shape
    y, x = start_y, start_x
    out = []
    for _ in range(max_rows):
        y2 = y + step
        if not (0 <= y2 < H): break
        x0, x1 = max(0, x - dx_win), min(W - 1, x + dx_win)
        row = prob[y2, x0:x1+1]
        if row.size == 0: break
        off = int(np.argmax(row))
        x2 = x0 + off
        p = float(prob[y2, x2])
        if p < prob_min: break
        out.append((y2, x2))
        y, x = y2, x2
    return out

def _merge_pairwise(tracks, prob, max_gap_rows=6, max_dx=4, prob_bridge_min=0.10):
    """
    Greedy merge: if end of A is close in y/x to start of B (and prob along the
    straight bridge is reasonable), concatenate.
    Returns a new list of merged tracks.
    """
    # index starts/ends
    def start(t): return t.points[0]
    def end(t):   return t.points[-1]
    used = [False]*len(tracks)
    merged = []
    # sort by start y
    order = sorted(range(len(tracks)), key=lambda i: start(tracks[i])[0])
    for i in order:
        if used[i]: continue
        ti = tracks[i]
        changed = True
        while changed:
            changed = False
            ey, ex = end(ti)
            # candidates that start shortly after ey
            for j in order:
                if used[j] or j == i: continue
                sjy, sjx = start(tracks[j])
                gap = sjy - ey
                if 0 < gap <= max_gap_rows and abs(sjx - ex) <= max_dx:
                    # cheap bridge score along a straight line in (y,x)
                    n = max(1, gap)
                    ys = np.linspace(ey, sjy, n+2, dtype=int)[1:-1]
                    xs = np.linspace(ex, sjx, n+2, dtype=int)[1:-1]
                    bridge_p = float(prob[ys, xs].mean()) if len(ys) else 1.0
                    if bridge_p >= prob_bridge_min:
                        # merge
                        ti = type(ti)(points=ti.points + tracks[j].points, id=ti.id)
                        used[j] = True
                        changed = True
                        break
        merged.append(ti)
        used[i] = True
    return merged

def refine_tracks(tracks, prob, *, extend_rows=10, dx_win=3, prob_min=0.12,
                  max_gap_rows=6, max_dx=4, prob_bridge_min=0.10,
                  min_track_len_for_merge=8):
    """
    1) Extend both ends using the prob map (keeps 1 px/row).
    2) Merge collinear/nearby segments across small gaps.
    """
    refined = []
    for t in tracks:
        if not t.points: continue
        # ensure y-sorted
        pts = sorted(t.points, key=lambda p: (p[0], p[1]))
        # extend head (toward smaller y) and tail (larger y)
        hy, hx = pts[0]
        ty, tx = pts[-1]
        head_ext = _extend_one_end(prob, hy, hx, step=-1, max_rows=extend_rows, dx_win=dx_win, prob_min=prob_min)
        tail_ext = _extend_one_end(prob, ty, tx, step=+1, max_rows=extend_rows, dx_win=dx_win, prob_min=prob_min)
        pts = list(reversed(head_ext)) + pts + tail_ext
        refined.append(type(t)(points=pts, id=t.id))

    # merge short segments first, they’re the ones that need help
    short_first = sorted(refined, key=lambda t: _track_len_rows(t) < max(min_track_len_for_merge, 8), reverse=True)
    merged = _merge_pairwise(short_first, prob,
                             max_gap_rows=max_gap_rows, max_dx=max_dx, prob_bridge_min=prob_bridge_min)
    # keep one-point-per-row property
    for i, t in enumerate(merged):
        merged[i] = type(t)(points=enforce_one_point_per_row(t.points), id=t.id)
    return merged

def _prep_gray(img_path: Union[str, Path]) -> np.ndarray:
    gray = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
    if gray is None:
        raise FileNotFoundError(img_path)
    return gray

def _clahe(img: np.ndarray, clip=2.0, tile=(8, 8)) -> np.ndarray:
    clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=tile)
    return clahe.apply(img)

def _denoise(img: np.ndarray, ksize: int = 3) -> np.ndarray:
    if ksize <= 1:
        return img
    return cv2.GaussianBlur(img, (ksize | 1, ksize | 1), 0)

def preprocess_image(
    image_path: Union[str, Path],
    verbose: bool = False,
    script_name: str = "Run_Kymobutler_Save_Image.wls",  # API-compat; unused
    output_suffix: str = "_processed_2.png",
    use_clahe: bool = True,
    denoise_ksize: int = 0,
) -> Path:
    image_path = Path(image_path)
    proc_dir = image_path.parent / image_path.stem / "proc"
    proc_dir.mkdir(parents=True, exist_ok=True)

    gray = _prep_gray(image_path)
    proc = gray
    if use_clahe:
        proc = _clahe(proc, clip=2.0, tile=(8, 8))
    if denoise_ksize > 1:
        proc = _denoise(proc, denoise_ksize)

    out_path = proc_dir / f"{image_path.stem}{output_suffix}"
    cv2.imwrite(str(out_path), proc)
    if verbose:
        print(f"[preprocess] wrote {out_path} (H,W={proc.shape})")
    return out_path

# ---------------------------
# Helpers for segmentation thresholding
# ---------------------------
def _auto_threshold(prob: np.ndarray,
                    sweep=(0.12, 0.30, 19),
                    target_mask_pct=(15.0, 25.0)) -> float:
    lo, hi, n = sweep
    cand = np.linspace(lo, hi, int(n))
    target = 0.01 * (target_mask_pct[0] + target_mask_pct[1]) / 2.0
    best_t, best_err = float(cand[0]), 1e9
    for t in cand:
        pct = (prob > t).mean()
        err = abs(pct - target)
        if err < best_err:
            best_t, best_err = float(t), err
    return best_t

# ---------------------------
# KymoButler ONNX pipeline
# ---------------------------
def _scale_tracks_to_original(
    tracks: List[Track], seg_hw: Tuple[int, int], orig_hw: Tuple[int, int]
) -> List[Track]:
    segH, segW = seg_hw
    H, W = orig_hw
    sy, sx = H / segH, W / segW
    scaled: List[Track] = []
    for t in tracks:
        pts = [(int(round(y * sy)), int(round(x * sx))) for (y, x) in t.points]
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
    verbose: bool = True,
    script_name: str = "Run_Kymobutler.wls",  # API-compat; unused
    export_dir: Union[str, Path] = None,      # None -> auto-discover via KymoButlerPT
    seg_size: int = 256,
    thr: float = 0.20,                         # legacy single threshold
    min_component_px: int = 5,
    *,
    force_mode: Optional[str] = "bi",          # "uni" | "bi" | None
    thr_uni: Optional[float] = None,           # overrides thr for uni
    thr_bi: Optional[float] = None,            # overrides thr for bi
    # WL-parity knobs:
    morph_close_open_close: bool = True,
    comp_min_px: int = 10,                     # WL SelectComponents count
    comp_min_rows: int = 10,                   # WL SelectComponents vertical span
    prune_iters: int = 3,                      # WL Pruning[..., 3]
    # --- auto-threshold controls ---
    auto_threshold: bool = True,
    auto_target_pct: Tuple[float,float] = (15.0, 25.0),
    auto_sweep: Tuple[float,float,int] = (0.12, 0.30, 19),
    auto_trigger_pct: Tuple[float,float] = (5.0, 35.0),

    hysteresis_enable: bool = True,
    hysteresis_low: float = 0.10,
    hysteresis_high: float = 0.20,
    directional_close: bool = True,
    fuse_uni_into_bi: bool = True,
    fuse_uni_weight: float = 0.7
) -> Path:
    """
    Pure-Python replacement for the Mathematica runner.
    Steps: classify -> segment -> (auto)threshold -> morph -> component filter ->
           thin+prune -> crossing-aware tracking.

    Outputs .npy tracks in:
        <heatmap_dir>/<heatmap_stem>/kymobutler_output/
    Returns base_dir.
    """
    heatmap_path = Path(heatmap_path)
    base_dir = heatmap_path.parent / heatmap_path.stem
    out_dir = base_dir / "kymobutler_output"
    out_dir.mkdir(parents=True, exist_ok=True)

    gray_orig = cv2.imread(str(heatmap_path), cv2.IMREAD_GRAYSCALE)
    if gray_orig is None:
        raise FileNotFoundError(heatmap_path)
    H, W = gray_orig.shape

    # Load models (ORT backend)
    kb = KymoButlerPT(export_dir=export_dir, seg_size=seg_size)

    # 1) classify (we often force 'bi' to mirror WL)
    cls = kb.classify(gray_orig)
    mode = "bi" if cls["label"] == 1 else "uni"
    if force_mode in {"uni", "bi"}:
        mode = force_mode
    if verbose:
        print(f"[classify] probs={cls['probs']} -> mode={mode}" + (" [forced]" if force_mode else ""))

    # per-mode thresholds
    t_uni = thr if thr_uni is None else thr_uni
    t_bi  = thr if thr_bi  is None else thr_bi

    # 2) segmentation (wrapper handles WL preproc & resizing)
    if mode == "uni":
        out = kb.segment_uni(gray_orig)
        prob = np.maximum(out["ant"], out["ret"])
        used_thr = t_uni
    else:
        prob_bi = kb.segment_bi(gray_orig)
        if fuse_uni_into_bi:   # pass cfg in or use function arg
            outu = kb.segment_uni(gray_orig)
            prob_uni = np.maximum(outu["ant"], outu["ret"])
            w = fuse_uni_weight
            prob = np.maximum(prob_bi, w * prob_uni)
        else:
            prob = prob_bi
        used_thr = t_bi

    # 3) threshold + WL-ish cleanups (all in seg space)
    #    auto-tune threshold if extremely sparse or dense
    mask0 = prob_to_mask(prob, thr=used_thr)

    #   Optional: hysteresis bridges weak links connected to strong pixels
    if hysteresis_enable:
        # apply on probs directly; cast to uint8 mask
        hmask = apply_hysteresis_threshold(prob, hysteresis_low, hysteresis_high)
        mask0 = (hmask.astype(np.uint8))

    pct0 = float(mask0.mean()) * 100.0
    if auto_threshold and (pct0 < auto_trigger_pct[0] or pct0 > auto_trigger_pct[1]):
        used_thr = _auto_threshold(prob, sweep=auto_sweep, target_mask_pct=auto_target_pct)
        mask0 = prob_to_mask(prob, thr=used_thr)
        pct0 = float(mask0.mean()) * 100.0

    # Optional morphology: prefer directional closing to preserve diagonals
    mask = mask0.copy()
    if morph_close_open_close:
        # (kept for compatibility, but generally too aggressive for thin streaks)
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  k)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
    elif directional_close:
        """
        kv = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 3))  # vertical
        kh = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 1))  # horizontal
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kv)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kh)
        """
        # stronger directional close
        kv = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 5))  # vertical
        kh = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 1))  # horizontal
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kv)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kh)

        # diagonal bridge with proper 3×3 kernels (center anchor)
        kdl = np.array([[1, 0, 1],
                        [0, 1, 0],
                        [1, 0, 1]], dtype=np.uint8)   # connects / and \ gaps
        kdr = np.array([[0, 1, 0],
                        [1, 1, 1],
                        [0, 1, 0]], dtype=np.uint8)   # connects + gaps robustly
        mask_dl = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kdl)
        mask_dr = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kdr)
        mask = np.maximum(mask, np.maximum(mask_dl, mask_dr)).astype(np.uint8)

    # WL: drop tiny blobs before skeletonization
    mask_f = filter_components(mask, min_px=comp_min_px, min_rows=comp_min_rows)

    # --- skeletonization: THIN preserves obliques better than Zhang–Suen in our case
    skel = _thin(mask_f.astype(bool)).astype(np.uint8)

    # optional: close tiny 1-px voids on the skeleton (bridges tiny breaks), then re-thin
    # (you can skip this if skeleton.png already looks clean)
    skel_closed = cv2.morphologyEx(skel, cv2.MORPH_CLOSE,
                                cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))
    skel = _thin(skel_closed.astype(bool)).astype(np.uint8)

    # --- prune endpoints lightly (equivalent to Pruning[..., prune_iters] in WL)
    if prune_iters > 0:
        skel = prune_endpoints(skel, iterations=prune_iters)  # see helper below

    if verbose:
        dbg = base_dir / "debug"
        dbg.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(dbg / "prob.png"), (prob * 255).astype(np.uint8))
        cv2.imwrite(str(dbg / "mask_raw.png"), (mask0 * 255))
        cv2.imwrite(str(dbg / "mask_clean.png"), (mask * 255))
        cv2.imwrite(str(dbg / "mask_filtered.png"), (mask_f * 255))
        cv2.imwrite(str(dbg / "skeleton.png"), (skel * 255))
        cv2.imwrite(str(dbg / "mask_hysteresis.png"), (hmask.astype(np.uint8) * 255))
        cv2.imwrite(str(dbg / "mask_after_dirclose.png"), (mask * 255))
        print("[debug] prob range:", float(prob.min()), "→", float(prob.max()),
              "  mask_raw %:", round(pct0, 2),
              "  thr_used:", used_thr,
              "  mask_clean %:", round(float(mask.mean()) * 100.0, 2),
              "  mask_filtered %:", round(float(mask_f.mean()) * 100.0, 2),
              "  skel_px:", int(skel.sum()))

    # 4) crossing-aware tracking in seg space (use WL-preprocessed seg-gray)
    gray_seg = kb.preproc_for_seg(gray_orig)
    tracker = CrossingTracker(
        kb,
        max_branch_steps=256,
        min_track_len=max(5, min_length // 3),   # start permissive; final save gate still uses min_length
        decision_recent_tail=16,
    )
    tracks_seg = tracker.extract_tracks(gray_seg, skel)

    # Post-process: extend ends and merge across small gaps using prob
    tracks_seg = refine_tracks(
        tracks_seg, prob,
        extend_rows=20,      # was 16
        dx_win=4,
        prob_min=0.11,
        max_gap_rows=12,     # was 10
        max_dx=6,
        prob_bridge_min=0.09 # was 0.10 (slightly easier to join)
    )
    if verbose:
        lengths = [_track_len_rows(t) for t in tracks_seg]
        if lengths:
            print(f"[debug] track rows: n={len(lengths)}  "
                f"p50={int(np.percentile(lengths,50))}  "
                f"p90={int(np.percentile(lengths,90))}  "
                f"max={max(lengths)}")

    # 5) scale tracks back to original resolution
    tracks = _scale_tracks_to_original(tracks_seg, seg_hw=(seg_size, seg_size), orig_hw=(H, W))

    # 6) save .npy like before (final gate)
    saved = _save_npy_tracks(tracks, out_dir, min_length=min_length)
    if verbose:
        print(f"[kymobutler_pt] extracted {len(tracks)} track(s); saved {saved} ≥ {min_length} to {out_dir}")

    # Optional overlay (uint8)
    if verbose:
        overlay = cv2.cvtColor(gray_orig, cv2.COLOR_GRAY2BGR)
        for t in tracks:
            for y, x in t.points:
                cv2.circle(overlay, (int(x), int(y)), 1, (0, 255, 0), -1)
        cv2.imwrite(str(base_dir / "overlay_tracks.png"), overlay)

    return base_dir
