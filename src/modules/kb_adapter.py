from __future__ import annotations
from pathlib import Path
from typing import Union, List, Tuple, Optional
import numpy as np
import cv2
from skimage.filters import apply_hysteresis_threshold
from skimage.morphology import thin as _thin

# ORT-backed wrappers + helpers (make sure these exist in kymobutler_pt.py)
from .kymobutler_pt import (
    KymoButlerPT,
    prob_to_mask,
    filter_components,
    prune_endpoints,
)

# tracker
from .tracker import CrossingTracker, Track, enforce_one_point_per_row


# ---------------------------
# Small utilities
# ---------------------------

def _track_score(prob: np.ndarray, t) -> float:
    """Median prob along track (robust to a few weak pixels)."""
    if not t.points: return 0.0
    ys, xs = zip(*t.points)
    return float(np.median(prob[tuple(np.array(ys)), tuple(np.array(xs))]))

def _row_overlap(a_pts, b_pts) -> float:
    """Fraction of overlapped rows between two y-sorted tracks."""
    ay = {y for y,_ in a_pts}
    by = {y for y,_ in b_pts}
    if not ay or not by: return 0.0
    inter = len(ay & by)
    return inter / float(min(len(ay), len(by)))

def _mean_dx_on_overlap(a_pts, b_pts) -> float:
    """Mean |dx| where rows overlap (nearest x if multiple per row)."""
    from collections import defaultdict
    ax = defaultdict(list); bx = defaultdict(list)
    for y,x in a_pts: ax[y].append(x)
    for y,x in b_pts: bx[y].append(x)
    ys = sorted(set(ax) & set(bx))
    if not ys: return 1e9
    diffs = []
    for y in ys:
        xa = min(ax[y], key=lambda v: abs(v - np.median(ax[y])))
        xb = min(bx[y], key=lambda v: abs(v - np.median(bx[y])))
        diffs.append(abs(xa - xb))
    return float(np.mean(diffs)) if diffs else 1e9

def filter_and_dedupe_tracks(tracks, prob,
                             min_rows=30,           # same as save gate, or looser
                             min_score=0.11,        # prob floor for acceptance
                             overlap_iou=0.80,      # row-coverage threshold to consider duplicates
                             dx_tol=2.5):           # average pixel distance on overlaps
    """Keep strong tracks, remove near-duplicates (keep the stronger by score)."""
    # precompute score/length
    enriched = []
    for t in tracks:
        pts = sorted(t.points, key=lambda p: (p[0], p[1]))
        if len(pts) < min_rows: 
            continue
        score = _track_score(prob, t)
        if score < min_score:
            continue
        enriched.append((t, pts, score, len(pts)))

    # sort by score desc then length desc
    enriched.sort(key=lambda z: (z[2], z[3]), reverse=True)

    kept = []
    for t, pts, score, _ in enriched:
        dup = False
        for kt, kpts, kscore, _ in kept:
            ov = _row_overlap(pts, kpts)
            if ov >= overlap_iou and _mean_dx_on_overlap(pts, kpts) <= dx_tol:
                dup = True
                break
        if not dup:
            kept.append((t, pts, score, _))

    return [z[0] for z in kept]

# --- graph / skeleton helpers (adaptive cleaning) ---
_OFFSETS_8 = [(-1,-1), (-1,0), (-1,1),
              ( 0,-1),         ( 0,1),
              ( 1,-1), ( 1,0), ( 1,1)]

def _neighbors8(y, x, H, W):
    for dy, dx in _OFFSETS_8:
        ny, nx = y + dy, x + dx
        if 0 <= ny < H and 0 <= nx < W:
            yield ny, nx

def _degree_map(skel: np.ndarray) -> np.ndarray:
    """3x3 neighbor count (degree) for binary skeleton."""
    k = np.ones((3, 3), np.uint8)
    k[1, 1] = 0
    # cv2.filter2D with uint8 is fine here; result ≤ 8
    deg = cv2.filter2D(skel.astype(np.uint8), ddepth=cv2.CV_8U, kernel=k, borderType=cv2.BORDER_CONSTANT)
    return deg

def _spur_prune_by_len(skel: np.ndarray, max_len: int = 5) -> np.ndarray:
    """Remove endpoint→junction twigs of length ≤ max_len (iterative)."""
    sk = skel.copy().astype(np.uint8)
    H, W = sk.shape
    changed = True
    while changed:
        changed = False
        ys, xs = np.where(sk == 1)
        for y0, x0 in zip(ys, xs):
            if sk[y0, x0] == 0:
                continue
            # endpoint?
            deg0 = 0
            for ny, nx in _neighbors8(y0, x0, H, W):
                if sk[ny, nx] == 1:
                    deg0 += 1
            if deg0 != 1:
                continue
            # walk forward until junction/end or exceed max_len
            py, px = -1, -1
            y, x = y0, x0
            trail = []
            steps = 0
            while steps <= max_len:
                trail.append((y, x))
                steps += 1
                nbrs = [(ny, nx) for ny, nx in _neighbors8(y, x, H, W) if sk[ny, nx] == 1 and (ny, nx) != (py, px)]
                if len(nbrs) == 0:
                    # dead end -> remove trail
                    for yy, xx in trail:
                        sk[yy, xx] = 0
                    changed = True
                    break
                if len(nbrs) >= 2:
                    # hit junction: remove if short
                    if steps - 1 <= max_len:
                        for yy, xx in trail:
                            sk[yy, xx] = 0
                        changed = True
                    break
                # len(nbrs) == 1: continue walking
                py, px = y, x
                y, x = nbrs[0]
    return sk

def _junction_nms(skel: np.ndarray, prob: np.ndarray) -> np.ndarray:
    """Only suppress non-maxima at junctions (deg ≥ 3) in 3x3 windows."""
    H, W = skel.shape
    deg = _degree_map(skel)
    keep = skel.copy().astype(np.uint8)
    ys, xs = np.where((skel == 1) & (deg >= 3))
    for y, x in zip(ys, xs):
        p0 = prob[y, x]
        # if any skeleton neighbor in 3x3 has strictly higher prob, drop center
        if np.any((skel[max(0,y-1):min(H,y+2), max(0,x-1):min(W,x+2)] == 1) &
                  (prob[max(0,y-1):min(H,y+2), max(0,x-1):min(W,x+2)] > p0)):
            keep[y, x] = 0
    return keep


def _track_len_rows(t: Track) -> int:
    if not t.points:
        return 0
    ys = [p[0] for p in t.points]
    return int(max(ys) - min(ys) + 1)

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


# ---------------------------
# Public: preprocessing wrapper (kept for API parity)
# ---------------------------

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
# Track refinement helpers
# ---------------------------

def _extend_one_end(prob: np.ndarray, start_y: int, start_x: int, step: int,
                    max_rows: int = 8, dx_win: int = 3, prob_min: float = 0.12) -> List[Tuple[int,int]]:
    """
    Greedy 1-pixel-per-row extension along +step in y. Keeps to a small lateral window.
    Returns list of (y,x) appended in the extension direction.
    """
    H, W = prob.shape
    y, x = start_y, start_x
    out: List[Tuple[int,int]] = []
    for _ in range(max_rows):
        y2 = y + step
        if not (0 <= y2 < H):
            break
        x0, x1 = max(0, x - dx_win), min(W - 1, x + dx_win)
        if x1 < x0:
            break
        row = prob[y2, x0:x1 + 1]
        if row.size == 0:
            break
        off = int(np.argmax(row))
        x2 = x0 + off
        p = float(prob[y2, x2])
        if p < prob_min:
            break
        out.append((y2, x2))
        y, x = y2, x2
    return out

def _merge_pairwise(tracks: List[Track], prob: np.ndarray,
                    max_gap_rows: int = 6, max_dx: int = 4,
                    prob_bridge_min: float = 0.10) -> List[Track]:
    """
    Greedy merge: if end of A is close in y/x to start of B (and prob along a straight
    bridge is reasonable), concatenate.
    """
    if not tracks:
        return []

    def start(t: Track) -> Tuple[int,int]: return t.points[0]
    def end(t: Track)   -> Tuple[int,int]: return t.points[-1]

    used = [False] * len(tracks)
    merged: List[Track] = []

    order = sorted(range(len(tracks)), key=lambda i: start(tracks[i])[0])
    for i in order:
        if used[i]:
            continue
        ti = tracks[i]
        changed = True
        while changed:
            changed = False
            ey, ex = end(ti)
            for j in order:
                if used[j] or j == i:
                    continue
                sjy, sjx = start(tracks[j])
                gap = sjy - ey
                if 0 < gap <= max_gap_rows and abs(sjx - ex) <= max_dx:
                    n = max(1, gap)
                    ys = np.linspace(ey, sjy, n + 2, dtype=int)[1:-1]
                    xs = np.linspace(ex, sjx, n + 2, dtype=int)[1:-1]
                    bridge_p = float(prob[ys, xs].mean()) if len(ys) else 1.0
                    if bridge_p >= prob_bridge_min:
                        ti = type(ti)(points=ti.points + tracks[j].points, id=ti.id)
                        used[j] = True
                        changed = True
                        break
        merged.append(ti)
        used[i] = True
    return merged

def refine_tracks(tracks: List[Track], prob: np.ndarray, *,
                  extend_rows: int = 10, dx_win: int = 3, prob_min: float = 0.12,
                  max_gap_rows: int = 6, max_dx: int = 4, prob_bridge_min: float = 0.10,
                  min_track_len_for_merge: int = 8) -> List[Track]:
    """
    1) Extend both ends using the prob map (keeps 1 px/row).
    2) Merge collinear/nearby segments across small gaps.
    """
    if not tracks:
        return []

    refined: List[Track] = []
    for t in tracks:
        if not t.points:
            continue
        pts = sorted(t.points, key=lambda p: (p[0], p[1]))
        hy, hx = pts[0]
        ty, tx = pts[-1]
        head_ext = _extend_one_end(prob, hy, hx, step=-1, max_rows=extend_rows, dx_win=dx_win, prob_min=prob_min)
        tail_ext = _extend_one_end(prob, ty, tx, step=+1, max_rows=extend_rows, dx_win=dx_win, prob_min=prob_min)
        pts = list(reversed(head_ext)) + pts + tail_ext
        refined.append(type(t)(points=pts, id=t.id))

    short_first = sorted(
        refined,
        key=lambda t: _track_len_rows(t) < max(min_track_len_for_merge, 8),
        reverse=True
    )
    merged = _merge_pairwise(
        short_first, prob,
        max_gap_rows=max_gap_rows, max_dx=max_dx, prob_bridge_min=prob_bridge_min
    )

    # enforce one-point-per-row
    for i, t in enumerate(merged):
        merged[i] = type(t)(points=enforce_one_point_per_row(t.points), id=t.id)
    return merged


# ---------------------------
# Threshold helper
# ---------------------------

def _auto_threshold(prob: np.ndarray,
                    sweep: Tuple[float, float, int] = (0.12, 0.30, 19),
                    target_mask_pct: Tuple[float, float] = (15.0, 25.0)) -> float:
    lo, hi, n = sweep
    cand = np.linspace(lo, hi, int(n))
    target = 0.01 * (target_mask_pct[0] + target_mask_pct[1]) / 2.0
    best_t, best_err = float(cand[0]), 1e9
    for t in cand:
        pct = float((prob > t).mean())
        err = abs(pct - target)
        if err < best_err:
            best_t, best_err = float(t), err
    return best_t


# ---------------------------
# Geometry helpers
# ---------------------------

def _scale_tracks_to_original(
    tracks: List[Track], seg_hw: Tuple[int, int], orig_hw: Tuple[int, int]
) -> List[Track]:
    """Map (y,x) from segmentation size -> original size."""
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


# ---------------------------
# Main entry
# ---------------------------

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
    thr_uni: Optional[float] = None,
    thr_bi: Optional[float] = None,

    # WL-ish cleaning knobs
    morph_close_open_close: bool = False,
    comp_min_px: int = 10,
    comp_min_rows: int = 10,
    prune_iters: int = 2,

    # auto-threshold controls
    auto_threshold: bool = True,
    auto_target_pct: Tuple[float, float] = (15.0, 25.0),
    auto_sweep: Tuple[float, float, int] = (0.12, 0.30, 19),
    auto_trigger_pct: Tuple[float, float] = (5.0, 35.0),

    # hysteresis & closing
    hysteresis_enable: bool = True,
    hysteresis_low: float = 0.10,
    hysteresis_high: float = 0.20,
    directional_close: bool = True,

    # fusion
    fuse_uni_into_bi: bool = True,
    fuse_uni_weight: float = 0.7
) -> Path:
    """
    Pure-Python replacement for the Mathematica runner.

    Steps: classify → segment → (auto/hysteresis) threshold → morphology → component filter
           → thin(+prune) → crossing-aware tracking → refine/merge → save .npy tracks
    """
    heatmap_path = Path(heatmap_path)
    base_dir = heatmap_path.parent / heatmap_path.stem
    out_dir = base_dir / "kymobutler_output"
    out_dir.mkdir(parents=True, exist_ok=True)

    # original grayscale
    gray_orig = cv2.imread(str(heatmap_path), cv2.IMREAD_GRAYSCALE)
    if gray_orig is None:
        raise FileNotFoundError(heatmap_path)
    H, W = gray_orig.shape

    # models
    kb = KymoButlerPT(export_dir=export_dir, seg_size=seg_size)

    # classify (we usually force 'bi')
    cls = kb.classify(gray_orig)
    mode = "bi" if cls["label"] == 1 else "uni"
    if force_mode in {"uni", "bi"}:
        mode = force_mode
    if verbose:
        print(f"[classify] probs={cls['probs']} -> mode={mode}" + (" [forced]" if force_mode else ""))

    # per-mode thresholds
    t_uni = thr if thr_uni is None else thr_uni
    t_bi  = thr if thr_bi  is None else thr_bi

    # ---- segmentation (prob in seg space) ----
    if mode == "uni":
        out = kb.segment_uni_full(gray_orig)
        prob = np.maximum(out["ant"], out["ret"]).astype(np.float32)
        used_thr = t_uni
    else:
        prob_bi = kb.segment_bi_full(gray_orig).astype(np.float32)

        if fuse_uni_into_bi:
            outu = kb.segment_uni_full(gray_orig)
            prob_uni = np.maximum(outu["ant"], outu["ret"]).astype(np.float32)

            # safety: ensure both maps have identical H×W before fusing
            if prob_uni.shape != prob_bi.shape:
                prob_uni = cv2.resize(
                    prob_uni,
                    (prob_bi.shape[1], prob_bi.shape[0]),
                    interpolation=cv2.INTER_LINEAR,
                )
            prob = np.maximum(prob_bi, float(fuse_uni_weight) * prob_uni)
        else:
            prob = prob_bi

        used_thr = t_bi

    # ---- threshold + WL-like shaping ----
    mask0 = prob_to_mask(prob, thr=used_thr)

    # hysteresis (keeps weak connected to strong)
    hmask = None
    if hysteresis_enable:
        try:
            hmask = apply_hysteresis_threshold(prob.astype(np.float32), hysteresis_low, hysteresis_high)
            mask0 = (hmask.astype(np.uint8))
        except Exception:
            # fall back to plain threshold if skimage isn't happy
            hmask = None

    pct0 = float(mask0.mean()) * 100.0
    if auto_threshold and (pct0 < auto_trigger_pct[0] or pct0 > auto_trigger_pct[1]):
        used_thr = _auto_threshold(prob, sweep=auto_sweep, target_mask_pct=auto_target_pct)
        mask0 = prob_to_mask(prob, thr=used_thr)
        pct0 = float(mask0.mean()) * 100.0

    # morphology: prefer directional close (v/h + diagonals) over open/close/open
    mask = mask0.copy()
    if morph_close_open_close:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  k)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
    elif directional_close:
        kv = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 5))  # vertical
        kh = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 1))  # horizontal
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kv)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kh)

        # diagonal / cross bridges (3x3 with center anchor)
        kdl = np.array([[1, 0, 1],
                        [0, 1, 0],
                        [1, 0, 1]], dtype=np.uint8)   # connects diag gaps
        kcr = np.array([[0, 1, 0],
                        [1, 1, 1],
                        [0, 1, 0]], dtype=np.uint8)   # plus connectivity
        mask_dl = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kdl)
        mask_cr = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kcr)
        mask = np.maximum(mask, np.maximum(mask_dl, mask_cr)).astype(np.uint8)

    # component filter prior to skeletonization
    mask_f = filter_components(mask, min_px=comp_min_px, min_rows=comp_min_rows)

    # --- skeletonization: THIN to preserve obliques
    skel_base = _thin(mask_f.astype(bool)).astype(np.uint8)

    # Adaptive cleaning guardrails
    BASE_PX = int(skel_base.sum())
    MIN_KEEP_RATIO = 0.60     # never drop below 60% of base skeleton pixels
    MIN_KEEP_PX    = max(2000, int(MIN_KEEP_RATIO * BASE_PX))  # absolute floor, protects sparse images

    # 1) Probability floor on corridors only (do NOT trim junctions)
    skel = skel_base.copy()
    deg  = _degree_map(skel)
    corridor = (skel == 1) & (deg <= 2)

    if np.any(skel):
        vals = prob[skel == 1]
        # dynamic floor: 10th percentile, clipped to [0.06, 0.10]
        p10 = float(np.percentile(vals, 10.0)) if vals.size else 0.08
        PROB_FLOOR = min(0.10, max(0.06, p10))
    else:
        PROB_FLOOR = 0.08

    to_drop = corridor & (prob < PROB_FLOOR)
    skel[to_drop] = 0
    skel = _thin(skel.astype(bool)).astype(np.uint8)

    # Guardrail: if we removed too much, back off the floor
    if int(skel.sum()) < MIN_KEEP_PX:
        skel = skel_base.copy()

    # 2) Junction-only NMS (reduce hair only where necessary)
    skel_nms = _junction_nms(skel, prob)
    skel_nms = _thin(skel_nms.astype(bool)).astype(np.uint8)

    # Guardrail: if this step nukes too much, skip it
    if int(skel_nms.sum()) >= MIN_KEEP_PX:
        skel = skel_nms
    # else keep 'skel' from previous step

    # 3) Spur pruning with backoff if too destructive
    for spur_len in [5, 4, 3]:
        skel_try = _spur_prune_by_len(skel, max_len=spur_len)
        if int(skel_try.sum()) >= MIN_KEEP_PX:
            skel = skel_try
            break  # accepted
    # optional small endpoint prune (your existing knob)
    if prune_iters > 0:
        skel = prune_endpoints(skel, iterations=prune_iters)

    if verbose:
        print("[debug.skel] base_px:", BASE_PX,
            " after_floor:", int((_thin((skel_base & corridor & (prob >= PROB_FLOOR)).astype(bool)).sum())) if BASE_PX else 0,
            " final_px:", int(skel.sum()),
            " keep_floor:", MIN_KEEP_PX)

    if verbose:
        dbg = base_dir / "debug"
        dbg.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(dbg / "prob.png"), (prob * 255).astype(np.uint8))
        cv2.imwrite(str(dbg / "mask_raw.png"), (mask0 * 255))
        cv2.imwrite(str(dbg / "mask_clean.png"), (mask * 255))
        cv2.imwrite(str(dbg / "mask_filtered.png"), (mask_f * 255))
        cv2.imwrite(str(dbg / "skeleton.png"), (skel * 255))
        if hmask is not None:
            cv2.imwrite(str(dbg / "mask_hysteresis.png"), (hmask.astype(np.uint8) * 255))
        cv2.imwrite(str(dbg / "mask_after_dirclose.png"), (mask * 255))
        print("[debug] prob range:", float(prob.min()), "→", float(prob.max()),
              "  mask_raw %:", round(pct0, 2),
              "  thr_used:", used_thr,
              "  mask_clean %:", round(float(mask.mean()) * 100.0, 2),
              "  mask_filtered %:", round(float(mask_f.mean()) * 100.0, 2),
              "  skel_px:", int(skel.sum()))

    # ---- tracking on seg-sized, WL-preprocessed gray ----
    gray_seg = kb.preproc_for_seg(gray_orig, hw=prob.shape)
    tracker = CrossingTracker(
        kb,
        max_branch_steps=256,
        min_track_len=max(5, min_length // 3),
        decision_recent_tail=16,
    )
    tracks_seg = tracker.extract_tracks(gray_seg, skel)

    # refine (extend + merge) using prob
    tracks_seg = refine_tracks(
        tracks_seg, prob,
        extend_rows=22,
        dx_win=4,
        prob_min=0.11,
        max_gap_rows=13,
        max_dx=6,
        prob_bridge_min=0.11
    )

    # quality filter + dedupe
    tracks_seg = filter_and_dedupe_tracks(
        tracks_seg, prob,
        min_rows=min_length,        # match save gate
        min_score=0.11,             # tune between 0.10–0.14
        overlap_iou=0.80,
        dx_tol=2.5
    )

    if verbose and tracks_seg:
        lengths = [_track_len_rows(t) for t in tracks_seg]
        print(f"[debug] track rows: n={len(lengths)}  "
              f"p50={int(np.percentile(lengths, 50))}  "
              f"p90={int(np.percentile(lengths, 90))}  "
              f"max={max(lengths)}")

    # scale back to original resolution
    tracks = _scale_tracks_to_original(
        tracks_seg,
        seg_hw=prob.shape,          # <-- was (seg_size, seg_size)
        orig_hw=(H, W),
    )

    # final save gate
    saved = _save_npy_tracks(tracks, out_dir, min_length=min_length)
    if verbose:
        print(f"[kymobutler_pt] extracted {len(tracks)} track(s); saved {saved} ≥ {min_length} to {out_dir}")

    # overlay for quick QA
    if verbose:
        overlay = cv2.cvtColor(gray_orig, cv2.COLOR_GRAY2BGR)
        for t in tracks:
            for y, x in t.points:
                cv2.circle(overlay, (int(x), int(y)), 1, (0, 255, 0), -1)
        cv2.imwrite(str(base_dir / "overlay_tracks.png"), overlay)

    return base_dir
