from __future__ import annotations
from pathlib import Path
import shutil
from typing import Union, List, Tuple, Optional, Dict

import numpy as np
import cv2
from skimage.filters import apply_hysteresis_threshold
from skimage.morphology import thin as _thin

# ORT-backed wrappers + helpers (must exist in kymobutler_pt.py)
from .kymobutler_pt import (
    KymoButlerPT,
    prob_to_mask,
    filter_components,
    prune_endpoints,
)

# tracker bits
from .tracker import CrossingTracker, Track, enforce_one_point_per_row


# ======================================================
# New: Morphology helpers (WL "classic" + directional)
# ======================================================

def _to_cv(mask01: np.ndarray) -> np.ndarray:
    """0/1 -> 0/255 uint8 for OpenCV morphology."""
    return (mask01.astype(np.uint8) * 255)

def _from_cv(mask255: np.ndarray) -> np.ndarray:
    """0/255 -> 0/1 uint8."""
    return (mask255 > 0).astype(np.uint8)

def morph_classic(mask01: np.ndarray, k: int = 3) -> np.ndarray:
    """WL-like SmoothBin: close -> open -> close with 3x3 ellipse."""
    se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (int(k), int(k)))
    m = _to_cv(mask01)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, se, iterations=1)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN,  se, iterations=1)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, se, iterations=1)
    return _from_cv(m)

def morph_directional(mask01: np.ndarray, kv: int, kh: int, diag_bridge: bool) -> np.ndarray:
    """Anisotropic vertical+horizontal close; optional gentle 3x3 diagonal bridge."""
    m = _to_cv(mask01)
    v = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(1, int(kv))))
    h = cv2.getStructuringElement(cv2.MORPH_RECT, (max(1, int(kh)), 1))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, v, iterations=1)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, h, iterations=1)
    if diag_bridge:
        d = np.ones((3, 3), np.uint8)  # gentle 3x3 bridge
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, d, iterations=1)
    return _from_cv(m)

def weak_only_shave(mask01: np.ndarray, prob: np.ndarray, p_shave: float = 0.12) -> np.ndarray:
    """
    Open only the low-confidence pixels (prob < p_shave) with tiny (1x3, 3x1) kernels,
    then merge back with strong pixels. Shaves whiskers; preserves strong ridges.
    """
    weak = (prob < float(p_shave))
    m = mask01.astype(bool)
    sub = (m & weak).astype(np.uint8) * 255
    k1 = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 3))
    k2 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 1))
    sub = cv2.morphologyEx(sub, cv2.MORPH_OPEN, k1, iterations=1)
    sub = cv2.morphologyEx(sub, cv2.MORPH_OPEN, k2, iterations=1)
    m_weak = (sub > 0)
    m_strong = m & (~weak)
    return (m_weak | m_strong).astype(np.uint8)

def apply_morphology(
    mask01: np.ndarray,
    prob: np.ndarray,
    *,
    mode: str = "classic",          # "classic" | "directional" | "none"
    classic_kernel: int = 3,
    dir_kv: int = 4,
    dir_kh: int = 3,
    diag_bridge: bool = True,
    weak_shave_enable: bool = True,
    p_shave: float = 0.12,
) -> np.ndarray:
    """Router to classic/directional morphology with optional weak-only shave."""
    mode = (mode or "classic").lower()
    if mode == "none":
        m = mask01
    elif mode == "classic":
        m = morph_classic(mask01, k=int(classic_kernel))
    else:
        m = morph_directional(mask01, kv=int(dir_kv), kh=int(dir_kh), diag_bridge=bool(diag_bridge))
        if weak_shave_enable:
            m = weak_only_shave(m, prob, p_shave=float(p_shave))
    return m


# ---------------------------
# Small helpers
# ---------------------------

def _auto_threshold(prob: np.ndarray,
                    sweep: Tuple[float, float, int] = (0.12, 0.30, 19),
                    target_mask_pct: Tuple[float, float] = (15.0, 25.0)) -> float:
    """
    Sweep thresholds in [lo, hi] (N steps). Pick thr whose mask density is
    closest to the midpoint of the target band.
    """
    lo, hi, N = float(sweep[0]), float(sweep[1]), int(sweep[2])
    thr_candidates = np.linspace(lo, hi, max(2, N))
    target_mid = 0.5 * (float(target_mask_pct[0]) + float(target_mask_pct[1]))
    best_thr, best_err = thr_candidates[0], 1e9
    for t in thr_candidates:
        m = prob_to_mask(prob, thr=float(t))  # expected 0/1 mask
        pct = float(m.mean()) * 100.0
        err = abs(pct - target_mid)
        if err < best_err:
            best_thr, best_err = float(t), err
    return float(best_thr)


# ---------------------------
# Track quality / dedupe helpers
# ---------------------------

def _track_score(prob: np.ndarray, t: Track) -> float:
    """Median prob along track (robust to a few weak pixels)."""
    if not t.points:
        return 0.0
    ys, xs = zip(*t.points)
    ys = np.asarray(ys, dtype=int)
    xs = np.asarray(xs, dtype=int)
    return float(np.median(prob[ys, xs]))

def _row_overlap(a_pts: List[Tuple[int,int]], b_pts: List[Tuple[int,int]]) -> float:
    """Fraction of overlapped rows between two y-sorted tracks."""
    ay = {y for y, _ in a_pts}
    by = {y for y, _ in b_pts}
    if not ay or not by:
        return 0.0
    inter = len(ay & by)
    return inter / float(min(len(ay), len(by)))

def _mean_dx_on_overlap(a_pts: List[Tuple[int,int]], b_pts: List[Tuple[int,int]]) -> float:
    """Mean |dx| where rows overlap (nearest x if multiple per row)."""
    from collections import defaultdict
    ax = defaultdict(list); bx = defaultdict(list)
    for y, x in a_pts: ax[y].append(x)
    for y, x in b_pts: bx[y].append(x)
    ys = sorted(set(ax) & set(bx))
    if not ys:
        return 1e9
    diffs = []
    for y in ys:
        xa = min(ax[y], key=lambda v: abs(v - np.median(ax[y])))
        xb = min(bx[y], key=lambda v: abs(v - np.median(bx[y])))
        diffs.append(abs(xa - xb))
    return float(np.mean(diffs)) if diffs else 1e9

def filter_and_dedupe_tracks(
    tracks: List[Track],
    prob: np.ndarray,
    *,
    min_rows: int = 30,
    min_score: float = 0.11,
    overlap_iou: float = 0.80,
    dx_tol: float = 2.5,
) -> List[Track]:
    """Keep strong tracks, remove near-duplicates (keep the stronger by score)."""
    enriched = []
    for t in tracks:
        pts = sorted(t.points, key=lambda p: (p[0], p[1]))
        if len(pts) < min_rows:
            continue
        score = _track_score(prob, t)
        if score < min_score:
            continue
        enriched.append((t, pts, score, len(pts)))

    enriched.sort(key=lambda z: (z[2], z[3]), reverse=True)  # score desc, len desc

    kept = []
    for t, pts, score, _ in enriched:
        dup = False
        for kt, kpts, kscore, _ in kept:
            if _row_overlap(pts, kpts) >= overlap_iou and _mean_dx_on_overlap(pts, kpts) <= dx_tol:
                dup = True
                break
        if not dup:
            kept.append((t, pts, score, _))

    return [z[0] for z in kept]


# ---------------------------
# Skeleton helpers (adaptive clean)
# ---------------------------

_OFFSETS_8 = [(-1,-1), (-1,0), (-1,1),
              ( 0,-1),         ( 0,1),
              ( 1,-1), ( 1,0), ( 1,1)]

def _neighbors8(y: int, x: int, H: int, W: int):
    for dy, dx in _OFFSETS_8:
        ny, nx = y + dy, x + dx
        if 0 <= ny < H and 0 <= nx < W:
            yield ny, nx

def _degree_map(skel: np.ndarray) -> np.ndarray:
    """3x3 neighbor count (degree) for binary skeleton."""
    k = np.ones((3, 3), np.uint8); k[1, 1] = 0
    return cv2.filter2D(skel.astype(np.uint8), ddepth=cv2.CV_8U, kernel=k, borderType=cv2.BORDER_CONSTANT)

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
            # follow twig
            py, px = -1, -1
            y, x = y0, x0
            trail = []
            steps = 0
            while steps <= max_len:
                trail.append((y, x))
                steps += 1
                nbrs = [(ny, nx) for ny, nx in _neighbors8(y, x, H, W) if sk[ny, nx] == 1 and (ny, nx) != (py, px)]
                if len(nbrs) == 0:
                    for yy, xx in trail: sk[yy, xx] = 0
                    changed = True
                    break
                if len(nbrs) >= 2:
                    if steps - 1 <= max_len:
                        for yy, xx in trail: sk[yy, xx] = 0
                        changed = True
                    break
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
        y0, y1 = max(0, y - 1), min(H, y + 2)
        x0, x1 = max(0, x - 1), min(W, x + 2)
        win = (skel[y0:y1, x0:x1] == 1) & (prob[y0:y1, x0:x1] > p0)
        if np.any(win):
            keep[y, x] = 0
    return keep

def _spur_prune_by_prob(skel: np.ndarray, prob: np.ndarray, L: int = 10, p_min: float = 0.13) -> np.ndarray:
    """
    Remove leaf branches up to length L whose mean probability < p_min.
    Confidence-guided variant that preserves strong terminals.
    """
    S = skel.copy().astype(np.uint8)
    H, W = S.shape
    changed = True
    while changed:
        changed = False
        deg = _degree_map(S)
        ys, xs = np.where((S == 1) & (deg == 1))
        for y0, x0 in zip(ys, xs):
            path = [(y0, x0)]
            py, px = -1, -1
            y, x = y0, x0
            for _ in range(L):
                nbrs = [(ny, nx) for ny, nx in _neighbors8(y, x, H, W) if S[ny, nx] == 1 and (ny, nx) != (py, px)]
                if len(nbrs) != 1:
                    break
                py, px = y, x
                y, x = nbrs[0]
                path.append((y, x))
            vals = [prob[yy, xx] for (yy, xx) in path]
            if len(path) <= L and (np.mean(vals) < p_min):
                for (yy, xx) in path:
                    S[yy, xx] = 0
                changed = True
    return S

def _endpoints(skel: np.ndarray) -> List[Tuple[int,int]]:
    H, W = skel.shape
    out = []
    for y, x in zip(*np.where(skel == 1)):
        deg = 0
        for dy, dx in _OFFSETS_8:
            ny, nx = y + dy, x + dx
            if 0 <= ny < H and 0 <= nx < W and skel[ny, nx] == 1:
                deg += 1
        if deg == 1:
            out.append((y, x))
    return out

def _bresenham(y0, x0, y1, x1):
    """Integer line coords from (y0,x0) to (y1,x1)."""
    pts = []
    dy = abs(y1 - y0); dx = abs(x1 - x0)
    sy = 1 if y0 < y1 else -1
    sx = 1 if x0 < x1 else -1
    err = (dy - dx)
    while True:
        pts.append((y0, x0))
        if y0 == y1 and x0 == x1: break
        e2 = 2 * err
        if e2 > -dx: err -= dx; y0 += sy
        if e2 <  dy: err += dy; x0 += sx
    return pts

def _bridge_skeleton_gaps(
    skel: np.ndarray,
    prob: np.ndarray,
    *,
    max_gap_rows: int = 18,
    max_dx: int = 7,
    prob_min: float = 0.11,
    max_bridges: int = 2000
) -> np.ndarray:
    """
    Connect pairs of endpoints if:
      - 0 < Δy ≤ max_gap_rows
      - |Δx| ≤ max_dx
      - mean(prob) along straight line ≥ prob_min
      - line passes mostly through empty skeleton
    Then re-thin to 1 px.
    """
    H, W = skel.shape
    eps = 1e-6
    ends = _endpoints(skel)
    if not ends: return skel

    # bucket endpoints by row for quick lookups
    by_row: Dict[int, List[Tuple[int,int]]] = {}
    for y, x in ends:
        by_row.setdefault(y, []).append((y, x))

    bridges = 0
    sk = skel.copy().astype(np.uint8)

    for y0, x0 in sorted(ends):
        for dy in range(1, max_gap_rows + 1):
            y1 = y0 + dy
            if y1 >= H or y1 not in by_row: break
            # search candidates near x0
            for (yy, xx) in by_row[y1]:
                if abs(xx - x0) > max_dx: continue
                pts = _bresenham(y0, x0, yy, xx)
                yyv, xxv = zip(*pts)
                # require the path to cross mostly empty skeleton
                if sk[yyv, xxv].mean() > 0.25:  # already occupied → likely same component
                    continue
                pmean = float(prob[yyv, xxv].mean()) if pts else 0.0
                if pmean < prob_min: continue
                sk[yyv, xxv] = 1
                bridges += 1
                if bridges >= max_bridges:
                    break
            if bridges >= max_bridges:
                break
        if bridges >= max_bridges:
            break

    # keep skeleton 1-px thin
    sk = _thin(sk.astype(bool)).astype(np.uint8)
    return sk

# ---------------------------
# Track refinement (extend + merge)
# ---------------------------

def _extend_one_end(prob: np.ndarray, start_y: int, start_x: int, step: int,
                    max_rows: int = 8, dx_win: int = 3, prob_min: float = 0.12) -> List[Tuple[int,int]]:
    """Greedy 1-px/row extension along ±y with small lateral window."""
    H, W = prob.shape
    y, x = start_y, start_x
    out: List[Tuple[int,int]] = []
    for _ in range(max_rows):
        y2 = y + step
        if not (0 <= y2 < H): break
        x0, x1 = max(0, x - dx_win), min(W - 1, x + dx_win)
        if x1 < x0: break
        row = prob[y2, x0:x1 + 1]
        if row.size == 0: break
        x2 = x0 + int(np.argmax(row))
        if float(prob[y2, x2]) < prob_min: break
        out.append((y2, x2)); y, x = y2, x2
    return out

def _merge_pairwise(tracks: List[Track], prob: np.ndarray,
                    max_gap_rows: int = 6, max_dx: int = 4,
                    prob_bridge_min: float = 0.10) -> List[Track]:
    """Greedy merge: if end of A is near start of B and bridge prob is OK, concat."""
    if not tracks: return []
    def start(t): return t.points[0]
    def end(t):   return t.points[-1]

    used = [False] * len(tracks)
    merged: List[Track] = []
    order = sorted(range(len(tracks)), key=lambda i: start(tracks[i])[0])

    for i in order:
        if used[i]: continue
        ti = tracks[i]
        changed = True
        while changed:
            changed = False
            ey, ex = end(ti)
            for j in order:
                if used[j] or j == i: continue
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
        merged.append(ti); used[i] = True
    return merged

def refine_tracks(tracks: List[Track], prob: np.ndarray, *,
                  extend_rows: int = 10, dx_win: int = 3, prob_min: float = 0.12,
                  max_gap_rows: int = 6, max_dx: int = 4, prob_bridge_min: float = 0.10,
                  min_track_len_for_merge: int = 8) -> List[Track]:
    """1) Extend both ends using prob  2) Merge across small gaps  3) enforce one-px/row."""
    if not tracks: return []
    refined: List[Track] = []

    for t in tracks:
        if not t.points: continue
        pts = sorted(t.points, key=lambda p: (p[0], p[1]))
        hy, hx = pts[0]; ty, tx = pts[-1]
        head_ext = _extend_one_end(prob, hy, hx, step=-1, max_rows=extend_rows, dx_win=dx_win, prob_min=prob_min)
        tail_ext = _extend_one_end(prob, ty, tx, step=+1, max_rows=extend_rows, dx_win=dx_win, prob_min=prob_min)
        pts = list(reversed(head_ext)) + pts + tail_ext
        refined.append(type(t)(points=pts, id=t.id))

    short_first = sorted(refined, key=lambda t: (_track_len_rows(t) < max(min_track_len_for_merge, 8)), reverse=True)
    merged = _merge_pairwise(short_first, prob, max_gap_rows=max_gap_rows, max_dx=max_dx, prob_bridge_min=prob_bridge_min)
    return [type(t)(points=enforce_one_point_per_row(t.points), id=t.id) for t in merged]


# ---------------------------
# Geometry + IO
# ---------------------------

def _track_len_rows(t: Track) -> int:
    if not t.points: return 0
    ys = [p[0] for p in t.points]
    return int(max(ys) - min(ys) + 1)

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

def _find_run_dir_from(path: Path) -> Path | None:
    """Walk up until we find the run root (has run.json)."""
    p = Path(path).resolve()
    for anc in [p] + list(p.parents):
        if (anc / "run.json").exists():
            return anc
    return None

def _publish_debug_to_runs(heatmap_path: Path, base_dir: Path) -> Path | None:
    """
    Copy <base_dir>/debug/*.png to runs/<run_id>/output/<layer>.png
    Returns the destination directory if anything got copied.
    """
    src_debug = base_dir / "debug"
    if not src_debug.exists():
        return None

    run_dir = _find_run_dir_from(heatmap_path)
    if run_dir is None:
        return None

    dest = run_dir / "output"
    dest.mkdir(parents=True, exist_ok=True)

    copied = False
    for name in ("prob", "mask_raw", "mask_clean", "mask_filtered", "skeleton", "mask_hysteresis"):
        src = src_debug / f"{name}.png"
        if src.exists():
            shutil.copy2(src, dest / f"{name}.png")
            copied = True

    # optional: keep stats for quick inspection
    stats = src_debug / "stats.txt"
    if stats.exists():
        shutil.copy2(stats, dest / "debug_stats.txt")

    return dest if copied else None

# ---------------------------
# Main ONNX-backed runner
# ---------------------------

def run_kymobutler(
    heatmap_path: Union[str, Path],
    min_length: int = 30,
    verbose: bool = True,
    script_name: str = "Run_Kymobutler.wls",  # API-compat; unused
    export_dir: Union[str, Path] = None,
    seg_size: int = 256,
    thr: float = 0.20,
    min_component_px: int = 5,   # legacy compat (not used directly if comp_min_px provided)
    *,
    force_mode: Optional[str] = "bi",          # "uni" | "bi" | None
    thr_uni: Optional[float] = None,
    thr_bi: Optional[float] = None,

    # WL-ish cleaning knobs
    morph_close_open_close: bool = False,       # if True => classic mode
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
    directional_close: bool = True,            # if True (and not classic) => directional mode

    # directional kernel hints (optional; passed by extract.py if present)
    dir_kv: int = 5,               # vertical kernel height
    dir_kh: int = 5,               # horizontal kernel width
    diag_bridge: bool = True,      # enable diagonal/cross bridging
    weak_shave_enable: bool = True,# NEW: shave low-prob whiskers after directional bridging
    weak_shave_p: float = 0.12,    # NEW: probability threshold for weak-only shave

    # fusion
    fuse_uni_into_bi: bool = True,
    fuse_uni_weight: float = 0.7,

    # skeleton guardrails (to avoid over-trimming)
    skel_keep_ratio: float = 0.60,             # keep at least this fraction of base skel px
    skel_keep_min_px: Optional[int] = None,    # absolute min; if None, computed from base
    skel_prob_floor_min: float = 0.06,         # clamp floor lower bound
    skel_prob_floor_max: float = 0.10,         # clamp floor upper bound

    # refine/merge toggles
    refine_enable: bool = True,
    extend_rows: int = 22,
    dx_win: int = 4,
    refine_prob_min: float = 0.11,
    max_gap_rows: int = 13,
    max_dx: int = 6,
    prob_bridge_min: float = 0.11,

    # dedupe toggles
    dedupe_enable: bool = True,
    dedupe_min_rows: Optional[int] = None,  # default to min_length if None
    dedupe_min_score: float = 0.11,
    dedupe_overlap_iou: float = 0.80,
    dedupe_dx_tol: float = 2.5,

    # decision network gate (parity-critical)
    decision_thr: float = 0.50,    # NEW: gate low-confidence branch picks

    # debug file outputs
    debug_save_images: bool = True,

    # accept unknown kwargs without breaking (future-proof with YAML)
    **_
) -> Path:
    """
    ONNX-based KymoButler runner.

    Steps: classify → segment → (auto/hysteresis) threshold → morphology → component filter
           → thin(+adaptive prune) → crossing-aware tracking → refine/merge → dedupe → save
    """
    heatmap_path = Path(heatmap_path)
    base_dir = heatmap_path.parent / heatmap_path.stem
    out_dir = base_dir / "kymobutler_output"
    out_dir.mkdir(parents=True, exist_ok=True)

    gray_orig = cv2.imread(str(heatmap_path), cv2.IMREAD_GRAYSCALE)
    if gray_orig is None:
        raise FileNotFoundError(heatmap_path)
    H, W = gray_orig.shape

    kb = KymoButlerPT(export_dir=export_dir, seg_size=seg_size)

    # classify (often forced to 'bi' to mirror WL)
    cls = kb.classify(gray_orig)
    mode = "bi" if cls["label"] == 1 else "uni"
    if force_mode in {"uni", "bi"}:
        mode = force_mode
    if verbose:
        print(f"[classify] probs={cls['probs']} -> mode={mode}" + (" [forced]" if force_mode else ""))

    # thresholds
    t_uni = thr if thr_uni is None else float(thr_uni)
    t_bi  = thr if thr_bi  is None else float(thr_bi)

    # segmentation (original-size prob maps)
    if mode == "uni":
        out = kb.segment_uni_full(gray_orig)
        prob = np.maximum(out["ant"], out["ret"]).astype(np.float32)
        used_thr = float(t_uni)
    else:
        prob_bi = kb.segment_bi_full(gray_orig).astype(np.float32)
        if fuse_uni_into_bi:
            outu = kb.segment_uni_full(gray_orig)
            prob_uni = np.maximum(outu["ant"], outu["ret"]).astype(np.float32)
            if prob_uni.shape != prob_bi.shape:
                prob_uni = cv2.resize(prob_uni, (prob_bi.shape[1], prob_bi.shape[0]), interpolation=cv2.INTER_LINEAR)
            prob = np.maximum(prob_bi, float(fuse_uni_weight) * prob_uni)
        else:
            prob = prob_bi
        used_thr = float(t_bi)

    # threshold (+ optional hysteresis / auto-threshold)
    mask0 = prob_to_mask(prob, thr=used_thr)  # expected 0/1 uint8

    hmask = None
    if hysteresis_enable:
        try:
            hmask = apply_hysteresis_threshold(prob.astype(np.float32), hysteresis_low, hysteresis_high)
            mask0 = (hmask.astype(np.uint8))
        except Exception:
            hmask = None

    pct0 = float(mask0.mean()) * 100.0
    if auto_threshold and (pct0 < auto_trigger_pct[0] or pct0 > auto_trigger_pct[1]):
        used_thr = _auto_threshold(prob, sweep=auto_sweep, target_mask_pct=auto_target_pct)
        mask0 = prob_to_mask(prob, thr=used_thr)
        pct0 = float(mask0.mean()) * 100.0

    # morphology (WL-parity router)
    morph_mode = "classic" if morph_close_open_close else ("directional" if directional_close else "none")
    mask = apply_morphology(
        mask0, prob,
        mode=morph_mode,
        classic_kernel=3,
        dir_kv=dir_kv, dir_kh=dir_kh, diag_bridge=diag_bridge,
        weak_shave_enable=bool(weak_shave_enable),
        p_shave=float(weak_shave_p),
    )

    # component filter prior to skeletonization
    mask_f = filter_components(mask, min_px=comp_min_px, min_rows=comp_min_rows)

    # skeletonization (thin) + adaptive cleanup with guardrails
    skel_base = _thin(mask_f.astype(bool)).astype(np.uint8)
    BASE_PX = int(skel_base.sum())
    if skel_keep_min_px is None:
        MIN_KEEP_PX = max(2000, int(float(skel_keep_ratio) * max(1, BASE_PX)))
    else:
        MIN_KEEP_PX = int(skel_keep_min_px)

    skel = skel_base.copy()
    if BASE_PX > 0:
        deg = _degree_map(skel)
        corridor = (skel == 1) & (deg <= 2)
        vals = prob[skel == 1]
        p10 = float(np.percentile(vals, 10.0)) if vals.size else 0.08
        # clamp using YAML-configured min/max
        lo = float(skel_prob_floor_min)
        hi = float(skel_prob_floor_max)
        if hi < lo:  # safety swap
            hi, lo = lo, hi
        PROB_FLOOR = max(lo, min(hi, p10))
        to_drop = corridor & (prob < PROB_FLOOR)
        skel[to_drop] = 0
        skel = _thin(skel.astype(bool)).astype(np.uint8)
        if int(skel.sum()) < MIN_KEEP_PX:
            skel = skel_base.copy()

        skel_nms = _junction_nms(skel, prob)
        skel_nms = _thin(skel_nms.astype(bool)).astype(np.uint8)
        if int(skel_nms.sum()) >= MIN_KEEP_PX:
            skel = skel_nms

        # prune short twigs by length (iterative, conservative)
        for spur_len in [7, 6, 5]:
            skel_try = _spur_prune_by_len(skel, max_len=spur_len)
            if int(skel_try.sum()) >= MIN_KEEP_PX:
                skel = skel_try
                break

        # NEW: probability-aware spur pruning (confidence-gated)
        skel_prob = _spur_prune_by_prob(skel, prob, L=10, p_min=0.13)
        if int(skel_prob.sum()) >= MIN_KEEP_PX:
            skel = skel_prob

    if prune_iters > 0:
        skel = prune_endpoints(skel, iterations=prune_iters)
    
    skel = _bridge_skeleton_gaps(
        skel, prob,
        max_gap_rows=max_gap_rows,
        max_dx=max_dx,
        prob_min=prob_bridge_min
    )
    

    if verbose:
        print("[debug.skel] base_px:", BASE_PX,
              " final_px:", int(skel.sum()),
              " keep_floor:", MIN_KEEP_PX)
        print("[debug] prob range:", float(prob.min()), "→", float(prob.max()),
              "  thr_used:", used_thr)

    if debug_save_images:
        dbg = base_dir / "debug"
        dbg.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(dbg / "prob.png"), (prob * 255).astype(np.uint8))
        cv2.imwrite(str(dbg / "mask_raw.png"), (mask0 * 255))
        cv2.imwrite(str(dbg / "mask_clean.png"), (mask * 255))
        cv2.imwrite(str(dbg / "mask_filtered.png"), (mask_f * 255))
        cv2.imwrite(str(dbg / "skeleton.png"), (skel * 255))
        if hmask is not None:
            cv2.imwrite(str(dbg / "mask_hysteresis.png"), (hmask.astype(np.uint8) * 255))
        with open(dbg / "stats.txt", "w") as f:
            pct_raw = float(mask0.mean()) * 100.0
            pct_clean = float(mask.mean()) * 100.0
            pct_filt = float(mask_f.mean()) * 100.0
            f.write(f"prob_min={float(prob.min()):.6f} prob_max={float(prob.max()):.6f}\n")
            f.write(f"thr_used={used_thr:.6f}\n")
            f.write(f"mask_raw_pct={pct_raw:.2f} mask_clean_pct={pct_clean:.2f} mask_filtered_pct={pct_filt:.2f}\n")
            f.write(f"skel_px_base={BASE_PX} skel_px_final={int(skel.sum())} keep_floor={MIN_KEEP_PX}\n")

        # NEW: mirror debug images into runs/<run_id>/output/overlay/debug/<image_id>/
        _publish_debug_to_runs(heatmap_path, base_dir)

    # tracking on seg-sized, WL-preprocessed gray aligned to prob shape
    gray_seg = kb.preproc_for_seg(gray_orig, hw=prob.shape)  # ensure your KymoButlerPT supports hw=(H,W)
    tracker = CrossingTracker(
        kb,
        max_branch_steps=256,
        min_track_len=max(5, min_length // 3),
        decision_recent_tail=16,
        decision_thr=float(decision_thr),   # NEW: confidence gate for branch selection
    )
    tracks_seg = tracker.extract_tracks(gray_seg, skel)

    # refine & merge (optional)
    if refine_enable and tracks_seg:
        tracks_seg = refine_tracks(
            tracks_seg, prob,
            extend_rows=extend_rows,
            dx_win=dx_win,
            prob_min=refine_prob_min,
            max_gap_rows=max_gap_rows,
            max_dx=max_dx,
            prob_bridge_min=prob_bridge_min,
        )

    # dedupe (optional)
    if dedupe_enable and tracks_seg:
        tracks_seg = filter_and_dedupe_tracks(
            tracks_seg, prob,
            min_rows=(dedupe_min_rows if dedupe_min_rows is not None else min_length),
            min_score=dedupe_min_score,
            overlap_iou=dedupe_overlap_iou,
            dx_tol=dedupe_dx_tol,
        )

    if verbose and tracks_seg:
        lengths = [_track_len_rows(t) for t in tracks_seg]
        print(f"[debug] track rows: n={len(lengths)}  "
              f"p50={int(np.percentile(lengths, 50))}  "
              f"p90={int(np.percentile(lengths, 90))}  "
              f"max={max(lengths)}")

    # scale back to original resolution (use prob shape as seg space)
    tracks = _scale_tracks_to_original(tracks_seg, seg_hw=prob.shape, orig_hw=(H, W))

    saved = _save_npy_tracks(tracks, out_dir, min_length=min_length)
    if verbose:
        print(f"[kymobutler_pt] extracted {len(tracks)} track(s); saved {saved} ≥ {min_length} to {out_dir}")

    if debug_save_images:
        overlay = cv2.cvtColor(gray_orig, cv2.COLOR_GRAY2BGR)
        for t in tracks:
            for y, x in t.points:
                cv2.circle(overlay, (int(x), int(y)), 1, (0, 255, 0), -1)
        cv2.imwrite(str(base_dir / "overlay_tracks.png"), overlay)

    return base_dir
