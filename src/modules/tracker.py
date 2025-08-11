# tracker.py
from __future__ import annotations
import os, csv
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Iterable

import numpy as np
import cv2

from kymobutler_pt import KymoButlerPT, prob_to_mask, skeletonize

Coord = Tuple[int, int]  # (y, x)

# ------------------- small geometry helpers -------------------
_OFFSETS_8 = [(-1,-1), (-1,0), (-1,1),
              ( 0,-1),         ( 0,1),
              ( 1,-1), ( 1,0), ( 1,1)]

def neighbors8(y: int, x: int, H: int, W: int) -> Iterable[Coord]:
    for dy, dx in _OFFSETS_8:
        ny, nx = y + dy, x + dx
        if 0 <= ny < H and 0 <= nx < W:
            yield ny, nx

def degree_at(skel: np.ndarray, y: int, x: int) -> int:
    H, W = skel.shape
    return sum(1 for ny, nx in neighbors8(y, x, H, W) if skel[ny, nx] == 1)

def find_endpoints_and_junctions(skel: np.ndarray) -> Tuple[List[Coord], List[Coord]]:
    H, W = skel.shape
    endpoints, junctions = [], []
    ys, xs = np.where(skel == 1)
    for y, x in zip(ys, xs):
        d = degree_at(skel, y, x)
        if d == 1:
            endpoints.append((y, x))
        elif d >= 3:
            junctions.append((y, x))
    return endpoints, junctions

def crop_with_pad(arr: np.ndarray, cy: int, cx: int, hh: int, hw: int) -> np.ndarray:
    """Crop [cy-hh:cy+hh, cx-hw:cx+hw], pad with zeros if near border."""
    H, W = arr.shape[:2]
    y0, y1 = cy - hh, cy + hh
    x0, x1 = cx - hw, cx + hw
    pad_top   = max(0, -y0)
    pad_left  = max(0, -x0)
    pad_bot   = max(0, y1 - H + 1)
    pad_right = max(0, x1 - W + 1)
    y0c, y1c = max(0, y0), min(H - 1, y1)
    x0c, x1c = max(0, x0), min(W - 1, x1)
    crop = arr[y0c:y1c+1, x0c:x1c+1]
    if pad_top or pad_bot or pad_left or pad_right:
        crop = np.pad(crop,
                      ((pad_top, pad_bot), (pad_left, pad_right)),
                      mode='constant', constant_values=0)
    return crop

def path_mask(shape: Tuple[int,int], pts: List[Coord], radius: int = 0) -> np.ndarray:
    """Binary mask with 1s at pts (optionally dilated by 'radius')."""
    m = np.zeros(shape, dtype=np.uint8)
    if not pts: return m
    ys, xs = zip(*pts)
    m[ys, xs] = 1
    if radius > 0:
        k = 2*radius + 1
        m = cv2.dilate(m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    return m

def enforce_one_point_per_row(track: List[Coord]) -> List[Coord]:
    """Keep a single (y,x) per y by taking the median x for each row (preserves order)."""
    if not track: return track
    rows: Dict[int, List[int]] = {}
    for y, x in track:
        rows.setdefault(y, []).append(x)
    ys_sorted = sorted(rows.keys())
    cleaned: List[Coord] = []
    prev_x: Optional[int] = None
    for y in ys_sorted:
        xs = rows[y]
        if prev_x is None:
            x = int(np.median(xs))
        else:
            # choose x closest to previous x to smooth zigzags
            x = min(xs, key=lambda v: abs(v - prev_x))
        cleaned.append((y, x))
        prev_x = x
    return cleaned

def line_len(track: List[Coord]) -> int:
    return len(track)

# ------------------- crossing-aware tracker -------------------
@dataclass
class Track:
    points: List[Coord]
    id: int

class CrossingTracker:
    def __init__(self, kb: KymoButlerPT,
                 decision_crop_hw: Tuple[int,int]=(48,48),
                 max_branch_steps: int = 64,
                 min_track_len: int = 8,
                 decision_recent_tail: int = 12,
                 decision_thr: float = 0.0):
        """
        decision_recent_tail: how many last points to include in 'current' channel for decision crops
        decision_thr: optional extra threshold on decision probabilities when scoring branches
        """
        self.kb = kb
        self.ch, self.cw = decision_crop_hw
        self.max_branch_steps = max_branch_steps
        self.min_track_len = min_track_len
        self.decision_recent_tail = decision_recent_tail
        self.decision_thr = decision_thr

    # --- branch scoring using the decision net ---
    def _score_branches(self, raw01: np.ndarray, skel_all: np.ndarray,
                        curr_pts: List[Coord], junction: Coord,
                        cand_starts: List[Coord]) -> Optional[Coord]:
        H, W = raw01.shape
        jy, jx = junction

        # Build 'current path' mask using last K points (focus on local direction)
        tail_pts = curr_pts[-self.decision_recent_tail:] if len(curr_pts) > 0 else []
        curr_mask = path_mask((H, W), tail_pts, radius=0)

        # Build the decision crop inputs centered at junction
        hh, hw = self.ch // 2, self.cw // 2
        crop_raw  = crop_with_pad(raw01, jy, jx, hh, hw)
        crop_all  = crop_with_pad(skel_all, jy, jx, hh, hw)
        crop_curr = crop_with_pad(curr_mask, jy, jx, hh, hw)

        # Query decision net -> probability map for "continue" class
        prob = self.kb.decision_map(crop_raw, crop_all, crop_curr)  # [Hc,Wc] in 0..1

        # For each candidate, walk a short branch and sum decision prob along it
        best_start, best_score = None, -1.0
        for sy, sx in cand_starts:
            # Walk from candidate start up to max_branch_steps or next junction
            branch_pts = self._walk_branch_preview(skel_all, junction, (sy, sx), self.max_branch_steps)
            if not branch_pts:
                continue

            # Map those points into crop coords
            scores = []
            for y, x in branch_pts:
                cy, cx = y - (jy - hh), x - (jx - hw)  # translate into crop
                if 0 <= cy < prob.shape[0] and 0 <= cx < prob.shape[1]:
                    p = prob[cy, cx]
                    if p >= self.decision_thr:
                        scores.append(p)
            score = float(np.sum(scores)) if scores else 0.0

            if score > best_score:
                best_score, best_start = score, (sy, sx)

        return best_start

    def _walk_branch_preview(self, skel: np.ndarray, prev: Coord, start: Coord, max_steps: int) -> List[Coord]:
        """From 'start', proceed along skeleton until endpoint/junction/visited or step budget exhausted."""
        H, W = skel.shape
        path: List[Coord] = []
        prev_y, prev_x = prev
        y, x = start
        steps = 0
        while steps < max_steps and skel[y, x] == 1:
            path.append((y, x))
            steps += 1
            deg = degree_at(skel, y, x)
            nbrs = [(ny, nx) for ny, nx in neighbors8(y, x, H, W) if skel[ny, nx] == 1 and (ny, nx) != (prev_y, prev_x)]

            if deg != 2:  # endpoint (1) or junction (>=3)
                break
            if not nbrs:
                break
            # Single-forward continuation
            prev_y, prev_x = y, x
            y, x = nbrs[0]
        return path

    # --- main path grower ---
    def extract_tracks(self, raw_gray: np.ndarray, skel: np.ndarray) -> List[Track]:
        """
        raw_gray: grayscale image (uint8 or float 0..1)
        skel: binary skeleton (0/1)
        """
        raw01 = raw_gray.astype(np.float32)
        if raw01.max() > 1.0: raw01 /= 255.0
        H, W = skel.shape
        visited = np.zeros_like(skel, dtype=np.uint8)
        endpoints, _ = find_endpoints_and_junctions(skel)

        tracks: List[Track] = []
        tid = 0

        for seed in endpoints:
            if visited[seed] == 1:  # already consumed
                continue
            # Start a new track
            curr: List[Coord] = [seed]
            visited[seed] = 1

            # Choose initial step (the single neighbor)
            cy, cx = seed
            nbrs = [(ny, nx) for ny, nx in neighbors8(cy, cx, H, W) if skel[ny, nx] == 1 and visited[ny, nx] == 0]
            prev = seed
            if not nbrs:
                continue
            y, x = nbrs[0]
            curr.append((y, x))
            visited[y, x] = 1
            prev = (cy, cx)

            # Walk forward
            while True:
                deg = degree_at(skel, y, x)
                nbrs = [(ny, nx) for ny, nx in neighbors8(y, x, H, W) if skel[ny, nx] == 1 and (ny, nx) != prev and visited[ny, nx] == 0]

                if deg == 1:
                    # endpoint
                    break
                elif deg == 2:
                    if not nbrs:
                        break
                    prev = (y, x)
                    y, x = nbrs[0]
                    curr.append((y, x))
                    visited[y, x] = 1
                    continue
                else:
                    # junction: decide branch with decision net
                    # Candidate starts are neighbors (excluding prev & visited)
                    cand = [(ny, nx) for ny, nx in neighbors8(y, x, H, W) if skel[ny, nx] == 1 and (ny, nx) != prev and visited[ny, nx] == 0]
                    if not cand:
                        break
                    chosen = self._score_branches(raw01, skel, curr, (y, x), cand)
                    if chosen is None:
                        # fallback: pick the neighbor that is most straight (nearest to direction prev->(y,x))
                        vy, vx = y - prev[0], x - prev[1]
                        def straightness(n):
                            dy, dx = n[0]-y, n[1]-x
                            return -(vy*dy + vx*dx)  # more negative = better dot product (min)
                        chosen = min(cand, key=straightness)

                    prev = (y, x)
                    y, x = chosen
                    curr.append((y, x))
                    visited[y, x] = 1

            curr = enforce_one_point_per_row(curr)
            if line_len(curr) >= self.min_track_len:
                tracks.append(Track(points=curr, id=tid))
                tid += 1

        return tracks

    # --- CSV export ---
    @staticmethod
    def save_tracks_csv(tracks: List[Track], out_csv: str):
        """
        CSV columns: track_id, frame (y), x
        """
        os.makedirs(os.path.dirname(out_csv), exist_ok=True)
        with open(out_csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["track_id", "frame", "x"])
            for t in tracks:
                for (y, x) in t.points:
                    w.writerow([t.id, y, x])

# ------------------- convenience: full pipeline -------------------
def run_full_pipeline(image_path: str,
                      kb: KymoButlerPT,
                      out_dir: str = "outputs",
                      thr_uni: float = 0.20,
                      thr_bi: float = 0.20,
                      min_component_px: int = 5) -> List[Track]:
    """
    1) classify kymograph (uni vs bi)
    2) segment (uni -> 2 maps; bi -> 1 map)
    3) threshold + skeletonize
    4) track with crossing resolution (decision net used automatically)
    5) write CSVs and debug PNGs
    """
    os.makedirs(out_dir, exist_ok=True)

    gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if gray is None:
        raise FileNotFoundError(image_path)

    cls = kb.classify(gray)
    mode = "bi" if cls["label"] == 1 else "uni"  # adjust if your label order is reversed
    print(f"[classifier] probs={cls['probs']}, mode={mode}")

    if mode == "uni":
        out = kb.segment_uni(gray)                 # dict {ant, ret}
        mask = ((out["ant"] > thr_uni) | (out["ret"] > thr_uni)).astype(np.uint8)
        cv2.imwrite(os.path.join(out_dir, "uni_ant.png"), (out["ant"]*255).astype(np.uint8))
        cv2.imwrite(os.path.join(out_dir, "uni_ret.png"), (out["ret"]*255).astype(np.uint8))
    else:
        prob = kb.segment_bi(gray)
        mask = prob_to_mask(prob, thr_bi)
        cv2.imwrite(os.path.join(out_dir, "bi_prob.png"), (prob*255).astype(np.uint8))

    skel = skeletonize(mask, min_component_px=min_component_px)
    cv2.imwrite(os.path.join(out_dir, "skeleton.png"), (skel*255))

    tracker = CrossingTracker(kb)
    tracks = tracker.extract_tracks(gray, skel)
    print(f"[tracker] extracted {len(tracks)} tracks")

    # overlays + CSV
    overlay = cv2.cvtColor(_norm01(gray), cv2.COLOR_GRAY2BGR)
    for t in tracks:
        for y, x in t.points:
            cv2.circle(overlay, (int(x), int(y)), 1, (0,255,0), -1)
    cv2.imwrite(os.path.join(out_dir, "overlay_tracks.png"), (overlay*255).astype(np.uint8))

    CrossingTracker.save_tracks_csv(tracks, os.path.join(out_dir, "tracks.csv"))
    return tracks

def _norm01(img: np.ndarray) -> np.ndarray:
    f = img.astype(np.float32)
    if f.max() > 1: f /= 255.0
    return np.clip(f, 0, 1)
