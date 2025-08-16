# src/modules/kymobutler_pt.py  (ORT backend with WL-like preprocessing + skeleton helpers)

import os
import math
from pathlib import Path

import cv2
import numpy as np
import onnx
import onnxruntime as ort

from skimage.morphology import skeletonize as _skel
from skimage.morphology import thin as _thin
from skimage.measure import label, regionprops


REQUIRED_ONNX = ["uni_seg.onnx", "bi_seg.onnx", "classifier.onnx", "decision.onnx"]


# ---------------------------
# Paths / I/O helpers
# ---------------------------

def _find_export_dir(user_path=None) -> Path:
    if user_path is not None:
        p = Path(user_path)
        if p.is_dir():
            return p
    env = os.getenv("KYMO_EXPORT_DIR")
    if env:
        p = Path(env)
        if p.is_dir():
            return p
    here = Path(__file__).resolve()
    for c in [here.parents[2] / "export", here.parents[1] / "export", here.parent / "export", Path.cwd() / "export"]:
        if c.is_dir():
            return c
    raise FileNotFoundError("Could not locate ONNX export directory. Set KYMO_EXPORT_DIR or pass export_dir.")


# ---------------------------
# Basic image helpers
# ---------------------------

def _to_01_gray(img: np.ndarray) -> np.ndarray:
    g = img.astype(np.float32)
    if g.max() > 1.0:
        g /= 255.0
    return np.clip(g, 0.0, 1.0)

def _resize_hw(img01: np.ndarray, hw: tuple[int, int]) -> np.ndarray:
    h, w = hw
    return cv2.resize(img01, (w, h), interpolation=cv2.INTER_AREA)

def prob_to_mask(prob: np.ndarray, thr: float = 0.20) -> np.ndarray:
    return (prob > thr).astype(np.uint8)


# ---------------------------
# Skeleton helpers (WL-parity)
# ---------------------------

def skeletonize(mask01: np.ndarray, min_component_px: int = 5, algo: str = "thin") -> np.ndarray:
    """Binary mask -> 1px skeleton + small component filter."""
    if algo == "thin":
        sk = _thin(mask01.astype(bool)).astype(np.uint8)
    else:
        sk = _skel(mask01.astype(bool)).astype(np.uint8)
    lab = label(sk, connectivity=2)
    keep = np.zeros_like(sk, dtype=np.uint8)
    for r in regionprops(lab):
        if r.area >= min_component_px:
            rr, cc = zip(*r.coords)
            keep[tuple(rr), tuple(cc)] = 1
    return keep

def thin_and_prune(mask: np.ndarray, prune_iters: int = 3) -> np.ndarray:
    """
    WL parity: Pruning[Thinning@mask, prune_iters]
    1) Thin to 1-px skeleton.
    2) Iteratively remove endpoints 'prune_iters' times.
    """
    sk = _thin(mask.astype(bool)).astype(np.uint8)

    H, W = sk.shape
    nb = [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]

    def endpoints(arr):
        ys, xs = np.where(arr == 1)
        idx = []
        for y, x in zip(ys, xs):
            deg = 0
            for dy, dx in nb:
                ny, nx = y + dy, x + dx
                if 0 <= ny < H and 0 <= nx < W and arr[ny, nx] == 1:
                    deg += 1
            if deg == 1:
                idx.append((y, x))
        return idx

    for _ in range(int(prune_iters)):
        eps = endpoints(sk)
        if not eps:
            break
        ys, xs = zip(*eps)
        sk[ys, xs] = 0

    return sk

def prune_endpoints(skel: np.ndarray, iterations: int = 1) -> np.ndarray:
    sk = skel.copy().astype(np.uint8)
    H, W = sk.shape
    nbrs = [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]
    for _ in range(max(0, int(iterations))):
        ys, xs = np.where(sk == 1)
        to_zero = []
        for y, x in zip(ys, xs):
            deg = 0
            for dy, dx in nbrs:
                ny, nx = y + dy, x + dx
                if 0 <= ny < H and 0 <= nx < W and sk[ny, nx] == 1:
                    deg += 1
            if deg <= 1:  # endpoint or isolated pixel
                to_zero.append((y, x))
        if not to_zero:
            break
        for y, x in to_zero:
            sk[y, x] = 0
    return sk

def filter_components(mask: np.ndarray, min_px: int, min_rows: int) -> np.ndarray:
    """
    WL parity: SelectComponents[..., (#Count >= min_px && vertical_span >= min_rows)]
    Apply on a binary mask BEFORE thinning to drop tiny junk.
    """
    lab = label(mask.astype(np.uint8), connectivity=2)
    keep = np.zeros_like(mask, dtype=np.uint8)
    for r in regionprops(lab):
        y0, x0, y1, x1 = r.bbox
        span = (y1 - y0)
        if r.area >= min_px and span >= min_rows:
            yy, xx = r.coords.T
            keep[yy, xx] = 1
    return keep


# ---------------------------
# Math helpers
# ---------------------------

def _softmax(x, axis=-1):
    x = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=axis, keepdims=True)

def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def _as_prob(y: np.ndarray) -> np.ndarray:
    """Return probabilities in [0,1]. If already in [0,1], trust it; else apply sigmoid."""
    m, M = float(y.min()), float(y.max())
    if m >= -1e-6 and M <= 1.0 + 1e-6:
        return y.astype(np.float32, copy=False)
    return _sigmoid(y).astype(np.float32, copy=False)


# ---------------------------
# WL-like preprocessing
# ---------------------------

def _is_negated_like_wl(img01: np.ndarray) -> bool:
    # WL: n1 = Total@Binarize[kym, .5], n2 = Total@Binarize[ColorNegate@kym, .5]; n1>=n2 => invert
    gt = (img01 > 0.5).sum()
    lt = (img01 < 0.5).sum()
    return gt >= lt  # True => invert

def _normlines_like_wl(img01: np.ndarray) -> np.ndarray:
    # Map rows: row / mean(row) if mean>0
    eps = 1e-8
    row_means = img01.mean(axis=1, keepdims=True)
    scale = 1.0 / np.maximum(row_means, eps)
    out = img01 * scale
    # Clamp to [0,1] and light percentile stretch
    out = np.clip(out, 0.0, None)
    vmax = np.percentile(out, 99.0)
    if vmax > 0:
        out = np.clip(out / vmax, 0.0, 1.0)
    return out.astype(np.float32, copy=False)

def _preproc_like_wl(img_gray: np.ndarray) -> np.ndarray:
    """ImageAdjust + ColorConvert + RemoveAlpha handled by caller; we get a gray uint8/float."""
    x = _to_01_gray(img_gray)
    if _is_negated_like_wl(x):
        x = 1.0 - x
    x = _normlines_like_wl(x)
    return x


# ---------------------------
# Tiling / blending helpers
# ---------------------------

def _round_up(v, base):
    return int(base * math.ceil(float(v) / base))

def _hann1d(n: int) -> np.ndarray:
    # Hann with floor to avoid 0-weight borders (prevents divide-by-zero at edges)
    if n <= 1:
        return np.ones((n,), dtype=np.float32)
    w = 0.5 - 0.5 * np.cos(2.0 * np.pi * np.arange(n, dtype=np.float32) / (n - 1))
    return np.maximum(w.astype(np.float32), 0.25)  # clamp to 0.25 at edges

def _hann2d(h: int, w: int) -> np.ndarray:
    return np.outer(_hann1d(h), _hann1d(w)).astype(np.float32)

def _ensure_min_hw(img01: np.ndarray, min_h: int, min_w: int) -> np.ndarray:
    h, w = img01.shape[:2]
    if h >= min_h and w >= min_w:
        return img01
    return cv2.resize(img01, (max(min_w, w), max(min_h, h)), interpolation=cv2.INTER_CUBIC)


# ---------------------------
# ORT thin wrapper
# ---------------------------

class ORTModel:
    def __init__(self, path: str, providers=None):
        self.path = str(path)
        self.model = onnx.load(self.path)
        self.init_names = {init.name for init in self.model.graph.initializer}

        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.sess = ort.InferenceSession(
            self.path,
            sess_options=so,
            providers=(providers or ["CPUExecutionProvider"]),
        )

        self.inputs_meta = self.sess.get_inputs()
        self.outputs = [o.name for o in self.sess.get_outputs()]

        def _rank(shape):
            return 0 if shape is None else sum(1 for d in shape if d is not None)
        candidates = sorted(self.inputs_meta, key=lambda a: _rank(a.shape or []), reverse=True)
        for a in candidates:
            nm = a.name.lower()
            if "input" in nm or _rank(a.shape or []) >= 3:
                self.data_input = a
                break
        else:
            self.data_input = self.inputs_meta[0]

    def _default_for(self, arg: ort.NodeArg):
        name = arg.name
        if name in self.init_names:
            return None
        t = (arg.type or "").lower()
        shp = arg.shape
        if not shp:
            shp = ()
        elif any(d is None for d in shp):
            shp = tuple(1 for _ in shp)
        lname = name.lower()
        if lname in {"trainingmode", "is_training", "training", "train"} or "trainingmode" in lname:
            return np.array(False, dtype=np.bool_)
        if lname.endswith("ratio") or lname.startswith("ratio"):
            return np.array(0.0, dtype=np.float32)
        if "bool" in t:
            return np.zeros(shp, dtype=np.bool_)
        if "float16" in t:
            return np.zeros(shp, dtype=np.float16)
        if "float" in t or "tensor(float" in t:
            return np.zeros(shp, dtype=np.float32)
        if "double" in t:
            return np.zeros(shp, dtype=np.float64)
        if "int64" in t:
            return np.zeros(shp, dtype=np.int64)
        if "int32" in t:
            return np.zeros(shp, dtype=np.int32)
        if "int16" in t:
            return np.zeros(shp, dtype=np.int16)
        if "int8" in t:
            return np.zeros(shp, dtype=np.int8)
        if "uint8" in t:
            return np.zeros(shp, dtype=np.uint8)
        return np.array(0.0, dtype=np.float32)

    def run(self, x: np.ndarray) -> dict[str, np.ndarray]:
        feed = {self.data_input.name: x.astype(np.float32, copy=False)}
        for arg in self.inputs_meta:
            if arg.name == self.data_input.name:
                continue
            default = self._default_for(arg)
            if default is not None:
                feed[arg.name] = default
        outs = self.sess.run(self.outputs, feed)
        return {name: arr for name, arr in zip(self.outputs, outs)}


# ---------------------------
# Main ONNX runner
# ---------------------------

class KymoButlerPT:
    """ORT-based runner for KymoButler ONNX models, with WL-like preprocessing."""
    def __init__(self, export_dir=None, seg_size=256,
                 tile_stride: int = 128,    # overlap = seg_size - stride
                 tile_round_to: int = 16,   # round full-res canvas to multiple of this
                 use_tiling: bool = True,
                 providers=None):
        export_dir = _find_export_dir(export_dir)
        paths = {
            k: export_dir / f
            for k, f in {
                "uni": "uni_seg.onnx",
                "bi": "bi_seg.onnx",
                "clf": "classifier.onnx",
                "dec": "decision.onnx",
            }.items()
        }
        missing = [k for k, p in paths.items() if not Path(p).exists()]
        if missing:
            raise FileNotFoundError(f"Missing ONNX files for {missing}. Looked in: {export_dir}")

        self.uni = ORTModel(paths["uni"], providers=providers)
        self.bi  = ORTModel(paths["bi"],  providers=providers)
        self.clf = ORTModel(paths["clf"], providers=providers)
        self.dec = ORTModel(paths["dec"], providers=providers)

        # ONNX input tile size (fixed by export)
        self.seg_hw = (int(seg_size), int(seg_size))
        self.uni_hw = self.seg_hw
        self.bi_hw  = self.seg_hw
        self.clf_hw = (64, 64)
        self.dec_hw = (48, 48)

        # tiling controls
        self.tile_stride   = int(tile_stride)
        self.tile_round_to = int(tile_round_to)
        self.use_tiling    = bool(use_tiling)

    # --- public helper so tracker can get preprocessed gray at arbitrary hw ---
    def preproc_for_seg(self, img_gray: np.ndarray, hw: tuple[int, int] | None = None) -> np.ndarray:
        """
        Return WL-like preprocessed grayscale resized to 'hw' if provided;
        otherwise returns (seg_size, seg_size).
        """
        x = _preproc_like_wl(img_gray)
        if hw is None:
            hw = self.seg_hw
        return _resize_hw(x, hw)

    # --- internal prep to NCHW float32 ---
    def _prep_gray(self, img: np.ndarray, hw: tuple[int, int], *, wl_preproc: bool) -> np.ndarray:
        img01 = _preproc_like_wl(img) if wl_preproc else _to_01_gray(img)
        img01 = _resize_hw(img01, hw)
        return img01[None, None, ...].astype(np.float32)  # NCHW

    # ========== TILED INFERENCE CORE ==========
    def _tile_infer_2d(self, img01: np.ndarray, run_fn, *, out_kind: str):
        """
        Slide a (seg_h, seg_w) window with stride across img01 (H',W').
        'run_fn(tile_nchw)->dict' should return ORT outputs for that tile.
        out_kind: 'bi' -> single map; 'uni' -> two maps ('ant','ret' or 2ch)
        Returns:
          if 'bi' : prob (H',W')
          if 'uni': dict {'ant': (H',W'), 'ret': (H',W')}
        """
        seg_h, seg_w = self.seg_hw
        H, W = img01.shape
        # Ensure canvas >= tile size
        img01 = _ensure_min_hw(img01, seg_h, seg_w)
        H, W = img01.shape

        # Round canvas up to multiple of tile_round_to to stabilize coverage
        Hr = _round_up(H, self.tile_round_to)
        Wr = _round_up(W, self.tile_round_to)
        if (Hr, Wr) != (H, W):
            img01 = cv2.resize(img01, (Wr, Hr), interpolation=cv2.INTER_AREA)
            H, W = Hr, Wr

        stride = self.tile_stride
        wy = list(range(0, max(1, H - seg_h + 1), stride))
        wx = list(range(0, max(1, W - seg_w + 1), stride))
        if wy[-1] != H - seg_h: wy.append(H - seg_h)
        if wx[-1] != W - seg_w: wx.append(W - seg_w)

        w2d = _hann2d(seg_h, seg_w)  # soft blend; floored to 0.25 inside _hann1d
        eps = 1e-6

        if out_kind == "bi":
            acc = np.zeros((H, W), dtype=np.float32)
            wsum = np.zeros((H, W), dtype=np.float32)
        else:
            acc_ant = np.zeros((H, W), dtype=np.float32)
            acc_ret = np.zeros((H, W), dtype=np.float32)
            wsum    = np.zeros((H, W), dtype=np.float32)

        for y in wy:
            for x in wx:
                tile = img01[y:y+seg_h, x:x+seg_w]
                tile_nchw = tile[None, None, ...].astype(np.float32)
                out = run_fn(tile_nchw)

                if out_kind == "bi":
                    ypred = list(out.values())[0]  # [1,1,h,w] or [1,h,w] or [1,h,w,1]
                    ypred = np.squeeze(ypred)
                    prob = _as_prob(ypred).astype(np.float32)
                    acc[y:y+seg_h, x:x+seg_w] += prob * w2d
                    wsum[y:y+seg_h, x:x+seg_w] += w2d
                else:
                    if "ant" in out and "ret" in out:
                        ant = _as_prob(out["ant"]).squeeze().astype(np.float32)
                        ret = _as_prob(out["ret"]).squeeze().astype(np.float32)
                    else:
                        ypred = list(out.values())[0]
                        ypred = np.squeeze(ypred)  # [2,h,w] or [h,w,2]
                        if ypred.ndim == 3 and ypred.shape[0] == 2:
                            ant = _as_prob(ypred[0]).astype(np.float32)
                            ret = _as_prob(ypred[1]).astype(np.float32)
                        elif ypred.ndim == 3 and ypred.shape[-1] == 2:
                            ant = _as_prob(ypred[..., 0]).astype(np.float32)
                            ret = _as_prob(ypred[..., 1]).astype(np.float32)
                        else:
                            raise RuntimeError(f"Unexpected uni_seg tile out shape: {ypred.shape}")
                    acc_ant[y:y+seg_h, x:x+seg_w] += ant * w2d
                    acc_ret[y:y+seg_h, x:x+seg_w] += ret * w2d
                    wsum[y:y+seg_h, x:x+seg_w]    += w2d

        if out_kind == "bi":
            prob_full = acc / np.maximum(wsum, eps)
            return prob_full[:H, :W].astype(np.float32)
        else:
            ant_full = acc_ant / np.maximum(wsum, eps)
            ret_full = acc_ret / np.maximum(wsum, eps)
            return {
                "ant": ant_full[:H, :W].astype(np.float32),
                "ret": ret_full[:H, :W].astype(np.float32),
            }

    # ========== CLASSIFIER / DECISION ==========
    def classify(self, img_gray: np.ndarray) -> dict:
        x = self._prep_gray(img_gray, self.clf_hw, wl_preproc=True)
        out = self.clf.run(x)
        y = list(out.values())[0]  # expected [1,2]
        logits = y[0]
        probs = _softmax(logits, axis=-1)
        label = int(np.argmax(probs))
        return {"logits": logits, "probs": probs, "label": label}

    def decision_map(self, crop_raw01: np.ndarray, crop_skel_all: np.ndarray, crop_skel_curr: np.ndarray) -> np.ndarray:
        x = np.stack([
            _resize_hw(_to_01_gray(crop_raw01), self.dec_hw),
            _resize_hw(_to_01_gray(crop_skel_all), self.dec_hw),
            _resize_hw(_to_01_gray(crop_skel_curr), self.dec_hw),
        ], axis=0)[None, ...].astype(np.float32)
        out = self.dec.run(x)
        y = list(out.values())[0]  # [1,48,48,2] or [1,2,48,48]
        if y.ndim == 4 and y.shape[-1] == 2:
            prob = _softmax(y, axis=-1)[0, ..., 1]
        elif y.ndim == 4 and y.shape[1] == 2:
            prob = _softmax(np.moveaxis(y, 1, -1), axis=-1)[0, ..., 1]
        else:
            raise RuntimeError(f"Unexpected decision output shape: {y.shape}")
        return prob.astype(np.float32)

    # ========== FULL-RES SEGMENTATION ==========
    def segment_bi_full(self, img_gray: np.ndarray, *, crop_to_original: bool = True) -> np.ndarray:
        """
        Full-resolution bidirectional segmentation with tiling/blending.
        Returns prob map at original size (if crop_to_original=True).
        """
        img01 = _preproc_like_wl(img_gray)
        if not self.use_tiling:
            x = self._prep_gray(img_gray, self.bi_hw, wl_preproc=True)
            out = self.bi.run(x)
            y = list(out.values())[0]
            y = np.squeeze(y)
            prob = _as_prob(y).astype(np.float32)
            if crop_to_original:
                H0, W0 = img01.shape
                return cv2.resize(prob, (W0, H0), interpolation=cv2.INTER_LINEAR)
            return prob

        H0, W0 = img01.shape
        Ht = max(self.seg_hw[0], _round_up(H0, self.tile_round_to))
        Wt = max(self.seg_hw[1], _round_up(W0, self.tile_round_to))
        img01r = cv2.resize(img01, (Wt, Ht), interpolation=cv2.INTER_AREA)

        def _run(tile_nchw):
            return self.bi.run(tile_nchw)

        prob = self._tile_infer_2d(img01r, _run, out_kind="bi")
        return prob[:H0, :W0] if crop_to_original else prob

    def segment_uni_full(self, img_gray: np.ndarray, *, crop_to_original: bool = True) -> dict:
        """
        Full-resolution unidirectional segmentation with tiling/blending.
        Returns dict {'ant': HxW, 'ret': HxW} (cropped if crop_to_original=True).
        """
        img01 = _preproc_like_wl(img_gray)
        if not self.use_tiling:
            x = self._prep_gray(img_gray, self.uni_hw, wl_preproc=True)
            out = self.uni.run(x)
            if "ant" in out and "ret" in out:
                ant = _as_prob(out["ant"]).squeeze().astype(np.float32)
                ret = _as_prob(out["ret"]).squeeze().astype(np.float32)
            else:
                y = list(out.values())[0]
                if y.ndim == 4 and y.shape[1] == 2:
                    ant = _as_prob(y[:, 0]).squeeze().astype(np.float32)
                    ret = _as_prob(y[:, 1]).squeeze().astype(np.float32)
                elif y.ndim == 4 and y.shape[-1] == 2:
                    ant = _as_prob(y[..., 0]).squeeze().astype(np.float32)
                    ret = _as_prob(y[..., 1]).squeeze().astype(np.float32)
                else:
                    raise RuntimeError(f"Unexpected uni_seg output shape: {y.shape}")
            if crop_to_original:
                H0, W0 = img01.shape
                ant = cv2.resize(ant, (W0, H0), interpolation=cv2.INTER_LINEAR)
                ret = cv2.resize(ret, (W0, H0), interpolation=cv2.INTER_LINEAR)
            return {"ant": ant, "ret": ret}

        H0, W0 = img01.shape
        Ht = max(self.seg_hw[0], _round_up(H0, self.tile_round_to))
        Wt = max(self.seg_hw[1], _round_up(W0, self.tile_round_to))
        img01r = cv2.resize(img01, (Wt, Ht), interpolation=cv2.INTER_AREA)

        def _run(tile_nchw):
            return self.uni.run(tile_nchw)

        out_full = self._tile_infer_2d(img01r, _run, out_kind="uni")
        if crop_to_original:
            return {
                "ant": out_full["ant"][:H0, :W0],
                "ret": out_full["ret"][:H0, :W0],
            }
        return out_full

    # ========== LEGACY (kept for compatibility) ==========
    def segment_bi(self, img_gray: np.ndarray) -> np.ndarray:
        """Legacy: 256×256 output for existing code paths."""
        x = self._prep_gray(img_gray, self.bi_hw, wl_preproc=True)
        out = self.bi.run(x)
        y = list(out.values())[0]  # [1,1,H,W] or [1,H,W] or [1,H,W,1]
        y = y.squeeze()
        return _as_prob(y)

    def segment_uni(self, img_gray: np.ndarray) -> dict:
        """Legacy: 256×256 output for existing code paths."""
        x = self._prep_gray(img_gray, self.uni_hw, wl_preproc=True)
        out = self.uni.run(x)
        if "ant" in out and "ret" in out:
            ant = _as_prob(out["ant"]).squeeze()
            ret = _as_prob(out["ret"]).squeeze()
        else:
            y = list(out.values())[0]
            if y.ndim == 4 and y.shape[1] == 2:        # NCHW
                ant = _as_prob(y[:, 0]).squeeze()
                ret = _as_prob(y[:, 1]).squeeze()
            elif y.ndim == 4 and y.shape[-1] == 2:     # NHWC
                ant = _as_prob(y[..., 0]).squeeze()
                ret = _as_prob(y[..., 1]).squeeze()
            else:
                raise RuntimeError(f"Unexpected uni_seg output shape: {y.shape}")
        return {"ant": ant, "ret": ret}
