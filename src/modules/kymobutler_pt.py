# src/modules/kymobutler_pt.py  (ORT backend)

import os
from pathlib import Path
import cv2
import numpy as np
import onnx
import onnxruntime as ort

from skimage.morphology import skeletonize as _skel
from skimage.measure import label, regionprops

REQUIRED_ONNX = ["uni_seg.onnx","bi_seg.onnx","classifier.onnx","decision.onnx"]

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
    for c in [here.parents[2]/"export", here.parents[1]/"export", here.parent/"export", Path.cwd()/"export"]:
        if c.is_dir():
            return c
    raise FileNotFoundError("Could not locate ONNX export directory. Set KYMO_EXPORT_DIR or pass export_dir.")

def _to_01_gray(img: np.ndarray) -> np.ndarray:
    g = img.astype(np.float32)
    if g.max() > 1.0: g /= 255.0
    return np.clip(g, 0.0, 1.0)

def _resize_hw(img01: np.ndarray, hw: tuple[int,int]) -> np.ndarray:
    h,w = hw
    return cv2.resize(img01, (w,h), interpolation=cv2.INTER_AREA)

def prob_to_mask(prob: np.ndarray, thr: float = 0.20) -> np.ndarray:
    return (prob > thr).astype(np.uint8)

def skeletonize(mask01: np.ndarray, min_component_px: int = 5) -> np.ndarray:
    sk = _skel(mask01.astype(bool)).astype(np.uint8)
    lab = label(sk, connectivity=2)
    keep = np.zeros_like(sk, dtype=np.uint8)
    for r in regionprops(lab):
        if r.area >= min_component_px:
            rr, cc = zip(*r.coords)
            keep[tuple(rr), tuple(cc)] = 1
    return keep

def _softmax(x, axis=-1):
    x = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=axis, keepdims=True)

def _sigmoid(x):
    return 1/(1+np.exp(-x))

class ORTModel:
    def __init__(self, path: str):
        self.path = str(path)
        # Load model to introspect initializers (some exporters put them in graph inputs)
        self.model = onnx.load(self.path)
        self.init_names = {init.name for init in self.model.graph.initializer}

        # ORT session
        self.sess = ort.InferenceSession(self.path, providers=["CPUExecutionProvider"])
        self.inputs_meta = self.sess.get_inputs()
        self.outputs = [o.name for o in self.sess.get_outputs()]

        # Pick the primary tensor input (NCHW) heuristically
        def _rank(shape):
            return 0 if shape is None else sum(1 for d in shape if d is not None)
        candidates = sorted(self.inputs_meta, key=lambda a: _rank(a.shape or []), reverse=True)
        # prefer names like "Input" or containing "input"
        for a in candidates:
            nm = a.name.lower()
            if "input" in nm or _rank(a.shape or []) >= 3:
                self.data_input = a
                break
        else:
            self.data_input = self.inputs_meta[0]

    def _default_for(self, arg: ort.NodeArg):
        """Return a default NumPy value for non-data inputs (scalars by dtype), or None to skip."""
        name = arg.name
        # If it's actually an initializer, ORT doesn't need us to feed it
        if name in self.init_names:
            return None

        t = (arg.type or "").lower()
        shp = arg.shape  # may be None or contain None dims
        # Treat scalars as (), 1D-unknown as (1,)
        if not shp:
            shp = ()
        elif any(d is None for d in shp):
            # if it's clearly not the data tensor, make it a scalar or 1
            shp = tuple(1 for _ in shp)

        # Heuristics by name
        if name.lower() in {"trainingmode", "is_training", "training", "train"} or "trainingmode" in name.lower():
            return np.array(False, dtype=np.bool_)
        if name.lower().endswith("ratio") or name.lower().startswith("ratio"):
            return np.array(0.0, dtype=np.float32)

        # By dtype
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

        # Fallback: scalar float
        return np.array(0.0, dtype=np.float32)

    def run(self, x: np.ndarray) -> dict[str, np.ndarray]:
        feed = {}
        # Feed the main data input
        feed[self.data_input.name] = x.astype(np.float32, copy=False)

        # Feed defaults for the rest
        for arg in self.inputs_meta:
            if arg.name == self.data_input.name:
                continue
            default = self._default_for(arg)
            if default is not None:
                feed[arg.name] = default

        outs = self.sess.run(self.outputs, feed)
        return {name: arr for name, arr in zip(self.outputs, outs)}

class KymoButlerPT:
    """
    ORT-based runner for KymoButler ONNX models.
    """
    def __init__(self, export_dir=None, seg_size=256):
        export_dir = _find_export_dir(export_dir)
        paths = {k: export_dir/f for k,f in {
            "uni":"uni_seg.onnx", "bi":"bi_seg.onnx", "clf":"classifier.onnx", "dec":"decision.onnx"}.items()}
        missing = [k for k,p in paths.items() if not Path(p).exists()]
        if missing:
            raise FileNotFoundError(f"Missing ONNX files for {missing}. Looked in: {export_dir}")
        self.uni = ORTModel(paths["uni"])
        self.bi  = ORTModel(paths["bi"])
        self.clf = ORTModel(paths["clf"])
        self.dec = ORTModel(paths["dec"])
        self.uni_hw = (seg_size, seg_size)
        self.bi_hw  = (seg_size, seg_size)
        self.clf_hw = (64, 64)
        self.dec_hw = (48, 48)

    def _prep_gray(self, img: np.ndarray, hw: tuple[int,int]) -> np.ndarray:
        img01 = _to_01_gray(img)
        img01 = _resize_hw(img01, hw)
        return img01[None, None, ...].astype(np.float32)  # NCHW

    def _prep_decision_crop(self, raw01: np.ndarray, sk_all: np.ndarray, sk_curr: np.ndarray) -> np.ndarray:
        H,W = self.dec_hw
        def P(a): 
            a = a.astype(np.float32)
            if a.max()>1:a/=255.0
            return _resize_hw(a,(H,W))
        c0, c1, c2 = P(raw01), P(sk_all), P(sk_curr)
        x = np.stack([c0,c1,c2],axis=0)[None,...].astype(np.float32)  # [1,3,H,W]
        return x

    # --- classifier: returns dict with logits, probs, label (0=uni?,1=bi?) ---
    def classify(self, img_gray: np.ndarray) -> dict:
        x = self._prep_gray(img_gray, self.clf_hw)
        out = self.clf.run(x)
        # take first output
        y = list(out.values())[0]  # shape [1,2] (per your report)
        logits = y[0]
        probs = _softmax(logits, axis=-1)
        label = int(np.argmax(probs))
        return {"logits": logits, "probs": probs, "label": label}

    # --- unidirectional segmentation: returns {"ant": HxW, "ret": HxW} probs ---
    def segment_uni(self, img_gray: np.ndarray) -> dict:
        x = self._prep_gray(img_gray, self.uni_hw)
        out = self.uni.run(x)
        # your export had named outputs "ant" and "ret"
        if "ant" in out and "ret" in out:
            ant = _sigmoid(out["ant"]).squeeze()
            ret = _sigmoid(out["ret"]).squeeze()
        else:
            # fallback: single tensor [1,2,H,W]
            y = list(out.values())[0]
            ant = _sigmoid(y[:,0]).squeeze()
            ret = _sigmoid(y[:,1]).squeeze()
        return {"ant": ant, "ret": ret}

    # --- bidirectional segmentation: returns HxW prob map ---
    def segment_bi(self, img_gray: np.ndarray) -> np.ndarray:
        x = self._prep_gray(img_gray, self.bi_hw)
        out = self.bi.run(x)
        y = list(out.values())[0]  # [1,1,H,W] or [1,H,W]
        y = y.squeeze()
        return _sigmoid(y)

    # --- decision map on 48x48 crop: returns HxW prob of foreground (class 1) ---
    def decision_map(self, crop_raw01: np.ndarray, crop_skel_all: np.ndarray, crop_skel_curr: np.ndarray) -> np.ndarray:
        x = self._prep_decision_crop(crop_raw01, crop_skel_all, crop_skel_curr)
        out = self.dec.run(x)
        y = list(out.values())[0]  # [1,48,48,2] in your report
        if y.ndim == 4 and y.shape[-1] == 2:  # NHWC
            prob = _softmax(y, axis=-1)[0,...,1]
        elif y.ndim == 4 and y.shape[1] == 2: # NCHW
            prob = _softmax(np.moveaxis(y,1,-1), axis=-1)[0,...,1]
        else:
            raise RuntimeError(f"Unexpected decision output shape: {y.shape}")
        return prob.astype(np.float32)
