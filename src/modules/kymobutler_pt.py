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
        self.sess = ort.InferenceSession(self.path, providers=["CPUExecutionProvider"])
        self.inputs = [i.name for i in self.sess.get_inputs()]
        self.outputs = [o.name for o in self.sess.get_outputs()]
        # assume single input
        self.in_name = self.inputs[0]

    def run(self, x: np.ndarray) -> dict[str,np.ndarray]:
        out = self.sess.run(self.outputs, {self.in_name: x})
        return {name: arr for name,arr in zip(self.outputs, out)}

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
