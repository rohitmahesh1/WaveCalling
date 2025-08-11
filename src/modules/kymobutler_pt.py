import os
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from onnx2pytorch import ConvertModel
import onnx

from skimage.morphology import skeletonize as _skel
from skimage.measure import label, regionprops

# ---- I/O utils ----
def _load_onnx_as_torch(path: str) -> torch.nn.Module:
    model = ConvertModel(onnx.load(path))
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model

def _to_01_gray(img: np.ndarray) -> np.ndarray:
    if img.dtype != np.float32:
        img = img.astype(np.float32)
    if img.max() > 1.0:
        img /= 255.0
    return np.clip(img, 0.0, 1.0)

def _resize_hw(img01: np.ndarray, hw: tuple[int, int]) -> np.ndarray:
    h, w = hw
    return cv2.resize(img01, (w, h), interpolation=cv2.INTER_AREA)

def _ensure_nchw(x: torch.Tensor) -> torch.Tensor:
    # Accept [N,H,W] or [N,1,H,W] and return [N,1,H,W]
    if x.dim() == 3:
        x = x.unsqueeze(1)
    return x

# ---- post-processing ----
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

# ---- main wrapper ----
class KymoButlerPT:
    """
    Paths:
      uni_seg.onnx        -> outputs 2 logits maps (anterograde / retrograde) or equivalent
      bi_seg.onnx         -> outputs 1 logits map
      classifier.onnx     -> outputs [N,2] logits (uni vs bi)
      decision.onnx       -> outputs per-pixel 2-class logits for 48x48 crops
    """
    def __init__(self, export_dir="export", device=None, seg_size=256):
        self.paths = {
            "uni": os.path.join(export_dir, "uni_seg.onnx"),
            "bi": os.path.join(export_dir, "bi_seg.onnx"),
            "clf": os.path.join(export_dir, "classifier.onnx"),
            "dec": os.path.join(export_dir, "decision.onnx"),
        }
        for k, p in self.paths.items():
            if not os.path.exists(p):
                raise FileNotFoundError(f"Missing {k} model: {p}")
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # load models
        self.uni = _load_onnx_as_torch(self.paths["uni"]).to(self.device)
        self.bi  = _load_onnx_as_torch(self.paths["bi"]).to(self.device)
        self.clf = _load_onnx_as_torch(self.paths["clf"]).to(self.device)
        self.dec = _load_onnx_as_torch(self.paths["dec"]).to(self.device)

        # fixed input shapes (from your report)
        self.uni_hw = (seg_size, seg_size)      # (256,256)
        self.bi_hw  = (seg_size, seg_size)      # (256,256)
        self.clf_hw = (64, 64)
        self.dec_hw = (48, 48)                  # crops for decision (3-channel)

    # ---- preprocess helpers ----
    def _prep_gray(self, img: np.ndarray, hw: tuple[int, int]) -> torch.Tensor:
        img01 = _to_01_gray(img)
        img01 = _resize_hw(img01, hw)
        x = torch.from_numpy(img01).float()[None, None, ...]  # [1,1,H,W]
        return x.to(self.device)

    def _prep_decision_crop(self, raw01: np.ndarray, skel_all: np.ndarray, skel_curr: np.ndarray) -> torch.Tensor:
        H, W = self.dec_hw
        def _prep(ch, hw=(H, W)):
            c = ch.astype(np.float32)
            if c.max() > 1.0: c = c / 255.0
            return _resize_hw(c, hw)
        c0 = _prep(raw01)       # [H,W] in 0..1
        c1 = _prep(skel_all)    # binary
        c2 = _prep(skel_curr)   # binary
        crop = np.stack([c0, c1, c2], axis=0)   # [3,H,W]
        x = torch.from_numpy(crop).float()[None, ...]  # [1,3,H,W]
        return x.to(self.device)

    # ---- classifier ----
    @torch.inference_mode()
    def classify(self, img_gray: np.ndarray) -> dict:
        x = self._prep_gray(img_gray, self.clf_hw)
        y = self.clf(_ensure_nchw(x))
        # normalize possible shapes: [N,2] or [N,2,1,1]
        if y.dim() == 4:
            y = y.squeeze(-1).squeeze(-1)
        probs = F.softmax(y, dim=1).detach().cpu().numpy()[0]
        label = int(np.argmax(probs))
        return {"logits": y.detach().cpu().numpy()[0], "probs": probs, "label": label}  # 0=class0, 1=class1

    # ---- segmentation: unidirectional -> 2-channel prob maps (anterograde, retrograde) ----
    @torch.inference_mode()
    def segment_uni(self, img_gray: np.ndarray) -> dict:
        x = self._prep_gray(img_gray, self.uni_hw)
        out = self.uni(_ensure_nchw(x))
        # onnx2pytorch might return dict or list; handle both
        ant, ret = None, None
        if isinstance(out, dict):
            # your report showed keys "ant" and "ret"
            ant = out.get("ant", None)
            ret = out.get("ret", None)
        elif isinstance(out, (list, tuple)):
            if len(out) == 2:
                ant, ret = out
        else:
            # single tensor with 2 channels -> [N,2,H,W]
            if out.dim() == 4 and out.shape[1] == 2:
                ant, ret = out[:,0], out[:,1]

        if ant is None or ret is None:
            raise RuntimeError("Couldn't parse uni_seg outputs (expected 'ant' and 'ret').")

        ant = torch.sigmoid(ant).squeeze().detach().cpu().numpy()
        ret = torch.sigmoid(ret).squeeze().detach().cpu().numpy()
        return {"ant": ant, "ret": ret}

    # ---- segmentation: bidirectional -> 1 prob map ----
    @torch.inference_mode()
    def segment_bi(self, img_gray: np.ndarray) -> np.ndarray:
        x = self._prep_gray(img_gray, self.bi_hw)
        out = self.bi(_ensure_nchw(x))
        # normalize to [N,1,H,W]
        if isinstance(out, (list, tuple)):
            out = out[0]
        if out.dim() == 3:
            out = out.unsqueeze(1)
        prob = torch.sigmoid(out).squeeze().detach().cpu().numpy()
        return prob

    # ---- decision net on a 48x48 crop (3 channels: raw, skel_all, skel_curr) ----
    @torch.inference_mode()
    def decision_map(self, crop_raw01: np.ndarray, crop_skel_all: np.ndarray, crop_skel_curr: np.ndarray) -> np.ndarray:
        x = self._prep_decision_crop(crop_raw01, crop_skel_all, crop_skel_curr)
        y = self.dec(_ensure_nchw(x))  # may return [N,2,H,W] or [N,H,W,2]
        if y.dim() == 4 and y.shape[1] == 2:
            logits = y                       # [N,2,H,W] (NCHW)
        elif y.dim() == 4 and y.shape[-1] == 2:
            logits = y.permute(0, 3, 1, 2)   # NHWC -> NCHW
        else:
            raise RuntimeError(f"Unexpected decision output shape: {tuple(y.shape)}")

        prob = torch.softmax(logits, dim=1)[:, 1]  # foreground class probability
        return prob.squeeze().detach().cpu().numpy()  # [H,W]

