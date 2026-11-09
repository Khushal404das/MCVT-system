"""Appearance feature extraction (FR-RI-01) and cheap colour descriptors."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import cv2
import numpy as np
import torch

from .resnet_ibn import ResNetIBNReID

log = logging.getLogger("reid")


class FeatureExtractor:
    """Turns vehicle crops into L2-normalised ResNet-IBN embeddings."""

    def __init__(self, weights: str | Path, input_size=(256, 256), device: str = "cpu",
                 adapter: str | Path | None = None):
        self.device = torch.device(device)
        self.adapter = ReIDAdapter(adapter) if adapter and Path(adapter).exists() else None
        self.height, self.width = int(input_size[0]), int(input_size[1])
        self.model = ResNetIBNReID()
        missing, unused = self.model.load_fastreid_checkpoint(str(weights))
        real_missing = [m for m in missing if not m.endswith("num_batches_tracked")]
        if real_missing:
            raise RuntimeError(f"ReID checkpoint does not match the model, missing: {real_missing[:5]}")
        log.info("ReID weights loaded (%d tensors ignored: classifier/training only)", len(unused))
        self.model.eval().to(self.device)
        torch.set_grad_enabled(False)
        self.dim = self.adapter.dim if self.adapter else 2048

    def _prep(self, crops: list[np.ndarray]) -> torch.Tensor:
        batch = np.empty((len(crops), self.height, self.width, 3), dtype=np.float32)
        for i, crop in enumerate(crops):
            rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            batch[i] = cv2.resize(rgb, (self.width, self.height), interpolation=cv2.INTER_LINEAR)
        return torch.from_numpy(batch).permute(0, 3, 1, 2).to(self.device)

    @torch.inference_mode()
    def extractFeatures(self, crops: list[np.ndarray] | np.ndarray, camera_ids: list[str] | None = None) -> np.ndarray:
        """Return an (N, dim) float32 array of unit-length embeddings (dim = 2048, or the
        adapter's size when a camera adapter is loaded; ``camera_ids`` selects each crop's
        camera correction)."""
        if isinstance(crops, np.ndarray) and crops.ndim == 3:
            crops = [crops]
        crops = [c for c in crops]
        if not crops:
            return np.zeros((0, self.dim), dtype=np.float32)
        feats = self.model(self._prep(crops))
        feats = torch.nn.functional.normalize(feats, dim=1).cpu().numpy().astype(np.float32)
        if self.adapter is not None:
            feats = self.adapter(feats, camera_ids or [None] * len(feats))
        return feats

    # snake_case alias
    extract_features = extractFeatures


class ReIDAdapter:
    """Camera-aware adaptation of the ResNet-IBN features, learned on this project's cameras
    (engine/training/train_reid_adapter.py): subtract the camera's mean feature, project with
    the learned matrix, normalise. Also converts match thresholds into the adapted space."""

    def __init__(self, path: str | Path):
        z = np.load(path)
        self.path = str(path)
        self.cam_means = {str(c): m for c, m in zip(z["cameras"], z["cam_means"])}
        self.mean_all = z["mean_all"]
        self.W = z["W"] if z["W"].size else None
        self.dim = self.W.shape[1] if self.W is not None else self.mean_all.shape[0]
        self.raw_neg, self.ada_neg = np.sort(z["raw_neg"]), np.sort(z["ada_neg"])
        self.info = json.loads(str(z["info"]))

    def __call__(self, feats: np.ndarray, camera_ids: list[str | None]) -> np.ndarray:
        x = feats - np.stack([self.cam_means.get(str(c), self.mean_all) for c in camera_ids])
        if self.W is not None:
            x = x @ self.W
        return (x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-9)).astype(np.float32)

    def margin(self, raw: float) -> float:
        """A similarity *difference* in plain units converted by the spread of different-vehicle pairs."""
        return float(raw * self.ada_neg.std() / max(self.raw_neg.std(), 1e-9))

    def threshold(self, raw: float) -> float:
        """Adapted threshold accepting the same share of different-vehicle pairs as ``raw``."""
        fpr = float((self.raw_neg >= raw).mean())
        return float(np.quantile(self.ada_neg, 1.0 - fpr)) if fpr > 0 else float(self.ada_neg.max() + 1e-3)


def color_histogram(crop: np.ndarray, bins=(8, 4, 4)) -> np.ndarray:
    """Hellinger-normalised HSV histogram of the crop centre.

    Used as the DeepSORT appearance descriptor on CPU-only machines where running
    the ResNet-IBN network on every detection of every frame is too slow.
    """
    h, w = crop.shape[:2]
    if h < 4 or w < 4:
        return np.full(int(np.prod(bins)), 1.0 / np.sqrt(np.prod(bins)), dtype=np.float32)
    c = crop[h // 6: h - h // 6, w // 6: w - w // 6]
    hsv = cv2.cvtColor(c, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1, 2], None, list(bins), [0, 180, 0, 256, 0, 256]).flatten()
    hist = np.sqrt(hist / max(hist.sum(), 1e-6))
    return (hist / max(np.linalg.norm(hist), 1e-6)).astype(np.float32)
