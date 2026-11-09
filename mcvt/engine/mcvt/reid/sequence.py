"""Sequence ReID: a GRU turns a track's time-ordered crop features into one vehicle signature
(trained by engine/training/kaggle_train_gru.py). Used instead of the plain average of the
crop features when reid.sequence_model is set."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

log = logging.getLogger("reid")


class SeqReID(nn.Module):
    """Sequence of crop features (B x L x d_in) -> d_out signature (same layers as the training script)."""

    def __init__(self, d_in=2048, d_h=512, d_out=512, **_):
        super().__init__()
        self.inp = nn.Sequential(nn.Linear(d_in, d_h), nn.LayerNorm(d_h), nn.GELU(), nn.Dropout(0.1))
        self.gru = nn.GRU(d_h, d_h, batch_first=True, bidirectional=True)
        self.att = nn.Linear(2 * d_h, 1)
        self.out = nn.Linear(2 * d_h + d_in, d_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # one sequence: 1 x L x d_in
        h, _ = self.gru(self.inp(x))
        a = self.att(h).squeeze(-1).softmax(1)
        pooled = (a.unsqueeze(-1) * h).sum(1)
        mean = F.normalize(x.mean(1), dim=1)
        return self.out(torch.cat([pooled, mean], 1))


class SequenceAggregator:
    """signature(list of crop features, oldest first) -> unit-length vector."""

    def __init__(self, path: str | Path):
        ckpt = torch.load(str(path), map_location="cpu", weights_only=True)
        self.config = dict(ckpt.get("config") or {})
        self.max_len = int(self.config.get("max_len", 8))
        self.model = SeqReID(**self.config)
        self.model.load_state_dict(ckpt["state_dict"])
        self.model.eval()
        self.dim = int(self.config.get("d_out", 512))
        self.calibration = ckpt.get("calibration") or {}
        self.path = str(path)

    @torch.inference_mode()
    def signature(self, vectors) -> np.ndarray:
        vecs = [np.asarray(v, dtype=np.float32).ravel() for v in vectors][-self.max_len:]
        x = torch.from_numpy(np.stack(vecs))[None]
        x = F.normalize(x, dim=2)
        out = F.normalize(self.model(x), dim=1)[0]
        return out.numpy().astype(np.float32)

    def convert(self, avg_threshold: float) -> float:
        """A threshold set for the plain average of crop features -> GRU units: the GRU threshold that
        lets through the same share of DIFFERENT-vehicle pairs (measured on our cameras' test tracks)."""
        a, g = self.calibration.get("average"), self.calibration.get("gru")
        if not a or not g or avg_threshold >= 1.0:
            return avg_threshold
        an, gn = np.sort(np.asarray(a["neg"])), np.sort(np.asarray(g["neg"]))
        fpr = float((an >= avg_threshold).mean())
        return float(np.quantile(gn, 1.0 - fpr)) if fpr > 0 else float(gn.max() + 1e-3)

    def convert_margin(self, avg_margin: float) -> float:
        """A similarity difference in average units -> GRU units (by the spread of different-vehicle pairs)."""
        a, g = self.calibration.get("average"), self.calibration.get("gru")
        if not a or not g:
            return avg_margin
        return float(avg_margin * np.std(g["neg"]) / max(np.std(a["neg"]), 1e-9))
