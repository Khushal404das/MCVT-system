"""Adapt the ReID features to this project's cameras (camera-aware metric learning).

Fine-tuning the ResNet50-IBN backbone itself is not practical on a CPU (hours per epoch and
the live engine would have to stop), so the frozen network's 2048-d features are adapted
with a light, learned layer trained on the project's own footage:

1. camera bias removal: every camera shifts all features in its own way (viewpoint,
   lens, light). The mean feature of each camera is subtracted (camera-aware centring);
2. LDA projection (2048 -> --dim): directions in which different vehicles differ a lot
   while one vehicle's crops (other moments, other cameras) differ little.

Identities come from make_reid_dataset.py tracklets (one moving vehicle in one camera).
Tracklets of one camera that a short occlusion split are merged, and cameras listed in
reid.overlapping_cameras (same place, same moment) give cross-camera identities.

The later part of the video (--train-fraction) is held out to measure cross-camera
matching before / after. Match thresholds of the raw feature space are converted to the
adapted space (same rate of accepted different-vehicle pairs) and stored with the adapter.

Usage:  python training/train_reid_adapter.py [--dim 128] [--verified VH-1:VH-2,VH-3:VH-4]
Output: engine/models/reid_adapter.npz  (enable with reid.adapter in config.yaml)
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mcvt.config import load_config  # noqa: E402


def l2n(x: np.ndarray) -> np.ndarray:
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-9)


class UnionFind:
    def __init__(self, n: int):
        self.p = list(range(n))

    def find(self, a: int) -> int:
        while self.p[a] != a:
            self.p[a] = self.p[self.p[a]]
            a = self.p[a]
        return a

    def union(self, a: int, b: int) -> None:
        self.p[self.find(a)] = self.find(b)


def load_tracklets(data: Path) -> list[dict]:
    out = []
    for f in sorted(data.glob("tracklets*.json")):
        out.extend(json.loads(f.read_text())["tracklets"])
    return out


def extract(data: Path, tracklets: list[dict], cfg, threads: int) -> dict[str, np.ndarray]:
    import torch
    from mcvt.reid.extractor import FeatureExtractor

    cache_file = data / "features_raw.npz"
    cache = dict(np.load(cache_file)) if cache_file.exists() else {}
    todo = [c["file"] for t in tracklets for c in t["crops"] if c["file"] not in cache]
    if todo:
        torch.set_num_threads(threads)
        ex = FeatureExtractor(cfg.path("reid.weights"), cfg.get("reid.input_size", [256, 256]), "cpu")
        t0 = time.time()
        for i in range(0, len(todo), 8):
            batch = todo[i:i + 8]
            feats = ex.extractFeatures([cv2.imread(str(data / f)) for f in batch])
            cache.update(zip(batch, feats))
            if (i // 8) % 25 == 0:
                print(f"  features {i + len(batch)}/{len(todo)} ({time.time() - t0:.0f} s)", flush=True)
                np.savez(cache_file, **cache)
        np.savez(cache_file, **cache)
    return cache


def identities(tracklets, feats, overlaps, split_gap, split_sim, overlap_min_s, overlap_min_sim):
    """Union-find over tracklets: occlusion splits inside a camera, overlapping cameras."""
    n = len(tracklets)
    uf = UnionFind(n)
    mean = [l2n(np.mean([feats[c["file"]] for c in t["crops"]], axis=0)) for t in tracklets]
    by_cam: dict[str, list[int]] = {}
    for i, t in enumerate(tracklets):
        by_cam.setdefault(t["camera"], []).append(i)
    splits = 0
    for idx in by_cam.values():
        idx.sort(key=lambda i: tracklets[i]["t0"])
        for a_pos, a in enumerate(idx):
            for b in idx[a_pos + 1:]:
                gap = tracklets[b]["t0"] - tracklets[a]["t1"]
                if gap > split_gap:
                    break
                if gap >= -0.5 and float(mean[a] @ mean[b]) >= split_sim:
                    uf.union(a, b)
                    splits += 1
    links = 0
    for ca, cb in overlaps:
        A, B = by_cam.get(ca, []), by_cam.get(cb, [])
        cand = {}
        for a in A:
            for b in B:
                ov = min(tracklets[a]["t1"], tracklets[b]["t1"]) - max(tracklets[a]["t0"], tracklets[b]["t0"])
                if ov >= overlap_min_s:
                    cand[(a, b)] = float(mean[a] @ mean[b])
        # mutual best pairs (time overlap + most similar) are the same vehicle
        best_a, best_b = {}, {}
        for (a, b), s in cand.items():
            if s > best_a.get(a, (None, -1))[1]:
                best_a[a] = (b, s)
            if s > best_b.get(b, (None, -1))[1]:
                best_b[b] = (a, s)
        for a, (b, s) in best_a.items():
            if best_b.get(b, (None,))[0] == a and s >= overlap_min_sim:
                uf.union(a, b)
                links += 1
    labels = np.array([uf.find(i) for i in range(n)])
    return labels, splits, links, uf, mean


def transit_links(tracklets, mean, uf, overlaps, offsets, window, min_sim, margin) -> int:
    """Spatio-temporal pseudo-labels between NON-overlapping cameras: a vehicle that leaves one
    camera and a vehicle that enters another within ``window`` seconds (shared clock), which are
    each other's most similar candidate by ``margin``, are taken as the same vehicle."""
    ov = {frozenset(p) for p in overlaps}
    clock = [(t["t0"] + offsets.get(t["camera"], 0.0), t["t1"] + offsets.get(t["camera"], 0.0)) for t in tracklets]
    cand: dict[int, list[tuple[float, int]]] = {}
    rev: dict[int, list[tuple[float, int]]] = {}
    for a, ta in enumerate(tracklets):
        for b, tb in enumerate(tracklets):
            if ta["camera"] == tb["camera"] or frozenset((ta["camera"], tb["camera"])) in ov:
                continue
            dt = clock[b][0] - clock[a][1]
            if 0 < dt <= window:
                s = float(mean[a] @ mean[b])
                cand.setdefault(a, []).append((s, b))
                rev.setdefault(b, []).append((s, a))
    links = 0
    for a, lst in cand.items():
        lst.sort(reverse=True)
        s, b = lst[0]
        second = lst[1][0] if len(lst) > 1 else -1.0
        back = sorted(rev[b], reverse=True)
        back_second = back[1][0] if len(back) > 1 else -1.0
        if s >= min_sim and back[0][1] == a and s - second >= margin and s - back_second >= margin:
            uf.union(a, b)
            links += 1
    return links


class Adapter:
    def __init__(self, cam_means: dict[str, np.ndarray], mean_all: np.ndarray, W: np.ndarray | None):
        self.cam_means, self.mean_all, self.W = cam_means, mean_all, W

    def __call__(self, X: np.ndarray, cams: list[str]) -> np.ndarray:
        X = l2n(X) - np.stack([self.cam_means.get(c, self.mean_all) for c in cams])
        if self.W is not None:
            X = X @ self.W
        return l2n(X)


def fit(X: np.ndarray, cams: list[str], labels: np.ndarray, dim: int, reg: float, per_camera: bool = True) -> Adapter:
    Xn = l2n(X)
    mean_all = Xn.mean(0)
    cam_means = ({c: Xn[[i for i, k in enumerate(cams) if k == c]].mean(0) for c in set(cams)} if per_camera
                 else {c: mean_all for c in set(cams)})
    Z = Xn - np.stack([cam_means[c] for c in cams])
    if dim <= 0:
        return Adapter(cam_means, mean_all, None)
    d = Z.shape[1]
    Sw = np.zeros((d, d))
    for lab in np.unique(labels):
        m = labels == lab
        if m.sum() < 2:
            continue
        D = Z[m] - Z[m].mean(0)
        Sw += D.T @ D
    Sw /= len(Z)
    St = np.cov(Z, rowvar=False)
    Sb = St - Sw
    Sw += np.eye(d) * reg * np.trace(Sw) / d
    w, U = np.linalg.eigh(Sw)
    P = U / np.sqrt(np.maximum(w, 1e-12))
    e, V = np.linalg.eigh(P.T @ Sb @ P)
    W = P @ V[:, np.argsort(e)[::-1][:dim]]
    return Adapter(cam_means, mean_all, W.astype(np.float32))


def evaluate(ad: Adapter | None, feats, tracklets, labels, idx, overlaps):
    """Cross-camera retrieval on held-out tracklets (overlap-camera identities) + pair AUC."""
    desc = {}
    for i in idx:
        X = np.stack([feats[c["file"]] for c in tracklets[i]["crops"]])
        cams = [tracklets[i]["camera"]] * len(X)
        Y = l2n(X) if ad is None else ad(X, cams)
        desc[i] = l2n(Y.mean(0))
    r1, aps, pos, neg = [], [], [], []
    for ca, cb in overlaps:
        for qa, ga in ((ca, cb), (cb, ca)):
            Q = [i for i in idx if tracklets[i]["camera"] == qa]
            G = [i for i in idx if tracklets[i]["camera"] == ga]
            if not G:
                continue
            Gm = np.stack([desc[g] for g in G])
            for q in Q:
                truth = np.array([labels[g] == labels[q] for g in G])
                if not truth.any():
                    continue
                s = Gm @ desc[q]
                order = np.argsort(-s)
                hits = truth[order]
                r1.append(float(hits[0]))
                prec = np.cumsum(hits) / (np.arange(len(hits)) + 1)
                aps.append(float((prec * hits).sum() / hits.sum()))
                pos.extend(s[truth].tolist())
                neg.extend(s[~truth].tolist())
    pos, neg = np.array(pos), np.array(neg)
    auc = float((pos[:, None] > neg[None, :]).mean()) if len(pos) and len(neg) else float("nan")
    return {"queries": len(r1), "rank1": float(np.mean(r1)) if r1 else float("nan"),
            "mAP": float(np.mean(aps)) if aps else float("nan"), "auc": auc,
            "pos_mean": float(pos.mean()) if len(pos) else float("nan"),
            "neg_mean": float(neg.mean()) if len(neg) else float("nan")}, neg


def cross_negatives(ad: Adapter | None, feats, tracklets, labels, idx) -> np.ndarray:
    """Similarities of different vehicles in different cameras (for threshold conversion)."""
    D, C, L = [], [], []
    for i in idx:
        X = np.stack([feats[c["file"]] for c in tracklets[i]["crops"]])
        Y = l2n(X) if ad is None else ad(X, [tracklets[i]["camera"]] * len(X))
        D.append(l2n(Y.mean(0)))
        C.append(tracklets[i]["camera"])
        L.append(labels[i])
    D, C, L = np.stack(D), np.array(C), np.array(L)
    S = D @ D.T
    m = (C[:, None] != C[None, :]) & (L[:, None] != L[None, :])
    return np.sort(S[np.triu(m, 1)])


def verified_ranks(ad: Adapter | None, db_path: Path, pairs: list[tuple[str, str]], window: float):
    """Rank of the true match among all vehicles the engine saw around the same time (from the DB)."""
    c = sqlite3.connect(db_path)
    out = []
    for q, p in pairs:
        rows = c.execute("SELECT vec, camera_id, ts FROM embeddings WHERE gid=?", (q,)).fetchall()
        if not rows:
            continue
        t = rows[0][2]
        X = np.stack([np.frombuffer(r[0], np.float32) for r in rows])
        qd = l2n((l2n(X) if ad is None else ad(X, [r[1] for r in rows])).mean(0))
        gal = {}
        for g, v, cam in c.execute("SELECT gid, vec, camera_id FROM embeddings WHERE ts BETWEEN ? AND ? AND gid != ?",
                                   (t - window, t + window, q)):
            gal.setdefault(g, ([], []))
            gal[g][0].append(np.frombuffer(v, np.float32))
            gal[g][1].append(cam)
        if p not in gal:
            continue
        sims = {}
        for g, (vs, cams) in gal.items():
            Y = l2n(np.stack(vs)) if ad is None else ad(np.stack(vs), cams)
            sims[g] = float(l2n(Y.mean(0)) @ qd)
        same_car = {b for a, b in pairs if a == q}  # other ids of the same car are not competitors
        rank = 1 + sum(s > sims[p] for g, s in sims.items() if g not in same_car)
        out.append((q, p, rank, len(sims), sims[p]))
    return out


def convert(raw_neg: np.ndarray, ada_neg: np.ndarray, tau: float) -> float:
    """Adapted threshold that accepts the same share of different-vehicle pairs as ``tau`` did."""
    fpr = float((raw_neg >= tau).mean())
    return float(np.quantile(ada_neg, 1.0 - fpr)) if fpr > 0 else float(ada_neg.max() + 1e-3)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--data", default=None)
    ap.add_argument("--dim", type=int, default=128)
    ap.add_argument("--reg", type=float, default=0.1, help="within-class scatter regularisation")
    ap.add_argument("--train-fraction", type=float, default=0.6, help="earlier part of the video used for training")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--verified", default="", help="comma list of gidA:gidB pairs checked by eye (DB embeddings)")
    ap.add_argument("--transit-window", type=float, default=120, help="0 = no transit pseudo-labels")
    ap.add_argument("--transit-min-sim", type=float, default=0.6)
    ap.add_argument("--transit-margin", type=float, default=0.05)
    ap.add_argument("--dims", default="64,128,256")
    ap.add_argument("--final", action="store_true", help="after evaluating, refit on ALL tracklets for the saved adapter")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    data = Path(args.data) if args.data else cfg.path("paths.storage") / "training" / "reid"
    tracklets = load_tracklets(data)
    print(f"{len(tracklets)} tracklets, {sum(len(t['crops']) for t in tracklets)} crops")
    feats = extract(data, tracklets, cfg, args.threads)
    overlaps = [tuple(map(str, p)) for p in (cfg.get("reid.overlapping_cameras") or []) if len(p) == 2]
    labels, splits, links, uf, means = identities(tracklets, feats, overlaps, split_gap=4.0, split_sim=0.8,
                                                  overlap_min_s=1.0, overlap_min_sim=0.3)
    tl = 0
    if args.transit_window > 0:
        # shared clock: each camera's playback starts this many seconds into its recording
        offsets = {s.id: float(s.extra.get("analysis_start_seconds" if s.analysis_source else "start_seconds", 0.0))
                   for s in cfg.camera_specs()}
        tl = transit_links(tracklets, means, uf, overlaps, offsets, args.transit_window, args.transit_min_sim,
                           args.transit_margin)
        labels = np.array([uf.find(i) for i in range(len(tracklets))])
    print(f"identities: {len(set(labels))} (merged {splits} occlusion splits, {links} overlap-camera links, "
          f"{tl} transit links)")

    t_end = max(t["t1"] for t in tracklets)
    split_t = args.train_fraction * t_end
    train = [i for i, t in enumerate(tracklets) if t["t1"] < split_t]
    test = [i for i, t in enumerate(tracklets) if t["t0"] >= split_t]
    files = [(c["file"], tracklets[i]["camera"], labels[i]) for i in train for c in tracklets[i]["crops"]]
    X = np.stack([feats[f] for f, _, _ in files])
    cams = [c for _, c, _ in files]
    labs = np.array([l for _, _, l in files])
    print(f"train: {len(train)} tracklets / {len(X)} crops (video < {split_t:.0f} s); test: {len(test)} tracklets")

    db = cfg.path("paths.database") if cfg.get("paths.database") else cfg.path("paths.storage") / "mcvt.sqlite"
    pairs = [tuple(p.split(":")) for p in args.verified.split(",") if ":" in p]
    results = {}
    candidates = {"raw": None, "camera-centred": fit(X, cams, labs, 0, args.reg)}
    for d in sorted({int(x) for x in args.dims.split(",") if x}):
        candidates[f"centred+LDA{d}"] = fit(X, cams, labs, d, args.reg)
        candidates[f"LDA{d}"] = fit(X, cams, labs, d, args.reg, per_camera=False)
    for name, ad in candidates.items():
        m, _ = evaluate(ad, feats, tracklets, labels, test, overlaps)
        vr = verified_ranks(ad, db, pairs, 600) if pairs else []
        m["verified_rank1"] = float(np.mean([r[2] == 1 for r in vr])) if vr else float("nan")
        m["verified_mean_rank"] = float(np.mean([r[2] for r in vr])) if vr else float("nan")
        results[name] = (m, vr)
        print(f"{name:16s} queries={m['queries']:3d} rank-1={m['rank1']:.2f} mAP={m['mAP']:.2f} AUC={m['auc']:.3f} "
              f"same={m['pos_mean']:.2f} diff={m['neg_mean']:.2f} | verified rank-1={m['verified_rank1']:.2f} "
              f"mean rank={m['verified_mean_rank']:.1f}", flush=True)
        for q, p, r, n, s in vr:
            print(f"    {q} -> {p}: rank {r}/{n} (similarity {s:.2f})")

    best = max((k for k in results if k != "raw"), key=lambda k: (results[k][0]["mAP"], results[k][0]["auc"]))
    print(f"best: {best}")
    ad = candidates[best]
    if args.final:
        allf = [(c["file"], t["camera"], labels[i]) for i, t in enumerate(tracklets) for c in t["crops"]]
        dim = 0 if ad.W is None else ad.W.shape[1]
        ad = fit(np.stack([feats[f] for f, _, _ in allf]), [c for _, c, _ in allf], np.array([l for _, _, l in allf]),
                 dim, args.reg, per_camera=not best.startswith("LDA"))
    # threshold conversion (same accepted share of different-vehicle pairs)
    everything = list(range(len(tracklets)))
    raw_neg = cross_negatives(None, feats, tracklets, labels, everything)
    ada_neg = cross_negatives(ad, feats, tracklets, labels, everything)
    names = ["similarity_threshold", "handoff_threshold", "follow_threshold", "follow_min_similarity"]
    conv = {k: convert(raw_neg, ada_neg, float(cfg.get(f"reid.{k}"))) for k in names if cfg.get(f"reid.{k}") is not None}
    for k, v in conv.items():
        print(f"  {k}: {cfg.get(f'reid.{k}')} -> {v:.3f}")
    out = Path(args.out) if args.out else cfg.path("reid.weights").parent / "reid_adapter.npz"
    cam_ids = sorted(ad.cam_means)
    np.savez(out, cameras=np.array(cam_ids), cam_means=np.stack([ad.cam_means[c] for c in cam_ids]).astype(np.float32),
             mean_all=ad.mean_all.astype(np.float32), W=ad.W if ad.W is not None else np.zeros((0, 0), np.float32),
             raw_neg=raw_neg[:: max(1, len(raw_neg) // 5000)].astype(np.float32),
             ada_neg=ada_neg[:: max(1, len(ada_neg) // 5000)].astype(np.float32),
             info=json.dumps({"method": best, "trained_at": time.time(), "tracklets": len(tracklets),
                              "metrics": {k: v[0] for k, v in results.items()}, "thresholds": conv}))
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
