"""Train a ReID feature head on this project's cameras: triplet loss + ID loss.

The ResNet50-IBN (VeRi-776 weights) stays frozen - fine-tuning it on a CPU takes hours per
epoch - and a learned linear head maps its 2048-d feature to a 256-d embedding in which

* crops of ONE vehicle (other moments, other camera, far-away or close-up) are close, and
* crops of DIFFERENT vehicles are pushed apart (batch-hard triplet loss, margin --margin),

plus an ID (classification) loss over the training vehicles, as in the BoT / FastReID recipe.

Training data (no hand labels), from make_reid_dataset.py:
* one tracklet = one moving vehicle in one camera; crops of it are positives;
* cameras in reid.overlapping_cameras (same place, same moment): the time-overlapping, mutually
  most similar tracklets are the same vehicle (cross-camera positives);
* every crop is also fed as low-resolution copies (shrunk to --lowres-min..--lowres-max px and
  blurred back up), so a tiny far-away view learns to match the close-up of the same car.

The later part of the video (--train-fraction) is held out and the head is compared with the
plain ResNet features on it. ``--final`` then refits on all data for the saved head.

Usage:  python engine/training/train_reid_head.py [--final]
Output: engine/models/reid_head.npz (loaded like a camera adapter: reid.adapter in config.yaml)
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from mcvt.config import load_config  # noqa: E402
from train_reid_adapter import extract, identities, l2n, load_tracklets  # noqa: E402


def lowres_features(data: Path, tracklets, cfg, copies: int, lo: int, hi: int, threads: int, seed: int = 0):
    """Features of shrunk + blurred copies of every crop (cached)."""
    import torch
    from mcvt.reid.extractor import FeatureExtractor

    cache_file = data / f"features_lowres_{copies}x_{lo}-{hi}.npz"
    cache = dict(np.load(cache_file)) if cache_file.exists() else {}
    files = [c["file"] for t in tracklets for c in t["crops"]]
    todo = [(f, k) for f in files for k in range(copies) if f"{f}#{k}" not in cache]
    if not todo:
        return cache
    torch.set_num_threads(threads)
    ex = FeatureExtractor(cfg.path("reid.weights"), cfg.get("reid.input_size", [256, 256]), "cpu")
    rng = np.random.default_rng(seed)
    t0 = time.time()
    for i in range(0, len(todo), 16):
        batch = todo[i:i + 16]
        imgs = []
        for f, _k in batch:
            img = cv2.imread(str(data / f))
            h, w = img.shape[:2]
            side = int(rng.integers(lo, hi + 1))
            s = side / max(h, w)
            if s < 1.0:
                small = cv2.resize(img, (max(4, int(w * s)), max(4, int(h * s))), interpolation=cv2.INTER_AREA)
                img = cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)
            imgs.append(img)
        feats = ex.extractFeatures(imgs)
        cache.update({f"{f}#{k}": v for (f, k), v in zip(batch, feats)})
        if (i // 16) % 10 == 0:
            print(f"  low-res features {i + len(batch)}/{len(todo)} ({time.time() - t0:.0f} s)", flush=True)
            np.savez(cache_file, **cache)
    np.savez(cache_file, **cache)
    return cache


def samples(idx, tracklets, labels, feats, low, copies):
    """(features, identity, tracklet, is_lowres) for the crops of the given tracklets."""
    X, y, tr, lr = [], [], [], []
    for i in idx:
        for c in tracklets[i]["crops"]:
            X.append(feats[c["file"]]); y.append(labels[i]); tr.append(i); lr.append(0)
            for k in range(copies):
                key = f"{c['file']}#{k}"
                if key in low:
                    X.append(low[key]); y.append(labels[i]); tr.append(i); lr.append(1)
    return np.stack(X).astype(np.float32), np.array(y), np.array(tr), np.array(lr)


def train_head(X, y, dim, epochs, margin, lr, wd, seed=0, log=True):
    import torch
    import torch.nn.functional as F

    torch.set_grad_enabled(True)  # FeatureExtractor switches autograd off process-wide
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    mean = X.mean(0)  # one mean for all cameras (per-camera centring hurt cameras it was not trained on)
    Xc = torch.from_numpy((X - mean).astype(np.float32))
    classes, yi = np.unique(y, return_inverse=True)
    yt = torch.from_numpy(yi)
    W = torch.nn.Parameter(torch.randn(X.shape[1], dim) * (1.0 / np.sqrt(X.shape[1])))
    C = torch.nn.Parameter(torch.randn(len(classes), dim) * 0.01)  # ID-loss class centres (cosine softmax)
    opt = torch.optim.Adam([W, C], lr=lr, weight_decay=wd)
    by_id = {c: np.where(yi == c)[0] for c in range(len(classes))}
    multi = [c for c, v in by_id.items() if len(v) >= 2]
    P, K = 16, 4  # identities x samples per batch
    for ep in range(epochs):
        tot, nb = 0.0, 0
        rng.shuffle(multi)
        for b in range(0, len(multi) - P + 1, P):
            ids = multi[b:b + P]
            pick = np.concatenate([rng.choice(by_id[c], K, replace=len(by_id[c]) < K) for c in ids])
            z = F.normalize(Xc[pick] @ W, dim=1)
            lab = yt[pick]
            d = 2 - 2 * z @ z.T  # squared euclidean on the unit sphere
            same = lab[:, None] == lab[None, :]
            hardest_pos = (d * same).max(1).values
            hardest_neg = (d + same * 10.0).min(1).values
            trip = F.relu(hardest_pos - hardest_neg + margin).mean()
            logits = 16.0 * z @ F.normalize(C, dim=1).T
            idl = F.cross_entropy(logits, lab)
            loss = trip + idl
            opt.zero_grad(); loss.backward(); opt.step()
            tot += float(loss); nb += 1
        if log and (ep % 10 == 0 or ep == epochs - 1):
            print(f"  epoch {ep:3d} loss {tot / max(nb, 1):.3f}", flush=True)
    return mean.astype(np.float32), W.detach().numpy().astype(np.float32)


def embed(X, head):
    if head is None:
        return l2n(X)
    mean, W = head
    return l2n((X - mean) @ W)


def evaluate(head, tracklets, labels, idx, feats, low, copies, overlaps, thr_grid):
    """Held-out tracklets: cross-camera retrieval (overlapping cameras), low-res -> sharp
    retrieval, how much different vehicles overlap, and the live ID rule."""
    def tmean(i, use_low=False):
        vs = []
        for c in tracklets[i]["crops"]:
            if use_low:
                vs += [low[f"{c['file']}#{k}"] for k in range(copies) if f"{c['file']}#{k}" in low]
            else:
                vs.append(feats[c["file"]])
        return l2n(embed(np.stack(vs), head).mean(0)) if vs else None

    out = {}
    M = {i: tmean(i) for i in idx}
    L = {i: tmean(i, True) for i in idx}
    cams = {i: tracklets[i]["camera"] for i in idx}
    # 1) cross-camera: tracklets of camera B searched among camera A (overlap pairs)
    r1, rows = [], []
    for ca, cb in overlaps:
        gal = [i for i in idx if cams[i] == ca]
        if not gal:
            continue
        G = np.stack([M[i] for i in gal])
        for q in (i for i in idx if cams[i] == cb):
            s = G @ M[q]
            truth = {j for j, g in enumerate(gal) if labels[g] == labels[q]}
            rows.append((s, truth))
            if truth:
                r1.append(int(np.argmax(s)) in truth)
    out["cross_rank1"] = float(np.mean(r1)) if r1 else float("nan")
    out["cross_queries"] = len(r1)
    # 2) far-away (low-res) view of a vehicle searched among all sharp tracklets (other ones too)
    G = np.stack([M[i] for i in idx])
    r1l = [int(np.argmax(G @ L[q])) == idx.index(q) or labels[idx[int(np.argmax(G @ L[q]))]] == labels[q]
           for q in idx if L[q] is not None]
    out["lowres_rank1"] = float(np.mean(r1l)) if r1l else float("nan")
    # 3) feature overlap of DIFFERENT vehicles vs SAME vehicle
    ids = list(idx)
    S = G @ G.T
    diff = [S[a, b] for a in range(len(ids)) for b in range(a + 1, len(ids)) if labels[ids[a]] != labels[ids[b]]]
    same = [float(M[i] @ L[i]) for i in idx if L[i] is not None]  # one vehicle: sharp vs far-away view
    same += [S[a, b] for a in range(len(ids)) for b in range(a + 1, len(ids)) if labels[ids[a]] == labels[ids[b]]]
    diff, same = np.array(diff), np.array(same)
    out["diff_p99"] = float(np.percentile(diff, 99))
    out["same_p10"] = float(np.percentile(same, 10))
    out["same_median"] = float(np.median(same))
    out["auc"] = float((same[:, None] > diff[None, ::max(1, len(diff) // 4000)]).mean())
    # 4) the live ID rule on the cross-camera queries: best >= threshold
    rule = {}
    for thr in thr_grid:
        c = {"right": 0, "wrong": 0, "dup": 0, "newok": 0}
        for s, truth in rows:
            j = int(np.argmax(s))
            if s[j] >= thr:
                c["right" if j in truth else "wrong"] += 1
            else:
                c["dup" if truth else "newok"] += 1
        n = max(1, len(rows))
        rule[thr] = {k: v / n for k, v in c.items()}
    out["rule"] = rule
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--dim", type=int, default=256)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--margin", type=float, default=0.3)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--copies", type=int, default=2, help="low-resolution copies per crop")
    ap.add_argument("--lowres-min", type=int, default=18)
    ap.add_argument("--lowres-max", type=int, default=48)
    ap.add_argument("--overlap-min-sim", type=float, default=0.15)
    ap.add_argument("--train-fraction", type=float, default=0.6)
    ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--final", action="store_true", help="after evaluating, refit on ALL tracklets for the saved head")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    data = cfg.path("paths.storage") / "training" / "reid"
    tracklets = load_tracklets(data)
    feats = extract(data, tracklets, cfg, args.threads)
    low = lowres_features(data, tracklets, cfg, args.copies, args.lowres_min, args.lowres_max, args.threads)
    overlaps = [tuple(map(str, p)) for p in (cfg.get("reid.overlapping_cameras") or []) if len(p) == 2]
    labels, splits, links, _uf, _m = identities(tracklets, feats, overlaps, split_gap=4.0, split_sim=0.8,
                                                overlap_min_s=1.0, overlap_min_sim=args.overlap_min_sim)
    print(f"{len(tracklets)} tracklets, {len(set(labels))} vehicles ({splits} occlusion splits, {links} "
          f"overlapping-camera links {overlaps})")
    t_end = max(t["t1"] for t in tracklets)
    split_t = args.train_fraction * t_end
    train = [i for i, t in enumerate(tracklets) if t["t1"] < split_t]
    test = [i for i, t in enumerate(tracklets) if t["t0"] >= split_t]
    X, y, _tr, _lr = samples(train, tracklets, labels, feats, low, args.copies)
    print(f"train: {len(train)} tracklets / {len(X)} samples (incl. low-res copies); held out: {len(test)} tracklets")

    t0 = time.time()
    head = train_head(X, y, args.dim, args.epochs, args.margin, args.lr, args.wd)
    print(f"trained in {time.time() - t0:.0f} s")
    grid = [0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]
    res = {}
    for name, h in (("plain ResNet50-IBN", None), ("trained head", head)):
        m = evaluate(h, tracklets, labels, test, feats, low, args.copies, overlaps, grid)
        res[name] = m
        print(f"\n{name}: cross-camera rank-1 {m['cross_rank1']:.2f} ({m['cross_queries']} queries), far-away->sharp "
              f"rank-1 {m['lowres_rank1']:.2f}, AUC {m['auc']:.3f}")
        print(f"   different vehicles: 99% below {m['diff_p99']:.2f} | same vehicle: 90% above {m['same_p10']:.2f}, "
              f"median {m['same_median']:.2f}")
        for thr, r in m["rule"].items():
            print(f"   threshold {thr:.2f}: right id {r['right']:.0%}  WRONG id {r['wrong']:.0%}  duplicate {r['dup']:.0%}"
                  f"  new id ok {r['newok']:.0%}")

    if args.final:
        X, y, _tr, _lr = samples(list(range(len(tracklets))), tracklets, labels, feats, low, args.copies)
        head = train_head(X, y, args.dim, args.epochs, args.margin, args.lr, args.wd, log=False)
        print("refit on all tracklets")
    mean, W = head
    # different-vehicle similarities (for the adapter's threshold conversion, unused when
    # reid.adapter_threshold is set)
    allidx = list(range(len(tracklets)))
    raw = l2n(np.stack([np.mean([feats[c["file"]] for c in tracklets[i]["crops"]], 0) for i in allidx]))
    ada = embed(np.stack([np.mean([feats[c["file"]] for c in tracklets[i]["crops"]], 0) for i in allidx]), head)
    neg = [(a, b) for a in allidx for b in allidx[a + 1:] if labels[a] != labels[b]][::7]
    raw_neg = np.array([raw[a] @ raw[b] for a, b in neg], np.float32)
    ada_neg = np.array([ada[a] @ ada[b] for a, b in neg], np.float32)
    cams = sorted({t["camera"] for t in tracklets})
    out = Path(args.out) if args.out else cfg.path("reid.weights").parent / "reid_head.npz"
    np.savez(out, cameras=np.array(cams), cam_means=np.stack([mean for _ in cams]), mean_all=mean, W=W,
             raw_neg=raw_neg, ada_neg=ada_neg,
             info=json.dumps({"method": f"triplet+ID head {W.shape[1]}-d (low-res copies {args.copies})",
                              "trained_at": time.time(), "tracklets": len(tracklets),
                              "metrics": {k: {kk: vv for kk, vv in v.items() if kk != "rule"} for k, v in res.items()},
                              "args": vars(args)}))
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
