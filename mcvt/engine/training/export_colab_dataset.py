"""Pack this project's labelled vehicle crops for fine-tuning the ReID network on a GPU
(Google Colab / Kaggle, notebook engine/training/colab_finetune_reid.ipynb).

Labels (no hand labelling) are built exactly as train_reid_adapter.py / train_reid_head.py do:
one tracklet = one vehicle in one camera, occlusion-split tracklets joined, and cameras in
reid.overlapping_cameras (same place, same moment) joined by time overlap + mutual best match.
The later part of the video (--train-fraction) is marked "test" so the notebook can measure the
fine-tuned model on footage it was not trained on.

Usage:  python engine/training/export_colab_dataset.py
Output: storage/training/colab/mcvt_reid_dataset.zip  (crops/ + labels.json)
"""

from __future__ import annotations

import argparse
import json
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from mcvt.config import load_config  # noqa: E402
from train_reid_adapter import extract, identities, load_tracklets  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--train-fraction", type=float, default=0.6)
    ap.add_argument("--overlap-min-sim", type=float, default=0.15)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)
    data = cfg.path("paths.storage") / "training" / "reid"
    tracklets = load_tracklets(data)
    feats = extract(data, tracklets, cfg, args.threads)
    overlaps = [tuple(map(str, p)) for p in (cfg.get("reid.overlapping_cameras") or []) if len(p) == 2]
    labels, splits, links, _uf, _m = identities(tracklets, feats, overlaps, split_gap=4.0, split_sim=0.8,
                                                overlap_min_s=1.0, overlap_min_sim=args.overlap_min_sim)
    t_end = max(t["t1"] for t in tracklets)
    split_t = args.train_fraction * t_end
    ids = {lab: n for n, lab in enumerate(sorted(set(labels.tolist())))}
    rows = []
    for i, t in enumerate(tracklets):
        split = "train" if t["t1"] < split_t else "test" if t["t0"] >= split_t else "skip"
        for c in t["crops"]:
            rows.append({"file": c["file"], "camera": t["camera"], "tracklet": i, "vehicle": ids[int(labels[i])],
                         "t": c.get("t", t["t0"]), "split": split})
    out = Path(args.out) if args.out else cfg.path("paths.storage") / "training" / "colab" / "mcvt_reid_dataset.zip"
    out.parent.mkdir(parents=True, exist_ok=True)
    meta = {"overlapping_cameras": overlaps, "train_fraction": args.train_fraction, "vehicles": len(ids),
            "tracklets": len(tracklets), "occlusion_splits": splits, "overlap_links": links, "crops": rows}
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("labels.json", json.dumps(meta))
        for r in rows:
            z.write(data / r["file"], r["file"])
    n_tr = sum(r["split"] == "train" for r in rows)
    n_te = sum(r["split"] == "test" for r in rows)
    print(f"{len(rows)} crops, {len(ids)} vehicles ({links} overlapping-camera links); train {n_tr} / test {n_te} crops")
    print(f"Saved {out} ({out.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
