"""Build the Google Colab version of the end-to-end ResNet + GRU (RNN) training:

  storage/training/rnn_colab/colab_train_rnn.ipynb     the notebook (open in colab.research.google.com)
  storage/training/rnn_colab/mcvt_rnn_inputs.zip       everything it needs from this project (~100 MB):
      kaggle_train_gru_e2e.py, resnet_ibn.py, mcvt_reid_finetuned.pth, mcvt_reid_dataset.zip
VeRi-776 is downloaded inside Colab from Kaggle (abhyudaya12/veri-vehicle-re-identification-dataset).

Usage: python engine/training/make_colab_rnn.py
"""

import json
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "storage" / "training" / "rnn_colab"
INPUTS = {
    "kaggle_train_gru_e2e.py": ROOT / "engine" / "training" / "kaggle_train_gru_e2e.py",
    "resnet_ibn.py": ROOT / "engine" / "mcvt" / "reid" / "resnet_ibn.py",
    "mcvt_reid_finetuned.pth": ROOT / "engine" / "models" / "mcvt_reid_finetuned.pth",
    "mcvt_reid_dataset.zip": ROOT / "storage" / "training" / "colab" / "mcvt_reid_dataset.zip",
}


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.strip("\n").splitlines(True)}


def code(text):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
            "source": text.strip("\n").splitlines(True)}


CELLS = [
    md("""
# MCVT - train the RNN (ResNet50-IBN + GRU) end to end on a Colab GPU

1. **Runtime -> Change runtime type -> T4 GPU**, then **Runtime -> Run all**.
2. Cell 2 asks for a file: choose **`mcvt_rnn_inputs.zip`** (from `D:\\MCVT-system\\mcvt\\storage\\training\\rnn_colab`).
3. Training takes about 1-1.5 h. Keep this tab open.
4. The last cell downloads **`mcvt_rnn_results.zip`** - put it in
   `D:\\MCVT-system\\mcvt\\storage\\training\\rnn_colab` (or send it back in the chat).
"""),
    code("""
# 1) GPU check
import torch, subprocess
print(subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True).stdout or "no nvidia-smi")
assert torch.cuda.is_available(), "No GPU: Runtime -> Change runtime type -> T4 GPU, then run again"
print("GPU:", torch.cuda.get_device_name(0))
"""),
    code("""
# 2) This project's files: upload mcvt_rnn_inputs.zip (about 100 MB)
import os, zipfile
from google.colab import files
os.makedirs("/content/inputs", exist_ok=True)
if not os.path.exists("/content/inputs/kaggle_train_gru_e2e.py"):
    up = files.upload()
    name = next(n for n in up if n.endswith(".zip"))
    zipfile.ZipFile(name).extractall("/content/inputs")
print(sorted(os.listdir("/content/inputs")))
"""),
    code("""
# 3) VeRi-776 from Kaggle (public dataset; if Kaggle asks for a sign-in, run kagglehub.login() first)
import glob, kagglehub
VERI = kagglehub.dataset_download("abhyudaya12/veri-vehicle-re-identification-dataset")
train_dirs = glob.glob(f"{VERI}/**/image_train", recursive=True)
print("VeRi:", VERI, "| image_train:", train_dirs[:1])
assert train_dirs, "VeRi image_train not found"
"""),
    code("""
# 4) Train + evaluate (prints progress every 200 steps, then the comparison table)
import os
os.environ.update({
    "MCVT_ROOTS": f"{VERI};/content/inputs",          # VeRi first, then this project's files
    "MCVT_WEIGHTS": "/content/inputs/mcvt_reid_finetuned.pth",
    "MCVT_MODEL_PY": "/content/inputs/resnet_ibn.py",
    "MCVT_WORK": "/content/out",
    "MCVT_WORKERS": "2",                                # Colab has 2 CPU cores
})
!python -u /content/inputs/kaggle_train_gru_e2e.py 2>&1 | tee /content/train.log
"""),
    code("""
# 5) Download the results: mcvt_rnn_results.zip (models + metrics + log, about 120 MB)
import zipfile, os
from google.colab import files
with zipfile.ZipFile("/content/mcvt_rnn_results.zip", "w") as z:
    for f in ["mcvt_reid_e2e.pth", "mcvt_reid_gru_e2e.pt", "e2e_metrics.json"]:
        p = f"/content/out/{f}"
        if os.path.exists(p):
            z.write(p, f)
    z.write("/content/train.log", "train.log")
files.download("/content/mcvt_rnn_results.zip")
"""),
]


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    missing = [str(p) for p in INPUTS.values() if not p.exists()]
    if missing:
        sys.exit(f"missing: {missing}")
    nb = {"nbformat": 4, "nbformat_minor": 5, "cells": CELLS,
          "metadata": {"accelerator": "GPU", "colab": {"provenance": []},
                       "kernelspec": {"name": "python3", "display_name": "Python 3"}}}
    (OUT / "colab_train_rnn.ipynb").write_text(json.dumps(nb, indent=1), encoding="utf-8")
    with zipfile.ZipFile(OUT / "mcvt_rnn_inputs.zip", "w", zipfile.ZIP_STORED) as z:
        for name, path in INPUTS.items():
            z.write(path, name)
    for p in sorted(OUT.iterdir()):
        print(f"{p.name}: {p.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
