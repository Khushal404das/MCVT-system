"""Start the MCVT AI engine.

Usage:  python run_engine.py [--config ../config.yaml]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from mcvt.config import load_config  # noqa: E402
from mcvt.pipeline import Pipeline  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="MCVT multi-camera vehicle tracking engine")
    parser.add_argument("--config", default=None, help="path to config.yaml (default: project root)")
    args = parser.parse_args()
    cfg = load_config(args.config)
    Pipeline(cfg).run()


if __name__ == "__main__":
    main()
