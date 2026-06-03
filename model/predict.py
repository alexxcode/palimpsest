"""
Single-pair inference CLI.

Usage (inside the Docker container):
  python model/predict.py \\
      --before data/datasets/test/A/test_1.png \\
      --after  data/datasets/test/B/test_1.png \\
      --output /tmp/mask_test_1.png

Saves a grayscale PNG where 255 = changed, 0 = unchanged.
Prints the detected change percentage and inference time.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

import numpy as np
from PIL import Image

from app.services.detector import ChangeDetector
from model.dataset import LEVIRDataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Palimpsest single-pair change detection")
    parser.add_argument("--before",     required=True, help="Before image path (PNG or GeoTIFF)")
    parser.add_argument("--after",      required=True, help="After image path (PNG or GeoTIFF)")
    parser.add_argument("--output",     required=True, help="Output mask path (.png)")
    parser.add_argument("--checkpoint", default="checkpoints/changeformer_levir.pth",
                        help="Model checkpoint (default: checkpoints/changeformer_levir.pth)")
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # Load images
    # ------------------------------------------------------------------
    before = LEVIRDataset.load_image(Path(args.before))   # (H, W, 4) float32 [0,1]
    after  = LEVIRDataset.load_image(Path(args.after))

    if before.shape != after.shape:
        raise ValueError(
            f"Image size mismatch: before={before.shape[:2]}  after={after.shape[:2]}"
        )

    print(f"Image size  : {before.shape[:2]}")
    print(f"Checkpoint  : {args.checkpoint}")

    # ------------------------------------------------------------------
    # Load model + run inference
    # ------------------------------------------------------------------
    detector = ChangeDetector()

    t0   = time.time()
    mask = detector.predict_large(before, after)   # (H, W) uint8  0|1
    ms   = (time.time() - t0) * 1000

    changed_pct = mask.mean() * 100
    print(f"Changed     : {changed_pct:.2f}%  ({int(mask.sum()):,} pixels)")
    print(f"Inference   : {ms:.0f} ms")

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(mask * 255).save(out_path)
    print(f"Mask saved  : {out_path}")


if __name__ == "__main__":
    main()
