"""Step 2: run the authors' pretrained A-YOLOM on a few images to confirm it works end to end.

Usage (from the repo root):
    python step2_predict.py --weights weights\\v4s.pt --source test_images
"""
import argparse
import functools
import time
from pathlib import Path

import torch

# PyTorch 2.6+ defaults torch.load(weights_only=True), which rejects old Ultralytics
# checkpoints (they pickle full model objects). Allow full loading for this run only.
# Only do this for weight files from a source you trust (here: the A-YOLOM authors).
_orig_load = torch.load
torch.load = functools.partial(_orig_load, weights_only=False)

from ultralytics import YOLO  # noqa: E402  (import after the patch)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True, help="path to A-YOLOM .pt file")
    ap.add_argument("--source", default="test_images", help="image file or folder")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--name", default="step2_test")
    args = ap.parse_args()

    if not Path(args.weights).exists():
        raise SystemExit(f"Weights not found: {args.weights}")
    if not Path(args.source).exists():
        raise SystemExit(f"Source not found: {args.source}")

    device = 0 if torch.cuda.is_available() else "cpu"
    print(f"Loading {args.weights} on {'GPU ' + torch.cuda.get_device_name(0) if device == 0 else 'CPU'}")

    model = YOLO(args.weights, task="multi")

    t0 = time.time()
    results = model.predict(
        source=args.source,
        imgsz=(args.imgsz, args.imgsz),
        device=[device] if device == 0 else device,
        conf=args.conf,
        iou=0.45,
        half=(device == 0),   # FP16 on GPU
        batch=1,              # keep memory low on a 4 GB card
        save=True,
        name=args.name,
        show_labels=False,
    )
    dt = time.time() - t0

    print(f"\nDone in {dt:.1f}s")
    print(f"Result object type: {type(results)}  (multi-task output: one entry per task head)")
    try:
        print(f"Number of entries: {len(results)}")
    except TypeError:
        pass
    print(f"Look for saved images under: {Path('runs') / 'multi' / args.name}  (also check the newest folder in runs\\)")


if __name__ == "__main__":  # required on Windows for dataloader worker processes
    main()
