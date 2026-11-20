"""Long lanedet training on the laptop, with crash-safe checkpoints and resume.

Run from the repo root:
    python tools\\train_laptop.py                      start a new run
    python tools\\train_laptop.py --resume             continue after a crash, Ctrl+C, reboot or power cut

What gets saved (runs\\multi\\<name>\\weights):
    last.pt   every --ckpt-minutes during an epoch, at the end of every epoch, and when you press Ctrl+C
    best.pt   after each epoch's validation, when it is the best so far
A resumed run continues from the exact batch where last.pt was written, with the same optimizer,
learning-rate schedule, EMA and image order, so at most --ckpt-minutes of work is lost.
"""
import argparse
import sys
import time
from pathlib import Path

import torch

from ultralytics import YOLO


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default='datasets_local/bdd_demo/lanedet_data.yaml')
    ap.add_argument('--weights', default='weights/lanedet_s_init.pt', help='starting weights for a new run')
    ap.add_argument('--name', default='bdd_laptop', help='run folder under runs\\multi')
    ap.add_argument('--epochs', type=int, default=14)
    ap.add_argument('--batch', type=int, default=4, help='4 fits in 4 GB at 640; use 2 if CUDA runs out of memory')
    ap.add_argument('--imgsz', type=int, default=640)
    ap.add_argument('--workers', type=int, default=0, help='try 2 if the GPU sits idle waiting for images')
    ap.add_argument('--ckpt-minutes', type=float, default=10, help='mid-epoch save interval (0 = epoch end only)')
    ap.add_argument('--resume', action='store_true', help='continue runs\\multi\\<name>\\weights\\last.pt')
    args = ap.parse_args()

    run = Path('runs/multi') / args.name
    last = run / 'weights' / 'last.pt'
    t0 = time.time()

    if args.resume:
        if not last.exists():
            sys.exit(f'Nothing to resume: {last} not found. Start a new run without --resume.')
        ck = torch.load(last, map_location='cpu', weights_only=False)
        if ck.get('optimizer') is None:
            sys.exit(f'{last} is from a finished run. Start a new run (use a new --name) or raise --epochs.')
        it = ck.get('iter', -1)
        where = f'epoch {ck["epoch"] + 1}, after batch {it + 1}' if it is not None and it >= 0 \
            else f'start of epoch {ck["epoch"] + 2}'
        print(f'Resuming {last} from {where} (saved {ck.get("date", "?")[:19]})')
        del ck
        # settings (batch, epochs, data, ...) come from the checkpoint, so the image order lines up
        YOLO(str(last), task='multi').train(resume=True)
    else:
        for p in (args.data, args.weights):
            if not Path(p).exists():
                sys.exit(f'Not found: {p}  (run convert_bdd.py / step7_check.py first)')
        if last.exists():
            sys.exit(f'{run} already has a checkpoint. Add --resume to continue it, or pick a new --name.')
        dev = 0 if torch.cuda.is_available() else 'cpu'
        YOLO(args.weights, task='multi').train(
            data=args.data, epochs=args.epochs, imgsz=args.imgsz, batch=args.batch, workers=args.workers,
            amp=dev == 0, device=dev, project='runs/multi', name=args.name, exist_ok=True,
            seed=0, patience=100, ckpt_minutes=args.ckpt_minutes,
            warmup_bias_lr=0.01,  # gentler warmup for the warm-started heads (avoids the epoch-1 lane IoU dip)
            verbose=True)

    print(f'\nThis session: {(time.time() - t0) / 3600:.2f} h.  Results: {run}  '
          f'(results_detection-object.csv, results_seg-lane.csv, weights\\best.pt)')


if __name__ == '__main__':
    main()
