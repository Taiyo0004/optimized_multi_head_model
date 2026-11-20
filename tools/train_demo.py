"""~20-minute laptop demo: fine-tune the lanedet model on a small real BDD100K subset.

Run from the repo root, after convert_bdd.py:
    python tools\\train_demo.py
    python tools\\train_demo.py --epochs 10 --batch 2      (if it is too slow, or runs out of GPU memory)

Starts from weights\\lanedet_s_init.pt (A-YOLOM warm start), trains detection + solid/dashed lanes,
then prints the key numbers and saves predictions on validation images to runs\\multi\\demo_predict.
These are demo numbers from 300 images - they show the pipeline works on real data, not final accuracy.
"""
import argparse
import csv
import sys
from pathlib import Path

import torch

from ultralytics import YOLO


def last_row(path):
    with open(path) as f:
        rows = list(csv.reader(f))
    head = [h.strip() for h in rows[0]]
    return [dict(zip(head, map(float, r))) for r in rows[1:] if len(r) == len(head)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default='datasets_local/bdd_demo/lanedet_data.yaml')
    ap.add_argument('--weights', default='weights/lanedet_s_init.pt')
    ap.add_argument('--epochs', type=int, default=14)
    ap.add_argument('--batch', type=int, default=4)
    ap.add_argument('--imgsz', type=int, default=640)
    ap.add_argument('--name', default='demo')
    args = ap.parse_args()

    for p in (args.data, args.weights):
        if not Path(p).exists():
            sys.exit(f'Not found: {p}  (run convert_bdd.py / step7_check.py first)')
    dev = 0 if torch.cuda.is_available() else 'cpu'
    run = Path('runs/multi') / args.name

    YOLO(args.weights, task='multi').train(
        data=args.data, epochs=args.epochs, imgsz=args.imgsz, batch=args.batch, workers=0,
        amp=dev == 0, device=dev, project='runs/multi', name=args.name, exist_ok=True,
        seed=0, patience=1000, verbose=True)

    print('\n' + '=' * 80 + '\nRESULTS (last epoch, on the 100 validation images)\n' + '=' * 80)
    det = last_row(run / 'results_detection-object.csv')
    lane = last_row(run / 'results_seg-lane.csv')
    d0, d1, l0, l1 = det[0], det[-1], lane[0], lane[-1]
    for k in ('metrics/mAP50(B)', 'metrics/mAP50-95(B)'):
        if k in d1:
            print(f'  detection {k:<24} epoch 1: {d0[k]:.3f}   last: {d1[k]:.3f}')
    for k in ('metrics/IoU', 'metrics/IoU_lane_solid', 'metrics/IoU_lane_dashed', 'metrics/mIoU'):
        if k in l1:
            print(f'  lane      {k:<24} epoch 1: {l0[k]:.3f}   last: {l1[k]:.3f}')
    print('  (metrics/IoU = any lane line vs background, comparable to A-YOLOM\'s lane IoU)')

    best = run / 'weights' / 'best.pt'
    imgs = sorted((Path(args.data).parent / 'images' / 'val').glob('*.jpg'))[:8]
    out = Path('runs/multi/demo_predict')
    for img in imgs:
        YOLO(str(best), task='multi').predict(source=str(img), imgsz=args.imgsz, conf=0.25, save=True,
                                              project='runs/multi', name='demo_predict', exist_ok=True,
                                              device=dev, verbose=False)
    print(f'\nPredictions on {len(imgs)} validation images: {out}  (blue = solid, orange = dashed)')
    print(f'Per-class detection results and plots: {run}')


if __name__ == '__main__':
    main()
