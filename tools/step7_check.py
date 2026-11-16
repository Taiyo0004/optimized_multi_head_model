"""Step 7 check: three-class lane head (background, solid, dashed) end to end.

Run from the repo root (the folder with setup.py), after unzipping step7_patch.zip:
    python step7_check.py --weights weights\\v4s.pt

What it does
  1. Builds the model: lane head now outputs 3 channels.
  2. Transfers A-YOLOM(s) weights. The lane head's last layer goes 2 -> 3 channels, so it is
     warm-started: background <- background, solid <- line, dashed <- line. The model therefore
     starts out finding lane lines exactly like A-YOLOM and only has to learn solid vs dashed.
     Proof: "any line" from the new model must equal A-YOLOM's lane mask.
     Saves weights\\lanedet_s_init.pt (overwrites the Step 6 one) - the starting point for training.
  3. Unit-tests the new multi-class lane loss (cross-entropy + Dice).
  4. Writes a small synthetic dataset in the real on-disk format (solid and dashed lines, a "car"),
     loads it through the real dataset code, and checks the masks hold 0/1/2. Saves a preview.
  5. Runs a short real training (trainer, loss, validator, metrics, CSV) on that synthetic data.
  6. Runs the predictor on a synthetic validation image.
Everything printed is also saved to step7_check.txt.
"""
import argparse
import csv
import datetime
import shutil
import sys
from copy import deepcopy
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml

from ultralytics.nn.modules import Segment
from ultralytics.nn.tasks import MultiModel

MODEL_YAML = Path('ultralytics/models/v8/yolov8-lanedet-s.yaml')
DATA_YAML = Path('ultralytics/datasets/lanedet.yaml')
OLD_DRIVABLE = set(range(34, 46)) | {47}
OLD_TO_NEW = {46: 34, 48: 35}
LANE_HEAD = 'model.35.cv3.'
results, warnings = [], []


class Tee:
    """Copy a stream (stdout or stderr) into the log file, so errors and warnings end up in it too."""
    f = None

    def __init__(self, path, stream):
        if Tee.f is None:
            Tee.f = open(path, 'w', encoding='utf-8')
        self.stream = stream

    def write(self, s):
        self.stream.write(s)
        Tee.f.write(s)
        Tee.f.flush()

    def flush(self):
        self.stream.flush()
        Tee.f.flush()

    def isatty(self):
        return False


def check(name, passed, detail=''):
    results.append(bool(passed))
    print(f"[{'OK' if passed else 'FAIL'}] {name}{' - ' + str(detail) if detail != '' else ''}")


def warn(name, detail=''):
    warnings.append(name)
    print(f"[WARN] {name}{' - ' + str(detail) if detail != '' else ''}")


def header(t):
    print('\n' + '=' * 80 + f'\n{t}\n' + '=' * 80)


def remap_old_state(old_sd):
    out = {}
    for k, v in old_sd.items():
        parts = k.split('.')
        if parts[0] != 'model' or not parts[1].isdigit():
            continue
        i = int(parts[1])
        if i in OLD_DRIVABLE:
            continue
        parts[1] = str(OLD_TO_NEW.get(i, i))
        out['.'.join(parts)] = v
    return out


# ------------------------------------------------------------------------------------------ synthetic data
def band(p0, p1, thick, w, h):
    """Polygon (4 points, normalised) covering a thick line from p0 to p1 (pixels)."""
    p0, p1 = np.array(p0, float), np.array(p1, float)
    d = p1 - p0
    n = np.array([-d[1], d[0]]) / (np.linalg.norm(d) + 1e-9) * thick / 2
    pts = np.array([p0 + n, p1 + n, p1 - n, p0 - n])
    pts[:, 0] = np.clip(pts[:, 0] / w, 0, 1)
    pts[:, 1] = np.clip(pts[:, 1] / h, 0, 1)
    return pts


def make_image(rng, w=512, h=320):
    img = np.zeros((h, w, 3), np.uint8)
    img[: h // 2] = (200, 170, 120)                                  # sky (BGR)
    img[h // 2:] = (90, 90, 90)                                      # road
    img = np.clip(img.astype(int) + rng.integers(-12, 12, img.shape), 0, 255).astype(np.uint8)
    hz = h // 2 + int(rng.integers(-10, 10))
    left = ((w * rng.uniform(0.05, 0.2), h), (w * rng.uniform(0.42, 0.47), hz))
    right = ((w * rng.uniform(0.8, 0.95), h), (w * rng.uniform(0.53, 0.58), hz))
    solid_side = rng.integers(2)  # randomise which side is solid so position is no cue
    lanes = []
    for side, (p0, p1) in enumerate((left, right)):
        style = 'solid' if side == solid_side else 'dashed'
        if style == 'solid':
            cv2.line(img, tuple(map(int, p0)), tuple(map(int, p1)), (235, 235, 235), 6)
        else:
            for t in np.arange(0, 1, 0.16):  # painted dashes
                a = np.array(p0) + (np.array(p1) - np.array(p0)) * t
                b = np.array(p0) + (np.array(p1) - np.array(p0)) * (t + 0.08)
                cv2.line(img, tuple(map(int, a)), tuple(map(int, b)), (235, 235, 235), 6)
        lanes.append((17 if style == 'solid' else 18, band(p0, p1, 8, w, h)))  # label covers the whole line
    cw, ch = int(rng.integers(40, 80)), int(rng.integers(25, 45))
    cx, cy = int(rng.integers(cw, w - cw)), int(hz + rng.integers(10, 50))
    cv2.rectangle(img, (cx - cw // 2, cy - ch // 2), (cx + cw // 2, cy + ch // 2), (30, 30, 160), -1)
    box = (10, cx / w, cy / h, cw / w, ch / h)  # class 10 = car
    return img, box, lanes


def write_dataset(root, n_train, n_val, seed=0):
    rng = np.random.default_rng(seed)
    if root.exists():
        shutil.rmtree(root)
    for split, n in (('train', n_train), ('val', n_val)):
        for sub in ('images', 'detection-object/labels', 'seg-lane/labels'):
            (root / sub / split).mkdir(parents=True, exist_ok=True)
        for i in range(n):
            img, box, lanes = make_image(rng)
            stem = f'{split}_{i:03d}'
            cv2.imwrite(str(root / 'images' / split / f'{stem}.jpg'), img)
            (root / 'detection-object/labels' / split / f'{stem}.txt').write_text(
                f'{box[0]} {box[1]:.6f} {box[2]:.6f} {box[3]:.6f} {box[4]:.6f}\n')
            (root / 'seg-lane/labels' / split / f'{stem}.txt').write_text(''.join(
                f'{c} ' + ' '.join(f'{v:.6f}' for v in pts.reshape(-1)) + '\n' for c, pts in lanes))
    data = yaml.safe_load(open(DATA_YAML))
    data['path'] = str(root.resolve()).replace('\\', '/')
    data['test'] = 'images/val'
    out = root / 'synthetic.yaml'
    with open(out, 'w') as f:
        yaml.safe_dump(data, f, sort_keys=False)
    return out


def read_csv(path):
    with open(path) as f:
        rows = list(csv.reader(f))
    head = [h.strip() for h in rows[0]]
    return [{k: float(v) for k, v in zip(head, r)} for r in rows[1:] if len(r) == len(head)]


# ------------------------------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--weights', default='weights/v4s.pt')
    ap.add_argument('--out', default='weights/lanedet_s_init.pt')
    ap.add_argument('--epochs', type=int, default=25)
    ap.add_argument('--n-train', type=int, default=48)
    ap.add_argument('--skip-train', action='store_true')
    args = ap.parse_args()

    sys.stdout = Tee('step7_check.txt', sys.__stdout__)
    sys.stderr = Tee('step7_check.txt', sys.__stderr__)
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'torch {torch.__version__}, device {dev}')
    torch.manual_seed(0)

    # ---------------------------------------------------------------- 1. build
    header('1. BUILD')
    model = MultiModel(str(MODEL_YAML), verbose=False)
    lane_head = model.model[35]
    check('lane head is a Segment at layer 35', isinstance(lane_head, Segment))
    check('lane head outputs 3 channels (background, solid, dashed)',
          lane_head.cv3.conv.out_channels == 3, lane_head.cv3.conv.out_channels)
    check('detection still 17 classes', model.model[34].nc == 17, model.model[34].nc)
    print(f'     parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.3f} M')
    model.to(dev).eval()
    with torch.no_grad():
        out = model(torch.zeros(1, 3, 384, 640, device=dev))
    check('lane output 1x3x384x640', tuple(out[1].shape) == (1, 3, 384, 640), tuple(out[1].shape))

    # ---------------------------------------------------------------- 2. transfer + warm start
    header('2. WEIGHT TRANSFER + LANE HEAD WARM START')
    if not Path(args.weights).exists():
        check('pretrained weights found', False, args.weights)
        return
    ckpt = torch.load(args.weights, map_location='cpu', weights_only=False)  # trusted (A-YOLOM authors)
    old_model = (ckpt.get('ema') or ckpt['model']).float()
    mapped = remap_old_state(old_model.state_dict())
    new_sd = model.state_dict()
    ok = {k: v for k, v in mapped.items() if k in new_sd and v.shape == new_sd[k].shape}
    mismatch = sorted(k for k, v in mapped.items() if k in new_sd and v.shape != new_sd[k].shape)
    warm = []
    for k in mismatch:
        if k.startswith(LANE_HEAD):
            o, n = mapped[k], new_sd[k].clone().cpu()
            n[0], n[1], n[2] = o[0], o[1], o[1]  # bg <- bg, solid <- line, dashed <- line
            if k.endswith('conv.weight'):
                # Break the solid/dashed symmetry: with identical channels the gradient that separates
                # them cancels exactly and training gets stuck (found in step7_diag2/3).
                g = torch.Generator().manual_seed(0)
                s = o[1].std()
                n[1] += torch.randn(n[1].shape, generator=g) * 0.3 * s
                n[2] += torch.randn(n[2].shape, generator=g) * 0.3 * s
            ok[k] = n
            warm.append(k)
    model.load_state_dict(ok, strict=False)
    det_new = [k for k in mismatch if k.startswith('model.34.')]
    for k in mismatch:
        print(f"     {k}: {tuple(mapped[k].shape)} -> {tuple(new_sd[k].shape)}"
              f"{'   (warm-started)' if k in warm else '   (new)'}")
    check('only expected layers differ: 6 detection class tensors + 5 lane-head tensors',
          len(det_new) == 6 and len(warm) == 5 and len(mismatch) == 11, f'{len(mismatch)} differ')
    check('all weights transferred or warm-started', len(ok) == len(new_sd) - len(det_new),
          f'{len(ok)}/{len(new_sd)}')

    old_model.to(dev).eval()
    model.eval()
    with torch.no_grad():
        x = torch.rand(2, 3, 384, 640, device=dev)
        old_mask = old_model(x)[2].argmax(1) == 1
        new_out = model(x)[1]
    new_mask = new_out.argmax(1) > 0
    diff = (old_mask != new_mask).float().mean().item()
    # small differences are expected now that solid/dashed get a little noise to break their symmetry
    check('"any line" from new head ~= A-YOLOM lane mask', diff < 0.01, f'{diff * 100:.4f}% pixels differ')
    d_ch = (lane_head.cv3.conv.weight[1] - lane_head.cv3.conv.weight[2]).abs().max().item()
    check('solid and dashed channels are not identical (symmetry broken)', d_ch > 1e-3, f'max diff {d_ch:.3f}')
    old_model.cpu()
    del old_model

    names = yaml.safe_load(open(DATA_YAML))['names']
    save_model = deepcopy(model).cpu().half()
    save_model.names = {int(k): v for k, v in names.items()}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save({'model': save_model, 'ema': None, 'optimizer': None, 'best_fitness': None, 'epoch': -1,
                'train_args': {'task': 'multi', 'model': str(MODEL_YAML)},
                'date': datetime.datetime.now().isoformat(),
                'note': 'A-YOLOM(s) -> lanedet: drivable removed, 17 det classes, 3-class lane head (warm-started)'},
               args.out)
    check('starting checkpoint saved', Path(args.out).exists(), args.out)

    # ---------------------------------------------------------------- 3. loss unit test
    header('3. MULTI-CLASS LANE LOSS')
    from ultralytics.yolo.cfg import get_cfg
    from ultralytics.yolo.v8.DecSeg.train import SegLoss
    model.train()
    model.args = get_cfg()
    print(f'     gains: seg_ce={model.args.seg_ce} seg_dice={model.args.seg_dice} '
          f'seg_line_w={model.args.seg_line_w} lane_lambda={model.args.lane_lambda} fliplr={model.args.fliplr}')
    check('flips disabled by default (fliplr = 0)', model.args.fliplr == 0.0, model.args.fliplr)
    seg_loss = SegLoss(model, overlap=True, count=-1, task_name='seg-lane', map={'17': '0', '18': '1'})
    h, w = 256, 384
    masks = torch.zeros(2, h, w)
    masks[:, h // 2:, 100:106] = 1  # solid
    masks[:, h // 2:, 280:286] = 2  # dashed
    preds = model(torch.rand(2, 3, h, w, device=dev))[1]
    loss, items = seg_loss(preds, {'masks': masks, 'im_file': ['a', 'b']})
    print(f'     dice, ce = {[round(v, 4) for v in items.tolist()]}')
    check('lane loss finite and has 2 parts (dice, ce)', torch.isfinite(loss).item() and len(items) == 2)
    model.zero_grad()
    loss.backward()
    g = lane_head.cv3.conv.weight.grad
    check('gradient reaches all 3 lane-head channels', g is not None and bool((g.abs().sum((1, 2, 3)) > 0).all()))
    perfect = torch.full((1, 3, 8, 8), -20.0, device=dev)
    tgt = torch.zeros(1, 8, 8)
    tgt[:, :, 2] = 1
    tgt[:, :, 5] = 2
    for c in range(3):
        perfect[0, c][tgt[0].to(dev) == c] = 20.0
    lp, ip = seg_loss(perfect, {'masks': tgt, 'im_file': ['p']})
    check('a perfect prediction gives ~0 loss', lp.item() < 0.05, f'{lp.item():.4f}')

    # ---------------------------------------------------------------- 4. dataset pipeline
    header('4. DATA PIPELINE (synthetic dataset in the real format)')
    root = Path('step7_synthetic')
    data_yaml = write_dataset(root, args.n_train, max(8, args.n_train // 4))
    print(f'     wrote {data_yaml}')
    from ultralytics.yolo.data import build_yolo_dataset
    from ultralytics.yolo.data.utils import check_det_dataset
    print('     checking dataset yaml ...')
    data = check_det_dataset(str(data_yaml))
    print(f"     train: {data['train']}\n     val:   {data['val']}")
    for mode in ('val', 'train'):
        print(f'     building {mode} dataset ...')
        cfg = get_cfg(overrides={'task': 'multi', 'imgsz': 320})
        ds = build_yolo_dataset(cfg, data[mode], 4, data, mode=mode, stride=32)
        print(f'     {mode}: {len(ds)} images, loading one sample ...')
        item = ds[0]
        lane = item[1]
        m = lane['masks']
        vals = sorted(int(v) for v in torch.unique(m))
        print(f'     {mode}: image {tuple(lane["img"].shape)}, lane mask {tuple(m.shape)}, values {vals}, '
              f'lane classes in labels {sorted(set(int(c) for c in lane["cls"].reshape(-1).tolist()))}')
        if mode == 'val':
            check('val: lane mask is class-valued with solid (1) and dashed (2)', vals == [0, 1, 2], vals)
        else:  # mosaic + random crop can occasionally cut one class out of a sample
            check('train (augmented): lane mask values within {0, 1, 2}', set(vals) <= {0, 1, 2} and len(vals) > 1,
                  vals)
        check(f'{mode}: lane classes remapped 17/18 -> 0/1',
              set(int(c) for c in lane['cls'].reshape(-1).tolist()) <= {0, 1})
        det_cls = sorted(set(int(c) for c in item[0]['cls'].reshape(-1).tolist()))
        check(f'{mode}: detection class untouched (10 = car)', det_cls in ([10], []), det_cls)
        im = lane['img'].permute(1, 2, 0).numpy()[:, :, ::-1].copy()  # RGB->BGR
        mk = m[0].numpy()
        im[mk == 1] = (0.5 * im[mk == 1] + 0.5 * np.array([255, 0, 0])).astype(np.uint8)    # solid: blue
        im[mk == 2] = (0.5 * im[mk == 2] + 0.5 * np.array([0, 165, 255])).astype(np.uint8)  # dashed: orange
        cv2.imwrite(f'step7_mask_preview_{mode}.jpg', im)
    print('     previews: step7_mask_preview_val.jpg, step7_mask_preview_train.jpg (blue = solid, orange = dashed)')

    # ---------------------------------------------------------------- 5. short real training
    if args.skip_train:
        warn('training skipped (--skip-train)')
    else:
        header(f'5. SHORT TRAINING RUN ({args.epochs} epochs on synthetic data)')
        from ultralytics import YOLO
        run = Path('runs/multi/step7_smoke')
        if run.exists():
            shutil.rmtree(run)
        YOLO(args.out, task='multi').train(
            data=str(data_yaml), epochs=args.epochs, imgsz=320, batch=4, workers=0, amp=False,
            device=0 if dev.type == 'cuda' else 'cpu', project='runs/multi', name='step7_smoke', exist_ok=True,
            warmup_epochs=0, patience=1000, seed=0, verbose=True)
        lane_csv, det_csv = run / 'results_seg-lane.csv', run / 'results_detection-object.csv'
        check('per-task CSVs written', lane_csv.exists() and det_csv.exists(),
              f'{lane_csv.name}, {det_csv.name}')
        if lane_csv.exists():
            rows = read_csv(lane_csv)
            keys = list(rows[0].keys()) if rows else []
            print(f'     lane CSV columns: {keys}')
            need = ['metrics/IoU', 'metrics/mIoU', 'metrics/IoU_lane_solid', 'metrics/IoU_lane_dashed']
            check('lane CSV has merged and per-class IoU', all(k in keys for k in need))
            if rows and all(k in keys for k in need):
                f, l = rows[0], rows[-1]
                for k in need:
                    print(f'     {k:<26} epoch 1: {f[k]:.3f}   last: {l[k]:.3f}')
                if l['metrics/IoU_lane_dashed'] > f['metrics/IoU_lane_dashed'] and \
                        l['metrics/IoU_lane_solid'] > f['metrics/IoU_lane_solid']:
                    check('solid and dashed IoU both improved during training', True)
                else:
                    warn('solid/dashed IoU did not both improve - try --epochs 50; send me the log')
        best = run / 'weights' / 'best.pt'
        check('best.pt saved', best.exists(), best)

        # ------------------------------------------------------------ 6. predictor
        header('6. PREDICTOR')
        if best.exists():
            src = sorted((root / 'images' / 'val').glob('*.jpg'))[0]
            res = YOLO(str(best), task='multi').predict(source=str(src), imgsz=320, conf=0.25, save=True,
                                                         project='runs/multi', name='step7_predict', exist_ok=True,
                                                         device=0 if dev.type == 'cuda' else 'cpu')
            check('predictor runs with 2 outputs', len(res) == 2, len(res))
            lane = res[1]
            vals = sorted(int(v) for v in torch.unique(lane))
            print(f'     predicted lane classes on {src.name}: {vals}  (0 bg, 1 solid, 2 dashed)')
            print('     image saved under runs\\multi\\step7_predict (blue = solid, orange = dashed)')

    print('\n' + (f'ALL CHECKS PASSED{" (with warnings)" if warnings else ""} - Step 7 code is ready.'
                  if all(results) else f'{results.count(False)} CHECK(S) FAILED - send me step7_check.txt'))


if __name__ == '__main__':
    try:
        main()
    except BaseException:
        import traceback
        print('\n' + '!' * 80 + '\nCRASHED - full error below. Send me step7_check.txt\n' + '!' * 80)
        traceback.print_exc(file=sys.stdout)
        raise
