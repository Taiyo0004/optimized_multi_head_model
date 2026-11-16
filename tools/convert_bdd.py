"""Convert BDD100K (original 2018 label JSON) into the lanedet dataset format, with an optional small subset.

Reads   bdd100k_labels_images_<split>.json   (boxes + lane polylines with solid/dashed)
Writes  <out>/images/{train,val}/*.jpg
        <out>/detection-object/labels/{train,val}/*.txt   YOLO boxes, our 17-class ids
        <out>/seg-lane/labels/{train,val}/*.txt           YOLO polygons: 17 = solid lane, 18 = dashed lane
        <out>/lanedet_data.yaml                            ready for training
        <out>/preview/*.jpg                                a few images with labels drawn, CHECK THESE

Demo (both groups taken from BDD's val split, which A-YOLOM never saw; no overlap between them):
    python tools\\convert_bdd.py --json bdd100k\\labels\\bdd100k_labels_images_val.json ^
        --images bdd100k\\images\\val --out datasets_local\\bdd_demo --train-n 300 --val-n 100

Class mapping (option A: car includes bus and truck):
    person -> pedestrian(9)   car/truck/bus -> car(10)   bike -> bicycle(11)   motor -> two_wheeler(12)
    traffic light red/yellow/green -> light_red(5)/light_amber(6)/light_green(7)
    dropped: rider, train, traffic sign (no speed value), traffic lights with colour "none"
Lanes kept: laneDirection == parallel (along the road) and laneType not crosswalk / road curb.
    Vertical markings are stop lines / crosswalks, and curbs are road edges, not painted lane lines.
    Each kept polyline is drawn THICK px wide (at 1280x720) and stored as a polygon, solid=17 dashed=18.
"""
import argparse
import json
import random
import shutil
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import yaml

DET_MAP = {'person': 9, 'car': 10, 'truck': 10, 'bus': 10, 'bike': 11, 'motor': 12}
LIGHT_MAP = {'red': 5, 'yellow': 6, 'green': 7}
LANE_CLS = {'solid': 17, 'dashed': 18}
SKIP_LANE_TYPES = {'crosswalk', 'road curb'}
DATA_TEMPLATE = Path('ultralytics/datasets/lanedet.yaml')


def bezier(p0, c1, c2, p3, n=16):
    t = np.linspace(0, 1, n)[:, None]
    return (1 - t) ** 3 * p0 + 3 * (1 - t) ** 2 * t * c1 + 3 * (1 - t) * t ** 2 * c2 + t ** 3 * p3


def polyline_points(poly):
    """BDD poly2d -> dense list of points. 'L' = line vertex, 'C' = cubic Bezier control point
    (two control points followed by an end point, as in BDD's own rendering)."""
    v = np.asarray(poly['vertices'], dtype=float)
    types = poly.get('types') or 'L' * len(v)
    pts = [v[0]]
    i = 1
    while i < len(v):
        if types[i] == 'C' and i + 1 < len(v):
            end = v[i + 2] if i + 2 < len(v) else v[i + 1]
            pts.extend(bezier(pts[-1], v[i], v[i + 1], end)[1:])
            i += 3
        else:
            pts.append(v[i])
            i += 1
    return np.asarray(pts)


def lane_polygons(labels, w, h, thick):
    """Return {class_id: [polygon (N,2) in pixels]} for kept lanes, via a drawn mask per class."""
    masks = {c: np.zeros((h, w), np.uint8) for c in LANE_CLS.values()}
    kept = Counter()
    for lb in labels:
        if lb.get('category') != 'lane':
            continue
        a = lb.get('attributes') or {}
        if a.get('laneDirection') != 'parallel' or a.get('laneType') in SKIP_LANE_TYPES:
            continue
        cls = LANE_CLS.get(a.get('laneStyle'))
        if cls is None:
            continue
        for poly in lb.get('poly2d') or []:
            pts = polyline_points(poly)
            if len(pts) < 2:
                continue
            cv2.polylines(masks[cls], [np.round(pts).astype(np.int32)], False, 255, thick, cv2.LINE_8)
            kept[cls] += 1
    out = {}
    for cls, m in masks.items():
        cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        polys = []
        for c in cnts:
            c = cv2.approxPolyDP(c, 1.0, True).reshape(-1, 2)
            if len(c) >= 3 and cv2.contourArea(c) >= 4:
                polys.append(c)
        out[cls] = polys
    return out, kept


def det_boxes(labels):
    boxes = []
    for lb in labels:
        b = lb.get('box2d')
        if not b:
            continue
        cat = lb.get('category')
        if cat == 'traffic light':
            cls = LIGHT_MAP.get((lb.get('attributes') or {}).get('trafficLightColor'))
        else:
            cls = DET_MAP.get(cat)
        if cls is None:
            continue
        boxes.append((cls, b['x1'], b['y1'], b['x2'], b['y2']))
    return boxes


def write_sample(fr, img_src, out, split, thick, stats):
    name = fr['name']
    img = cv2.imread(str(img_src))
    if img is None:
        stats['missing_image'] += 1
        return False
    h, w = img.shape[:2]
    stem = Path(name).stem
    shutil.copy2(img_src, out / 'images' / split / name)

    lines = []
    for cls, x1, y1, x2, y2 in det_boxes(fr['labels']):
        x1, x2 = np.clip([x1, x2], 0, w)
        y1, y2 = np.clip([y1, y2], 0, h)
        if x2 - x1 < 1 or y2 - y1 < 1:
            continue
        lines.append(f'{cls} {(x1 + x2) / 2 / w:.6f} {(y1 + y2) / 2 / h:.6f} {(x2 - x1) / w:.6f} {(y2 - y1) / h:.6f}')
        stats[f'box_{cls}'] += 1
    (out / 'detection-object' / 'labels' / split / f'{stem}.txt').write_text('\n'.join(lines) + ('\n' if lines else ''))

    polys, kept = lane_polygons(fr['labels'], w, h, thick)
    seg = []
    for cls, plist in polys.items():
        for p in plist:
            xy = np.clip(p / [w, h], 0, 1).reshape(-1)
            seg.append(f'{cls} ' + ' '.join(f'{v:.6f}' for v in xy))
            stats[f'lane_poly_{cls}'] += 1
    for cls, n in kept.items():
        stats[f'lane_line_{cls}'] += n
    (out / 'seg-lane' / 'labels' / split / f'{stem}.txt').write_text('\n'.join(seg) + ('\n' if seg else ''))
    return True


def preview(out, split, names, k=6):
    pdir = out / 'preview'
    pdir.mkdir(exist_ok=True)
    colors = {17: (255, 0, 0), 18: (0, 165, 255)}  # BGR: solid blue, dashed orange
    for name in names[:k]:
        stem = Path(name).stem
        img = cv2.imread(str(out / 'images' / split / name))
        h, w = img.shape[:2]
        over = img.copy()
        for ln in (out / 'seg-lane' / 'labels' / split / f'{stem}.txt').read_text().splitlines():
            v = ln.split()
            cls, xy = int(v[0]), np.array(v[1:], float).reshape(-1, 2) * [w, h]
            cv2.fillPoly(over, [xy.astype(np.int32)], colors[cls])
        img = cv2.addWeighted(over, 0.7, img, 0.3, 0)
        for ln in (out / 'detection-object' / 'labels' / split / f'{stem}.txt').read_text().splitlines():
            c, x, y, bw, bh = map(float, ln.split())
            p1 = (int((x - bw / 2) * w), int((y - bh / 2) * h))
            p2 = (int((x + bw / 2) * w), int((y + bh / 2) * h))
            cv2.rectangle(img, p1, p2, (0, 255, 0), 2)
            cv2.putText(img, str(int(c)), (p1[0], max(p1[1] - 3, 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv2.putText(img, 'blue = solid lane, orange = dashed lane, green = boxes (class id)', (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.imwrite(str(pdir / f'{split}_{stem}.jpg'), img)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--json', required=True, help='bdd100k_labels_images_<split>.json')
    ap.add_argument('--images', required=True, help='folder with that split\'s images')
    ap.add_argument('--out', default='datasets_local/bdd_demo')
    ap.add_argument('--train-n', type=int, default=300, help='0 = all remaining images')
    ap.add_argument('--val-n', type=int, default=100)
    ap.add_argument('--thick', type=int, default=8, help='lane line width in pixels at 1280x720')
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()

    out = Path(args.out)
    if out.exists():
        shutil.rmtree(out)
    for split in ('train', 'val'):
        for sub in ('images', 'detection-object/labels', 'seg-lane/labels'):
            (out / sub / split).mkdir(parents=True, exist_ok=True)

    print(f'Loading {args.json} ...')
    frames = json.load(open(args.json, encoding='utf-8'))
    img_dir = Path(args.images)

    def has_lane(fr, style=None):
        for lb in fr.get('labels') or []:
            a = lb.get('attributes') or {}
            if lb.get('category') == 'lane' and a.get('laneDirection') == 'parallel' \
                    and a.get('laneType') not in SKIP_LANE_TYPES and (style is None or a.get('laneStyle') == style):
                return True
        return False

    # find images anywhere under --images (Kaggle downloads often add an extra folder level)
    on_disk = {p.name: p for p in img_dir.rglob('*') if p.suffix.lower() in ('.jpg', '.jpeg', '.png')}
    matched = [f for f in frames if f['name'] in on_disk]
    usable = [f for f in matched if has_lane(f)]
    print(f'{len(on_disk)} images found under {img_dir}; {len(matched)} of the {len(frames)} JSON entries match by name; '
          f'{len(usable)} of those have at least one kept lane')
    if not matched:
        print('\nNO MATCHES. Example names on disk :', [p for p in list(on_disk)[:3]])
        print('                Example names in JSON:', [f['name'] for f in frames[:3]])
        if on_disk:
            print('                Example image path   :', next(iter(on_disk.values())))
        raise SystemExit('Point --images at the folder holding the photos named in the JSON.')
    rng = random.Random(args.seed)
    both = [f for f in usable if has_lane(f, 'solid') and has_lane(f, 'dashed')]
    both_names = {f['name'] for f in both}
    rest = [f for f in usable if f['name'] not in both_names]
    rng.shuffle(both)
    rng.shuffle(rest)
    pool = both + rest  # images with both solid and dashed lanes first
    val = pool[:args.val_n]
    train = pool[args.val_n:args.val_n + args.train_n] if args.train_n else pool[args.val_n:]
    print(f'Selected train {len(train)}, val {len(val)} ({len(both)} images have both solid and dashed)')

    stats = Counter()
    for split, group in (('train', train), ('val', val)):
        ok = 0
        for i, fr in enumerate(group, 1):
            ok += write_sample(fr, on_disk[fr['name']], out, split, args.thick, stats)
            if i % 200 == 0:
                print(f'  {split}: {i}/{len(group)}')
        print(f'  {split}: wrote {ok} images')

    data = yaml.safe_load(open(DATA_TEMPLATE))
    data['path'] = str(out.resolve()).replace('\\', '/')
    data['test'] = 'images/val'
    with open(out / 'lanedet_data.yaml', 'w') as f:
        yaml.safe_dump(data, f, sort_keys=False)

    preview(out, 'val', [f['name'] for f in val])
    names = data['names']
    print('\nBoxes written per class:')
    for c in sorted(int(k.split('_')[1]) for k in stats if k.startswith('box_')):
        print(f'  {c:>2} {names[c]:<14} {stats[f"box_{c}"]}')
    print('Lanes: ' + ', '.join(f'{names[c]}: {stats[f"lane_line_{c}"]} lines -> {stats[f"lane_poly_{c}"]} polygons'
                                 for c in (17, 18)))
    if stats['missing_image']:
        print(f'WARNING: {stats["missing_image"]} images could not be read')
    print(f'\nDataset: {out}\nData yaml: {out / "lanedet_data.yaml"}\nCheck the label previews in {out / "preview"}')


if __name__ == '__main__':
    main()