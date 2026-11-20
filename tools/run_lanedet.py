"""Run the trained lanedet model on an image, a folder of images, a video file or a live camera.

Run from the repo root (conda activate ayolom):
    python tools\\run_lanedet.py --source test.jpg
    python tools\\run_lanedet.py --source test_images\\
    python tools\\run_lanedet.py --source drive.mp4
    python tools\\run_lanedet.py --source 0                  (camera 0 = the first webcam / USB camera)
    python tools\\run_lanedet.py --source 0 --save           (also record the camera view to a video)

Options:
    --weights   model to use (default: best.pt of the bdd_laptop run)
    --json      also write lane coordinates and boxes, one JSON line per image/frame
    --show / --no-show   live window (default: on for camera and video, off for images); press q to stop
    --conf      detection confidence threshold (default 0.25)

Output (runs\\lanedet_out\\<name>): drawn images / videos, and with --json a <name>.jsonl file.
Drawing: boxes with class names; lane pixels tinted blue (solid) / orange (dashed); the extracted lane
lines drawn on top as thin lines with dots, so you can see the coordinates the JSON contains.
"""
import argparse
import json
import platform
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

LANE_NAMES = {1: 'solid', 2: 'dashed'}
LANE_BGR = {1: (255, 0, 0), 2: (0, 165, 255)}  # blue, orange
IMG_EXT = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
VID_EXT = {'.mp4', '.avi', '.mov', '.mkv', '.m4v', '.wmv'}


# ----------------------------------------------------------------------------------------------------
# Lane mask -> lane lines (coordinates)
# ----------------------------------------------------------------------------------------------------
def extract_lanes(mask, step=10, min_points=4, min_run=2, max_jump=None, max_gap=None):
    """Turn the lane class mask (H, W; 0 = background, 1 = solid, 2 = dashed) into lane lines.

    Scans rows from the bottom of the image upwards every `step` px. In each row, every horizontal
    run of lane pixels gives one point (its centre). Points are chained into lanes by nearest x, so
    two lanes that come close near the horizon stay separate. Short pieces are dropped.

    Returns [{'type': 'solid'|'dashed', 'points': [[x, y], ...] (bottom to top), 'fit': [a, b, c]}],
    where fit gives x = a*y^2 + b*y + c in image pixels (None if the lane is too short to fit).
    """
    h, w = mask.shape
    max_jump = max_jump or max(20, w // 30)  # how far x may move between scanned rows
    max_gap = max_gap or max(60, h // 8)  # how many px of rows a lane may skip (e.g. a dash gap)
    lanes = []
    for value, name in LANE_NAMES.items():
        m = mask == value
        if m.sum() < min_points * min_run:
            continue
        open_tracks, done = [], []
        for y in range(h - 1, -1, -step):
            row = m[y]
            if not row.any():
                xs = []
            else:
                d = np.diff(np.concatenate(([0], row.astype(np.int8), [0])))
                starts, ends = np.flatnonzero(d == 1), np.flatnonzero(d == -1)
                xs = [(s + e - 1) / 2 for s, e in zip(starts, ends) if e - s >= min_run]
            # match points to open tracks, nearest first
            pairs = sorted(((abs(x - t['pts'][-1][0]), i, j) for i, x in enumerate(xs)
                            for j, t in enumerate(open_tracks)), key=lambda p: p[0])
            used_x, used_t = set(), set()
            for dist, i, j in pairs:
                if dist > max_jump * (1 + (open_tracks[j]['pts'][-1][1] - y) / step / 2) or i in used_x or j in used_t:
                    continue
                open_tracks[j]['pts'].append((xs[i], y))
                used_x.add(i)
                used_t.add(j)
            for i, x in enumerate(xs):
                if i not in used_x:
                    open_tracks.append({'pts': [(x, y)]})
            still = []
            for t in open_tracks:
                (done if t['pts'][-1][1] - y > max_gap else still).append(t)
            open_tracks = still
        for t in done + open_tracks:
            if len(t['pts']) < min_points:
                continue
            pts = np.array(t['pts'])
            fit = np.polyfit(pts[:, 1], pts[:, 0], 2).tolist() if len(pts) >= 5 else None
            lanes.append({'type': name, 'points': [[round(float(x), 1), int(y)] for x, y in pts], 'fit': fit})
    lanes.sort(key=lambda l: l['points'][0][0])  # left to right at the bottom
    return lanes


# ----------------------------------------------------------------------------------------------------
# Model
# ----------------------------------------------------------------------------------------------------
class LaneDet:
    def __init__(self, weights, imgsz=640, conf=0.25, device=None):
        from ultralytics import YOLO
        self.gpu = torch.cuda.is_available() if device is None else device != 'cpu'
        self.device = 0 if self.gpu else 'cpu'
        self.model = YOLO(str(weights), task='multi')
        self.imgsz, self.conf = imgsz, conf
        self.names = None

    def __call__(self, frame):
        """frame: BGR image (H, W, 3). Returns (boxes, lane_mask): boxes = [(x1, y1, x2, y2, conf, cls)],
        lane_mask = (H, W) uint8 with 0 background, 1 solid, 2 dashed."""
        r = self.model.predict(source=frame, imgsz=self.imgsz, conf=self.conf, iou=0.45, device=self.device,
                               half=self.gpu, verbose=False, save=False)
        det = r[0][0]
        if self.names is None:
            self.names = det.names
        boxes = det.boxes.data.cpu().numpy() if det.boxes is not None else np.zeros((0, 6))
        mask = r[1][0].to(torch.uint8).cpu().numpy()
        if mask.shape != frame.shape[:2]:
            mask = cv2.resize(mask, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST)
        return boxes, mask


def draw(frame, boxes, mask, lanes, names, fps=None):
    out = frame.copy()
    for value, colour in LANE_BGR.items():  # tint lane pixels
        sel = mask == value
        out[sel] = (0.5 * out[sel] + 0.5 * np.array(colour)).astype(np.uint8)
    for lane in lanes:  # extracted lane lines (what the JSON contains)
        pts = np.array(lane['points'], dtype=np.int32)
        colour = LANE_BGR[1 if lane['type'] == 'solid' else 2]
        cv2.polylines(out, [pts], False, (255, 255, 255), 3)
        cv2.polylines(out, [pts], False, colour, 1)
        for x, y in pts:
            cv2.circle(out, (int(x), int(y)), 2, (255, 255, 255), -1)
    for x1, y1, x2, y2, conf, cls in boxes:
        p1, p2 = (int(x1), int(y1)), (int(x2), int(y2))
        cv2.rectangle(out, p1, p2, (0, 255, 0), 2)
        label = f'{names.get(int(cls), int(cls)) if names else int(cls)} {conf:.2f}'
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(out, (p1[0], p1[1] - th - 6), (p1[0] + tw + 4, p1[1]), (0, 255, 0), -1)
        cv2.putText(out, label, (p1[0] + 2, p1[1] - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    if fps is not None:
        cv2.putText(out, f'{fps:.1f} FPS', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 4)
        cv2.putText(out, f'{fps:.1f} FPS', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
    return out


def record(i, boxes, lanes, names, size):
    return {'frame': i, 'width': size[1], 'height': size[0],
            'lanes': lanes,
            'objects': [{'class': names.get(int(c), str(int(c))) if names else int(c), 'class_id': int(c),
                         'conf': round(float(cf), 3), 'box': [round(float(v), 1) for v in (x1, y1, x2, y2)]}
                        for x1, y1, x2, y2, cf, c in boxes]}


# ----------------------------------------------------------------------------------------------------
# Sources
# ----------------------------------------------------------------------------------------------------
def run_images(net, files, out_dir, show, jsonl, step):
    for i, f in enumerate(files):
        frame = cv2.imread(str(f))
        if frame is None:
            print(f'  could not read {f}')
            continue
        t = time.time()
        boxes, mask = net(frame)
        lanes = extract_lanes(mask, step=step)
        dt = time.time() - t
        img = draw(frame, boxes, mask, lanes, net.names)
        cv2.imwrite(str(out_dir / f.name), img)
        if jsonl:
            jsonl.write(json.dumps({'file': f.name, **record(i, boxes, lanes, net.names, frame.shape)}) + '\n')
        print(f'  {f.name}: {len(boxes)} objects, {len(lanes)} lanes '
              f'({sum(l["type"] == "solid" for l in lanes)} solid, {sum(l["type"] == "dashed" for l in lanes)} dashed), '
              f'{dt * 1000:.0f} ms')
        if show:
            cv2.imshow('lanedet', img)
            if cv2.waitKey(0) & 0xFF == ord('q'):
                break


def run_stream(net, cap, out_path, show, jsonl, step, is_camera):
    writer, n, t0, fps = None, 0, time.time(), None
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if not is_camera else 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            t = time.time()
            boxes, mask = net(frame)
            lanes = extract_lanes(mask, step=step)
            dt = time.time() - t
            fps = 1 / dt if fps is None else 0.9 * fps + 0.1 / max(dt, 1e-6)
            img = draw(frame, boxes, mask, lanes, net.names, fps)
            if out_path is not None:
                if writer is None:
                    rec_fps = src_fps if not is_camera else min(src_fps, max(fps, 1))
                    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*'mp4v'), rec_fps,
                                             (img.shape[1], img.shape[0]))
                writer.write(img)
            if jsonl:
                jsonl.write(json.dumps(record(n, boxes, lanes, net.names, frame.shape)) + '\n')
            n += 1
            if not show and n % 50 == 0:
                print(f'  frame {n}' + (f'/{total}' if total else '') + f'  {fps:.1f} FPS')
            if show:
                cv2.imshow('lanedet  (q = quit)', img)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
    finally:  # always finish the video file, even on Ctrl+C
        cap.release()
        if writer is not None:
            writer.release()
    print(f'  {n} frames, average {n / max(time.time() - t0, 1e-6):.1f} FPS including drawing')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--source', required=True, help='image, folder, video file, or camera number (0, 1, ...)')
    ap.add_argument('--weights', default='runs/multi/bdd_laptop/weights/best.pt')
    ap.add_argument('--imgsz', type=int, default=640)
    ap.add_argument('--conf', type=float, default=0.25)
    ap.add_argument('--name', default=None, help='output folder name under runs\\lanedet_out')
    ap.add_argument('--save', action='store_true', help='camera only: also record a video (files are always saved)')
    ap.add_argument('--show', dest='show', action='store_true', default=None)
    ap.add_argument('--no-show', dest='show', action='store_false')
    ap.add_argument('--json', action='store_true', help='write lane coordinates + boxes to a .jsonl file')
    ap.add_argument('--step', type=int, default=10, help='lane points every N pixel rows')
    args = ap.parse_args()

    if not Path(args.weights).exists():
        sys.exit(f'Weights not found: {args.weights}  (pass --weights path\\to\\best.pt)')
    src = args.source
    is_camera = src.isdigit()
    p = Path(src)
    if not is_camera and not p.exists():
        sys.exit(f'Source not found: {src}')

    name = args.name or ('camera' + src if is_camera else p.stem)
    out_dir = Path('runs/lanedet_out') / name
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f'Loading {args.weights} on {"GPU " + torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"}')
    net = LaneDet(args.weights, args.imgsz, args.conf)
    jsonl = open(out_dir / f'{name}.jsonl', 'w') if args.json else None

    try:
        if is_camera:
            cap = cv2.VideoCapture(int(src), cv2.CAP_DSHOW) if platform.system() == 'Windows' \
                else cv2.VideoCapture(int(src))
            if not cap.isOpened():
                sys.exit(f'Could not open camera {src}. Try --source 1, or close other apps using the camera.')
            print('Camera running. Press q in the window to stop.')
            run_stream(net, cap, out_dir / f'{name}.mp4' if args.save else None,
                       True if args.show is None else args.show, jsonl, args.step, True)
        elif p.is_dir() or p.suffix.lower() in IMG_EXT:
            files = sorted(f for f in p.iterdir() if f.suffix.lower() in IMG_EXT) if p.is_dir() else [p]
            print(f'{len(files)} image(s)')
            run_images(net, files, out_dir, bool(args.show), jsonl, args.step)
        elif p.suffix.lower() in VID_EXT:
            cap = cv2.VideoCapture(str(p))
            if not cap.isOpened():
                sys.exit(f'Could not open video {p}')
            print('Processing video' + (' (press q in the window to stop early)' if args.show is not False else ''))
            run_stream(net, cap, out_dir / f'{p.stem}_lanedet.mp4', True if args.show is None else args.show,
                       jsonl, args.step, False)
        else:
            sys.exit(f'Unknown source type: {src}')
    finally:
        if jsonl:
            jsonl.close()
        cv2.destroyAllWindows()
    print(f'Output: {out_dir}')


if __name__ == '__main__':
    main()
