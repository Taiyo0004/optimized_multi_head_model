"""Step 4a: pull labelling frames out of your road videos, one folder per recording session.

Each video file is treated as one recording session (that's what the 80/10/10 split is by later).
Skips frames that are too blurry or nearly identical to the last saved one (e.g. stopped at a signal).

Usage:
    python extract_frames.py --videos D:\\drives --out frames --every 1.0
    python extract_frames.py --videos D:\\drives --out frames --every 0.5 --diff 6

Output:
    frames\\<session>\\<session>_000123.jpg
    frames\\sessions.csv   (session, video path, frames saved, duration) - keep this, it drives the split
"""
import argparse
import csv
import re
from pathlib import Path

import cv2
import numpy as np

VIDEO_EXT = {".mp4", ".mov", ".avi", ".mkv", ".m4v"}


def session_name(path: Path) -> str:
    return re.sub(r"[^A-Za-z0-9_-]+", "_", path.stem)


def sharpness(gray):
    return cv2.Laplacian(gray, cv2.CV_64F).var()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos", required=True, help="folder containing your recordings (searched recursively)")
    ap.add_argument("--out", default="frames")
    ap.add_argument("--every", type=float, default=1.0, help="seconds between candidate frames")
    ap.add_argument("--blur", type=float, default=60.0, help="min sharpness (Laplacian variance); 0 disables")
    ap.add_argument("--diff", type=float, default=4.0, help="min mean pixel change vs last saved frame; 0 disables")
    ap.add_argument("--quality", type=int, default=95)
    args = ap.parse_args()

    videos = sorted(p for p in Path(args.videos).rglob("*") if p.suffix.lower() in VIDEO_EXT)
    if not videos:
        raise SystemExit(f"No videos found in {args.videos}")

    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)
    rows, total = [], 0

    for vid in videos:
        sess = session_name(vid)
        cap = cv2.VideoCapture(str(vid))
        if not cap.isOpened():
            print(f"[skip] cannot open {vid}")
            continue
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        step = max(1, round(fps * args.every))
        sess_dir = out_root / sess
        sess_dir.mkdir(exist_ok=True)

        last_small, saved, blurry, dupes, idx = None, 0, 0, 0, 0
        while True:
            ok = cap.grab()
            if not ok:
                break
            if idx % step == 0:
                ok, frame = cap.retrieve()
                if not ok:
                    break
                small = cv2.cvtColor(cv2.resize(frame, (320, 180)), cv2.COLOR_BGR2GRAY)
                if args.blur and sharpness(small) < args.blur:
                    blurry += 1
                elif args.diff and last_small is not None and \
                        np.abs(small.astype(np.int16) - last_small).mean() < args.diff:
                    dupes += 1
                else:
                    cv2.imwrite(str(sess_dir / f"{sess}_{idx:06d}.jpg"), frame,
                                [cv2.IMWRITE_JPEG_QUALITY, args.quality])
                    last_small = small.astype(np.int16)
                    saved += 1
            idx += 1
        cap.release()

        dur = n / fps if fps else 0
        print(f"{sess:<30} saved {saved:>5}  (blurry {blurry}, near-duplicate {dupes}, {dur / 60:.1f} min)")
        rows.append([sess, str(vid), saved, f"{dur:.0f}"])
        total += saved

    with open(out_root / "sessions.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["session", "video", "frames_saved", "duration_s"])
        w.writerows(rows)

    print(f"\nTotal frames: {total} from {len(rows)} sessions -> {out_root}")
    if total and len(rows) < 10:
        print("Note: fewer than 10 sessions makes an 80/10/10 split by session lumpy - more, shorter drives help.")


if __name__ == "__main__":
    main()
