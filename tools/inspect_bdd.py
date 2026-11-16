"""Step 4: look inside the BDD100K label files before we write the converter.

Prints every category, every attribute and its values, counts, and one raw example per category,
so the converter matches the real file format instead of a guess.

Usage (point it at a JSON file OR a folder of JSON files):
    python inspect_bdd.py --labels D:\\bdd100k\\labels\\det_20\\det_val.json
    python inspect_bdd.py --labels D:\\bdd100k\\labels\\lane\\polygons\\lane_val.json
Writes the report to inspect_<name>.txt as well.
"""
import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path


def load_frames(path: Path):
    files = [path] if path.is_file() else sorted(path.rglob("*.json"))
    for f in files:
        with open(f, encoding="utf-8") as fh:
            data = json.load(fh)
        if isinstance(data, dict) and "frames" in data:
            data = data["frames"]
        if isinstance(data, dict):
            data = [data]
        yield from data


def short(v, n=60):
    s = json.dumps(v)
    return s if len(s) <= n else s[:n] + "..."


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", required=True)
    ap.add_argument("--max-frames", type=int, default=0, help="0 = all")
    args = ap.parse_args()

    path = Path(args.labels)
    out = open(f"inspect_{path.stem}.txt", "w", encoding="utf-8")

    def p(*a):
        line = " ".join(str(x) for x in a)
        print(line)
        out.write(line + "\n")

    n_frames, n_empty = 0, 0
    frame_keys = Counter()
    cats = Counter()
    geom = Counter()
    attrs = defaultdict(Counter)          # (category, attr) -> value counts
    example = {}
    poly_types = Counter()

    for fr in load_frames(path):
        n_frames += 1
        frame_keys.update(fr.keys())
        labels = fr.get("labels") or []
        if not labels:
            n_empty += 1
        for lb in labels:
            c = lb.get("category", "?")
            cats[c] += 1
            for g in ("box2d", "poly2d", "box3d"):
                if lb.get(g):
                    geom[(c, g)] += 1
            for k, v in (lb.get("attributes") or {}).items():
                attrs[(c, k)][short(v, 40)] += 1
            for poly in lb.get("poly2d") or []:
                if isinstance(poly, dict):
                    poly_types[short(poly.get("types", ""), 20)] += 1
            if c not in example:
                example[c] = lb
        if args.max_frames and n_frames >= args.max_frames:
            break

    p(f"File: {path}")
    p(f"Frames: {n_frames}   frames with no labels: {n_empty}")
    p(f"Frame-level keys: {dict(frame_keys)}")
    p("\nCATEGORIES")
    for c, n in cats.most_common():
        g = ", ".join(f"{k[1]}={v}" for k, v in geom.items() if k[0] == c)
        p(f"  {c:<28} {n:>8}   ({g})")
    p("\nATTRIBUTES (per category)")
    for (c, k), vals in sorted(attrs.items()):
        top = ", ".join(f"{v}: {n}" for v, n in vals.most_common(8))
        p(f"  {c:<22} {k:<20} {top}")
    if poly_types:
        p("\nPOLY2D 'types' strings (L = line point, C = curve control point)")
        for t, n in poly_types.most_common(8):
            p(f"  {t:<24} {n}")
    p("\nONE RAW EXAMPLE PER CATEGORY")
    for c, lb in example.items():
        p(f"  [{c}] {short(lb, 300)}")
    out.close()
    print(f"\nSaved to inspect_{path.stem}.txt")


if __name__ == "__main__":
    main()
