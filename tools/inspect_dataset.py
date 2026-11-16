"""Summarise an unknown dataset folder so we can tell what labels it really has.

Usage (from the repo root):
    python tools\\inspect_dataset.py bdd10k
Writes the same report to inspect_dataset.txt.
"""
import json
import sys
from collections import Counter
from pathlib import Path

MAX_JSON_MB = 1500


def main():
    root = Path(sys.argv[1] if len(sys.argv) > 1 else 'bdd10k')
    out = open('inspect_dataset.txt', 'w', encoding='utf-8')

    def p(*a):
        s = ' '.join(str(x) for x in a)
        print(s)
        out.write(s + '\n')

    p(f'DATASET: {root.resolve()}\n')

    p('=== FOLDER TREE (file counts by extension) ===')
    for d in sorted([root] + [x for x in root.rglob('*') if x.is_dir()]):
        depth = len(d.relative_to(root).parts)
        if depth > 5:
            continue
        files = [f for f in d.iterdir() if f.is_file()]
        exts = Counter(f.suffix.lower() for f in files)
        ext_s = ', '.join(f'{n} {e or "(none)"}' for e, n in exts.most_common()) or '-'
        p(f"{'  ' * depth}{d.name}/   [{ext_s}]")

    p('\n=== YAML / TXT CONFIG FILES (top level) ===')
    for f in sorted(root.glob('*.yaml')) + sorted(root.glob('*.yml')):
        p(f'--- {f.name} ---')
        p(f.read_text(encoding='utf-8', errors='replace')[:2000])

    p('\n=== SAMPLE LABEL .txt FILES (first 3 lines of 3 files per folder) ===')
    seen = 0
    for d in sorted({f.parent for f in root.rglob('*.txt')}):
        files = sorted(d.glob('*.txt'))[:3]
        p(f'--- {d.relative_to(root)} ({sum(1 for _ in d.glob("*.txt"))} txt files) ---')
        for f in files:
            lines = f.read_text(errors='replace').splitlines()
            p(f'  {f.name}: {len(lines)} lines')
            for ln in lines[:3]:
                p(f'     {ln[:160]}')
        seen += 1
        if seen >= 8:
            break

    p('\n=== JSON FILES (keys and first record) ===')
    for f in sorted(root.rglob('*.json')):
        mb = f.stat().st_size / 1e6
        p(f'--- {f.relative_to(root)} ({mb:.1f} MB) ---')
        if mb > MAX_JSON_MB:
            p('  (too large to load here)')
            continue
        try:
            data = json.load(open(f, encoding='utf-8'))
        except Exception as e:
            p(f'  could not parse: {e}')
            continue
        if isinstance(data, dict):
            p(f'  dict keys: {list(data.keys())[:20]}')
            for k in ('images', 'annotations', 'categories', 'frames'):
                if k in data and isinstance(data[k], list) and data[k]:
                    p(f'  {k}: {len(data[k])} items; first: {json.dumps(data[k][0])[:400]}')
        elif isinstance(data, list):
            p(f'  list of {len(data)} items; first: {json.dumps(data[0])[:600] if data else "-"}')
            cats = Counter()
            for item in data[:2000]:
                for lb in (item.get('labels') or []) if isinstance(item, dict) else []:
                    cats[lb.get('category')] += 1
            if cats:
                p(f'  categories (first 2000 items): {dict(cats.most_common(25))}')

    p('\n=== SAMPLE IMAGES ===')
    imgs = sorted(x for x in root.rglob('*') if x.suffix.lower() in ('.jpg', '.jpeg', '.png'))
    p(f'{len(imgs)} images in total; first names: {[i.name for i in imgs[:5]]}')
    try:
        import cv2
        for i in imgs[:2]:
            im = cv2.imread(str(i))
            p(f'  {i.relative_to(root)}: {None if im is None else im.shape}')
    except ImportError:
        pass
    out.close()
    print('\nSaved to inspect_dataset.txt')


if __name__ == '__main__':
    main()