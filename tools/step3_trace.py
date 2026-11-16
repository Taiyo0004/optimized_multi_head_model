"""Step 3: trace one image through A-YOLOM and print the tensor shape after every layer.

Usage (from the repo root):
    python step3_trace.py --weights weights\\v4s.pt
    python step3_trace.py --weights weights\\v4s.pt --image test_images\\test.jpg
Writes the same output to step3_trace.txt so you can keep it next to your notes.
"""
import argparse
import functools
import sys

import torch

# Same PyTorch 2.6+ loading fix as Step 2 (trusted weights only).
torch.load = functools.partial(torch.load, weights_only=False)

from ultralytics import YOLO  # noqa: E402


def describe(x):
    """Readable shape for a tensor, or a list/tuple/dict of tensors."""
    if isinstance(x, torch.Tensor):
        return "x".join(str(d) for d in x.shape)
    if isinstance(x, (list, tuple)):
        return "[" + ", ".join(describe(i) for i in x) + "]"
    if isinstance(x, dict):
        return "{" + ", ".join(f"{k}: {describe(v)}" for k, v in x.items()) + "}"
    return type(x).__name__


class Tee:
    def __init__(self, path):
        self.f = open(path, "w", encoding="utf-8")

    def write(self, s):
        sys.__stdout__.write(s)
        self.f.write(s)

    def flush(self):
        sys.__stdout__.flush()
        self.f.flush()


def load_image(path, size):
    import cv2
    import numpy as np

    img = cv2.imread(path)
    if img is None:
        raise SystemExit(f"Could not read image: {path}")
    h, w = img.shape[:2]
    r = size / max(h, w)
    nh, nw = round(h * r), round(w * r)
    img = cv2.resize(img, (nw, nh))
    canvas = np.full((size, size, 3), 114, dtype=np.uint8)  # letterbox, as in Section 4.1
    top, left = (size - nh) // 2, (size - nw) // 2
    canvas[top:top + nh, left:left + nw] = img
    t = torch.from_numpy(canvas[:, :, ::-1].copy()).permute(2, 0, 1).float() / 255.0
    return t.unsqueeze(0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--image", default=None, help="optional real image; otherwise random input")
    ap.add_argument("--imgsz", type=int, default=640)
    args = ap.parse_args()

    sys.stdout = Tee("step3_trace.txt")

    yolo = YOLO(args.weights, task="multi")
    net = yolo.model            # the nn.Module
    net.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    net.to(device).float()

    # 1) The config the weights were built from
    cfg = getattr(net, "yaml", None)
    print("=" * 90)
    print("MODEL CONFIG (from the checkpoint)")
    print("=" * 90)
    if isinstance(cfg, dict):
        for key in cfg:
            if isinstance(cfg[key], list):
                print(f"\n{key}:")
                for i, row in enumerate(cfg[key]):
                    print(f"  {i:>3}: {row}")
            else:
                print(f"{key}: {cfg[key]}")
    else:
        print("(no yaml stored in checkpoint)")

    # 2) Per-layer shapes via forward hooks
    layers = net.model  # nn.Sequential of numbered layers
    rows = []

    def hook(idx, mod, inputs, output):
        src = getattr(mod, "f", "?")
        rows.append((idx, type(mod).__name__, src, describe(inputs[0] if len(inputs) == 1 else list(inputs)),
                     describe(output)))

    handles = [m.register_forward_hook(functools.partial(hook, i)) for i, m in enumerate(layers)]

    x = load_image(args.image, args.imgsz) if args.image else torch.rand(1, 3, args.imgsz, args.imgsz)
    x = x.to(device)
    with torch.no_grad():
        out = net(x)
    for h in handles:
        h.remove()

    print("\n" + "=" * 90)
    print(f"LAYER-BY-LAYER TRACE  (input {describe(x)})")
    print("=" * 90)
    print(f"{'#':>3}  {'layer type':<22} {'from':<14} {'input shape':<34} output shape")
    print("-" * 90)
    for idx, name, src, ins, outs in rows:
        print(f"{idx:>3}  {name:<22} {str(src):<14} {ins[:34]:<34} {outs}")

    print("\n" + "=" * 90)
    print("FINAL MODEL OUTPUT")
    print("=" * 90)
    if isinstance(out, (list, tuple)):
        for i, o in enumerate(out):
            print(f"  output[{i}]: {describe(o)}")
    else:
        print(f"  {describe(out)}")

    n_params = sum(p.numel() for p in net.parameters())
    print(f"\nTotal parameters: {n_params / 1e6:.2f} M")
    print("Saved to step3_trace.txt")


if __name__ == "__main__":
    main()
