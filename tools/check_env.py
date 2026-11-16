"""Step 1 check: confirms PyTorch + CUDA + the A-YOLOM repo are set up.

Run from inside the cloned YOLOv8-multi-task folder:
    python check_env.py
"""
import sys
import glob
import os

ok = True

def report(name, passed, detail=""):
    global ok
    ok &= passed
    print(f"[{'OK' if passed else 'FAIL'}] {name}{' - ' + detail if detail else ''}")

report("Python version", sys.version_info >= (3, 8), sys.version.split()[0])

try:
    import torch
    report("PyTorch import", True, torch.__version__)
    cuda = torch.cuda.is_available()
    report("CUDA available", cuda, torch.version.cuda or "no CUDA build")
    if cuda:
        p = torch.cuda.get_device_properties(0)
        report("GPU", True, f"{p.name}, {p.total_memory / 1e9:.1f} GB")
        # tiny FP16 matmul to make sure the GPU actually runs work
        x = torch.randn(1024, 1024, device="cuda", dtype=torch.half)
        y = (x @ x).float().sum().item()
        report("FP16 GPU compute", y == y, "matmul ran")
except Exception as e:
    report("PyTorch import", False, repr(e))

try:
    import ultralytics
    path = os.path.dirname(ultralytics.__file__)
    in_repo = os.path.abspath(path).startswith(os.path.abspath("."))
    report("ultralytics import", True, path)
    report("ultralytics is the A-YOLOM fork (installed with -e from this repo)",
           in_repo, "" if in_repo else "pip uninstall ultralytics, then pip install -e .")
except Exception as e:
    report("ultralytics import", False, repr(e))

yamls = glob.glob("ultralytics/models/v8/*.yaml")
multi = [y for y in yamls if "bdd" in os.path.basename(y)]
report("A-YOLOM model configs found", bool(multi),
       f"{len(multi)} bdd config(s), e.g. {os.path.basename(multi[0])}" if multi else "run from repo root")

for pkg in ["cv2", "numpy", "yaml", "tqdm"]:
    try:
        __import__(pkg)
        report(f"package {pkg}", True)
    except ImportError:
        report(f"package {pkg}", False, "pip install it")

print("\nAll good - ready for Step 2." if ok else "\nFix the FAIL lines above, then re-run.")
