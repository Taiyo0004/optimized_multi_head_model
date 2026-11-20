"""Check that checkpoint + resume works on this machine (about 5 minutes on the laptop).

Run from the repo root, after the demo data exists (datasets_local\\bdd_demo):
    python tools\\test_resume.py

Part 1 (seconds, no GPU): the resumed image order is exactly the uninterrupted order.
Part 2 (a few minutes): starts a real 2-epoch training, hard-kills it mid-epoch like a crash would,
resumes it from last.pt, and checks it finishes from the right place.
Writes everything to runs\\multi\\resume_test (safe to delete afterwards).
"""
import shutil
import subprocess
import sys
import time
from pathlib import Path

import torch

DATA = 'datasets_local/bdd_demo/lanedet_data.yaml'
WEIGHTS = 'weights/lanedet_s_init.pt'
RUN = Path('runs/multi/resume_test')
LOG1, LOG2 = RUN.parent / 'resume_test_part1.log', RUN.parent / 'resume_test_part2.log'


class Ints(torch.utils.data.Dataset):
    def __len__(self):
        return 23

    def __getitem__(self, i):
        return i


def part1():
    from ultralytics.yolo.data.build import build_dataloader, set_loader_position
    ok = True
    for workers in (0, 2):
        def batches(loader, n):
            it = iter(loader)
            return [next(it).tolist() for _ in range(n)]

        ref = build_dataloader(Ints(), 4, workers, shuffle=True, rank=-1, seed=0)
        nb = len(ref)
        full = []
        for _ in range(3):  # three uninterrupted epochs
            full += [b.tolist() for b in ref]
        # crash after batch 2 of epoch 1 (0-based), resume in a fresh loader
        new = build_dataloader(Ints(), 4, workers, shuffle=True, rank=-1, seed=0)
        set_loader_position(new, 1, 3)
        import itertools
        got = full[:nb + 3] + [b.tolist() for b in itertools.islice(new, nb - 3)] + [b.tolist() for b in new]
        same = got == full
        epochs_ok = all(sorted(sum(full[e * nb:(e + 1) * nb], [])) == list(range(23)) for e in range(3))
        print(f'  workers={workers}: resumed order matches = {same}; every image once per epoch = {epochs_ok}')
        ok &= same and epochs_ok
    return ok


def train_cmd(resume):
    code = ('from ultralytics import YOLO\n'
            "if __name__ == '__main__':\n")
    if resume:
        code += f"    YOLO(r'{RUN / 'weights' / 'last.pt'}', task='multi').train(resume=True)\n"
    else:
        code += (f"    YOLO(r'{WEIGHTS}', task='multi').train(data=r'{DATA}', epochs=2, imgsz=640, batch=4, "
                 f"workers=0, device=0, project='runs/multi', name='resume_test', exist_ok=True, seed=0, "
                 f"ckpt_minutes=0.1, plots=False)\n")
    return [sys.executable, '-c', code]


def part2():
    if not torch.cuda.is_available():
        print('  no GPU, skipping')
        return True
    for p in (DATA, WEIGHTS):
        if not Path(p).exists():
            print(f'  {p} not found, skipping')
            return True
    shutil.rmtree(RUN, ignore_errors=True)
    last = RUN / 'weights' / 'last.pt'

    print('  starting training; it will be killed once a mid-epoch checkpoint appears ...')
    with open(LOG1, 'w') as log:
        p = subprocess.Popen(train_cmd(False), stdout=log, stderr=subprocess.STDOUT)
        t0, killed_at = time.time(), None
        while p.poll() is None and time.time() - t0 < 900:
            time.sleep(2)
            if last.exists():
                try:
                    ck = torch.load(last, map_location='cpu', weights_only=False)
                except Exception:
                    continue  # still being written
                if ck.get('iter', -1) >= 0:
                    killed_at = (ck['epoch'], ck['iter'])
                    p.kill()  # like a crash: no clean shutdown
                    p.wait()
                    break
    if killed_at is None:
        print(f'  no mid-epoch checkpoint was written (see {LOG1})')
        return False
    print(f'  killed; last.pt was at epoch {killed_at[0] + 1}, batch {killed_at[1] + 1}')

    print('  resuming ...')
    with open(LOG2, 'w') as log:
        r = subprocess.run(train_cmd(True), stdout=log, stderr=subprocess.STDOUT)
    text = LOG2.read_text(errors='ignore')
    want = f'from epoch {killed_at[0] + 1} batch {killed_at[1] + 2}'
    resumed = want in text
    finished = r.returncode == 0 and (RUN / 'weights' / 'best.pt').exists()
    rows = sum(1 for _ in open(RUN / 'results_seg-lane.csv')) - 1 if (RUN / 'results_seg-lane.csv').exists() else 0
    print(f'  resumed at the right batch = {resumed}; finished = {finished}; epochs in results CSV = {rows} (want 2)')
    if not (resumed and finished):
        print(f'  see {LOG2}')
    return resumed and finished and rows == 2


if __name__ == '__main__':
    print('Part 1: image order after resume')
    a = part1()
    print('Part 2: kill and resume a real training run')
    b = part2()
    print('\nRESUME TEST', 'PASSED' if a and b else 'FAILED')
