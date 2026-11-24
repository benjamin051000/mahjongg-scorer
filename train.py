"""Train the tile classifier on dataset/labeled/<tile>/*.jpg -> tile_cls.pt.

    .venv/bin/python train.py [epochs]
"""
import random, shutil, sys
from pathlib import Path

from ultralytics import YOLO

import tiles

LABELED, SPLIT = Path("dataset/labeled"), Path("dataset/_split")

shutil.rmtree(SPLIT, ignore_errors=True)
counts = {}
for d in sorted(p for p in LABELED.iterdir() if p.is_dir()):
    files = sorted(d.glob("*.jpg"))
    if not files:
        continue
    if d.name not in tiles.CLASSES:
        sys.exit(f"unknown class folder {d}: use one of {tiles.CLASSES}")
    random.Random(0).shuffle(files)
    nval = max(1, len(files) // 5)
    # shortcut: a class with one image is in train and val both; val numbers mean little until each class has ~5+
    parts = {"val": files[:nval], "train": files[nval:] or files[:nval]}
    for part, fs in parts.items():
        (SPLIT / part / d.name).mkdir(parents=True)
        for f in fs:
            (SPLIT / part / d.name / f.name).symlink_to(f.resolve())
    counts[d.name] = len(files)

if len(counts) < 2:
    sys.exit("need at least 2 labelled classes")
print("images per class:", counts)
missing = [t for t in tiles.TILES if t not in counts]
if missing:
    print("no examples yet for:", " ".join(missing))

res = YOLO("yolo11n-cls.pt").train(data=str(SPLIT.resolve()), epochs=int(sys.argv[1]) if len(sys.argv) > 1 else 30,
                                   imgsz=128, batch=16, project="runs", name="tile_cls", exist_ok=True,
                                   fliplr=0.0, plots=False)  # no flips: a mirrored tile is not the same tile
shutil.copy(Path(res.save_dir) / "weights" / "best.pt", "tile_cls.pt")
print("saved tile_cls.pt")
