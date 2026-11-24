"""Bulk-crop every tile face in a folder of photos, for labelling by hand.

    .venv/bin/python collect.py photos/*.jpg

Without a model, crops land in dataset/unlabeled/: drag each into
dataset/labeled/<tile>/ (folders are pre-made). With a model, crops land in
dataset/predicted/<tile>/: fix the wrong ones, then merge them into labeled/.
"""
import sys
from pathlib import Path

import cv2

import tiles
from read_hand import classify

for name in tiles.CLASSES:
    (Path("dataset/labeled") / name).mkdir(parents=True, exist_ok=True)

for photo in sys.argv[1:]:
    img = cv2.imread(photo)
    if img is None:
        print(f"skip {photo}: can't read")
        continue
    crops = [tiles.crop(img, b) for b in tiles.find_tiles(img)]  # every tile, not just the rack: more data
    for k, (crop, (tile, _)) in enumerate(zip(crops, classify(crops))):
        d = Path("dataset/unlabeled") if tile == "?" else Path("dataset/predicted") / tile
        d.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(d / f"{Path(photo).stem}_{k}.jpg"), crop)
    print(f"{photo}: {len(crops)} tiles")
