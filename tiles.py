"""Find upright mahjong tile faces in a photo with FastSAM (free, runs on CPU)."""
import itertools

import cv2
import numpy as np
from ultralytics import FastSAM

TILES = (
    [f"{n}{s}" for s in ("dot", "bam", "crack") for n in range(1, 10)]
    + ["east", "south", "west", "north", "red", "green", "white"]
    + [f"flower{n}" for n in range(1, 5)] + [f"season{n}" for n in range(1, 5)]
    + ["joker"]
)
CLASSES = TILES + ["junk"]  # junk = tile backs, misfires; lets the classifier say "not a tile"

_sam = None


def find_tiles(img: np.ndarray) -> list[tuple[int, int, int, int]]:
    """Every box that looks like an upright tile face: (x1, y1, x2, y2)."""
    global _sam
    _sam = _sam or FastSAM("FastSAM-s.pt")
    r = _sam.predict(img, imgsz=1024, conf=0.25, iou=0.7, retina_masks=True, verbose=False)[0]
    if r.masks is None:
        return []
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    white = (hsv[..., 1] < 50) & (hsv[..., 2] > 150)
    H = img.shape[0]
    boxes = []
    for m, b in zip(r.masks.data.cpu().numpy().astype(bool), r.boxes.xyxy.cpu().numpy().astype(int)):
        x1, y1, x2, y2 = b
        w, h = x2 - x1, y2 - y1
        # shortcut: tuned on one photo (tiles ~10% of image height, upright); retune if tiles are tiny or rotated
        if 0.65 < w / h < 0.95 and h > 0.04 * H and m.sum() / (w * h) > 0.7 and white[m].mean() > 0.45:
            boxes.append((x1, y1, x2, y2))
    return _dedupe(boxes)


def _dedupe(boxes):
    """Drop boxes mostly inside a bigger one (FastSAM returns face + whole tile)."""
    boxes = sorted(boxes, key=lambda b: -(b[2] - b[0]) * (b[3] - b[1]))
    keep = []
    for b in boxes:
        if all(_overlap(b, k) < 0.6 for k in keep):
            keep.append(b)
    return keep


def _overlap(a, b):
    """Intersection over the area of a."""
    iw = min(a[2], b[2]) - max(a[0], b[0])
    ih = min(a[3], b[3]) - max(a[1], b[1])
    return max(iw, 0) * max(ih, 0) / ((a[2] - a[0]) * (a[3] - a[1]))


def rack_row(boxes):
    """The hand: the most boxes whose centres lie on one line, ordered left to right.
    Loose tiles elsewhere on the table fall off the line."""
    if len(boxes) < 3:
        return sorted(boxes)
    c = np.array([((b[0] + b[2]) / 2, (b[1] + b[3]) / 2) for b in boxes])
    tol = 0.5 * np.median([b[3] - b[1] for b in boxes])
    best = []
    for i, j in itertools.combinations(range(len(c)), 2):
        d = c[j] - c[i]
        if abs(d[0]) < 1e-6 or abs(d[1] / d[0]) > 0.5:  # the rack is roughly horizontal
            continue
        v = c - c[i]
        dist = np.abs(d[0] * v[:, 1] - d[1] * v[:, 0]) / np.linalg.norm(d)
        on = [k for k in range(len(c)) if dist[k] < tol]
        if len(on) > len(best):
            best = on
    return sorted((boxes[k] for k in best), key=lambda b: b[0])


def groups_by_gap(boxes):
    """Split the row where the player left a gap wider than ~1/3 of a tile."""
    if not boxes:
        return []
    gap = 0.35 * np.median([b[2] - b[0] for b in boxes])
    groups = [[0]]
    for k in range(1, len(boxes)):
        if boxes[k][0] - boxes[k - 1][2] < gap:
            groups[-1].append(k)
        else:
            groups.append([k])
    return groups


def crop(img, b):
    return img[b[1]:b[3], b[0]:b[2]]
