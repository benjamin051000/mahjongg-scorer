"""Photo of a rack -> tiles -> human confirms -> saved as training data -> house score.

    .venv/bin/python read_hand.py PXL_20261123_231735254.jpg [--use_claude]

--use_claude names the tiles with a headless Claude Code session instead of the local model.
"""
import json, sys
from pathlib import Path

import cv2

import tiles

MODEL = Path("tile_cls.pt")
LABELED = Path("dataset/labeled")
SURE = 0.8  # below this the tile is shown with a "?" for the human to check


def classify(crops):
    """[(tile, confidence)], or '?' for everything before the first model is trained."""
    if not MODEL.exists():
        return [("?", 0.0)] * len(crops)
    from ultralytics import YOLO
    model = YOLO(MODEL)
    out = []
    for r in model.predict(crops, imgsz=128, verbose=False):
        out.append((model.names[r.probs.top1], float(r.probs.top1conf)))
    return out


CLAUDE_PROMPT = """Read the image rack.jpg in the current directory. It shows {n} mahjong tiles side by side,
numbered 1 to {n} above each tile. Name every tile, in order, using only these names:
{names}.
dot = circles, bam = bamboo sticks, crack = characters with 萬. The white dragon may be blank or a framed
rectangle. A flower or season tile shows a picture and a corner number; use that number."""


def classify_claude(crops):
    """Name the crops with a headless Claude Code session: uses the Claude subscription it's logged in with, not the API."""
    import subprocess, tempfile
    h = 300
    cells = []
    for k, c in enumerate(crops, 1):
        c = cv2.resize(c, (int(c.shape[1] * h / c.shape[0]), h))
        cell = cv2.copyMakeBorder(c, 70, 10, 10, 10, cv2.BORDER_CONSTANT, value=(255, 255, 255))
        cv2.putText(cell, str(k), (12, 55), cv2.FONT_HERSHEY_SIMPLEX, 1.8, (0, 0, 0), 4)
        cells.append(cell)
    schema = {"type": "object", "additionalProperties": False, "required": ["tiles"],
              "properties": {"tiles": {"type": "array", "items": {"enum": list(tiles.TILES)}}}}
    with tempfile.TemporaryDirectory() as d:
        cv2.imwrite(f"{d}/rack.jpg", cv2.hconcat(cells))
        # cwd is the temp dir and Read is the only tool, so the session can see nothing else
        out = subprocess.run(
            ["claude", "-p", CLAUDE_PROMPT.format(n=len(crops), names=", ".join(tiles.TILES)),
             "--tools", "Read", "--allowedTools", "Read", "--output-format", "json",
             "--json-schema", json.dumps(schema), "--no-session-persistence"],
            cwd=d, stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=600)  # else it eats our stdin
    try:
        names = json.loads(out.stdout)["structured_output"]["tiles"]
    except (json.JSONDecodeError, KeyError, TypeError):
        print("claude failed:", (out.stderr or out.stdout)[-500:])
        return [("?", 0.0)] * len(crops)
    if len(names) != len(crops):
        print(f"claude named {len(names)} tiles, expected {len(crops)}: check them all")
        return [(t, 0.0) for t in (names + ["?"] * len(crops))[:len(crops)]]
    return [(t, 1.0) for t in names]


def apply_edit(groups: list[list[dict]], cmd: str) -> None:
    """'5=north' renames tile 5, '-5' deletes it, '+north' appends one, '|5' starts a new group at tile 5."""
    if cmd.startswith("+"):
        if cmd[1:] not in tiles.TILES:
            raise ValueError(f"unknown tile {cmd[1:]!r}")
        groups[-1].append({"tile": cmd[1:], "conf": 1.0})
        return
    flat = [(g, i) for g, grp in enumerate(groups) for i in range(len(grp))]
    op = cmd[0] if cmd[0] in "-|" else "="
    num, _, tile = cmd.lstrip("-|").partition("=")
    g, i = flat[int(num) - 1]  # IndexError on a bad number
    if op == "-":
        del groups[g][i]
        groups[:] = [grp for grp in groups if grp]
    elif op == "|":
        groups[g:g + 1] = [grp for grp in (groups[g][:i], groups[g][i:]) if grp]
    else:
        if tile not in tiles.TILES:
            raise ValueError(f"unknown tile {tile!r}")
        groups[g][i].update(tile=tile, conf=1.0)


def show(groups) -> str:
    n, out = 0, []
    for grp in groups:
        cells = []
        for e in grp:
            n += 1
            cells.append(f"{n}:{e['tile']}{'' if e['conf'] >= SURE else '?'}")
        out.append("  ".join(cells))
    return "  |  ".join(out)


def confirm(groups):
    print("Edits: 5=north  -5 (not a tile)  +north (missed tile)  |5 (new group at 5). Enter to accept.")
    while True:
        print("\n" + show(groups))
        cmd = input("> ").strip()
        if not cmd:
            if any(e["tile"] == "?" for grp in groups for e in grp):
                print("  ? name every tile first")
                continue
            return groups
        try:
            apply_edit(groups, cmd)
        except (ValueError, IndexError) as e:
            print(f"  ? {e}")


def main(photo: str, use_claude: bool = False) -> None:
    path = Path(photo)
    img = cv2.imread(str(path))
    if img is None:
        sys.exit(f"can't read {path}")
    boxes = tiles.rack_row(tiles.find_tiles(img))
    crops = [tiles.crop(img, b) for b in boxes]
    entries = [{"tile": t, "conf": c, "box": [int(v) for v in b], "crop": cr}
               for (t, c), b, cr in zip((classify_claude if use_claude else classify)(crops), boxes, crops)]
    groups = confirm([[entries[k] for k in grp] for grp in tiles.groups_by_gap(boxes)])

    # every confirmed tile becomes a labelled training example; re-confirming a photo replaces its old crops
    for old in LABELED.glob(f"*/{path.stem}_*.jpg"):
        old.unlink()
    for k, e in enumerate(e for grp in groups for e in grp):
        if "crop" in e:
            d = LABELED / e["tile"]
            d.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(d / f"{path.stem}_{k}.jpg"), e["crop"])
    hand = [[e["tile"] for e in grp] for grp in groups]
    out = Path("labels") / f"{path.stem}.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps({"image": path.name, "groups": hand,
                               "boxes": [[e.get("box") for e in grp] for grp in groups]}, indent=1))
    print(f"Saved {out}")

    try:
        from house_rules import score
    except ImportError:
        print("No house_rules.py yet: write score(groups) there to get a score.")
        return
    print("Score:", score(hand))


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if a != "--use_claude"]
    if len(args) != 1:
        sys.exit(__doc__)
    main(args[0], use_claude="--use_claude" in sys.argv)
