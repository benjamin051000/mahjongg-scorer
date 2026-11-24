# Mahjongg Hand Scorer

Photo of your hand -> list of tiles -> your house scoring.

- **FastSAM** finds the tiles.
- A small **YOLO classifier** that you train on your own set names each tile.
- Each verified hand becomes more training data.

## Use

```sh
uv run read_hand.py photo.jpg
```

It shows `1:flower3  2:4dot? ...`. A `?` means the model isn't sure. Fix mistakes with:

| type      | does                                   |
|-----------|----------------------------------------|
| `5=north` | tile 5 is a North                      |
| `-5`      | tile 5 isn't a tile (or isn't in your hand) |
| `+north`  | add a tile it missed                   |
| `\|5`     | start a new set at tile 5              |
| Enter     | accept                                 |

Accepting saves `labels/<photo>.json` and adds each tile picture to `dataset/labeled/<tile>/`.
Then it calls `score(groups)` from `house_rules.py`, which you write. `groups` is a list of sets,
e.g. `[["2dot","2dot","2dot"], ["9dot","9dot","9dot"], ...]`.

Tile names: `1dot`..`9dot`, `1bam`..`9bam`, `1crack`..`9crack`, `east south west north`,
`red green white`, `flower1`..`flower4`, `season1`..`season4`, `joker`.

### `--use_claude`: no local model needed yet

```sh
uv run read_hand.py photo.jpg --use_claude
```

This names the tiles with headless Claude Code instead of the local model. It needs the
`claude` CLI installed and logged in.
The session can only read the one tile image, in a temp folder.

Confirmed hands still land in `dataset/labeled/`, so scoring hands with `--use_claude` is the easiest
way to build the training set. Train once you have ~10 pictures per tile, then drop the flag.

## Getting the model good

Before there's a model, every tile shows `?` and you type all of them in.

1. **Quick start:** lay out one of every tile on the racks, take ~5 photos (different light and angles), and run
   `uv run collect.py photos/*.jpg`. Drag each picture from `dataset/unlabeled/` into its
   `dataset/labeled/<tile>/` folder; anything that isn't a tile face goes in `junk/`.
2. `uv run train.py`: a few minutes on CPU. Writes `tile_cls.pt`.
3. From then on `read_hand.py` pre-fills the names. Retrain every so often as hands pile up.
   `collect.py` will now pre-sort new photos into `dataset/predicted/<tile>/`. Move the wrong ones, then
   `rsync -a --remove-source-files dataset/predicted/ dataset/labeled/`.

Aim for ~10+ pictures per tile. Tiles that come up rarely (flowers, seasons) need the layout photos.

## Limits

- Tile finding is tuned for upright tiles that fill about a tenth of the photo's height. Shoot the rack
  from the front, roughly like the sample photo. Tiles that are tilted, lying flat or very small are skipped.
- Gaps between sets are guessed from spacing. Tiles that touch merge into one set; split them with `|5`.
