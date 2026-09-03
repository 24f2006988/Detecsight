"""Convert the Small Object Aerial Person Detection Dataset into this
project's YOLO layout.

Source: Zenodo record 7740081, CC-BY-4.0 -- UAV frames over the University of
Cyprus campus and Civil Defense exercises, annotated for people, top-view, with
a deliberate small-object focus. 3,136 images shipped with YOLO, COCO and VOC
annotations; this reads the YOLO ones.

Why this dataset specifically: the deployed model misses nearly every genuine
person on the real UAV clips in this repo while putting boxes on vegetation.
VisDrone is urban aerial traffic and WiderPerson is clean ground-level street
pedestrians -- neither contains a person seen small from altitude against
natural terrain, which is the operational case. Measured on this set, the
median box is ~11 px across at imgsz 1280, matching the regime the detector
actually has to work in rather than the clean, large-target imagery it was
trained on.

Its single class ("people", per the record's labels.txt) is already class id 0,
which is `personnel` in this project's taxonomy -- so unlike
remap_visdrone.py / convert_widerperson.py, the label files need no class
remapping and are copied verbatim. This script still parses and revalidates
every line rather than trusting that, because a silent taxonomy mismatch would
poison the training set in a way that only shows up as a mysterious accuracy
regression later.

The source's Test split HAS ground-truth labels, so it is folded into train
(more on-domain data is worth more than a second held-out split here); the
Valid split becomes val. Run scripts/train.py against
data/battlesight_fpv.yaml afterwards.

    python scripts/convert_aerialperson.py
    python scripts/convert_aerialperson.py --raw datasets/AerialPerson_raw --out datasets/AerialPerson
"""
import argparse
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".JPG", ".JPEG", ".PNG"}
# Source split -> this project's split. Test carries real labels, so it is
# training data rather than a second evaluation set.
SPLIT_MAP = {"Train": "train", "Test": "train", "Valid": "val"}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--raw", default="datasets/AerialPerson_raw",
                   help="Directory holding Images.zip and Annotations.zip")
    p.add_argument("--out", default="datasets/AerialPerson",
                   help="Destination root (images/ and labels/ are created under it)")
    p.add_argument("--overwrite", action="store_true",
                   help="Rewrite files that already exist instead of skipping them")
    return p.parse_args()


def index_labels(ann_zip: Path):
    """stem -> (split, label text), for the YOLO annotation files only."""
    out = {}
    with zipfile.ZipFile(ann_zip) as z:
        for name in z.namelist():
            parts = name.split("/")
            # Annotations/Yolo/<Split>/<stem>.txt -- the archive also carries
            # VOC (.xml) and COCO (.json) copies of the same boxes.
            if len(parts) < 4 or parts[1] != "Yolo" or not name.endswith(".txt"):
                continue
            split = SPLIT_MAP.get(parts[2])
            if split is None:
                continue
            out[Path(name).stem] = (split, z.read(name).decode("utf-8", "replace"))
    return out


def clean_label(text: str, stats: Counter):
    """Validate every box and return the lines to write.

    Anything malformed, out of range, or not class 0 is dropped and counted --
    an unnoticed taxonomy mismatch here would look like a training regression
    with no obvious cause, so it is worth being loud about.
    """
    lines = []
    for raw in text.splitlines():
        parts = raw.split()
        if not parts:
            continue
        if len(parts) != 5:
            stats["malformed"] += 1
            continue
        try:
            cls = int(float(parts[0]))
            cx, cy, bw, bh = (float(v) for v in parts[1:])
        except ValueError:
            stats["malformed"] += 1
            continue
        if cls != 0:
            # The record documents exactly one class. Anything else means the
            # source layout changed and the mapping needs rechecking.
            stats["unexpected_class"] += 1
            continue
        if not (0.0 <= cx <= 1.0 and 0.0 <= cy <= 1.0):
            stats["out_of_range"] += 1
            continue
        if bw <= 0 or bh <= 0 or bw > 1.0 or bh > 1.0:
            stats["degenerate"] += 1
            continue
        stats["boxes"] += 1
        lines.append("0 {:.6f} {:.6f} {:.6f} {:.6f}".format(cx, cy, bw, bh))
    return lines


def main():
    args = parse_args()
    raw = Path(args.raw)
    out = Path(args.out)
    img_zip, ann_zip = raw / "Images.zip", raw / "Annotations.zip"
    for path in (img_zip, ann_zip):
        if not path.exists():
            raise SystemExit(
                "Missing {}. Download it from Zenodo record 7740081 first "
                "(see this file's docstring).".format(path))

    labels = index_labels(ann_zip)
    print("YOLO label files found: {}".format(len(labels)))
    if not labels:
        raise SystemExit("No Annotations/Yolo/<Split>/*.txt entries in the archive; "
                         "the source layout has changed.")

    for split in ("train", "val"):
        (out / "images" / split).mkdir(parents=True, exist_ok=True)
        (out / "labels" / split).mkdir(parents=True, exist_ok=True)

    stats = Counter()
    per_split = defaultdict(int)
    unmatched_images = 0

    with zipfile.ZipFile(img_zip) as z:
        for name in z.namelist():
            if name.endswith("/"):
                continue
            stem = Path(name).stem
            if Path(name).suffix not in IMAGE_EXTS:
                continue
            entry = labels.get(stem)
            if entry is None:
                unmatched_images += 1
                continue
            split, text = entry
            lines = clean_label(text, stats)
            if not lines:
                # A frame with no people in it is a legitimate background
                # image and worth keeping, but only if the source really said
                # so -- an empty file after dropping malformed rows is not the
                # same thing and would teach the model to miss real people.
                if text.strip():
                    stats["dropped_all_boxes"] += 1
                    continue
                stats["background_images"] += 1

            img_dest = out / "images" / split / Path(name).name
            lbl_dest = out / "labels" / split / (stem + ".txt")
            if args.overwrite or not img_dest.exists():
                img_dest.write_bytes(z.read(name))
            if args.overwrite or not lbl_dest.exists():
                lbl_dest.write_text("\n".join(lines) + ("\n" if lines else ""),
                                    encoding="utf-8")
            per_split[split] += 1
            stats["images"] += 1

    print("\nWrote to {}".format(out))
    for split in ("train", "val"):
        print("  {:<6} {} images".format(split, per_split[split]))
    print("  boxes kept          : {}".format(stats["boxes"]))
    print("  background images   : {}".format(stats["background_images"]))
    for key in ("malformed", "unexpected_class", "out_of_range", "degenerate",
                "dropped_all_boxes"):
        if stats[key]:
            print("  {:<20}: {}".format(key, stats[key]))
    if unmatched_images:
        print("  images with no label: {} (skipped)".format(unmatched_images))
    if stats["unexpected_class"]:
        print("\nWARNING: non-zero class ids were present. This dataset is "
              "documented as single-class; recheck the taxonomy before training.")


if __name__ == "__main__":
    main()
