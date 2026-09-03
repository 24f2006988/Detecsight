"""Convert WiderPerson annotations to YOLO format, folded into the personnel class.

Expects the official WiderPerson layout extracted to datasets/WiderPerson_raw/:
    Images/<id>.jpg
    Annotations/<id>.jpg.txt   (first line = box count, then "class x1 y1 x2 y2")
    train.txt / val.txt        (one image id per line; WiderPerson's test split
                                 ships with no ground truth, so it's not used)

WiderPerson classes: 1 pedestrian, 2 rider, 3 partially-visible person,
4 ignore region, 5 crowd. Classes 1-3 all become BattleSight class 0
(personnel) since a rider annotation is still a person-shaped box, not a
vehicle. 4 and 5 have no reliable per-instance box and are dropped.
"""
import shutil
from pathlib import Path

import cv2

RAW_ROOT = Path("datasets/WiderPerson_raw")
OUT_ROOT = Path("datasets/WiderPerson")
SPLITS = ["train", "val"]

PERSON_CLASSES = {1, 2, 3}
PERSONNEL_ID = 0  # BattleSight class 0, per data/battlesight.yaml


def convert_split(split):
    id_list = RAW_ROOT / f"{split}.txt"
    if not id_list.exists():
        print(f"  [skip] {split}: {id_list} not found")
        return

    img_out = OUT_ROOT / "images" / split
    lbl_out = OUT_ROOT / "labels" / split
    img_out.mkdir(parents=True, exist_ok=True)
    lbl_out.mkdir(parents=True, exist_ok=True)

    ids = [line.strip() for line in id_list.read_text().splitlines() if line.strip()]
    written, boxes = 0, 0

    for img_id in ids:
        src_img = RAW_ROOT / "Images" / f"{img_id}.jpg"
        ann_file = RAW_ROOT / "Annotations" / f"{img_id}.jpg.txt"
        if not src_img.exists() or not ann_file.exists():
            continue

        img = cv2.imread(str(src_img))
        if img is None:
            continue
        h, w = img.shape[:2]

        lines = ann_file.read_text().strip().splitlines()
        out_lines = []
        for line in lines[1:]:  # first line is the box count, not a box
            parts = line.split()
            if len(parts) != 5:
                continue
            cls, x1, y1, x2, y2 = map(float, parts)
            if int(cls) not in PERSON_CLASSES:
                continue
            cx = ((x1 + x2) / 2) / w
            cy = ((y1 + y2) / 2) / h
            bw = (x2 - x1) / w
            bh = (y2 - y1) / h
            if bw <= 0 or bh <= 0:
                continue
            out_lines.append(f"{PERSONNEL_ID} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
            boxes += 1

        (lbl_out / f"{img_id}.txt").write_text("\n".join(out_lines))
        shutil.copy(src_img, img_out / f"{img_id}.jpg")
        written += 1

    print(f"  {split}: {written} images, {boxes} personnel boxes")


if __name__ == "__main__":
    print("Converting WiderPerson -> BattleSight personnel-only labels...")
    for split in SPLITS:
        convert_split(split)
    print("Done.")
