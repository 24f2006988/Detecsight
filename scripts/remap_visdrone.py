"""Collapse VisDrone's 10 classes into 4 BattleSight tactical classes.

Originals are backed up to labels_visdrone_raw/ on first run, so this
script is safe to run more than once — it always remaps FROM the backup.
"""
import shutil
from pathlib import Path

DATASET_ROOT = Path("datasets/VisDrone")
SPLITS = ["train", "val", "test"]

# VisDrone class id  ->  BattleSight class id
CLASS_MAP = {
    0: 0,   # pedestrian       -> personnel
    1: 0,   # people           -> personnel
    2: 1,   # bicycle          -> two_wheeler
    3: 2,   # car              -> light_vehicle
    4: 2,   # van              -> light_vehicle
    5: 3,   # truck            -> heavy_vehicle
    6: 1,   # tricycle         -> two_wheeler
    7: 1,   # awning-tricycle  -> two_wheeler
    8: 3,   # bus              -> heavy_vehicle
    9: 1,   # motor            -> two_wheeler
}

NEW_NAMES = ["personnel", "two_wheeler", "light_vehicle", "heavy_vehicle"]


def backup_originals():
    """Copy labels/ -> labels_visdrone_raw/ once, so remapping is repeatable."""
    src = DATASET_ROOT / "labels"
    dst = DATASET_ROOT / "labels_visdrone_raw"
    if dst.exists():
        print(f"Backup already exists at {dst} — using it as the source.")
        return dst
    print(f"Backing up {src} -> {dst} ...")
    shutil.copytree(src, dst)
    return dst


def remap_split(source_dir, target_dir, split):
    src = source_dir / split
    dst = target_dir / split
    if not src.exists():
        print(f"  [skip] {split}: no labels found at {src}")
        return

    dst.mkdir(parents=True, exist_ok=True)
    counts = {i: 0 for i in range(len(NEW_NAMES))}
    files = 0

    for label_file in src.glob("*.txt"):
        out_lines = []
        for line in label_file.read_text().strip().splitlines():
            parts = line.split()
            if len(parts) != 5:
                continue
            old_id = int(parts[0])
            if old_id not in CLASS_MAP:
                continue
            new_id = CLASS_MAP[old_id]
            counts[new_id] += 1
            out_lines.append(" ".join([str(new_id)] + parts[1:]))

        (dst / label_file.name).write_text("\n".join(out_lines))
        files += 1

    print(f"  {split}: {files} label files written")
    for i, name in enumerate(NEW_NAMES):
        print(f"    {i} {name:<15} {counts[i]:>7} boxes")


if __name__ == "__main__":
    source = backup_originals()
    target = DATASET_ROOT / "labels"
    print("Remapping to 4 tactical classes...")
    for split in SPLITS:
        remap_split(source, target, split)
    print("Done.")
