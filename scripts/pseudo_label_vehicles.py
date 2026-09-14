"""Add vehicle pseudo-labels to a personnel-only dataset, so mixing it in
doesn't teach the model that vehicles are background.

See ENGINEERING_LOG.md for the measurements behind this.
"""
import argparse
import shutil
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
CLASS_NAMES = ["personnel", "two_wheeler", "light_vehicle", "heavy_vehicle"]
VEHICLE_CLASSES = (1, 2, 3)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", default="datasets/AerialPerson")
    p.add_argument("--splits", nargs="+", default=["train", "val"])
    p.add_argument("--weights", default="weights/drone_best.pt",
                   help="Checkpoint used to propose vehicles. The drone-view model "
                        "by default, since this imagery is top-down aerial.")
    p.add_argument("--conf", type=float, default=0.5,
                   help="Confidence floor for a pseudo-label. Deliberately well "
                        "above serving's 0.25: a false positive here becomes a "
                        "permanent wrong label, so precision beats recall.")
    p.add_argument("--imgsz", type=int, default=1280)
    p.add_argument("--iou-person", type=float, default=0.3,
                   help="Drop a pseudo-box overlapping a ground-truth person by "
                        "more than this. The human label always wins.")
    p.add_argument("--dry-run", action="store_true",
                   help="Report what would be added without writing anything")
    p.add_argument("--restore", action="store_true",
                   help="Restore the original person-only labels and exit")
    return p.parse_args()


def iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    union = ((ax2 - ax1) * (ay2 - ay1)) + ((bx2 - bx1) * (by2 - by1)) - inter
    return inter / union if union > 0 else 0.0


def to_corners(cx, cy, bw, bh):
    return (cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2)


def restore(dataset: Path, splits):
    restored = 0
    for split in splits:
        backup = dataset / "labels" / (split + ".orig")
        target = dataset / "labels" / split
        if not backup.is_dir():
            print("  no backup for {} -- nothing to restore".format(split))
            continue
        for src in backup.glob("*.txt"):
            shutil.copyfile(src, target / src.name)
            restored += 1
    print("restored {} label files".format(restored))


def main():
    args = parse_args()
    dataset = (REPO / args.dataset) if not Path(args.dataset).is_absolute() else Path(args.dataset)

    if args.restore:
        restore(dataset, args.splits)
        return

    from ultralytics import YOLO
    model = YOLO(str(REPO / args.weights) if not Path(args.weights).is_absolute()
                 else args.weights)

    grand = Counter()
    for split in args.splits:
        img_dir = dataset / "images" / split
        lbl_dir = dataset / "labels" / split
        if not img_dir.is_dir():
            print("skip {} (no images)".format(split))
            continue

        backup = dataset / "labels" / (split + ".orig")
        if not args.dry_run and not backup.exists():
            # Back up ONCE. Re-running must not overwrite the backup with
            # already-pseudo-labelled files, or the originals are gone.
            shutil.copytree(lbl_dir, backup)
            print("backed up original labels -> {}".format(backup))

        source = backup if backup.is_dir() else lbl_dir
        images = sorted(p for p in img_dir.iterdir() if p.is_file())
        print("\n{}: {} images".format(split, len(images)))

        stats = Counter()
        for n, img in enumerate(images, 1):
            lbl = source / (img.stem + ".txt")
            person_rows, person_boxes = [], []
            if lbl.exists():
                for line in lbl.read_text(encoding="utf-8").splitlines():
                    parts = line.split()
                    if len(parts) != 5:
                        continue
                    person_rows.append(line.strip())
                    _, cx, cy, bw, bh = parts
                    person_boxes.append(to_corners(float(cx), float(cy),
                                                   float(bw), float(bh)))

            result = model.predict(str(img), imgsz=args.imgsz, conf=args.conf,
                                   verbose=False, max_det=500)[0]
            added = []
            for cls, xywhn in zip(result.boxes.cls.cpu().numpy().astype(int),
                                  result.boxes.xywhn.cpu().numpy(), strict=True):
                if int(cls) not in VEHICLE_CLASSES:
                    stats["skipped_non_vehicle"] += 1
                    continue
                cx, cy, bw, bh = (float(v) for v in xywhn)
                if bw <= 0 or bh <= 0:
                    continue
                corners = to_corners(cx, cy, bw, bh)
                if any(iou(corners, pb) > args.iou_person for pb in person_boxes):
                    # A human said this is a person. Do not overrule them.
                    stats["dropped_overlaps_person"] += 1
                    continue
                added.append("{} {:.6f} {:.6f} {:.6f} {:.6f}".format(
                    int(cls), cx, cy, bw, bh))
                stats[CLASS_NAMES[int(cls)]] += 1

            stats["person_boxes_kept"] += len(person_rows)
            if not args.dry_run:
                out = person_rows + added
                (lbl_dir / (img.stem + ".txt")).write_text(
                    "\n".join(out) + ("\n" if out else ""), encoding="utf-8")
            if n % 250 == 0:
                print("  {}/{}".format(n, len(images)), flush=True)

        print("  person boxes kept      : {}".format(stats["person_boxes_kept"]))
        for name in CLASS_NAMES[1:]:
            print("  + {:<20} : {}".format(name, stats[name]))
        print("  dropped (overlaps person): {}".format(stats["dropped_overlaps_person"]))
        grand.update(stats)

    print("\n" + "=" * 60)
    if args.dry_run:
        print("DRY RUN -- nothing written.")
    print("person boxes preserved : {}".format(grand["person_boxes_kept"]))
    print("vehicle pseudo-labels  : {}".format(
        sum(grand[n] for n in CLASS_NAMES[1:])))
    print("\nUndo at any time with:  python scripts/pseudo_label_vehicles.py --restore")


if __name__ == "__main__":
    main()
