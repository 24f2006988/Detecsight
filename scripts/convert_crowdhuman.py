"""Convert CrowdHuman annotations to YOLO format, folded into the personnel class.

See ENGINEERING_LOG.md for the measurements behind this.
"""
import argparse
import json
import os
import shutil
from collections import Counter
from pathlib import Path

PERSONNEL_ID = 0  # BattleSight class 0, per data/battlesight.yaml

# Recorded from the real annotation files on 2026-09-03. Used only to warn --
# the script never silently "corrects" itself to hit these.
EXPECTED = {"train": (15000, 339565), "val": (4370, 99481)}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--raw", default="datasets/CrowdHuman_raw",
                   help="Directory holding Images/ and the .odgt files")
    p.add_argument("--out", default="datasets/CrowdHuman",
                   help="Destination root (images/ and labels/ are created under it)")
    p.add_argument("--splits", nargs="+", default=["train", "val"],
                   choices=["train", "val"], help="Which splits to convert")
    p.add_argument("--limit", type=int, default=0,
                   help="Convert at most N images per split (0 = all), for spot checks")
    p.add_argument("--link", action="store_true",
                   help="Hardlink images instead of copying them. Same NTFS volume "
                        "only; saves ~11 GB and is near-instant, but the converted "
                        "tree then shares bytes with CrowdHuman_raw.")
    p.add_argument("--overwrite", action="store_true",
                   help="Rewrite images that already exist instead of skipping them")
    return p.parse_args()


def image_size(path: Path):
    """(width, height) from the header alone -- do not decode 19,370 JPEGs."""
    try:
        from PIL import Image
        with Image.open(path) as im:
            return im.size
    except Exception:
        pass
    try:  # Pillow refuses some CrowdHuman files; cv2 is more permissive
        import cv2
        img = cv2.imread(str(path))
        if img is not None:
            h, w = img.shape[:2]
            return w, h
    except Exception:
        pass
    return None


def convert_boxes(gtboxes, w, h, stats: Counter):
    """CrowdHuman gtboxes -> YOLO lines, applying the drop rules above."""
    out = []
    for b in gtboxes:
        tag = b.get("tag")
        if tag != "person":
            stats["drop_mask" if tag == "mask" else "drop_other_tag"] += 1
            continue
        # `ignore` is absent on ordinary boxes, so default rather than assume 0.
        if int((b.get("extra") or {}).get("ignore", 0)) == 1:
            stats["drop_ignore"] += 1
            continue

        fbox = b.get("fbox")
        if not fbox or len(fbox) != 4:
            stats["drop_malformed"] += 1
            continue

        x, y, bw, bh = (float(v) for v in fbox)
        x1, y1, x2, y2 = x, y, x + bw, y + bh
        if x2 <= 0 or y2 <= 0 or x1 >= w or y1 >= h:
            stats["drop_outside_frame"] += 1
            continue

        cx1, cy1 = max(0.0, x1), max(0.0, y1)
        cx2, cy2 = min(float(w), x2), min(float(h), y2)
        if (cx1, cy1, cx2, cy2) != (x1, y1, x2, y2):
            stats["clamped"] += 1

        nw, nh = cx2 - cx1, cy2 - cy1
        if nw <= 0 or nh <= 0:
            stats["drop_degenerate_after_clamp"] += 1
            continue

        out.append("{} {:.6f} {:.6f} {:.6f} {:.6f}".format(
            PERSONNEL_ID, (cx1 + cx2) / 2 / w, (cy1 + cy2) / 2 / h, nw / w, nh / h))
        stats["kept"] += 1
    return out


def place_image(src: Path, dst: Path, link: bool, overwrite: bool, stats: Counter):
    if dst.exists() and not overwrite:
        stats["image_reused"] += 1
        return
    if dst.exists():
        dst.unlink()
    if link:
        try:
            os.link(src, dst)
            stats["image_linked"] += 1
            return
        except OSError:
            stats["link_fell_back_to_copy"] += 1
    shutil.copy(src, dst)
    stats["image_copied"] += 1


def convert_split(split, raw: Path, out: Path, limit, link, overwrite):
    ann = raw / f"annotation_{split}.odgt"
    if not ann.exists():
        print(f"  [skip] {split}: {ann} not found")
        return

    img_src_dir = raw / "Images"
    img_out = out / "images" / split
    lbl_out = out / "labels" / split
    img_out.mkdir(parents=True, exist_ok=True)
    lbl_out.mkdir(parents=True, exist_ok=True)

    stats = Counter()
    written = 0

    with open(ann, encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            if limit and written >= limit:
                break
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                # A truncated final line means the download is still in flight.
                print(f"  [warn] {ann.name}:{lineno} is not valid JSON -- "
                      f"file may be incomplete; stopping at {written} images")
                break

            stats["records"] += 1
            img_id = rec.get("ID")
            if not img_id:
                stats["drop_no_id"] += 1
                continue

            src_img = img_src_dir / f"{img_id}.jpg"
            if not src_img.exists():
                stats["image_missing"] += 1
                continue

            size = image_size(src_img)
            if size is None:
                stats["image_unreadable"] += 1
                continue
            w, h = size
            if w <= 0 or h <= 0:
                stats["image_unreadable"] += 1
                continue

            lines = convert_boxes(rec.get("gtboxes", []), w, h, stats)
            if not lines:
                # Never write an empty label: it teaches the model to miss people.
                stats["image_no_boxes_skipped"] += 1
                continue

            (lbl_out / f"{img_id}.txt").write_text("\n".join(lines) + "\n")
            place_image(src_img, img_out / f"{img_id}.jpg", link, overwrite, stats)
            written += 1

            if written % 1000 == 0:
                print(f"    {written} images...", flush=True)

    if written:
        print(f"  {split}: {written} images, {stats['kept']} personnel boxes "
              f"({stats['kept'] / written:.1f}/image)")
    else:
        print(f"  {split}: 0 images written")
    for k in ("drop_mask", "drop_ignore", "drop_outside_frame",
              "drop_degenerate_after_clamp", "drop_malformed", "drop_other_tag",
              "clamped", "image_missing", "image_unreadable",
              "image_no_boxes_skipped", "image_copied", "image_linked",
              "image_reused", "link_fell_back_to_copy"):
        if stats[k]:
            print(f"      {k}: {stats[k]}")

    if not limit and split in EXPECTED:
        exp_imgs, exp_boxes = EXPECTED[split]
        if stats["image_missing"]:
            print(f"      [warn] {stats['image_missing']} images referenced by the "
                  f".odgt are not on disk -- is the extract complete?")
        elif written != exp_imgs or stats["kept"] != exp_boxes:
            print(f"      [warn] expected {exp_imgs} images / {exp_boxes} boxes, "
                  f"got {written} / {stats['kept']}")
        else:
            print("      counts match the recorded expectation exactly")


def main():
    args = parse_args()
    raw, out = Path(args.raw), Path(args.out)
    print("Converting CrowdHuman -> BattleSight personnel-only labels")
    print(f"  raw={raw}  out={out}  images={'hardlink' if args.link else 'copy'}")
    for split in args.splits:
        convert_split(split, raw, out, args.limit, args.link, args.overwrite)
    print("Done.")
    print("\nNext: measure the unlabelled-vehicle contradiction BEFORE training --")
    print("  python scripts/pseudo_label_vehicles.py --dataset datasets/CrowdHuman \\")
    print("      --weights weights/best.pt --splits train --dry-run")


if __name__ == "__main__":
    main()
