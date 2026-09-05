"""Convert SARD (Search And Rescue Dataset) to this project's dataset layout.

Source: https://www.kaggle.com/datasets/nikolasgegenava/sard-search-and-rescue
Paper:  Sambolek & Ivasic-Kos, "Automatic Person Detection in Search and
        Rescue Operations Using Deep CNN Detectors".

Actors simulate exhausted and injured people -- running, walking, standing,
sitting and LYING DOWN -- filmed from a drone over macadam roads, quarries,
low and high grass, and forest shade.

WHY THIS DATASET. The one genuinely open failure in this project is personnel
missed in vegetation from a UAV (log, "Still outstanding"). Resolution, palette,
viewpoint and augmentation were each ruled out by direct experiment; what
remained was POSE, and nothing in the training mix contained a prone or
crawling person seen from above. This is that missing distribution.

The scale matches the real failure, which is the part that matters. On v10 the
missed person measures 50-60 px and the diagnosis explicitly records "not
resolution -- the people are 50-60 px, not tiny". SARD's median person box is
55.6 px. This dataset is aimed at the POSE gap, NOT at the <32 px far-field
recall gap (0.187) -- those are separate problems with separate fixes, and
expecting this data to move the size-stratified curve will lead to a wrong
conclusion about whether it worked.

DIFFICULTY, measured rather than assumed: a stock COCO yolo26n over 300 of
these tiles found 40 people where SARD labels roughly 400. A strong general
detector misses ~90% of them. That is the gap, and it is also the baseline.

UNLABELLED-VEHICLE CHECK -- ALREADY DONE, and it passes. Every person-only
dataset merged into this 4-class taxonomy risks presenting its unlabelled
vehicles to the trainer as confirmed negatives; that is what
scripts/pseudo_label_vehicles.py exists to prevent, after AerialPerson was
measured at 97.3 unlabelled vehicles per image. Measured here with a stock COCO
control over 300 tiles: 0.060 vehicles/image, 4.7% of images. Far below the
~1/image threshold at which pseudo-labelling is worth its false labels, so
SARD is merged AS-IS. Do not pseudo-label it.

INPUT. The Kaggle download is a Roboflow export, already in YOLO format, and is
read straight out of the zip -- there is nothing to unpack by hand:

    datasets/SARD_raw/sard.zip
        search-and-rescue/data.yaml            nc: 1, names: ['human']
        search-and-rescue/{train,valid,test}/images/*.jpg
        search-and-rescue/{train,valid,test}/labels/*.txt

It is the TILES_3x3 build: each 640x640 image is a tile of a larger source
frame, ~2.9 tiles per frame. Verified that no source frame appears in more than
one split, so the tiling introduces no train/test leakage.

CONVERSION DECISIONS

  * Class 0 'human' -> class 0 'personnel'. An identity map, but it is VERIFIED
    rather than assumed: any label carrying a non-zero class id aborts the run,
    because silently folding an unexpected class into personnel is the kind of
    error that only shows up as an unexplained metric after a 7-hour run.
  * Roboflow's 'valid' is renamed 'val', which is what every other dataset in
    this project uses.
  * EMPTY LABEL FILES ARE KEPT, and this deliberately departs from the rule
    convert_aerialperson.py and convert_crowdhuman.py follow. Their rule exists
    because an image whose boxes were DROPPED, but which still contains people,
    teaches the model to miss real people. That is not this case: 14.5% of SARD
    tiles genuinely contain no person, because tiling a frame produces empty
    tiles. They are true negatives from exactly the domain where this model
    hallucinates -- the v10 run put 245 phantom personnel boxes on vegetation
    and zero on the person -- so they are the hard negatives that failure calls
    for. Pass --drop-empty to exclude them.
  * The test split is converted but is NOT for training. It is the held-out
    on-domain set for the one failure this data exists to fix.

EXPECTED COUNTS, as gates rather than estimates (read from the archive):

    split   images   boxes   empty labels
    train     4041    5229            571
    val       1144    1463            177
    test       570     732             84

An order of magnitude off means the archive changed under you.

    python scripts/convert_sard.py
    python scripts/convert_sard.py --drop-empty --out datasets/SARD_nonneg
"""
import argparse
import zipfile
from collections import Counter
from pathlib import Path

ROOT = "search-and-rescue"
SPLIT_MAP = {"train": "train", "valid": "val", "test": "test"}
EXPECTED = {
    "train": {"images": 4041, "boxes": 5229, "empty": 571},
    "val": {"images": 1144, "boxes": 1463, "empty": 177},
    "test": {"images": 570, "boxes": 732, "empty": 84},
}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--zip", default="datasets/SARD_raw/sard.zip",
                   help="The Kaggle/Roboflow archive, read in place")
    p.add_argument("--out", default="datasets/SARD",
                   help="Destination in this project's layout")
    p.add_argument("--drop-empty", action="store_true",
                   help="Exclude tiles with no person. Default is to KEEP them: "
                        "they are true negatives from the domain where this "
                        "model hallucinates people in vegetation.")
    p.add_argument("--dry-run", action="store_true", help="Report only, write nothing")
    return p.parse_args()


def main():
    args = parse_args()
    zpath = Path(args.zip)
    if not zpath.exists():
        raise SystemExit(
            f"{zpath} not found.\n"
            f"Download it from "
            f"https://www.kaggle.com/datasets/nikolasgegenava/sard-search-and-rescue "
            f"(no credentials needed):\n"
            f"  curl -L -o {zpath} "
            f"https://www.kaggle.com/api/v1/datasets/download/nikolasgegenava/sard-search-and-rescue"
        )

    out = Path(args.out)
    z = zipfile.ZipFile(zpath)
    names = z.namelist()

    stats = {v: Counter() for v in SPLIT_MAP.values()}
    written = 0

    for src_split, dst_split in SPLIT_MAP.items():
        prefix = f"{ROOT}/{src_split}/images/"
        imgs = sorted(x for x in names if x.startswith(prefix) and x.endswith(".jpg"))
        if not imgs:
            raise SystemExit(f"no images under {prefix} -- archive layout changed")

        img_dir = out / "images" / dst_split
        lbl_dir = out / "labels" / dst_split
        if not args.dry_run:
            img_dir.mkdir(parents=True, exist_ok=True)
            lbl_dir.mkdir(parents=True, exist_ok=True)

        for ip in imgs:
            stem = Path(ip).stem
            lp = f"{ROOT}/{src_split}/labels/{stem}.txt"
            try:
                raw = z.read(lp).decode().strip()
            except KeyError:
                # No label file at all is not the same as an empty one: it means
                # the archive is inconsistent, so say so rather than inventing
                # an empty label and calling the image a negative.
                stats[dst_split]["missing_label"] += 1
                continue

            lines = []
            for line in raw.split("\n"):
                q = line.split()
                if not q:
                    continue
                if len(q) != 5:
                    raise SystemExit(f"{lp}: expected 5 columns, got {len(q)}: {line!r}")
                if q[0] != "0":
                    raise SystemExit(
                        f"{lp}: class id {q[0]!r}, expected only '0' (human).\n"
                        f"The archive carries a class this converter does not know how "
                        f"to map; folding it into personnel silently would corrupt the "
                        f"training set."
                    )
                lines.append(" ".join(q))

            if not lines:
                stats[dst_split]["empty"] += 1
                if args.drop_empty:
                    continue

            stats[dst_split]["images"] += 1
            stats[dst_split]["boxes"] += len(lines)

            if not args.dry_run:
                (img_dir / f"{stem}.jpg").write_bytes(z.read(ip))
                (lbl_dir / f"{stem}.txt").write_text(
                    ("\n".join(lines) + "\n") if lines else "")
            written += 1

    print(f"\n{'split':<8}{'images':>9}{'boxes':>9}{'empty':>9}   expected")
    ok = True
    for dst in ("train", "val", "test"):
        s = stats[dst]
        e = EXPECTED[dst]
        # With --drop-empty the image count legitimately falls by the empty count.
        want_imgs = e["images"] - (e["empty"] if args.drop_empty else 0)
        flag = "" if (s["boxes"] == e["boxes"] and s["images"] == want_imgs) else "  <-- MISMATCH"
        if flag:
            ok = False
        print(f"{dst:<8}{s['images']:>9}{s['boxes']:>9}{s['empty']:>9}   "
              f"{want_imgs}/{e['boxes']}{flag}")
        if s["missing_label"]:
            print(f"         {s['missing_label']} images had NO label file and were skipped")

    if not ok:
        print("\nCounts differ from the recorded archive. Check the download before training.")
    if args.dry_run:
        print("\nDRY RUN -- nothing written.")
    else:
        print(f"\nwrote {written} image/label pairs to {out}")
        print("\nRegister it by adding to SOURCES in scripts/prepare_training.py:")
        print('    ("SARD train", "SARD/images/train", "train",')
        print('     "prone and non-upright personnel from a UAV -- the missing pose"),')
        print("\nKeep SARD/images/test OUT of training: it is the held-out set for")
        print("the vegetation/pose failure this data exists to fix.")


if __name__ == "__main__":
    main()
