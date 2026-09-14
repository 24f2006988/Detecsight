"""Get the training set ready, then print the exact command to launch.

The point of this script is that the training run can be started at any time
without first working out what state the data is in. The AerialPerson download
(Zenodo 7740081, ~3.7 GB) is slow on this network and may or may not have
finished; this checks, converts it if it has, and writes a dataset yaml naming
only the sources that actually exist on disk. Ultralytics fails hard on a yaml
that lists a missing path, so generating it from reality rather than hope is
what makes "start it whenever" safe.

    python scripts/prepare_training.py            # check, convert, write yaml
    python scripts/prepare_training.py --imgsz 960 --batch 8

It never starts training itself -- it prints the command. Launching a multi-hour
GPU job is the user's call, not a side effect of a status check.
"""
import argparse
import os
import subprocess
import sys
import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
YAML_PATH = REPO / "data" / "battlesight_fpv.yaml"

# The dataset yamls are anchored to VisDrone and reach its siblings with `../`,
# so the generated file carries no absolute path at all -- see _yaml_entry().
# VisDrone is the anchor because it is the only source every yaml uses.
ANCHOR = "VisDrone"


def _datasets_root() -> Path:
    """Where the datasets actually live.

    Deliberately NOT repo-relative. They are large, shared between projects and
    read across a folder boundary, so the location is a property of the machine
    rather than of this checkout. Resolution order:

      1. BATTLESIGHT_DATASETS, for an explicit one-off override.
      2. ultralytics' own `datasets_dir` setting -- which is already what
         data/battlesight.yaml resolves its relative `path:` against, so
         honouring it here keeps every yaml pointing at the same root.
      3. REPO/datasets, only as a last resort if neither is set.

    Set it once with:  yolo settings datasets_dir="<path>"
    """
    env = os.getenv("BATTLESIGHT_DATASETS")
    if env:
        return Path(env)
    try:
        from ultralytics.utils import SETTINGS
        configured = SETTINGS.get("datasets_dir")
        if configured:
            return Path(configured)
    except Exception:  # noqa: BLE001
        pass  # ultralytics not importable or settings unreadable; fall through
    return REPO / "datasets"


def _yaml_entry(rel: str) -> str:
    """Turn a datasets-root-relative path into one relative to ANCHOR.

    ultralytics resolves a yaml's `path:` against its datasets_dir when the
    value does not exist relative to cwd, then joins each split onto it and
    calls .resolve() -- which collapses `..` normally. So anchoring at VisDrone
    and climbing back out with `../` reaches every sibling dataset without ever
    naming a drive or an absolute directory, and the file stays valid on any
    machine whose datasets_dir is set.

    Note `path: .` does NOT work for this: "." always exists relative to cwd,
    so ultralytics keeps it and never consults datasets_dir. Verified against
    check_det_dataset() in ultralytics 8.4.135.
    """
    prefix = ANCHOR + "/"
    return rel[len(prefix):] if rel.startswith(prefix) else "../" + rel


DATASETS = _datasets_root()

# (label, path relative to DATASETS, split, note)
SOURCES = [
    ("VisDrone train", "VisDrone/images/train", "train", "6471, aerial, 4 classes"),
    ("VisDrone test-dev", "VisDrone/images/test", "train",
     "1610 labelled, unused by train and val before this"),
    ("WiderPerson train", "WiderPerson/images/train", "train", "8000, ground personnel"),
    ("AerialPerson train", "AerialPerson/images/train", "train",
     "UAV personnel over natural terrain -- the missing domain"),
    # CrowdHuman is TEMPORARILY OUT of the mix, 2026-09-05. Two reasons, and
    # the second is the better one:
    #
    # 1. It broke the machine. With CrowdHuman in, the blend is 37,735 images
    #    and ~970,000 boxes, and training died of SYSTEM RAM exhaustion (32 GB)
    #    partway through epoch 1. Windows multiprocessing spawns rather than
    #    forks, so every dataloader worker holds a FULL copy of the label
    #    arrays -- CrowdHuman alone contributes 339,565 boxes over 15,000
    #    images, about 35% of the total. Lowering --workers helps but the disk
    #    here reads at ~15 MB/s, so starving the loader has its own cost.
    #
    # 2. Leaving it out makes this a SINGLE-VARIABLE run. SARD is then the only
    #    change against the deployed checkpoint's mix, so if the result moves,
    #    the cause is not ambiguous. Section 21 is a fresh reminder of what
    #    ambiguity costs.
    #
    # It also avoids compounding a known problem: CrowdHuman is personnel-only
    # and shifts the class balance from 44.8% to 59.8% personnel, diluting the
    # vehicle classes from 46.1% to 33.5% -- exactly when section 20 found
    # ground-level vehicle recall already collapsed.
    #
    # Put it back as its own run once SARD is settled. It is converted,
    # registered and measured clean (0.047 unlabelled vehicles/image, so no
    # pseudo-labelling needed); only this line is in the way.
    # ("CrowdHuman train", "CrowdHuman/images/train", "train",
    #  "15000, dense occluded ground-level personnel"),
    # SARD, added 2026-09-05 after log section 21. Prone and non-upright
    # personnel seen from a UAV over grass, forest shade and quarries -- the
    # pose distribution nothing else in this mix contains. Fine-tuning on it
    # ALONE took its own held-out recall 0.143 -> 0.870 and then failed every
    # promotion criterion by catastrophic forgetting (personnel mAP50
    # 0.706 -> 0.221). It belongs in the blend, not on its own: at 4,041 of
    # ~37,700 images it cannot dominate, and every other class keeps receiving
    # positive examples throughout.
    #
    # SARD's val AND test splits are deliberately absent from the val list, for
    # the same reason AerialPerson's is: the val set must stay byte-identical to
    # battlesight_multi.yaml's, or every mAP figure recorded in the log stops
    # being comparable. SARD performance is tracked separately and explicitly,
    # with scripts/eval_size_recall.py against SARD/images/test -- which is
    # where the 0.143 baseline was measured and where the verdict is read.
    ("SARD train", "SARD/images/train", "train",
     "4041 tiles, prone/non-upright personnel from a UAV -- the missing pose"),
    # BDD100K, added 2026-09-07 after log section 20. That section measured the
    # failure and named the cause: `personnel` has aerial AND ground-level
    # training data, the three VEHICLE classes have aerial data only. VisDrone
    # is their sole source and AerialPerson's pseudo-labels are aerial too, so
    # the model has learned "a vehicle is a small object seen from above" and a
    # large, close, horizontal-view car is off-distribution for the vehicle
    # classes specifically. Re-measured 2026-09-07 on the current post-SARD
    # checkpoint over shibuya frames 500-559, ~6 vehicles continuously present:
    # deployed 0.27 vehicles/frame at conf 0.25, 6.35 at conf 0.02, against a
    # stock COCO control at 5.60. It localises them and kills them at the
    # threshold, which is what a coverage gap looks like, not a capacity one.
    #
    # This is the first ground-level vehicle data in the mix. It is also the
    # first night (39%) and adverse-weather (15%) data, and the first genuine
    # empty-road negatives from a forward-facing camera -- which is what
    # sections 14 and 16 wanted for the v11 phantoms and could not get from
    # VisDrone or WiderPerson.
    #
    # 8,500 of ~31,200 images is ~27% of the mix, against SARD's 17.8%. That is
    # a bigger share than anything else added here, so if the SARD result moves
    # this is the first suspect: re-measure SARD test recall and the drone
    # personnel floor (log 22h) after this run, not only the promotion gate.
    # Cap it with `convert_bdd100k.py --max-train 6000` if it does dominate.
    ("BDD100K train", "BDD100K/images/train", "train",
     "8500 dashcam frames at 1280x720 -- the only ground-level vehicle data"),
    ("VisDrone val", "VisDrone/images/val", "val", "548"),
    ("WiderPerson val", "WiderPerson/images/val", "val", "1000"),
    # BDD100K's val split is DELIBERATELY NOT in this list either, for the
    # reason given below: the val set must stay byte-identical to
    # battlesight_multi.yaml's or every mAP figure in the log stops being
    # comparable. Its 1,500 held-out images are the ground-level vehicle val
    # split section 20 called a prerequisite, and they are read separately
    # through data/battlesight_bdd_val.yaml -- the same arrangement SARD test
    # has. Putting them in here would measure the fix and destroy the
    # regression guard in one move.
    # AerialPerson's val split is DELIBERATELY NOT in this list. Its labels are
    # people only, so every correctly-detected car in it would score as a false
    # positive and drag vehicle precision down for no real reason. Pseudo-
    # labelling it instead would be worse: the pseudo-labels come from
    # drone_best.pt, which is the baseline the rubric compares against, so the
    # metric would partly reward agreeing with the model under test.
    #
    # Leaving it out has a second benefit: the val set stays identical to
    # battlesight_multi.yaml's, so eval_rubric.py numbers remain directly
    # comparable to every figure recorded in README's history.
    #
    # The 523 images stay on disk with clean ground truth as a held-out
    # on-domain set -- evaluate personnel on it explicitly when wanted:
    #   yolo val model=... data=... classes=0
]

YAML_HEADER = """\
# BattleSight AR -- generated by scripts/prepare_training.py. Do not hand-edit;
# rerun that script instead, so the file always names paths that exist.
#
# Same 4-class taxonomy as battlesight.yaml / battlesight_multi.yaml.
# battlesight_multi.yaml is deliberately left untouched so the historical
# numbers in README's "Detection accuracy work" stay reproducible against it.
#
# VisDrone's test-dev split is TRAINING data here: all 1,610 of its images
# carry ground-truth labels and were used by neither train nor val, so this is
# a free ~25% increase in VisDrone training data at no cost to any held-out
# set. AerialPerson (Zenodo 7740081, CC-BY-4.0) is the only source containing a
# person seen small from altitude against natural terrain, which is the case
# the deployed model fails hardest on.
#
# AerialPerson contributes to TRAIN ONLY. Its val split is held out of the
# metric on purpose -- it labels people but not the many cars in its aerial
# imagery, so scoring vehicles against it would be meaningless, and pseudo-
# labelling it would partly measure agreement with the baseline model that
# generated those labels. The val set here is therefore identical to
# battlesight_multi.yaml's, which keeps eval_rubric.py numbers directly
# comparable to the figures recorded in README's history.
#
# AerialPerson's TRAIN labels DO carry vehicle pseudo-labels (see
# scripts/pseudo_label_vehicles.py and README 16g) -- without them its ~258,000
# unlabelled cars would train the model to treat aerial vehicles as background.
#
# BDD100K (added 2026-09-07, log section 20) is the only ground-level vehicle
# source here. Every other dataset in the mix teaches vehicles from above, and
# section 20 measured what that costs: 0.27 vehicles/frame on a street at the
# deployed threshold, against 6.35 from the same model at conf 0.02. It needs
# no pseudo-labelling -- BDD annotates cars, trucks, buses, bicycles,
# motorcycles, riders and pedestrians exhaustively, so the AerialPerson trap
# does not apply. Its val split is held out separately as
# data/battlesight_bdd_val.yaml rather than added below.
#
# Paths are anchored at VisDrone and reach its siblings with `../`, so this file
# names no drive and no absolute directory. It resolves against whatever
# `yolo settings datasets_dir` is set to -- set that once per machine.
"""


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--imgsz", type=int, default=1280,
                   help="Training resolution for the printed command. 1280 matches "
                        "config.IMGSZ, i.e. what the model is actually served at; "
                        "the deployed checkpoints were trained at 640, which is a "
                        "real train/serve mismatch on tiny targets.")
    p.add_argument("--batch", type=int, default=4,
                   help="Batch size for the printed command. Verify against the "
                        "VRAM probe before trusting a larger value on this 8 GB GPU.")
    p.add_argument("--epochs", type=int, default=8,
                   help="8, not 40. At imgsz 1280 / batch 4 this machine runs "
                        "~51 min/epoch over the full 18,694-image train set, so 8 "
                        "epochs is about 7 h -- the stated budget. This is a "
                        "fine-tune from a trained checkpoint, not a run from scratch.")
    p.add_argument("--name", default="battlesight_fpv")
    p.add_argument("--skip-convert", action="store_true",
                   help="Do not run convert_aerialperson.py even if the archive looks complete")
    return p.parse_args()


def archive_complete(path: Path) -> bool:
    """True if the zip is fully downloaded and readable.

    A partial file from an interrupted transfer still exists and still has a
    plausible size, so testing existence is not enough -- read the central
    directory, which lives at the END of a zip and is therefore exactly what a
    truncated download lacks.
    """
    if not path.exists():
        return False
    try:
        with zipfile.ZipFile(path) as z:
            return z.testzip() is None or True
    except zipfile.BadZipFile:
        return False


def main():
    args = parse_args()

    raw = DATASETS / "AerialPerson_raw"
    converted = DATASETS / "AerialPerson" / "images" / "train"
    images_zip = raw / "Images.zip"

    print("=" * 72)
    print("AerialPerson (Zenodo 7740081)")
    if converted.exists() and any(converted.iterdir()):
        print("  already converted -> {}".format(converted))
    elif not images_zip.exists():
        print("  Images.zip not downloaded yet -- training will proceed without it.")
    elif not archive_complete(images_zip):
        size_gb = images_zip.stat().st_size / 1e9
        print("  Images.zip is still downloading ({:.2f} GB of ~3.72 GB) --"
              .format(size_gb))
        print("  training will proceed without it. Rerun this script once it lands.")
    elif args.skip_convert:
        print("  archive complete, but --skip-convert was passed")
    else:
        print("  archive complete; converting...")
        result = subprocess.run(
            [sys.executable, str(REPO / "scripts" / "convert_aerialperson.py")],
            cwd=str(REPO))
        if result.returncode != 0:
            print("  conversion FAILED; continuing without this dataset")

    # AerialPerson annotates PEOPLE ONLY, over top-down aerial imagery full of
    # parked cars -- the same viewpoint VisDrone teaches vehicles from. Mixing
    # it in raw presents ~258,000 unlabelled vehicles as confirmed negatives.
    # Refuse to do that silently; see scripts/pseudo_label_vehicles.py.
    if (DATASETS / "AerialPerson" / "images" / "train").is_dir():
        if not (DATASETS / "AerialPerson" / "labels" / "train.orig").is_dir():
            print("=" * 72)
            print("!! WARNING: AerialPerson vehicle pseudo-labels are MISSING.")
            print("   That dataset labels people only, but its aerial imagery is")
            print("   full of unlabelled cars (~97/image measured). Training on it")
            print("   as-is teaches the model that aerial vehicles are background")
            print("   and will damage light_vehicle/heavy_vehicle. Run first:")
            print("       python scripts/pseudo_label_vehicles.py")
            print("   (or pass --skip-convert and drop AerialPerson from the yaml)")

    print("=" * 72)
    print("Dataset sources")
    present = {"train": [], "val": []}
    for label, rel, split, note in SOURCES:
        path = DATASETS / rel
        count = len(list(path.glob("*"))) if path.is_dir() else 0
        if count:
            present[split].append(rel)
            print("  [x] {:<20} {:>6} images  {}".format(label, count, note))
        else:
            print("  [ ] {:<20} {:>6}          MISSING -- omitted from the yaml"
                  .format(label, 0))

    if not present["train"] or not present["val"]:
        raise SystemExit("\nNo usable train or val sources; nothing to write.")

    lines = [YAML_HEADER, "path: {}".format(ANCHOR), "", "train:"]
    lines += ["  - {}".format(_yaml_entry(p)) for p in present["train"]]
    lines += ["", "val:"]
    lines += ["  - {}".format(_yaml_entry(p)) for p in present["val"]]
    lines += ["", "names:", "  0: personnel", "  1: two_wheeler",
              "  2: light_vehicle", "  3: heavy_vehicle", ""]
    YAML_PATH.write_text("\n".join(lines), encoding="utf-8")
    print("\nwrote {}".format(YAML_PATH))

    print("=" * 72)
    print("Start training with:\n")
    print("  python scripts/train.py \\\n"
          "      --model weights/best.pt \\\n"
          "      --data data/battlesight_fpv.yaml \\\n"
          "      --aug-profile fpv \\\n"
          "      --imgsz {} --batch {} --epochs {} \\\n"
          "      --lr0 0.002 --flipud 0 \\\n"
          "      --name {}".format(args.imgsz, args.batch, args.epochs, args.name))
    print("\nThen, before promoting anything:\n")
    print("  python scripts/eval_rubric.py runs/detect/{}/weights/best.pt \\\n"
          "      --baseline weights/best.pt --data data/battlesight_fpv.yaml "
          "--imgsz {}".format(args.name, args.imgsz))


if __name__ == "__main__":
    main()
