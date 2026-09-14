"""Convert BDD100K detection labels to this project's 4-class layout.

See ENGINEERING_LOG.md for the measurements behind this.
"""
import argparse
import json
import os
import random
import shutil
from collections import Counter, defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def datasets_root() -> Path:
    """Where the datasets actually live -- the same resolution order as
    scripts/prepare_training.py and scripts/fetch_bdd100k.py.

    Deliberately NOT repo-relative. There is no `datasets/` directory in this
    checkout; the data is on another drive and shared with another project, so
    a hardcoded relative default would silently resolve to nothing. That is the
    known failure mode for datasets_dir: pointing it at a root that lacks a
    converted dataset drops that dataset from the blend without an error.
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
        pass
    return REPO / "datasets"


DATASETS = datasets_root()
RAW_ROOT = DATASETS / "BDD100K_raw"
OUT_ROOT = DATASETS / "BDD100K"

# BDD100K label -> this project's class id (data/battlesight.yaml)
CLASS_MAP = {
    "pedestrian": 0,   # personnel
    "rider": 0,        # personnel -- a person-shaped box; the bike is separate
    "bicycle": 1,      # two_wheeler
    "motorcycle": 1,   # two_wheeler
    "car": 2,          # light_vehicle
    "truck": 3,        # heavy_vehicle
    "bus": 3,          # heavy_vehicle
}
# Deliberately dropped, and why: not in this taxonomy, or too few to be worth a
# class decision. Listed rather than implied so the omission is reviewable.
DROPPED = {"traffic light", "traffic sign", "train", "trailer",
           "other vehicle", "other person"}

NEW_NAMES = ["personnel", "two_wheeler", "light_vehicle", "heavy_vehicle"]
VAL_SIZE = 1500
SEED = 0


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--raw", type=Path, default=RAW_ROOT)
    p.add_argument("--out", type=Path, default=OUT_ROOT)
    p.add_argument("--val-size", type=int, default=VAL_SIZE,
                   help="images held out as the ground-level vehicle val split")
    p.add_argument("--max-train", type=int, default=0,
                   help="cap the train split (0 = all). SARD went into the mix "
                        "at 17.8 pct and worked; all 8,500 of these is ~27 pct.")
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--audit", action="store_true",
                   help="run the stock-COCO unlabelled-object control and exit")
    p.add_argument("--audit-n", type=int, default=300)
    p.add_argument("--symlink", action="store_true",
                   help="link images instead of copying (saves ~650 MB)")
    return p.parse_args()


def load_samples(raw):
    f = raw / "samples.json"
    if not f.exists():
        raise SystemExit(f"{f} not found -- run scripts/fetch_bdd100k.py first")
    return json.load(f.open(encoding="utf-8"))["samples"]


def attr(sample, key):
    """FiftyOne stores weather/timeofday/scene as Classification dicts."""
    v = sample.get(key)
    return v.get("label") if isinstance(v, dict) else (v or "undefined")


def to_yolo(det):
    """FiftyOne bounding_box is [x_tl, y_tl, w, h], already normalised 0-1.
    YOLO wants [cx, cy, w, h]. Clip: a few boxes run a hair past the edge."""
    x, y, w, h = det["bounding_box"]
    cx, cy = x + w / 2.0, y + h / 2.0
    cx, cy = min(max(cx, 0.0), 1.0), min(max(cy, 0.0), 1.0)
    w, h = min(w, 1.0), min(h, 1.0)
    return cx, cy, w, h


def split_samples(samples, val_size, max_train, seed):
    """Stratify the val split over (timeofday, scene) so it is not accidentally
    all daytime city street, which is what a plain random draw drifts towards
    and would hide exactly the night and highway cases this dataset was added
    for."""
    buckets = defaultdict(list)
    for s in samples:
        buckets[(attr(s, "timeofday"), attr(s, "scene"))].append(s)
    rng = random.Random(seed)
    val, train = [], []
    for key in sorted(buckets):
        group = sorted(buckets[key], key=lambda s: s["filepath"])
        rng.shuffle(group)
        take = round(val_size * len(group) / len(samples))
        val.extend(group[:take])
        train.extend(group[take:])
    rng.shuffle(train)
    if max_train:
        train = train[:max_train]
    return train, val


def write_split(samples, out, split, raw, symlink):
    img_dir = out / "images" / split
    lbl_dir = out / "labels" / split
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)

    counts = Counter()
    dropped = Counter()
    written = empty = missing = 0

    for s in samples:
        src = raw / s["filepath"]
        if not src.exists():
            missing += 1
            continue
        lines = []
        for det in (s.get("detections") or {}).get("detections") or []:
            label = det["label"]
            if label not in CLASS_MAP:
                dropped[label] += 1
                continue
            cid = CLASS_MAP[label]
            cx, cy, w, h = to_yolo(det)
            if w <= 0 or h <= 0:
                continue
            lines.append(f"{cid} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
            counts[cid] += 1

        stem = Path(s["filepath"]).stem
        dst = img_dir / f"{stem}.jpg"
        if not dst.exists():
            if symlink:
                dst.symlink_to(src.resolve())
            else:
                shutil.copy2(src, dst)
        # An image whose only annotations were traffic signs becomes a genuine
        # negative frame, and an empty .txt is how YOLO is told that. Kept on
        # purpose: empty road from a forward-facing camera is the negative
        # evidence section 16 could not get from VisDrone or WiderPerson.
        (lbl_dir / f"{stem}.txt").write_text(
            "\n".join(lines) + ("\n" if lines else ""))
        written += 1
        empty += not lines

    return written, empty, missing, counts, dropped


def audit(raw, samples, n):
    """Section 20 and 21's control: how many people and vehicles does a
    detector that has never seen this project's biases find that the labels do
    not mention? Above ~1/image, pseudo-labelling is worth its false labels."""
    from ultralytics import YOLO

    COCO = {0: 0, 1: 1, 3: 1, 2: 2, 5: 3, 7: 3}
    model = YOLO("yolo26n.pt")
    rng = random.Random(SEED)
    picked = rng.sample(samples, min(n, len(samples)))

    extra = Counter()
    imgs_with = done = 0
    for s in picked:
        src = raw / s["filepath"]
        if not src.exists():
            continue
        labelled = Counter(
            CLASS_MAP[d["label"]]
            for d in (s.get("detections") or {}).get("detections") or []
            if d["label"] in CLASS_MAP)
        r = model.predict(str(src), device=0, conf=0.35, imgsz=1280,
                          verbose=False)[0]
        found = Counter(COCO[int(c)] for c in r.boxes.cls.tolist()
                        if int(c) in COCO)
        surplus = {k: max(0, found[k] - labelled[k]) for k in range(4)}
        extra.update(surplus)
        imgs_with += any(surplus[k] for k in (1, 2, 3))
        done += 1

    if not done:
        raise SystemExit("no images found -- is the download finished?")
    print(f"\nUnlabelled-object control, stock COCO yolo26n, {done} images")
    for k in range(4):
        print(f"  surplus {NEW_NAMES[k]:14} {extra[k]/done:.3f} / image")
    veh = sum(extra[k] for k in (1, 2, 3))
    print(f"  surplus VEHICLES        {veh/done:.3f} / image "
          f"({100*imgs_with/done:.1f} pct of images)")
    print("  threshold for pseudo-labelling: ~1.0 / image  ->  "
          + ("PSEUDO-LABEL" if veh / done > 1.0 else "MERGE AS-IS"))


def main():
    args = parse_args()
    samples = load_samples(args.raw)
    print(f"{len(samples)} labelled images in {args.raw}")

    if args.audit:
        audit(args.raw, samples, args.audit_n)
        return

    train, val = split_samples(samples, args.val_size, args.max_train, args.seed)
    print(f"split: {len(train)} train / {len(val)} val "
          f"(val stratified over timeofday x scene, seed {args.seed})")

    total = Counter()
    for split, group in (("train", train), ("val", val)):
        written, empty, missing, counts, dropped = write_split(
            group, args.out, split, args.raw, args.symlink)
        total.update(counts)
        print(f"\n  {split}: {written} images, {empty} with no boxes (negatives)"
              + (f", {missing} NOT YET DOWNLOADED" if missing else ""))
        for k in range(4):
            print(f"    {NEW_NAMES[k]:16} {counts[k]:>8}  "
                  f"{counts[k]/max(written,1):>6.2f}/img")
        if split == "train" and dropped:
            print("    dropped labels: "
                  + ", ".join(f"{k} {v}" for k, v in dropped.most_common()))

    print(f"\nwrote {args.out}")
    print("boxes by class: "
          + ", ".join(f"{NEW_NAMES[k]} {total[k]}" for k in range(4)))
    print("\nNext, before training, run the unlabelled-object control:")
    print("    python scripts/convert_bdd100k.py --audit")


if __name__ == "__main__":
    main()
