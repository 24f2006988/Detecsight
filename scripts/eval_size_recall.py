"""Recall stratified by ground-truth box size -- are we losing distant targets?

This is the only tool here that measures the thing which actually limits this
system. Overall mAP50 hides it: 41% of the people in ground-level data are under
32 px, but they are a minority of the instance-weighted metric, so a model can
gain mAP50 while still missing most of the far field.

Measured 2026-09-02 on weights/best.pt (the battlesight_fpv checkpoint),
300 WiderPerson val images, imgsz 1280, conf 0.10 -- BASELINE TO BEAT:

    size(px)     GT   found   recall
       <16     1427     267    0.187
     16-32     2214    1378    0.622
     32-48     1426    1148    0.805
     48-64     1128     964    0.855
     64-96     1481    1388    0.937
      >96      1207    1134    0.940
    overall: 6279 found of 8883, precision 0.662

Near targets are effectively solved (0.94); all the loss is distance. Note that
raising imgsz does NOT fix it -- 1280->1536 gains +0.4% detections for +46% cost
and the LARGER buckets degrade, because above the trained size the model is
off-distribution for its own scale priors (README section 19).

Usage:
    $env:PYTHONPATH="."
    python scripts\\eval_size_recall.py
    python scripts\\eval_size_recall.py --images datasets/CrowdHuman/images/val --limit 500
    python scripts\\eval_size_recall.py --weights runs/detect/battlesight_crowd/weights/best.pt

Windows note: keep the `if __name__ == "__main__":` guard. A bare ultralytics
call without it deadlocks on this machine and looks like slow disk I/O -- the
tell is process memory staying perfectly static.
"""
import argparse
import glob
import os
import time

import numpy as np

EDGES = [0, 16, 32, 48, 64, 96, 10 ** 9]
NAMES = ["<16", "16-32", "32-48", "48-64", "64-96", ">96"]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--weights", default="weights/best.pt",
                   help="Checkpoint to measure (default: the deployed one)")
    p.add_argument("--images", default="datasets/WiderPerson/images/val",
                   help="Image directory; labels are resolved by swapping "
                        "'images' for 'labels' and the extension for .txt")
    p.add_argument("--limit", type=int, default=300,
                   help="Images to evaluate (300 keeps a run under ~2 min)")
    p.add_argument("--imgsz", type=int, default=1280)
    p.add_argument("--conf", type=float, default=0.10,
                   help="Match config.CONF_THRESHOLD_PERSONNEL_GROUND so the "
                        "number reflects what is actually served")
    p.add_argument("--cls", type=int, default=0, help="Class id to score (0=personnel)")
    p.add_argument("--device", default="0")
    p.add_argument("--batch", type=int, default=8)
    return p.parse_args()


def load_gt(label_path, w, h, cls):
    """YOLO-normalised label file -> pixel xyxy for one class."""
    if not os.path.exists(label_path):
        return np.zeros((0, 4))
    out = []
    for line in open(label_path):
        parts = line.split()
        if len(parts) < 5 or int(parts[0]) != cls:
            continue
        cx, cy, bw, bh = (float(v) for v in parts[1:5])
        out.append([(cx - bw / 2) * w, (cy - bh / 2) * h,
                    (cx + bw / 2) * w, (cy + bh / 2) * h])
    return np.array(out) if out else np.zeros((0, 4))


def main():
    args = parse_args()
    from ultralytics import YOLO

    imgs = sorted(glob.glob(os.path.join(args.images, "*.jpg")))
    imgs += sorted(glob.glob(os.path.join(args.images, "*.png")))
    imgs = sorted(imgs)[:args.limit]
    if not imgs:
        raise SystemExit(f"No images found under {args.images}")

    model = YOLO(args.weights)
    # Warm: the GPU idles at 270 MHz and a cold first call reports roughly
    # double the true latency, which would poison the ms/img figure below.
    model.predict(imgs[0], imgsz=args.imgsz, verbose=False, device=args.device)

    tp = np.zeros(len(NAMES))
    gt_n = np.zeros(len(NAMES))
    n_pred = 0
    t0 = time.perf_counter()

    for k in range(0, len(imgs), args.batch):
        batch = imgs[k:k + args.batch]
        results = model.predict(batch, imgsz=args.imgsz, conf=args.conf,
                                verbose=False, device=args.device, max_det=500)
        for img, r in zip(batch, results):
            h, w = r.orig_shape
            label = os.path.splitext(img.replace("images", "labels"))[0] + ".txt"
            gt = load_gt(label, w, h, args.cls)
            if not len(gt):
                continue

            boxes = r.boxes
            if boxes is not None and len(boxes):
                sel = boxes.cls.cpu().numpy().astype(int) == args.cls
                pred = boxes.xyxy.cpu().numpy()[sel]
            else:
                pred = np.zeros((0, 4))
            n_pred += len(pred)

            # Greedy IoU>=0.5 matching, each prediction claimed at most once.
            used = set()
            for box in gt:
                side = float(np.sqrt(max(1.0, (box[2] - box[0]) * (box[3] - box[1]))))
                b = min(max(int(np.digitize(side, EDGES)) - 1, 0), len(NAMES) - 1)
                gt_n[b] += 1
                if not len(pred):
                    continue
                ix1 = np.maximum(box[0], pred[:, 0]); iy1 = np.maximum(box[1], pred[:, 1])
                ix2 = np.minimum(box[2], pred[:, 2]); iy2 = np.minimum(box[3], pred[:, 3])
                inter = np.clip(ix2 - ix1, 0, None) * np.clip(iy2 - iy1, 0, None)
                union = ((box[2] - box[0]) * (box[3] - box[1])
                         + (pred[:, 2] - pred[:, 0]) * (pred[:, 3] - pred[:, 1]) - inter)
                iou = inter / np.maximum(union, 1e-9)
                for j in np.argsort(-iou):
                    if iou[j] < 0.5:
                        break
                    if j not in used:
                        used.add(int(j))
                        tp[b] += 1
                        break

    dt = time.perf_counter() - t0
    found, total = int(tp.sum()), int(gt_n.sum())
    precision = tp.sum() / n_pred if n_pred else 0.0
    recall = tp.sum() / gt_n.sum() if gt_n.sum() else 0.0

    print(f"\n{args.weights}  on {len(imgs)} images from {args.images}")
    print(f"imgsz={args.imgsz} conf={args.conf} class={args.cls}   "
          f"{1000 * dt / len(imgs):.1f} ms/img")
    print(f"{'size(px)':>10} {'GT':>7} {'found':>7} {'recall':>8}")
    for i, name in enumerate(NAMES):
        r = tp[i] / gt_n[i] if gt_n[i] else 0.0
        print(f"{name:>10} {int(gt_n[i]):>7} {int(tp[i]):>7} {r:>8.3f}")
    print(f"\noverall: {found} found of {total}  recall={recall:.3f}  "
          f"precision={precision:.3f}  preds={n_pred}")
    print("\nBaseline 2026-09-02 (weights/best.pt, WiderPerson val, 300 imgs):")
    print("  <16=0.187  16-32=0.622  32-48=0.805  48-64=0.855  64-96=0.937  >96=0.940")
    print("  overall 6279 found, precision 0.662")


if __name__ == "__main__":
    main()
