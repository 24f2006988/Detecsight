"""Head-to-head validation of candidate checkpoints at several resolutions.

Answers two questions with numbers instead of guesses:
  1. Which checkpoint on disk is actually the best one?
  2. How much of the loss is resolution, i.e. are targets simply too small at
     the imgsz being served?

    python scripts/diagnose.py
    python scripts/diagnose.py --data data/battlesight.yaml --imgsz 640 1280
"""
import argparse
import json
from pathlib import Path

CLASS_NAMES = ["personnel", "two_wheeler", "light_vehicle", "heavy_vehicle"]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", default="data/battlesight.yaml")
    p.add_argument("--weights", nargs="+", default=[
        "weights/best.pt",
        "runs/detect/battlesight_v1/weights/best.pt",
    ])
    p.add_argument("--imgsz", nargs="+", type=int, default=[640, 1280])
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--device", default="0")
    p.add_argument("--out", default="runs/diagnose.json")
    return p.parse_args()


def main():
    args = parse_args()
    from ultralytics import YOLO

    results = []
    for w in args.weights:
        if not Path(w).exists():
            print(f"skip {w} (missing)")
            continue
        for imgsz in args.imgsz:
            print(f"\n=== {w} @ {imgsz} ===")
            model = YOLO(w)
            m = model.val(
                data=args.data,
                imgsz=imgsz,
                batch=args.batch,
                device=args.device,
                verbose=False,
                plots=False,
            )
            row = {
                "weights": w,
                "imgsz": imgsz,
                "mAP50": round(float(m.box.map50), 4),
                "mAP50_95": round(float(m.box.map), 4),
                "precision": round(float(m.box.mp), 4),
                "recall": round(float(m.box.mr), 4),
                "per_class": {
                    CLASS_NAMES[c]: {
                        "P": round(float(m.box.p[i]), 4),
                        "R": round(float(m.box.r[i]), 4),
                        "mAP50": round(float(m.box.ap50[i]), 4),
                        "mAP50_95": round(float(m.box.ap[i]), 4),
                    }
                    for i, c in enumerate(m.box.ap_class_index)
                },
            }
            results.append(row)
            print(json.dumps(row, indent=2))

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 78)
    hdr = f"{'weights':<46}{'imgsz':>6}{'mAP50':>8}{'mAP50-95':>10}{'R':>8}"
    print(hdr)
    print("-" * 78)
    for r in results:
        print(f"{r['weights']:<46}{r['imgsz']:>6}{r['mAP50']:>8.4f}"
              f"{r['mAP50_95']:>10.4f}{r['recall']:>8.4f}")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
