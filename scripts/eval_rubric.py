"""Programmatic validation + a promotion rubric for BattleSight training runs.

Used by scripts/run_pipeline.py, or standalone:
    python scripts\\eval_rubric.py runs\\detect\\battlesight_v2\\weights\\best.pt

The rubric is a gate against shipping something WORSE than what's currently
deployed, not a bar for "good enough to stop iterating." All thresholds are
deliberately conservative and are CLI-adjustable.
"""
import argparse

from ultralytics import YOLO

DATA = "data/battlesight_multi.yaml"


def run_val(weights, data=DATA, imgsz=640, device="0"):
    """Validate one checkpoint and return overall + per-class metrics."""
    model = YOLO(weights)
    metrics = model.val(data=data, imgsz=imgsz, device=device, verbose=False, plots=False)
    box = metrics.box

    per_class = {}
    for idx, cls_id in enumerate(box.ap_class_index):
        p, r, ap50, ap = box.class_result(idx)
        per_class[metrics.names[int(cls_id)]] = {
            "precision": float(p), "recall": float(r),
            "map50": float(ap50), "map50_95": float(ap),
        }

    return {
        "weights": str(weights),
        "precision": float(box.mp),
        "recall": float(box.mr),
        "map50": float(box.map50),
        "map50_95": float(box.map),
        "per_class": per_class,
    }


def check_rubric(new: dict, baseline: dict, map50_tolerance: float = 0.02,
                  recall_floor: float = 0.3):
    """Check `new` against `baseline` on four criteria.

    Returns (passed: bool, report: list[(criterion, passed, detail)]).
    """
    report = []

    # 1. Overall accuracy must not regress beyond tolerance.
    d_overall = new["map50"] - baseline["map50"]
    ok = d_overall >= -map50_tolerance
    report.append(("overall_map50_no_regression", ok,
                    f"{baseline['map50']:.4f} -> {new['map50']:.4f} "
                    f"({d_overall:+.4f}, tolerance -{map50_tolerance})"))

    # 2. The class this training run was actually meant to improve (personnel,
    # via WiderPerson) must not regress either -- catching that class 0 got
    # worse would otherwise be masked by gains on the vehicle classes.
    base_p = baseline["per_class"].get("personnel", {}).get("map50", 0.0)
    new_p = new["per_class"].get("personnel", {}).get("map50", 0.0)
    d_p = new_p - base_p
    ok = d_p >= -map50_tolerance
    report.append(("personnel_map50_no_regression", ok,
                    f"{base_p:.4f} -> {new_p:.4f} ({d_p:+.4f}, tolerance -{map50_tolerance})"))

    # 3. No class quietly stopped being detected at all.
    collapsed = [name for name, m in new["per_class"].items() if m["map50"] <= 0.0]
    ok = len(collapsed) == 0
    report.append(("no_collapsed_class", ok,
                    "none" if ok else f"collapsed: {', '.join(collapsed)}"))

    # 4. Recall floor: catches a degenerate model that just predicts nothing
    # (which can still post a deceptively OK precision).
    ok = new["recall"] >= recall_floor
    report.append(("recall_floor", ok, f"{new['recall']:.4f} >= {recall_floor}"))

    passed = all(ok for _, ok, _ in report)
    return passed, report


def print_report(new: dict, baseline: dict, passed: bool, report: list):
    print(f"\n{'='*72}\nVALIDATION: {new['weights']}\n{'='*72}")
    print(f"{'class':<16}{'P':>8}{'R':>8}{'mAP50':>10}{'mAP50-95':>10}")
    print(f"{'ALL':<16}{new['precision']:>8.3f}{new['recall']:>8.3f}"
          f"{new['map50']:>10.3f}{new['map50_95']:>10.3f}")
    for name, m in new["per_class"].items():
        print(f"{name:<16}{m['precision']:>8.3f}{m['recall']:>8.3f}"
              f"{m['map50']:>10.3f}{m['map50_95']:>10.3f}")

    print(f"\nPROMOTION RUBRIC (vs. {baseline['weights']}):")
    for criterion, ok, detail in report:
        print(f"  [{'PASS' if ok else 'FAIL'}] {criterion}: {detail}")
    verdict = "PASS - safe to promote" if passed else "FAIL - do not promote without manual review"
    print(f"\nVerdict: {verdict}")


def parse_args():
    p = argparse.ArgumentParser(description="Validate a run and check it against the promotion rubric")
    p.add_argument("weights", help="New run's weights (e.g. runs/detect/battlesight_v2/weights/best.pt)")
    p.add_argument("--baseline", default="weights/best.pt", help="Current deployed weights to compare against")
    p.add_argument("--data", default=DATA)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--device", default="0")
    p.add_argument("--map50-tolerance", type=float, default=0.02)
    p.add_argument("--recall-floor", type=float, default=0.3)
    return p.parse_args()


def main():
    args = parse_args()
    print("Validating baseline...")
    baseline = run_val(args.baseline, args.data, args.imgsz, args.device)
    print("Validating new run...")
    new = run_val(args.weights, args.data, args.imgsz, args.device)
    passed, report = check_rubric(new, baseline, args.map50_tolerance, args.recall_floor)
    print_report(new, baseline, passed, report)
    raise SystemExit(0 if passed else 1)


if __name__ == "__main__":
    main()
