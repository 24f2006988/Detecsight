"""The promotion gate, as one command.

See ENGINEERING_LOG.md for the measurements behind this.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.eval_rubric import check_rubric, run_val  # noqa: E402
from scripts.eval_size_recall import measure  # noqa: E402

DEPLOYED = "weights/best.pt"

# The recorded size-stratified baseline, measured 2026-09-02 on the deployed
# checkpoint over 300 WiderPerson val images at imgsz 1280, conf 0.10. A
# candidate is compared against these rather than against a fresh baseline run,
# because re-measuring the baseline every time doubles a slow evaluation for a
# number that does not move.
SIZE_BASELINE = {
    "<16": 0.187, "16-32": 0.622, "32-48": 0.805,
    "48-64": 0.855, "64-96": 0.937, ">96": 0.940,
}


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("candidate", nargs="?", default=None,
                   help="Checkpoint to consider promoting. Omit to report on "
                        "the currently deployed one.")
    p.add_argument("--baseline", default=DEPLOYED,
                   help=f"What to measure against (default: {DEPLOYED})")
    p.add_argument("--data", default="data/battlesight_multi.yaml",
                   help="Val set. The default is deliberately the multi yaml: "
                        "it is byte-identical to the fpv yaml's val split, so "
                        "numbers stay comparable to every figure in the log.")
    p.add_argument("--imgsz", type=int, default=1280,
                   help="1280 is what production serves at; measuring at 640 "
                        "produces numbers that do not describe the deployment")
    p.add_argument("--device", default="0")
    p.add_argument("--map50-tolerance", type=float, default=0.02)
    p.add_argument("--recall-floor", type=float, default=0.3)
    p.add_argument("--size-images", default="datasets/WiderPerson/images/val")
    p.add_argument("--size-limit", type=int, default=300)
    p.add_argument("--no-size-recall", action="store_true",
                   help="Skip the size-stratified pass (~2 min)")
    p.add_argument("--json", default=None, help="Also write results to this path")
    return p.parse_args()


def print_metrics(label, m):
    print(f"\n{label}: {m['weights']}")
    print(f"  {'class':<16}{'P':>8}{'R':>8}{'mAP50':>10}{'mAP50-95':>10}")
    print(f"  {'ALL':<16}{m['precision']:>8.3f}{m['recall']:>8.3f}"
          f"{m['map50']:>10.3f}{m['map50_95']:>10.3f}")
    for name, c in m["per_class"].items():
        print(f"  {name:<16}{c['precision']:>8.3f}{c['recall']:>8.3f}"
              f"{c['map50']:>10.3f}{c['map50_95']:>10.3f}")


def print_size_table(m):
    print(f"\nSize-stratified recall  ({m['n_images']} images, "
          f"imgsz {m['imgsz']}, conf {m['conf']})")
    print(f"  {'size(px)':>10} {'GT':>7} {'found':>7} {'recall':>8} "
          f"{'baseline':>9} {'delta':>8}")
    for i, name in enumerate(m["buckets"]):
        r = m["recall_by_bucket"][i]
        base = SIZE_BASELINE.get(name)
        if base is None:
            print(f"  {name:>10} {m['gt'][i]:>7} {m['found'][i]:>7} {r:>8.3f}")
        else:
            print(f"  {name:>10} {m['gt'][i]:>7} {m['found'][i]:>7} {r:>8.3f} "
                  f"{base:>9.3f} {r - base:>+8.3f}")
    print(f"  overall recall {m['recall']:.3f}  precision {m['precision']:.3f}")


def main():
    args = parse_args()
    out = {}

    if args.candidate is None:
        # Self-report: no gate to run, just say what is deployed and how it
        # does on the metric that limits it.
        print(f"Reporting on the deployed checkpoint ({args.baseline}); "
              f"pass a candidate to run the promotion gate.")
        deployed = run_val(args.baseline, args.data, args.imgsz, args.device)
        print_metrics("DEPLOYED", deployed)
        out["deployed"] = deployed
        if not args.no_size_recall:
            sr = measure(args.baseline, args.size_images, args.size_limit,
                         args.imgsz, device=args.device)
            print_size_table(sr)
            out["size_recall"] = sr
        if args.json:
            Path(args.json).write_text(json.dumps(out, indent=2))
        return 0

    print(f"Baseline : {args.baseline}")
    print(f"Candidate: {args.candidate}")
    print(f"Val      : {args.data} at imgsz {args.imgsz}")

    baseline = run_val(args.baseline, args.data, args.imgsz, args.device)
    candidate = run_val(args.candidate, args.data, args.imgsz, args.device)
    print_metrics("BASELINE ", baseline)
    print_metrics("CANDIDATE", candidate)

    passed, report = check_rubric(candidate, baseline,
                                  args.map50_tolerance, args.recall_floor)

    print("\nPROMOTION RUBRIC")
    for criterion, ok, detail in report:
        print(f"  [{'PASS' if ok else 'FAIL'}] {criterion}: {detail}")

    out.update({"baseline": baseline, "candidate": candidate,
                "rubric": [{"criterion": c, "passed": ok, "detail": d}
                           for c, ok, d in report],
                "passed": passed})

    if not args.no_size_recall:
        sr = measure(args.candidate, args.size_images, args.size_limit,
                     args.imgsz, device=args.device)
        print_size_table(sr)
        out["size_recall"] = sr
        # Reported, deliberately not gated on. The size baseline was taken on
        # one 300-image sample and moves by a point or two between runs; making
        # it a hard criterion would block promotions on sampling noise. It is
        # here because a candidate that gains mAP while losing the far field is
        # the exact trade this project does not want to make silently.
        worse = [n for i, n in enumerate(sr["buckets"])
                 if n in SIZE_BASELINE
                 and sr["recall_by_bucket"][i] < SIZE_BASELINE[n] - 0.03]
        if worse:
            print(f"  NOTE: lost more than 3 points in {', '.join(worse)} -- "
                  f"look at annotated frames before promoting")

    if args.json:
        Path(args.json).write_text(json.dumps(out, indent=2))
        print(f"\nwrote {args.json}")

    print(f"\nVerdict: {'PASS - safe to promote' if passed else 'FAIL - do not promote'}")
    if passed:
        print("Promotion is not automatic. Copy a rollback pair first, then "
              "rebuild the TensorRT engine or serving keeps using the old model:")
        print("  copy weights\\best.pt weights\\best_pre_<name>.pt")
        print(f"  copy {args.candidate} weights\\best.pt")
        print("  python scripts\\export_engine.py --model weights\\best.pt --static")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
