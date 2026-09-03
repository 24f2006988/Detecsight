"""One command: convert WiderPerson -> fine-tune on the merged dataset ->
validate against the promotion rubric (new run vs. current weights) ->
optionally promote -> annotate a demo video with the result.

This just sequences the existing standalone scripts/CLI calls - nothing here
duplicates their logic. Run stages individually instead if you want to
inspect results between steps rather than running the whole thing blind.
"""
import argparse
import shutil
import subprocess
import sys
from pathlib import Path

from eval_rubric import check_rubric, print_report, run_val


def run(cmd):
    print(f"\n$ {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def parse_args():
    p = argparse.ArgumentParser(
        description="Convert WiderPerson, fine-tune, validate, and annotate a demo video"
    )
    p.add_argument("video", help="Video to run the trained model against at the end")
    p.add_argument("--name", default="battlesight_v2", help="Training run name")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--lr0", type=float, default=0.002,
                   help="Fine-tuning LR; lower than train.py's from-scratch default")
    p.add_argument("--base", default="weights/best.pt", help="Checkpoint to fine-tune from")
    p.add_argument("--skip-convert", action="store_true",
                   help="Skip the WiderPerson conversion step (already run once)")
    p.add_argument("--promote", action="store_true",
                   help="Overwrite weights/best.pt with the new run's best.pt IF it passes "
                        "the promotion rubric. Default: stop after validation so you can "
                        "check the numbers first; the demo video uses the new run's weights "
                        "either way.")
    p.add_argument("--force-promote", action="store_true",
                   help="Promote even if the rubric fails. Use only after manually reviewing "
                        "why it failed -- this skips the regression/collapse/recall checks.")
    p.add_argument("--map50-tolerance", type=float, default=0.02,
                   help="Rubric: max acceptable mAP50 drop vs. current weights/best.pt")
    p.add_argument("--recall-floor", type=float, default=0.3,
                   help="Rubric: minimum acceptable overall recall")
    return p.parse_args()


def main():
    args = parse_args()
    py = sys.executable

    if not args.skip_convert:
        run([py, "scripts/convert_widerperson.py"])
    else:
        print("Skipping WiderPerson conversion (--skip-convert)")

    run([
        py, "scripts/train.py",
        "--model", args.base,
        "--data", "data/battlesight_multi.yaml",
        "--epochs", str(args.epochs),
        "--batch", str(args.batch),
        "--imgsz", str(args.imgsz),
        "--lr0", str(args.lr0),
        "--flipud", "0",
        "--name", args.name,
    ])

    new_weights = Path("runs/detect") / args.name / "weights" / "best.pt"
    if not new_weights.exists():
        raise SystemExit(f"Training did not produce {new_weights}")

    print("\n=== Validating against the promotion rubric ===")
    baseline_metrics = run_val("weights/best.pt", "data/battlesight_multi.yaml", args.imgsz)
    new_metrics = run_val(str(new_weights), "data/battlesight_multi.yaml", args.imgsz)
    passed, report = check_rubric(new_metrics, baseline_metrics,
                                   args.map50_tolerance, args.recall_floor)
    print_report(new_metrics, baseline_metrics, passed, report)

    weights_for_demo = str(new_weights)
    if args.promote and passed:
        print(f"\nRubric PASSED. Promoting {new_weights} -> weights/best.pt")
        shutil.copy(new_weights, "weights/best.pt")
        weights_for_demo = "weights/best.pt"
    elif args.promote and not passed:
        if args.force_promote:
            print(f"\nRubric FAILED but --force-promote given. "
                  f"Promoting {new_weights} -> weights/best.pt anyway.")
            shutil.copy(new_weights, "weights/best.pt")
            weights_for_demo = "weights/best.pt"
        else:
            print("\nRubric FAILED -- refusing to promote. "
                  "Review the report above, or pass --force-promote to override.")
    else:
        print("\nNot promoted (pass --promote to promote automatically on a rubric pass). "
              "Demo below uses the new run's weights directly.")

    print(f"\n=== Annotating {args.video} ===")
    run([py, "scripts/annotate_video.py", args.video, "--weights", weights_for_demo])


if __name__ == "__main__":
    main()
