"""Fine-tune YOLO26 on the BattleSight 4-class drone dataset."""
import argparse
import sys
from datetime import datetime
from pathlib import Path

from ultralytics import YOLO
from ultralytics.utils.logger import ConsoleLogger

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fpv_augment


def parse_args():
    p = argparse.ArgumentParser(description="BattleSight detector fine-tuning")
    p.add_argument("--model", default="yolo26s.pt",
                   help="Pretrained checkpoint to start from")
    p.add_argument("--data", default="data/battlesight.yaml",
                   help="Dataset config yaml")
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--device", default="0", help="'0' for first GPU, 'cpu' to force CPU")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--name", default="battlesight_v1",
                   help="Run name; output lands in runs/detect/<name>")
    p.add_argument("--resume", action="store_true",
                   help="Resume an interrupted run from its last.pt")
    p.add_argument("--lr0", type=float, default=0.01,
                   help="Initial LR; use something lower (e.g. 0.001-0.003) "
                        "when fine-tuning from an existing checkpoint")
    p.add_argument("--flipud", type=float, default=0.5,
                   help="Vertical flip probability; tuned for top-down drone "
                        "imagery. Set 0 when the dataset mixes in ground-level "
                        "imagery (e.g. WiderPerson)")
    p.add_argument("--warmup-epochs", type=float, default=None,
                   help="LR warmup length. Default adapts to run length: 3.0 for a "
                        "long run, but min(3, epochs/10) for a short one -- a fixed "
                        "3.0 spends 30%% of a 10-epoch fine-tune warming up.")
    p.add_argument("--close-mosaic", type=int, default=None,
                   help="Epochs at the END of training with mosaic disabled. Default "
                        "adapts to run length. The old hardcoded 10 silently disables "
                        "mosaic for the WHOLE run when epochs <= 10, which is a real "
                        "trap for short fine-tunes.")
    p.add_argument("--patience", type=int, default=20,
                   help="Early-stop patience, in epochs with no improvement")
    p.add_argument("--aug-profile", default="none", choices=list(fpv_augment.PROFILES),
                   help="Extra pixel-level augmentation (scripts/fpv_augment.py). "
                        "'fpv' models this system's real input -- motion blur, low "
                        "resolution, sensor noise, compression, exposure swings -- "
                        "which nothing in VisDrone/WiderPerson resembles. Note "
                        "albumentations was absent from this venv until that profile "
                        "was added, so every earlier checkpoint had NO such "
                        "augmentation at all, not merely the ultralytics defaults.")
    p.add_argument("--log-file", default=None,
                   help="Console log destination (default: logs/<name>_<timestamp>.log). "
                        "Captures the full console output -- config dump, every epoch's "
                        "loss/mAP line, final summary -- while still printing live.")
    p.add_argument("--no-log", action="store_true",
                   help="Disable file logging (on by default)")
    return p.parse_args()


def main():
    args = parse_args()

    console_logger = None
    if not args.no_log:
        log_file = Path(args.log_file) if args.log_file else (
            Path("logs") / f"{args.name}_{datetime.now():%Y%m%d_%H%M%S}.log"
        )
        console_logger = ConsoleLogger(log_file)
        console_logger.start_capture()
        print(f"Logging console output to {log_file}")

    try:
        if args.resume:
            ckpt = Path("runs/detect") / args.name / "weights" / "last.pt"
            if not ckpt.exists():
                raise SystemExit(
                    f"Nothing to resume: {ckpt} does not exist. Check --name matches "
                    f"the interrupted run, and run from the project root."
                )
            print(f"Resuming from {ckpt}")
            model = YOLO(str(ckpt))
            model.train(resume=True)
            return

        model = YOLO(args.model)
        print(fpv_augment.describe(args.aug_profile))

        # Both of these were hardcoded, and both are wrong for a short
        # fine-tune. close_mosaic=10 with epochs=10 disables mosaic for every
        # epoch of the run rather than just the tail, and a 3-epoch warmup is
        # nearly a third of it. Scale them to the run actually being asked for.
        warmup = (args.warmup_epochs if args.warmup_epochs is not None
                  else min(3.0, max(1.0, args.epochs / 10.0)))
        close_mosaic = (args.close_mosaic if args.close_mosaic is not None
                        else min(10, max(1, args.epochs // 4)))
        print(f"warmup_epochs={warmup}  close_mosaic={close_mosaic}  "
              f"patience={args.patience}")

        model.train(
            # --- core ---
            data=args.data,
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            device=args.device,
            workers=args.workers,

            # --- run bookkeeping ---
            # No project= here: ultralytics resolves a relative project UNDER
            # runs/<task>/, so project="runs/detect" would land the run in
            # runs/detect/runs/detect/<name>. Omitting it gives runs/detect/<name>.
            name=args.name,
            exist_ok=False,
            plots=True,
            val=True,

            # --- fitting into 8 GB ---
            amp=True,
            cache=False,

            # --- optimisation ---
            optimizer="auto",
            lr0=args.lr0,
            lrf=0.01,
            warmup_epochs=warmup,
            patience=args.patience,

            # --- augmentation tuned for a top-down drone view ---
            mosaic=1.0,
            close_mosaic=close_mosaic,
            degrees=15.0,
            scale=0.5,
            fliplr=0.5,
            flipud=args.flipud,
            hsv_h=0.015,
            hsv_s=0.7,
            hsv_v=0.4,

            # --- domain augmentation (scripts/fpv_augment.py) ---
            # Ultralytics 8.4 accepts a custom Albumentations list here as a
            # first-class training arg; None leaves its own defaults alone.
            augmentations=fpv_augment.build(args.aug_profile),
        )

        print("\nTraining complete.")
        print(f"Best weights: runs/detect/{args.name}/weights/best.pt")
    finally:
        if console_logger:
            console_logger.stop_capture()


if __name__ == "__main__":
    main()
