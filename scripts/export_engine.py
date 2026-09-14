"""Export a .pt checkpoint to a TensorRT engine for GPU-accelerated inference.

See ENGINEERING_LOG.md for the measurements behind this.
"""
import argparse
from pathlib import Path

from ultralytics import YOLO

from app import config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=config.MODEL_PATH,
                         help="Checkpoint to export (default: config.MODEL_PATH)")
    parser.add_argument("--imgsz", type=int, nargs="+", default=[config.IMGSZ],
                        metavar=("H", "W"),
                        help="One value for a square engine, or H W for a "
                             "rectangular one. USE H W WHEN THE SOURCE IS NOT "
                             "SQUARE. A --static engine is locked to the shape "
                             "it is built for, and ultralytics predict() "
                             "letterboxes video to stride-32 rect (1280x720 -> "
                             "1280x736), so a square 1280x1280 engine is fed a "
                             "differently-padded tensor than the .pt path ever "
                             "sees. Measured on v10.mp4 (1280x720): the square "
                             "engine reported 26 personnel at conf 0.25 where "
                             "the checkpoint reported 15, and dropped the "
                             "light_vehicle boxes entirely. Rebuilt at "
                             "--imgsz 736 1280 it matches the checkpoint "
                             "detection for detection and runs 3.2x faster "
                             "(62.5 -> 19.7 ms/frame), because 736 rows is "
                             "57%% of the pixels. Note that `yolo val` and "
                             "scripts/evaluate.py cannot validate a "
                             "rectangular engine -- non-PyTorch val requires "
                             "square input -- so gate the .pt and rely on the "
                             "equivalence above.")
    parser.add_argument("--device", default=config.DEVICE)
    parser.add_argument("--workspace", type=float, default=4,
                         help="TensorRT builder workspace, in GiB")
    parser.add_argument("--no-half", dest="half", action="store_false",
                         help="Disable FP16 (default: FP16 on)")
    parser.add_argument("--static", dest="dynamic", action="store_false",
                         help="Export a fixed-shape engine instead of a "
                              "dynamic-shape one (smaller, marginally faster, "
                              "but see the note below on why dynamic is the "
                              "default)")
    parser.set_defaults(half=True, dynamic=True)
    args = parser.parse_args()

    model_path = Path(args.model)
    if not model_path.exists():
        raise SystemExit(f"No checkpoint at {model_path}. Nothing to export.")
    if model_path.suffix != ".pt":
        raise SystemExit(f"{model_path} is not a .pt checkpoint.")

    if not args.dynamic:
        print("WARNING: this codebase calls predict()/track() at two "
              "different image sizes -- config.IMGSZ for full-frame/tracked "
              "inference, config.MOTION_CROP_IMGSZ for the motion-gated crop "
              "path (app/detector.py:_detect_on_crops, only active when "
              "BATTLESIGHT_MOTION_GATED=1). A static engine only serves the "
              "size it was built for; the other call will raise inside "
              "ultralytics. Only pass --static if you have also set "
              "BATTLESIGHT_MOTION_GATED=0 permanently.")

    print(f"Exporting {model_path} -> TensorRT engine "
          f"(imgsz={args.imgsz}, half={args.half}, dynamic={args.dynamic}, "
          f"workspace={args.workspace}GiB, device={args.device})")
    model = YOLO(str(model_path))
    engine_path = model.export(
        format="engine",
        device=args.device,
        half=args.half,
        dynamic=args.dynamic,
        workspace=args.workspace,
        imgsz=args.imgsz[0] if len(args.imgsz) == 1 else args.imgsz,
    )
    print(f"Wrote {engine_path}")
    print("Restart the service (or call Detector.load() again) to pick it up.")


if __name__ == "__main__":
    main()
