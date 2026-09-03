"""Export a .pt checkpoint to a TensorRT engine for GPU-accelerated inference.

Run once per checkpoint, and again any time a new best.pt is promoted:

    python scripts/export_engine.py

app/detector.py's Detector.load() automatically prefers weights/best.engine
over weights/best.pt when it exists (see config.USE_TENSORRT) -- no other
config change is needed. Delete the .engine file (or set
BATTLESIGHT_USE_TENSORRT=0) to force the .pt path again.

Requires the `tensorrt` Python package matching your installed CUDA/driver
(https://pypi.org/project/tensorrt/, or the NVIDIA pip index for your CUDA
version). Install that yourself before running this: unlike onnx/onnxslim,
which ultralytics will auto-install as needed, the wrong tensorrt build will
fail to import rather than fail to export, so it isn't something to leave to
autoinstall.
"""
import argparse
from pathlib import Path

from ultralytics import YOLO

from app import config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=config.MODEL_PATH,
                         help="Checkpoint to export (default: config.MODEL_PATH)")
    parser.add_argument("--imgsz", type=int, default=config.IMGSZ)
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
        imgsz=args.imgsz,
    )
    print(f"Wrote {engine_path}")
    print("Restart the service (or call Detector.load() again) to pick it up.")


if __name__ == "__main__":
    main()
