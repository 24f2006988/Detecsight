"""Latency vs. accuracy at each inference resolution.

Raising imgsz is the single biggest accuracy lever on this dataset (targets are
tiny), but this is a realtime feed, so the question is what it costs per frame.
Run this before changing BATTLESIGHT_IMGSZ.

    python scripts/bench_imgsz.py --weights runs/detect/battlesight_v1/weights/best.pt
"""
import argparse
import time
from pathlib import Path

import cv2
import numpy as np


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--weights", default="runs/detect/battlesight_v1/weights/best.pt")
    p.add_argument("--imgsz", nargs="+", type=int, default=[640, 960, 1280, 1600])
    p.add_argument("--frames", type=int, default=40)
    p.add_argument("--warmup", type=int, default=8)
    p.add_argument("--device", default="0")
    p.add_argument("--source", default="v1.mp4", help="video to pull real frames from")
    p.add_argument("--conf", type=float, default=0.35)
    return p.parse_args()


def load_frames(source, n):
    """Real frames if the source exists, otherwise 1080p noise."""
    if Path(source).exists():
        cap = cv2.VideoCapture(source)
        frames = []
        while len(frames) < n:
            ok, f = cap.read()
            if not ok:
                break
            frames.append(f)
        cap.release()
        if frames:
            h, w = frames[0].shape[:2]
            print(f"{len(frames)} frames from {source} at {w}x{h}")
            return frames
    print("falling back to synthetic 1920x1080 frames")
    return [np.random.randint(0, 255, (1080, 1920, 3), dtype=np.uint8) for _ in range(n)]


def main():
    args = parse_args()
    import torch
    from ultralytics import YOLO

    frames = load_frames(args.source, args.frames)
    model = YOLO(args.weights)

    print(f"\n{'imgsz':>6}{'ms/frame':>10}{'fps':>8}{'p95 ms':>9}{'VRAM MiB':>10}{'dets/frame':>12}")
    print("-" * 55)
    for imgsz in args.imgsz:
        for f in frames[: args.warmup]:
            model.predict(f, imgsz=imgsz, device=args.device, conf=args.conf, verbose=False)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

        times, dets = [], []
        for f in frames:
            t0 = time.perf_counter()
            r = model.predict(f, imgsz=imgsz, device=args.device, conf=args.conf, verbose=False)
            torch.cuda.synchronize()
            times.append((time.perf_counter() - t0) * 1000)
            dets.append(len(r[0].boxes))

        mean_ms = float(np.mean(times))
        print(f"{imgsz:>6}{mean_ms:>10.1f}{1000 / mean_ms:>8.1f}"
              f"{np.percentile(times, 95):>9.1f}"
              f"{torch.cuda.max_memory_allocated() / 2**20:>10.0f}"
              f"{np.mean(dets):>12.1f}")


if __name__ == "__main__":
    main()
