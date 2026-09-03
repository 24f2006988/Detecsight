"""Per-frame detection/motion diagnostics across the sample clips, with
automatic burst detection.

This is the regression harness for anything that touches app/motion_filter.py
or the motion-related thresholds in app/config.py. The README's fixes 7-9 and
11 all cite a script by this name as what verified them; it was missing from
the checkout (see README "Still outstanding"), so this is a reconstruction --
treat its numbers as a fresh baseline, not as a continuation of the ones
quoted in those sections.

A "burst" is a frame whose count of a given class sits far above the local
rolling median for that clip -- the shape every explosion bug in this project
has had. Rolling median rather than a global mean because the clips have
genuinely busy and genuinely quiet stretches, and a global threshold flags the
busy ones as bugs.

    python scripts/diagnose_bursts.py                       # every vN.mp4, ground view
    python scripts/diagnose_bursts.py v10.mp4 --view drone
    python scripts/diagnose_bursts.py --motion-only         # no GPU, motion pass only
    python scripts/diagnose_bursts.py --json runs/bursts.json

--motion-only skips the classifier entirely and exercises just MotionDetector.
That is the half every burst fix so far has lived in, it runs on CPU, and it
therefore works while the GPU is busy with a training run. Use the full path
(the default) to confirm a fix against what actually gets served.
"""
import argparse
import json
import statistics
import sys
from collections import Counter
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def default_clips():
    """Every vN.mp4 in the repo root, in numeric order, sources only."""
    return sorted(
        (p.name for p in Path(".").glob("v*.mp4") if "_annotated" not in p.name),
        key=lambda n: int("".join(c for c in n if c.isdigit()) or 0),
    )


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("clips", nargs="*",
                   help="Clips to sweep (default: every vN.mp4 in the repo root)")
    p.add_argument("--view", default="ground", choices=["ground", "drone"],
                   help="Which checkpoint/coherence threshold to run under")
    p.add_argument("--motion-only", action="store_true",
                   help="Run only MotionDetector (CPU, no classifier). Faster, and "
                        "usable while the GPU is training.")
    p.add_argument("--window", type=int, default=31,
                   help="Rolling-median window, in frames, for the burst baseline")
    p.add_argument("--burst-min", type=int, default=6,
                   help="A frame needs at least this many boxes of a class to count "
                        "as a burst at all (an isolated 0->2 is not an explosion)")
    p.add_argument("--burst-factor", type=float, default=3.0,
                   help="...and at least this multiple of the local rolling median")
    p.add_argument("--max-frames", type=int, default=None)
    p.add_argument("--stride", type=int, default=1,
                   help="Process every Nth frame. Note this changes motion-filter "
                        "behaviour (it is stateful frame-to-frame), so keep it at 1 "
                        "for anything you intend to compare against another run.")
    p.add_argument("--json", default=None,
                   help="Also write the full per-frame series here")
    p.add_argument("--top", type=int, default=5,
                   help="Worst N frames to print per clip")
    return p.parse_args()


def rolling_median(series, window):
    """Median of the `window` frames centred on each index, clipped at the ends."""
    half = window // 2
    return [
        statistics.median(series[max(0, i - half):min(len(series), i + half + 1)])
        for i in range(len(series))
    ]


def find_bursts(series, window, min_count, factor):
    """Frames sitting far enough above their local baseline to be a burst."""
    if not series:
        return []
    base = rolling_median(series, window)
    out = []
    for i, (value, baseline) in enumerate(zip(series, base)):
        if value >= min_count and value >= max(factor * baseline, factor):
            out.append({"frame": i, "count": value, "baseline": round(baseline, 2)})
    return out


def sweep_clip(path, args, detector, motion_detector):
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        print("  !! cannot open " + str(path))
        return None

    source_id = "diag-" + Path(path).stem
    if detector is not None:
        detector.reset_source(source_id)
    else:
        motion_detector.reset(source_id)

    per_frame, idx, kept = [], 0, 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % args.stride:
            idx += 1
            continue
        idx += 1

        if detector is not None:
            dets = detector.track(frame, source_id, view=args.view)["detections"]
            counts = Counter(d["class_name"] for d in dets)
        else:
            blobs = motion_detector.detect(frame, source_id, view=args.view)
            counts = Counter({"moving_object": len(blobs)})

        row = {"frame": kept, "counts": dict(counts)}
        row.update(motion_detector.debug_info(source_id))
        if detector is not None:
            from app.overlay_mask import overlay_mask
            row.update(overlay_mask.debug_info(source_id))
        per_frame.append(row)
        kept += 1
        if args.max_frames and kept >= args.max_frames:
            break
    cap.release()

    classes = sorted({c for r in per_frame for c in r["counts"]})
    series = {c: [r["counts"].get(c, 0) for r in per_frame] for c in classes}
    bursts = {c: find_bursts(s, args.window, args.burst_min, args.burst_factor)
              for c, s in series.items()}
    dropped = Counter(r.get("dropped_reason") for r in per_frame
                      if r.get("dropped_reason"))

    return {"clip": str(path), "frames": kept, "view": args.view,
            "motion_only": args.motion_only, "series": series,
            "bursts": {c: b for c, b in bursts.items() if b},
            "dropped_reasons": dict(dropped), "per_frame": per_frame}


def print_clip(res, top):
    tag = ", motion-only" if res["motion_only"] else ""
    print("\n{}  ({} frames, view={}{})".format(
        res["clip"], res["frames"], res["view"], tag))
    print("  {:<16}{:>8}{:>11}{:>8}{:>8}".format(
        "class", "total", "max/frame", "mean", "bursts"))
    for cls, s in sorted(res["series"].items()):
        n_burst = len(res["bursts"].get(cls, []))
        print("  {:<16}{:>8}{:>11}{:>8.2f}{:>8}".format(
            cls, sum(s), max(s), sum(s) / max(1, len(s)), n_burst))
    if res["dropped_reasons"]:
        drops = ", ".join("{}={}".format(k, v)
                          for k, v in sorted(res["dropped_reasons"].items()))
        print("  motion frames dropped: " + drops)
    for cls, blist in sorted(res["bursts"].items()):
        worst = sorted(blist, key=lambda b: -b["count"])[:top]
        detail = ", ".join("f{}={}(base {})".format(
            b["frame"], b["count"], b["baseline"]) for b in worst)
        print("  BURST {}: {} frames | worst: {}".format(cls, len(blist), detail))


def main():
    args = parse_args()
    clips = args.clips or default_clips()
    if not clips:
        raise SystemExit("No vN.mp4 clips found in the repo root.")

    from app.motion_filter import motion_detector
    detector = None
    if not args.motion_only:
        from app.detector import detector as loaded
        loaded.load()
        detector = loaded

    results = []
    for clip in clips:
        if not Path(clip).exists():
            print("skip {} (missing)".format(clip))
            continue
        print("\n=== {} ===".format(clip), flush=True)
        res = sweep_clip(clip, args, detector, motion_detector)
        if res:
            results.append(res)
            print_clip(res, args.top)

    print("\n" + "=" * 78)
    print("{:<14}{:>8}{:>15}{:>6}{:>8}{:>9}".format(
        "clip", "frames", "moving_object", "max", "bursts", "dropped"))
    print("-" * 78)
    total_bursts = 0
    for r in results:
        s = r["series"].get("moving_object", [0])
        n_burst = len(r["bursts"].get("moving_object", []))
        total_bursts += sum(len(b) for b in r["bursts"].values())
        print("{:<14}{:>8}{:>15}{:>6}{:>8}{:>9}".format(
            Path(r["clip"]).name, r["frames"], sum(s), max(s), n_burst,
            sum(r["dropped_reasons"].values())))
    print("\nTotal burst frames across all classes: {}".format(total_bursts))

    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        print("wrote " + args.json)


if __name__ == "__main__":
    main()
