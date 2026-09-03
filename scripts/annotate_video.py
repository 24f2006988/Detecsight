"""Run a video file, camera or stream through the Detector, with boxes drawn.

Batch equivalent of scripts/test_stream.py: that script pushes frames over the
live WebSocket to test the running API; this one calls the Detector directly
(same class, same tracking/motion logic) so it works without a server running,
and produces something you can actually look at instead of console output.

    python scripts/annotate_video.py v1.mp4                  # file -> annotated file
    python scripts/annotate_video.py v1.mp4 --show           # ...and watch it go
    python scripts/annotate_video.py 0                       # webcam, live overlay
    python scripts/annotate_video.py rtsp://cam/stream       # network feed
    python scripts/annotate_video.py 0 --output live.mp4     # live, also recorded

A file source ends on its own; a camera or stream runs until 'q'. Note this is
the local-preview path -- the AR client gets normalised boxes over
WS /ws/track/{source_id} and draws its own overlay.
"""
import argparse
import os
import sys
from pathlib import Path

import cv2

# Running as `python scripts\annotate_video.py` puts scripts/ on sys.path,
# not the project root, so the `app` package needs to be added explicitly.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# BATTLESIGHT_MODEL is read at import time by app.config, so this has to be
# set before any `app.*` import happens.
def parse_args():
    p = argparse.ArgumentParser(description="Annotate a video with BattleSight detections")
    p.add_argument("video",
                   help="Video file, camera index (0, 1, ...), or stream URL "
                        "(rtsp://, http://). A camera or stream is live: it has no "
                        "end, so it runs until you press 'q' in the preview window, "
                        "and --show is implied unless you pass --output.")
    p.add_argument("--output", default=None,
                   help="Output path. For a file source this defaults to "
                        "<video>_annotated.mp4 next to the input; for a live source "
                        "nothing is recorded unless you pass this explicitly.")
    p.add_argument("--weights", default=None,
                   help="Override weights/best.pt (default: whatever app.config resolves)")
    p.add_argument("--view", default="ground", choices=["ground", "drone"],
                   help="Which loaded checkpoint classifies this clip: 'ground' (default, "
                        "VisDrone+WiderPerson) or 'drone' (VisDrone-only, specialised for "
                        "top-down aerial footage -- weights/drone_best.pt or "
                        "BATTLESIGHT_DRONE_MODEL). Falls back to the default model if that "
                        "checkpoint isn't present. Independent of --weights, which replaces "
                        "the default model itself rather than picking between the two.")
    p.add_argument("--source-id", default="demo", help="Feed identifier for tracking state")
    p.add_argument("--codec", default="avc1",
                   help="FourCC of the output encoder (default: avc1 = H.264). The old "
                        "default, mp4v, is MPEG-4 Part 2 and OpenCV gives it no quality "
                        "knob, so it writes ~8x larger files. Falls back to mp4v if the "
                        "requested codec cannot be opened.")
    p.add_argument("--show", action="store_true",
                   help="Also open a live preview window while processing. Runs at "
                        "inference speed (~16-20 fps on this GPU), not the source "
                        "video's native fps, so playback trails slightly behind real "
                        "time -- a realistic preview of a live camera feed. Press 'q' "
                        "in the window to stop early (the file written so far is kept).")
    return p.parse_args()


COLORS = {
    "personnel": (0, 220, 0),
    "two_wheeler": (0, 200, 255),
    "light_vehicle": (255, 160, 0),
    "heavy_vehicle": (60, 60, 255),
    "moving_object": (255, 0, 220),  # class-agnostic motion detection, unclaimed by the trained classes
}


def draw_detections(frame, detections):
    h, w = frame.shape[:2]
    for d in detections:
        x1, y1, x2, y2 = int(d["x1"] * w), int(d["y1"] * h), int(d["x2"] * w), int(d["y2"] * h)
        color = COLORS.get(d["class_name"], (200, 200, 200))
        thickness = 3 if d.get("moving") else 1
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

        label = f'{d["class_name"]} {d["confidence"]:.2f}'
        if d.get("track_id") is not None:
            label += f' #{d["track_id"]}'
        if d.get("moving"):
            label += " MOVING"

        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(frame, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
        cv2.putText(frame, label, (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    return frame


def main():
    args = parse_args()
    if args.weights:
        os.environ["BATTLESIGHT_MODEL"] = str(Path(args.weights).resolve())

    # Deferred import: must happen after BATTLESIGHT_MODEL is set above.
    from app.detector import detector

    # A camera index or a stream URL is a live source: unbounded, and with no
    # file to default an output name from.
    is_camera = args.video.isdigit()
    is_stream = "://" in args.video
    live = is_camera or is_stream
    source = int(args.video) if is_camera else args.video
    label = f"camera {source}" if is_camera else args.video

    if live:
        out_path = Path(args.output) if args.output else None
        show = args.show or out_path is None   # a live run with no output and no
                                               # window would show nothing at all
    else:
        video_path = Path(args.video)
        if not video_path.exists():
            raise SystemExit(f"No such file: {video_path}")
        out_path = Path(args.output) if args.output else video_path.with_name(
            f"{video_path.stem}_annotated.mp4"
        )
        show = args.show

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise SystemExit(f"Could not open {label}")

    writer = None
    if out_path is not None:
        fps = cap.get(cv2.CAP_PROP_FPS) or 25
        # A webcam often reports a nonsense fps (0, or 1000); the recording is
        # paced by inference anyway, so clamp it to something a player accepts.
        if not 1 <= fps <= 120:
            fps = 25
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*args.codec), fps, (w, h))
        if not writer.isOpened():
            print(f"Codec {args.codec!r} unavailable, falling back to mp4v")
            writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
        if not writer.isOpened():
            raise SystemExit(f"Could not open a video writer for {out_path}")

    print(f"Loading model ({os.environ.get('BATTLESIGHT_MODEL', 'default weights/best.pt')})...")
    detector.load()

    if show:
        window = f"BattleSight preview - {label} (q to quit)"
        cv2.namedWindow(window, cv2.WINDOW_NORMAL)

    frame_no, total_ms, consecutive_failures = 0, 0.0, 0
    while True:
        ok, frame = cap.read()
        if not ok:
            # A file has simply ended. A live source dropping a frame is a
            # glitch, not the end of the feed, so keep reading.
            if not live:
                break
            consecutive_failures += 1
            if consecutive_failures > 30:
                print("Live source stopped delivering frames; giving up.")
                break
            continue
        consecutive_failures = 0
        result = detector.track(frame, args.source_id, args.view)
        total_ms += result["inference_ms"]
        draw_detections(frame, result["detections"])
        cv2.putText(frame, f'frame {frame_no}  {result["inference_ms"]:.1f} ms',
                    (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
        if writer is not None:
            writer.write(frame)
        frame_no += 1

        if show:
            cv2.imshow(window, frame)
            # waitKey(1): show the frame and yield briefly; inference latency
            # already paces this loop, no extra delay needed to look "live".
            if cv2.waitKey(1) & 0xFF == ord("q"):
                print(f"Stopped early at frame {frame_no} (output file kept as-is)")
                break

    cap.release()
    if writer is not None:
        writer.release()
    if show:
        cv2.destroyAllWindows()
    detector.reset_source(args.source_id)

    avg = total_ms / frame_no if frame_no else 0
    if out_path is not None:
        print(f"Wrote {frame_no} frames to {out_path}")
    else:
        print(f"Processed {frame_no} frames from {label} (not recorded)")
    print(f"Average inference: {avg:.1f} ms/frame ({1000/avg:.1f} fps)" if avg else "No frames processed")


if __name__ == "__main__":
    main()
