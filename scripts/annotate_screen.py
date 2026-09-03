"""Feed a live screen capture through the Detector, with boxes drawn -- for
testing against anything already playing on your monitor (a YouTube video in
a browser, a video call, another app's preview window) without downloading it
first or wiring up a camera.

Sibling of scripts/annotate_video.py: same Detector, same drawing, same
--view/--weights/--output/--show conventions. The only difference is where
frames come from -- mss grabs a screen region instead of cv2.VideoCapture
reading a file/camera/stream.

    python scripts/annotate_screen.py                          # primary monitor, preview only
    python scripts/annotate_screen.py --window chrome          # just that window, tracks it if it moves
    python scripts/annotate_screen.py --region 100,100,1280,720
    python scripts/annotate_screen.py --window chrome --output capture.mp4
    python scripts/annotate_screen.py --view drone              # aerial-view checkpoint

Runs until 'q' in the preview window, or Ctrl+C if running with --output and
no --show (the file written so far is kept either way). Screen capture has no
native frame rate -- like the webcam/stream path in annotate_video.py, this
reads-infers-draws as fast as inference allows and does not pace itself to
match the source video's playback speed.
"""
import argparse
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np

# Running as `python scripts\annotate_screen.py` puts scripts/ on sys.path,
# so this sibling import works the same way annotate_video.py's own `app`
# import does one level up -- see its comment.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from annotate_video import COLORS, draw_detections  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description="Feed a live screen capture through BattleSight detection")
    p.add_argument("--monitor", type=int, default=1,
                   help="mss monitor index to capture (1 = primary, 2 = second, "
                        "0 = all monitors combined into one image). Default 1. "
                        "Ignored if --region or --window is given.")
    p.add_argument("--region", default=None,
                   help="Explicit pixel region 'x,y,w,h' to capture, e.g. "
                        "'100,100,1280,720'. Overrides --monitor.")
    p.add_argument("--window", default=None,
                   help="Capture the bounding box of the first visible window whose "
                        "title contains this text (case-insensitive), e.g. --window "
                        "chrome or --window 'YouTube'. Re-read every frame, so it "
                        "keeps tracking the window if it's moved or resized. "
                        "Overrides --monitor/--region.")
    p.add_argument("--output", default=None,
                   help="Also record the annotated capture to this file, e.g. "
                        "capture_annotated.mp4. Not recorded by default.")
    p.add_argument("--weights", default=None,
                   help="Override weights/best.pt (default: whatever app.config resolves)")
    p.add_argument("--view", default="ground", choices=["ground", "drone"],
                   help="Which loaded checkpoint classifies this feed: 'ground' "
                        "(default) or 'drone' (top-down aerial footage) -- see "
                        "config.DRONE_MODEL_PATH.")
    p.add_argument("--source-id", default="screen", help="Feed identifier for tracking state")
    p.add_argument("--codec", default="avc1",
                   help="FourCC of the output encoder (default: avc1 = H.264). "
                        "Falls back to mp4v if the requested codec cannot be opened.")
    p.add_argument("--show", action="store_true",
                   help="Open a live preview window. Implied if --output isn't given "
                        "(otherwise nothing would be visible at all). Press 'q' to stop.")
    p.add_argument("--fps", type=int, default=20,
                   help="Nominal fps written into the output file's header (default 20). "
                        "Screen capture has no native rate, so this only affects how the "
                        "recorded file's playback speed is labelled, not capture speed.")
    p.add_argument("--max-frames", type=int, default=None,
                   help="Stop automatically after this many frames instead of running "
                        "until 'q'/Ctrl+C. Useful for a fixed-length capture or a quick test.")
    return p.parse_args()


def _find_window(title_substr: str):
    """Return the pygetwindow Window object for the first visible, non-empty
    window whose title contains title_substr (case-insensitive).

    Returned once and re-queried for live .left/.top/.width/.height every
    frame by the caller -- pygetwindow's Window wraps a live OS handle, so
    re-reading its properties reflects the window's current position/size
    without a fresh search.
    """
    import pygetwindow as gw

    needle = title_substr.lower()
    candidates = [
        w for w in gw.getAllWindows()
        if needle in w.title.lower() and w.visible and w.width > 0 and w.height > 0
    ]
    if not candidates:
        titles = sorted({w.title for w in gw.getAllWindows() if w.title.strip()})
        raise SystemExit(
            f"No visible window with title containing {title_substr!r}.\n"
            f"Open windows:\n  " + "\n  ".join(titles[:40])
        )
    return candidates[0]


def main():
    args = parse_args()
    if args.weights:
        os.environ["BATTLESIGHT_MODEL"] = str(Path(args.weights).resolve())

    # Deferred import: must happen after BATTLESIGHT_MODEL is set above --
    # same reasoning as annotate_video.py.
    from app.detector import detector

    import mss

    window = _find_window(args.window) if args.window else None

    def current_region():
        if window is not None:
            return {"left": window.left, "top": window.top,
                     "width": window.width, "height": window.height}
        if args.region:
            x, y, w, h = (int(v) for v in args.region.split(","))
            return {"left": x, "top": y, "width": w, "height": h}
        return None  # resolved from sct.monitors[args.monitor] once mss is open

    sct = mss.MSS()
    monitor_region = current_region() or sct.monitors[args.monitor]

    label = (f"window '{args.window}'" if args.window
              else f"region {args.region}" if args.region
              else f"monitor {args.monitor}")

    writer = None
    writer_wh = None
    if args.output:
        region = current_region() or monitor_region
        writer_wh = (region["width"], region["height"])
        writer = cv2.VideoWriter(str(args.output), cv2.VideoWriter_fourcc(*args.codec),
                                  args.fps, writer_wh)
        if not writer.isOpened():
            print(f"Codec {args.codec!r} unavailable, falling back to mp4v")
            writer = cv2.VideoWriter(str(args.output), cv2.VideoWriter_fourcc(*"mp4v"),
                                      args.fps, writer_wh)
        if not writer.isOpened():
            raise SystemExit(f"Could not open a video writer for {args.output}")

    show = args.show or writer is None

    print(f"Capturing {label}. Loading model ({os.environ.get('BATTLESIGHT_MODEL', 'default weights/best.pt')})...")
    detector.load()

    if show:
        win_name = f"BattleSight live screen capture - {label} (q to quit)"
        cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)

    frame_no, total_ms = 0, 0.0
    try:
        while True:
            region = current_region() or monitor_region
            raw = np.array(sct.grab(region))  # BGRA, top-left origin
            if raw.size == 0:
                continue  # window minimised or momentarily off-screen
            frame = cv2.cvtColor(raw, cv2.COLOR_BGRA2BGR)

            # A tracked window that moved/resized can hand back a different
            # shape than the writer was opened with; drop that frame rather
            # than write a size mismatch cv2.VideoWriter will silently corrupt.
            if writer is not None and (frame.shape[1], frame.shape[0]) != writer_wh:
                continue

            result = detector.track(frame, args.source_id, args.view)
            total_ms += result["inference_ms"]
            draw_detections(frame, result["detections"])
            cv2.putText(frame, f'frame {frame_no}  {result["inference_ms"]:.1f} ms',
                        (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

            if writer is not None:
                writer.write(frame)
            frame_no += 1

            if show:
                cv2.imshow(win_name, frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    print(f"Stopped at frame {frame_no} (output file kept as-is)")
                    break
            if args.max_frames is not None and frame_no >= args.max_frames:
                print(f"Reached --max-frames {args.max_frames}")
                break
    except KeyboardInterrupt:
        print(f"\nInterrupted at frame {frame_no} (output file kept as-is)")
    finally:
        sct.close()
        if writer is not None:
            writer.release()
        if show:
            cv2.destroyAllWindows()
        detector.reset_source(args.source_id)

    avg = total_ms / frame_no if frame_no else 0
    if args.output:
        print(f"Wrote {frame_no} frames to {args.output}")
    else:
        print(f"Processed {frame_no} frames from {label} (not recorded)")
    print(f"Average inference: {avg:.1f} ms/frame ({1000/avg:.1f} fps)" if avg else "No frames processed")


if __name__ == "__main__":
    main()
