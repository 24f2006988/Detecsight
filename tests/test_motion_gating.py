"""Verify the stage-3 motion gate: the classifier must run ONLY on crops that
survived stages 1-2, and must not run at all on a frame where nothing moved.

That "not at all" is the whole point of the gate, so it is asserted directly by
counting calls into the model rather than by timing anything, which would be
flaky on a loaded machine. Uses the stock yolo26n checkpoint on CPU so this
needs no trained weights and no GPU.
"""
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Config reads the environment at import time, so this has to precede it.
os.environ.setdefault("BATTLESIGHT_MODEL", "yolo26n.pt")
os.environ.setdefault("BATTLESIGHT_DEVICE", "cpu")
os.environ.setdefault("BATTLESIGHT_MOTION_GATED", "1")

from app import config  # noqa: E402
from app.detector import Detector  # noqa: E402
from app.motion_filter import merge_boxes, pad_box  # noqa: E402

FRAME_SHAPE = (480, 640, 3)  # h, w, c


def make_frame(top_left, size=48):
    frame = np.zeros(FRAME_SHAPE, dtype=np.uint8)
    x, y = top_left
    frame[y:y + size, x:x + size] = 255
    return frame


def test_crop_geometry():
    # Padding grows the box and min_size floors it; both clip to the frame.
    x1, y1, x2, y2 = pad_box((100, 100, 120, 130), 640, 480, padding=0.6, min_size=128)
    assert (x2 - x1) >= 128 and (y2 - y1) >= 128, "min_size floor not applied"
    assert x1 >= 0 and y1 >= 0 and x2 <= 640 and y2 <= 480, "crop escaped the frame"

    # A blob in the corner must stay inside the frame rather than go negative.
    cx1, cy1, _, _ = pad_box((0, 0, 10, 10), 640, 480, padding=1.0, min_size=200)
    assert cx1 == 0 and cy1 == 0

    # Two heavily overlapping boxes become one; a distant third stays separate.
    merged = merge_boxes([(0, 0, 100, 100), (10, 10, 105, 105), (400, 400, 450, 450)],
                         iou_threshold=0.2)
    assert len(merged) == 2, f"expected 2 merged crops, got {merged}"
    assert (0, 0, 105, 105) in merged
    print(f"Crop geometry: pad+clip OK, merge {3} -> {len(merged)} crops")


def main():
    test_crop_geometry()

    det = Detector()
    det.load()

    # Count every call into the network, on both the batched-crop path and the
    # full-frame fallback.
    calls = {"n": 0}
    real_predict = det.model.predict

    def counting_predict(*args, **kwargs):
        calls["n"] += 1
        return real_predict(*args, **kwargs)

    det.model.predict = counting_predict

    # --- a target crossing the frame: the gate must open ---
    moving = None
    for i in range(12):
        moving = det.track(make_frame((40 + i * 30, 200)), "gated")
    print(f"Moving target:  {len(moving['detections'])} detection(s), "
          f"{calls['n']} model call(s), {moving['inference_ms']} ms")
    assert calls["n"] > 0, "the gate never opened for a coherently moving target"
    assert moving["detections"], "a coherently moving target produced no detection"
    assert any(d["track_id"] is not None for d in moving["detections"]), \
        "gated detections must carry the motion tracker's id"

    # --- a still scene: the gate must stay shut ---
    still = make_frame((300, 200))
    for _ in range(6):
        det.track(still, "gated")  # let MOG2 absorb the square into the background
    calls["n"] = 0
    quiet = None
    for _ in range(6):
        quiet = det.track(still, "gated")
    print(f"Still scene:    {len(quiet['detections'])} detection(s), "
          f"{calls['n']} model call(s), {quiet['inference_ms']} ms")
    assert calls["n"] == 0, \
        f"the classifier ran {calls['n']}x on a frame where nothing moved"
    assert quiet["detections"] == []

    # --- the ungated path must still work ---
    config.MOTION_GATED = False
    det.reset_source("ungated")
    ungated = det.track(make_frame((100, 100)), "ungated")
    print(f"Ungated path:   {len(ungated['detections'])} detection(s), "
          f"{ungated['inference_ms']} ms")
    config.MOTION_GATED = True

    print("PASS")


if __name__ == "__main__":
    main()
