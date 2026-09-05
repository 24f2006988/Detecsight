"""Verify the stage-3 motion gate: the classifier must run ONLY on crops that
survived stages 1-2, and must not run at all on a frame where nothing moved.

That "not at all" is the whole point of the gate, so it is asserted directly by
counting calls into the model rather than by timing anything, which would be
flaky on a loaded machine. Uses the stock yolo26n checkpoint on CPU so this
needs no trained weights and no GPU -- it is marked `model` rather than `gpu`
for that reason, and skips cleanly when the checkpoint is not on disk.

    pytest -m model tests/test_motion_gating.py

The crop-geometry assertions below are pure arithmetic on app.motion_filter and
carry no marker, so they run in the default CPU subset. Detector (and with it
torch) is imported inside the marked tests only -- keeping the default suite
installable without the model stack is a property CI enforces.
"""
from pathlib import Path

import numpy as np
import pytest

from app.motion_filter import merge_boxes, pad_box

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


@pytest.fixture
def gated_detector(monkeypatch):
    """A CPU Detector on the stock checkpoint, with the gate forced on.

    config reads the environment at import time, and pytest has usually already
    imported app.config via another test module by now, so the settings are
    patched as attributes rather than through os.environ.
    """
    from app import config
    from app.detector import Detector

    if not Path("yolo26n.pt").exists():
        pytest.skip("yolo26n.pt not present; fetch it or run from the repo root")

    monkeypatch.setattr(config, "MODEL_PATH", "yolo26n.pt")
    monkeypatch.setattr(config, "DRONE_MODEL_PATH", "yolo26n-absent.pt")
    monkeypatch.setattr(config, "DEVICE", "cpu")
    monkeypatch.setattr(config, "MOTION_GATED", True)

    det = Detector()
    det.load()
    yield det
    det.unload()


@pytest.mark.model
def test_gate_opens_for_a_moving_target(gated_detector):
    det = gated_detector
    calls = {"n": 0}
    real_predict = det.model.predict

    def counting_predict(*args, **kwargs):
        calls["n"] += 1
        return real_predict(*args, **kwargs)

    det.model.predict = counting_predict

    moving = None
    for i in range(12):
        moving = det.track(make_frame((40 + i * 30, 200)), "gated")

    assert calls["n"] > 0, "the gate never opened for a coherently moving target"
    assert moving["detections"], "a coherently moving target produced no detection"
    assert any(d["track_id"] is not None for d in moving["detections"]), \
        "gated detections must carry the motion tracker's id"


@pytest.mark.model
def test_gate_stays_shut_on_a_still_scene(gated_detector):
    det = gated_detector
    calls = {"n": 0}
    real_predict = det.model.predict

    def counting_predict(*args, **kwargs):
        calls["n"] += 1
        return real_predict(*args, **kwargs)

    det.model.predict = counting_predict

    still = make_frame((300, 200))
    for _ in range(6):
        det.track(still, "gated")  # let MOG2 absorb the square into the background
    calls["n"] = 0
    quiet = None
    for _ in range(6):
        quiet = det.track(still, "gated")

    assert calls["n"] == 0, \
        f"the classifier ran {calls['n']}x on a frame where nothing moved"
    assert quiet["detections"] == []


@pytest.mark.model
def test_ungated_path_still_works(gated_detector, monkeypatch):
    from app import config

    monkeypatch.setattr(config, "MOTION_GATED", False)
    det = gated_detector
    det.reset_source("ungated")
    ungated = det.track(make_frame((100, 100)), "ungated")
    assert "detections" in ungated
