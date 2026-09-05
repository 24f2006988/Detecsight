"""Verify MotionDetector separates coherent motion (a moving object) from
incoherent jitter (e.g. wind-blown foliage) via trajectory straightness.
No API/model needed -- pure app.motion_filter logic on synthetic frames.
"""
import itertools

import numpy as np

from app.motion_filter import MotionDetector

FRAME_SHAPE = (240, 320, 3)  # h, w, c


def make_frame(top_left, size=24):
    frame = np.zeros(FRAME_SHAPE, dtype=np.uint8)
    x, y = top_left
    frame[y:y + size, x:x + size] = 255
    return frame


def run(source_id, positions):
    md = MotionDetector()
    last = []
    for pos in positions:
        last = md.detect(make_frame(pos), source_id)
    return last


def test_coherent_walk_survives():
    """A square walking steadily in one direction -- a straight trajectory."""
    coherent = run("coherent", [(20 + i * 10, 100) for i in range(12)])
    assert len(coherent) >= 1, "a steadily-moving object should be detected"


def test_incoherent_jitter_is_filtered():
    """A square jittering back and forth around a fixed point -- foliage-like."""
    jitter_cycle = [(150, 100), (158, 92), (144, 108), (154, 96)]
    jitter = run("jitter", list(itertools.islice(itertools.cycle(jitter_cycle), 12)))
    assert len(jitter) == 0, "incoherent jitter should be filtered out"
