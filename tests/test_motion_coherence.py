"""Verify MotionDetector separates coherent motion (a moving object) from
incoherent jitter (e.g. wind-blown foliage) via trajectory straightness.
No API/model needed -- pure app.motion_filter logic on synthetic frames.
"""
import itertools
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

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


def main():
    # A square walking steadily in one direction -- a straight trajectory.
    coherent_positions = [(20 + i * 10, 100) for i in range(12)]
    coherent = run("coherent", coherent_positions)
    print(f"Coherent (steady walk):   {len(coherent)} blob(s) survived -> {coherent}")
    assert len(coherent) >= 1, "a steadily-moving object should be detected"

    # A square jittering back and forth around a fixed point -- foliage-like.
    jitter_cycle = [(150, 100), (158, 92), (144, 108), (154, 96)]
    jitter_positions = list(itertools.islice(itertools.cycle(jitter_cycle), 12))
    jitter = run("jitter", jitter_positions)
    print(f"Jitter (foliage-like):    {len(jitter)} blob(s) survived -> {jitter}")
    assert len(jitter) == 0, "incoherent jitter should be filtered out"

    print("PASS")


if __name__ == "__main__":
    main()
