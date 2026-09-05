"""Static HUD/OSD overlay rejection (app/overlay_mask.py).

Covers the three properties the filter is actually relied on for, since every
one of them is a safety property for a situational-awareness system:

  1. It masks a glyph-sized detection that never moves while the camera does.
  2. It NEVER masks a large box, however persistent -- that is what protects a
     real target a drone is deliberately holding centred in frame.
  3. It fails OPEN, not closed: past OVERLAY_MAX_FRACTION it disables itself
     rather than suppressing real contacts.

Plus the two conditions that must NOT be sufficient on their own: a static
camera provides no evidence at all, and a detection that moves across the frame
is never masked no matter how many frames it appears in.

    pytest tests/test_overlay_mask.py
"""
import numpy as np
import pytest

from app import config
from app.overlay_mask import OverlayMask


def box(cx, cy, size=0.01, class_id=2):
    """A detection centred at (cx, cy), `size` wide/tall in normalised units."""
    half = size / 2.0
    return {"class_id": class_id, "class_name": "light_vehicle", "confidence": 0.5,
            "x1": cx - half, "y1": cy - half, "x2": cx + half, "y2": cy + half}


def frame():
    return np.zeros((64, 64, 3), np.uint8)


def feed(mask, source, dets_per_frame, n, camera_moving=True):
    for _ in range(n):
        mask.observe(frame(), dets_per_frame, source, camera_moving)


@pytest.fixture
def warm():
    return int(config.OVERLAY_WARMUP_FRAMES) + 40


@pytest.fixture
def glyph():
    return [box(0.5, 0.5, size=0.01)]


def test_static_glyph_under_moving_camera_is_masked(warm, glyph):
    m = OverlayMask()
    feed(m, "hud", glyph, warm)
    assert len(m.filter(glyph, "hud")) == 0

    # ...and a real detection elsewhere in the same frame still survives.
    assert len(m.filter([box(0.2, 0.8, size=0.01)], "hud")) == 1


def test_large_persistent_box_is_never_masked(warm):
    # Size is an absolute veto. Same position, same persistence, big box.
    m = OverlayMask()
    big = [box(0.5, 0.5, size=0.5)]
    feed(m, "big", big, warm)
    assert len(m.filter(big, "big")) == 1


def test_static_camera_contributes_no_evidence(warm, glyph):
    m = OverlayMask()
    feed(m, "still", glyph, warm, camera_moving=False)
    assert len(m.filter(glyph, "still")) == 1


def test_box_traversing_the_frame_is_never_masked(warm):
    # A detection that traverses the frame is attached to the world.
    m = OverlayMask()
    for i in range(warm):
        moving = [box(0.05 + 0.9 * ((i % 40) / 40.0), 0.5, size=0.01)]
        m.observe(frame(), moving, "mover", True)
    still_there = [box(0.05 + 0.9 * ((warm % 40) / 40.0), 0.5, size=0.01)]
    assert len(m.filter(still_there, "mover")) == 1


def test_whole_frame_coverage_fails_open(warm):
    # Cover far more than OVERLAY_MAX_FRACTION with glyphs and the filter must
    # switch itself off rather than blind the detector.
    m = OverlayMask()
    grid = config.OVERLAY_GRID
    flood = [box((i % grid) / grid + 0.5 / grid,
                 (i // grid) / grid + 0.5 / grid, size=0.005)
             for i in range(grid * grid)]
    feed(m, "flood", flood, warm)
    info = m.debug_info("flood")
    assert len(m.filter(flood, "flood")) == len(flood)
    assert info["overlay_active"] is False


def test_nothing_masked_before_warmup(glyph):
    m = OverlayMask()
    feed(m, "cold", glyph, 5)
    assert len(m.filter(glyph, "cold")) == 1


def test_moving_object_is_never_masked(warm, glyph):
    # moving_object comes from the motion pass, which by construction cannot
    # fire on something painted onto the sensor.
    m = OverlayMask()
    feed(m, "mo", glyph, warm)
    blob = dict(glyph[0], class_id=-1, class_name="moving_object")
    assert len(m.filter([blob], "mo")) == 1


def test_state_is_per_source(warm, glyph):
    m = OverlayMask()
    feed(m, "feed-a", glyph, warm)
    assert len(m.filter(glyph, "feed-b")) == 1
