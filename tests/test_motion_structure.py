"""Illumination-vs-structure discriminator and the brief-appearance fast path.

    pytest tests/test_motion_structure.py

What these pin down, in the project's own terms:
  * a LIGHT (muzzle flash, headlight, glare) changes brightness while leaving
    structure intact -- it must score LOW and be rejected;
  * a real object moving into a region changes what is there -- it must score
    HIGH and survive;
  * the test FAILS OPEN wherever it cannot judge (no previous frame, tiny box,
    featureless patch), because a suppressed real contact is the failure this
    system must never have;
  * a target visible for only 2 frames is reportable via the fast path, which
    is impossible under MOTION_COHERENCE_MIN_POINTS=4 alone.
"""
import numpy as np
import pytest

from app import config
from app.motion_filter import MotionDetector

BOX = (20, 20, 100, 90)


def textured(seed=0, shape=(120, 160)):
    """A patch with real structure -- the test is undefined on flat images."""
    rng = np.random.default_rng(seed)
    return (rng.integers(40, 200, shape)).astype(np.uint8)


@pytest.fixture
def base():
    return textured()


@pytest.fixture
def brighter(base):
    return np.clip(base.astype(np.int32) + 45, 0, 255).astype(np.uint8)


# --- structure score -------------------------------------------------------

def test_uniform_brightening_scores_low(base, brighter):
    """A light: same scene, uniformly brighter. Structure is preserved."""
    s_light = MotionDetector._structure_score(base, brighter, BOX)
    assert s_light < config.MOTION_STRUCTURE_MIN


def test_contrast_only_change_scores_low(base):
    """Contrast change, also illumination -- structure still preserved."""
    contrast = np.clip(base.astype(np.float32) * 1.5, 0, 255).astype(np.uint8)
    assert MotionDetector._structure_score(base, contrast, BOX) < config.MOTION_STRUCTURE_MIN_FAST


def test_new_content_scores_high_and_outscores_a_light(base, brighter):
    """A real object: different content lands in the box."""
    obj = base.copy()
    obj[30:80, 30:90] = textured(seed=7, shape=(50, 60))
    s_obj = MotionDetector._structure_score(base, obj, BOX)
    s_light = MotionDetector._structure_score(base, brighter, BOX)
    assert s_obj > config.MOTION_STRUCTURE_MIN
    assert s_obj > s_light


# --- fails open when it cannot judge ---------------------------------------

def test_no_previous_frame_fails_open(base):
    assert MotionDetector._structure_score(None, base, BOX) == 1.0


def test_degenerate_box_fails_open(base, brighter):
    assert MotionDetector._structure_score(base, brighter, (10, 10, 11, 11)) == 1.0


def test_featureless_patch_fails_open():
    flat = np.full((120, 160), 128, np.uint8)
    assert MotionDetector._structure_score(flat, flat + 30, BOX) == 1.0


def test_shape_mismatch_fails_open(base):
    assert MotionDetector._structure_score(np.zeros((10, 10), np.uint8), base, BOX) == 1.0


# --- brief appearance survives the fast path -------------------------------

def test_two_frame_appearance_is_reported():
    """A bright square crossing a static textured scene, present for 2 frames.

    Under MOTION_COHERENCE_MIN_POINTS=4 this is unreportable by construction.
    """
    det = MotionDetector()
    rng = np.random.default_rng(3)
    bg = rng.integers(60, 180, (240, 320, 3)).astype(np.uint8)

    def frame_with_object(x):
        f = bg.copy()
        if x is not None:
            f[100:140, x:x + 40] = textured(seed=11, shape=(40, 40))[:, :, None]
        return f

    for _ in range(12):                      # let MOG2 learn the background
        det.detect(bg.copy(), "brief")

    seen_fast = False
    for x in (60, 100):                      # object visible for exactly 2 frames
        out = det.detect(frame_with_object(x), "brief")
        if any(d.get("fast") for d in out):
            seen_fast = True
    assert seen_fast is True


# --- config sanity ---------------------------------------------------------

def test_fast_path_needs_fewer_points():
    assert config.MOTION_COHERENCE_MIN_POINTS_FAST < config.MOTION_COHERENCE_MIN_POINTS


def test_fast_bar_is_stricter_than_the_normal_bar():
    assert config.MOTION_STRUCTURE_MIN_FAST > config.MOTION_STRUCTURE_MIN


def test_degraded_band_is_above_the_drop_threshold():
    assert config.EGO_RESIDUAL_DEGRADED_FACTOR > 1.0
