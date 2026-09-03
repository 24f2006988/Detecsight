"""Illumination-vs-structure discriminator and the brief-appearance fast path.

Run directly (no pytest in this venv):
    $env:PYTHONPATH="."; python tests\test_motion_structure.py

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
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import config                      # noqa: E402
from app.motion_filter import MotionDetector  # noqa: E402

FAILS = []


def check(name, got, want):
    ok = got == want
    print(f"  [{'PASS' if ok else 'FAIL'}] {name:<52} got={got} want={want}")
    if not ok:
        FAILS.append(name)


def check_cmp(name, got, op, bound):
    ok = {"<": got < bound, ">": got > bound, ">=": got >= bound, "<=": got <= bound}[op]
    print(f"  [{'PASS' if ok else 'FAIL'}] {name:<52} {got:.3f} {op} {bound}")
    if not ok:
        FAILS.append(name)


def textured(seed=0, shape=(120, 160)):
    """A patch with real structure -- the test is undefined on flat images."""
    rng = np.random.default_rng(seed)
    return (rng.integers(40, 200, shape)).astype(np.uint8)


print("structure score")
base = textured()
box = (20, 20, 100, 90)

# A light: same scene, uniformly brighter. Structure is preserved.
brighter = np.clip(base.astype(np.int32) + 45, 0, 255).astype(np.uint8)
s_light = MotionDetector._structure_score(base, brighter, box)
check_cmp("uniform brightening scores low (a light)", s_light, "<",
          config.MOTION_STRUCTURE_MIN)

# Contrast change, also illumination -- structure still preserved.
contrast = np.clip(base.astype(np.float32) * 1.5, 0, 255).astype(np.uint8)
s_contrast = MotionDetector._structure_score(base, contrast, box)
check_cmp("contrast-only change scores low", s_contrast, "<",
          config.MOTION_STRUCTURE_MIN_FAST)

# A real object: different content lands in the box.
obj = base.copy()
obj[30:80, 30:90] = textured(seed=7, shape=(50, 60))
s_obj = MotionDetector._structure_score(base, obj, box)
check_cmp("new content in the box scores high (an object)", s_obj, ">",
          config.MOTION_STRUCTURE_MIN)
check("object outscores light", s_obj > s_light, True)

print("fails open when it cannot judge")
check("no previous frame -> 1.0",
      MotionDetector._structure_score(None, base, box), 1.0)
check("degenerate box -> 1.0",
      MotionDetector._structure_score(base, brighter, (10, 10, 11, 11)), 1.0)
flat = np.full((120, 160), 128, np.uint8)
check("featureless patch -> 1.0",
      MotionDetector._structure_score(flat, flat + 30, box), 1.0)
check("shape mismatch -> 1.0",
      MotionDetector._structure_score(np.zeros((10, 10), np.uint8), base, box), 1.0)

print("brief appearance survives the fast path")
# A bright square crossing a static textured scene, present for 2 frames only.
# Under MOTION_COHERENCE_MIN_POINTS=4 this is unreportable by construction.
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
for i, x in enumerate((60, 100)):        # object visible for exactly 2 frames
    out = det.detect(frame_with_object(x), "brief")
    if any(d.get("fast") for d in out):
        seen_fast = True
check("2-frame appearance is reported at all", seen_fast, True)
check("fast path needs fewer points than the normal path",
      config.MOTION_COHERENCE_MIN_POINTS_FAST < config.MOTION_COHERENCE_MIN_POINTS,
      True)

print("config sanity")
check("fast bar is stricter than the normal bar",
      config.MOTION_STRUCTURE_MIN_FAST > config.MOTION_STRUCTURE_MIN, True)
check("degraded band is above the drop threshold",
      config.EGO_RESIDUAL_DEGRADED_FACTOR > 1.0, True)

print("\nFAIL" if FAILS else "\nPASS")
if FAILS:
    print("failed:", ", ".join(FAILS))
sys.exit(1 if FAILS else 0)
