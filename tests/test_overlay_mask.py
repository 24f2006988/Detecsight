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

    python tests/test_overlay_mask.py
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import config  # noqa: E402
from app.overlay_mask import OverlayMask  # noqa: E402


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


def check(label, got, want):
    status = "PASS" if got == want else "FAIL"
    print("  [{}] {:<52} got={} want={}".format(status, label, got, want))
    return got == want


def main():
    ok = True
    warm = int(config.OVERLAY_WARMUP_FRAMES) + 40

    # 1. A glyph-sized box pinned to one spot while the camera pans is masked.
    m = OverlayMask()
    glyph = [box(0.5, 0.5, size=0.01)]
    feed(m, "hud", glyph, warm)
    ok &= check("static glyph under a moving camera is masked",
                len(m.filter(glyph, "hud")), 0)

    # ...and a real detection elsewhere in the same frame still survives.
    ok &= check("a box in an unmasked cell survives",
                len(m.filter([box(0.2, 0.8, size=0.01)], "hud")), 1)

    # 2. Size is an absolute veto. Same position, same persistence, big box.
    m = OverlayMask()
    big = [box(0.5, 0.5, size=0.5)]
    feed(m, "big", big, warm)
    ok &= check("a LARGE persistent box is never masked",
                len(m.filter(big, "big")), 1)

    # 3. A static camera is not evidence: nothing may be masked from it.
    m = OverlayMask()
    feed(m, "still", glyph, warm, camera_moving=False)
    ok &= check("static camera contributes no evidence",
                len(m.filter(glyph, "still")), 1)

    # 4. A detection that traverses the frame is attached to the world.
    m = OverlayMask()
    for i in range(warm):
        moving = [box(0.05 + 0.9 * ((i % 40) / 40.0), 0.5, size=0.01)]
        m.observe(frame(), moving, "mover", True)
    still_there = [box(0.05 + 0.9 * ((warm % 40) / 40.0), 0.5, size=0.01)]
    ok &= check("a box traversing the frame is never masked",
                len(m.filter(still_there, "mover")), 1)

    # 5. Fail open. Cover far more than OVERLAY_MAX_FRACTION with glyphs and
    #    the filter must switch itself off rather than blind the detector.
    m = OverlayMask()
    grid = config.OVERLAY_GRID
    flood = [box((i % grid) / grid + 0.5 / grid,
                 (i // grid) / grid + 0.5 / grid, size=0.005)
             for i in range(grid * grid)]
    feed(m, "flood", flood, warm)
    info = m.debug_info("flood")
    ok &= check("whole-frame coverage disables the filter (fails OPEN)",
                len(m.filter(flood, "flood")), len(flood))
    ok &= check("...and reports itself inactive", info["overlay_active"], False)

    # 6. Nothing is masked before warmup completes.
    m = OverlayMask()
    feed(m, "cold", glyph, 5)
    ok &= check("nothing masked before warmup", len(m.filter(glyph, "cold")), 1)

    # 7. moving_object is exempt: it comes from the motion pass, which by
    #    construction cannot fire on something painted onto the sensor.
    m = OverlayMask()
    feed(m, "mo", glyph, warm)
    blob = dict(glyph[0], class_id=-1, class_name="moving_object")
    ok &= check("moving_object is never masked", len(m.filter([blob], "mo")), 1)

    # 8. State is per source, never shared between feeds.
    m = OverlayMask()
    feed(m, "feed-a", glyph, warm)
    ok &= check("another feed is unaffected", len(m.filter(glyph, "feed-b")), 1)

    print("\n" + ("PASS" if ok else "FAIL"))
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
