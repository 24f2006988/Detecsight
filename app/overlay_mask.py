"""Static HUD/OSD overlay rejection.

The operational footage this system is pointed at is not clean camera output:
it is an FPV/UAV feed with a heads-up display burned into it -- crosshair
reticles, telemetry readouts, range/zoom text, unit emblems. Those glyphs are
small, high-contrast and rectangular, which is exactly what a VisDrone-trained
model has learned a distant vehicle looks like from above.

Measured on `v11.mp4` (FPV drone over open terrain) before this filter existed:
54,658 `light_vehicle` boxes across 1,746 frames -- 31.3 per frame, on a clip
containing no vehicles at all. A spatial heatmap of those boxes reproduced the
HUD exactly: one box per dash of each dotted reticle line, one per character of
the "UEX10 002587" telemetry string, one per character of the range readout.
Median false box was 9x9 px, i.e. glyph-sized, and the boxes covered 1.6% of
the frame -- the overlay's footprint.

The distinguishing property is not appearance, it is ATTACHMENT: a HUD glyph is
painted onto the sensor output, so it holds the same image-space position while
the world slides underneath it. A real object -- moving or parked -- is attached
to the world and moves across the frame as the camera pans. So this tracks, per
source, which grid cells keep producing detections *while the camera is
established to be moving*, and treats a cell that always does as overlay.

Three conditions, each closing a different failure of the others:

  1. Detection persistence, accumulated ONLY on camera-moving frames. On a
     static camera every real parked car would look persistent too, so those
     frames contribute no evidence at all.
  2. Small boxes only. Evidence is gathered from, and suppression applied to,
     detections below OVERLAY_MAX_BOX_AREA of the frame. A HUD glyph is by
     nature tiny (9x9 px measured on v11.mp4); this is what stops a drone
     that deliberately holds a real target centred in frame from ever having
     that target masked, however persistent it looks.
  3. A hard cap (OVERLAY_MAX_FRACTION): if this would mask a large part of the
     frame the assumptions have broken down, and the whole filter disables
     itself rather than blinding the detector.

Failing open is the only acceptable direction for a situational-awareness
system -- a phantom contact is a nuisance, a suppressed real one is the thing
this must never do.

TRIED AND REJECTED: requiring the cell's pixels to be temporally static as
well, on the theory that a painted glyph doesn't change while a real target
does. It carries no information on this footage and was removed. A glyph sits
on top of a CHANGING background, and an analog FPV feed is noisy everywhere,
so HUD cells are not pixel-static at all. Measured on v11.mp4, 900 frames:
of the 40 cells with detection persistence >= 0.3, requiring cell std <= 0.35x
the frame's median cell std kept only 11 of them -- while loosening the ratio
far enough to keep them (1.0x) admitted 2048 cells, half the grid, which
discriminates nothing. Box size (condition 2) does the safety job that test
was there for, and does it on evidence that actually separates the two cases.
"""
from typing import Dict, List, Optional
import numpy as np

from app import config


class _SourceState:
    """One feed's accumulated overlay evidence."""

    __slots__ = ("hits", "frames", "mask", "masked_fraction")

    def __init__(self, grid: int):
        # Decayed count of frames in which each cell held a small detection,
        # and the decayed count of qualifying (camera-moving) frames to divide
        # it by.
        self.hits = np.zeros((grid, grid), np.float32)
        self.frames = 0.0
        self.mask: Optional[np.ndarray] = None
        self.masked_fraction = 0.0


class OverlayMask:
    """Per-source static-overlay detection and rejection.

    State is per source_id and never shared between feeds -- two cameras have
    different HUDs, same reasoning as the per-feed background models in
    app/motion_filter.py.
    """

    def __init__(self):
        self._state: Dict[str, _SourceState] = {}

    def reset(self, source_id: str) -> None:
        self._state.pop(source_id, None)

    def _get(self, source_id: str) -> _SourceState:
        st = self._state.get(source_id)
        if st is None:
            st = _SourceState(config.OVERLAY_GRID)
            self._state[source_id] = st
        return st

    def observe(self, frame: np.ndarray, detections: List[dict], source_id: str,
                camera_moving: bool) -> None:
        """Fold one frame's evidence in. Call before filter().

        `camera_moving` comes from the motion pass's own ego-motion estimate
        (app/motion_filter.py), so this reuses a signal the pipeline already
        computes rather than estimating camera motion a second time.
        """
        if not config.OVERLAY_FILTER:
            return
        # Persistence only counts while the world is actually sliding past. On
        # a still camera a real parked vehicle is indistinguishable from a
        # painted glyph by this test, so those frames are not evidence at all.
        if not camera_moving:
            return

        grid = config.OVERLAY_GRID
        st = self._get(source_id)
        decay = config.OVERLAY_DECAY
        st.hits *= decay
        st.frames = st.frames * decay + 1.0

        touched = np.zeros((grid, grid), bool)
        for d in detections:
            if not self._is_glyph_sized(d):
                continue
            # Centre cell only, not the whole box: a box's extent is noisy at
            # 9 px, and it is where the thing sits that identifies it.
            gy, gx = self._cell(d, grid)
            touched[gy, gx] = True
        st.hits += touched

        self._recompute(st)

    @staticmethod
    def _cell(d: dict, grid: int) -> tuple:
        cx = (float(d["x1"]) + float(d["x2"])) * 0.5
        cy = (float(d["y1"]) + float(d["y2"])) * 0.5
        return (min(grid - 1, max(0, int(cy * grid))),
                min(grid - 1, max(0, int(cx * grid))))

    @staticmethod
    def _is_glyph_sized(d: dict) -> bool:
        """Small enough to plausibly be a HUD element. Boxes above this are
        neither learned from nor ever suppressed -- see condition 2 in the
        module docstring. Coordinates are normalised, so this is a fraction of
        the frame's area."""
        area = (float(d["x2"]) - float(d["x1"])) * (float(d["y2"]) - float(d["y1"]))
        return 0.0 < area <= config.OVERLAY_MAX_BOX_AREA

    def _recompute(self, st: _SourceState) -> None:
        """Rebuild the mask from the persistence evidence, then apply the cap."""
        if st.frames < config.OVERLAY_WARMUP_FRAMES:
            st.mask = None
            st.masked_fraction = 0.0
            return

        mask = (st.hits / max(st.frames, 1e-6)) >= config.OVERLAY_PERSISTENCE
        if config.OVERLAY_DILATE_CELLS > 0 and mask.any():
            # A glyph's box jitters, so its centre falls in one of two or three
            # neighbouring cells and no single one accumulates enough evidence
            # on its own. Grow the mask to cover the cells the jitter reaches.
            # A plain max-filter rather than scipy: one less dependency for a
            # 64x64 boolean grid, and it is the same operation.
            k = 2 * config.OVERLAY_DILATE_CELLS + 1
            padded = np.pad(mask, config.OVERLAY_DILATE_CELLS, constant_values=False)
            grown = np.zeros_like(mask)
            for dy in range(k):
                for dx in range(k):
                    grown |= padded[dy:dy + mask.shape[0], dx:dx + mask.shape[1]]
            mask = grown
        fraction = float(mask.mean())
        if fraction > config.OVERLAY_MAX_FRACTION:
            # Too much of the frame -- the premise has failed (a genuinely
            # static scene, a feed where everything persists). Fail open.
            st.mask = None
            st.masked_fraction = fraction
            return
        st.mask = mask
        st.masked_fraction = fraction

    def filter(self, detections: List[dict], source_id: str) -> List[dict]:
        """Drop detections centred in a cell established as static overlay."""
        if not config.OVERLAY_FILTER or not detections:
            return detections
        st = self._state.get(source_id)
        if st is None or st.mask is None:
            return detections

        grid = config.OVERLAY_GRID
        kept = []
        for d in detections:
            # moving_object comes from the motion pass, which by construction
            # never fires on something that isn't moving relative to the
            # world -- it cannot be a painted glyph, so it is never masked.
            if d.get("class_id", 0) == -1 or not self._is_glyph_sized(d):
                kept.append(d)
                continue
            gy, gx = self._cell(d, grid)
            if not st.mask[gy, gx]:
                kept.append(d)
        return kept

    def debug_info(self, source_id: str) -> dict:
        """Mask state for this source -- diagnosis only, never read by filter()."""
        st = self._state.get(source_id)
        if st is None:
            return {}
        return {
            "overlay_frames": round(st.frames, 1),
            "overlay_active": st.mask is not None,
            "overlay_masked_fraction": round(st.masked_fraction, 4),
            "overlay_cells": int(st.mask.sum()) if st.mask is not None else 0,
        }


overlay_mask = OverlayMask()
