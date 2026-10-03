"""Removes detections that sit on a HUD or on-screen text burned into the video.

A painted glyph stays in the same spot while the camera moves, so a cell that keeps
producing small detections while the camera is panning is an overlay. It fails open.
"""
from typing import Dict, List, Optional

import numpy as np

from app import config


class _SourceState:
    """One feed's accumulated overlay evidence."""

    __slots__ = ("hits", "frames", "mask", "masked_fraction",
                 "big_n", "big_wm", "big_wm2", "big_hm", "big_hm2", "big_mask")

    def __init__(self, grid: int):
        # decayed count of frames where each cell held a small detection, and of camera-moving frames
        self.hits = np.zeros((grid, grid), np.float32)
        self.frames = 0.0
        self.mask: Optional[np.ndarray] = None
        self.masked_fraction = 0.0
        # big-box channel: running width and height stats per cell
        self.big_n = np.zeros((grid, grid), np.float32)
        self.big_wm = np.zeros((grid, grid), np.float32)
        self.big_wm2 = np.zeros((grid, grid), np.float32)
        self.big_hm = np.zeros((grid, grid), np.float32)
        self.big_hm2 = np.zeros((grid, grid), np.float32)
        self.big_mask: Optional[np.ndarray] = None


class OverlayMask:
    """Overlay detection and rejection, with separate state for each source."""

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
        """Add one frame's evidence. Call before filter(). camera_moving comes from the
        motion filter's own estimate, so camera motion is only worked out once."""
        if not config.OVERLAY_FILTER:
            return
        # only counts while the camera moves, a parked vehicle looks like a glyph on a still one
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
                if (config.OVERLAY_LARGE_FILTER
                        and d.get("class_id", 0) not in config.OVERLAY_LARGE_EXEMPT):
                    self._observe_large(st, d, grid)
                continue
            # centre cell only, a box's edges are noisy at this size
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
        """Small enough to be a HUD element. Bigger boxes are never learned from or
        suppressed. Coordinates are normalised, so this is a fraction of the frame."""
        area = (float(d["x2"]) - float(d["x1"])) * (float(d["y2"]) - float(d["y1"]))
        return 0.0 < area <= config.OVERLAY_MAX_BOX_AREA

    @staticmethod
    def _observe_large(st: _SourceState, d: dict, grid: int) -> None:
        """Fold one above-glyph-size box into its cell's geometry statistics."""
        gy, gx = OverlayMask._cell(d, grid)
        w = float(d["x2"]) - float(d["x1"])
        h = float(d["y2"]) - float(d["y1"])
        n = st.big_n[gy, gx] + 1.0
        st.big_n[gy, gx] = n
        dw = w - st.big_wm[gy, gx]
        st.big_wm[gy, gx] += dw / n
        st.big_wm2[gy, gx] += dw * (w - st.big_wm[gy, gx])
        dh = h - st.big_hm[gy, gx]
        st.big_hm[gy, gx] += dh / n
        st.big_hm2[gy, gx] += dh * (h - st.big_hm[gy, gx])

    @staticmethod
    def _rigid_cells(st: _SourceState) -> np.ndarray:
        """Cells whose large boxes are suspiciously identical. A real object changes
        apparent size as the camera moves, a painted one does not. Uses the
        coefficient of variation so it works at any box size."""
        n = st.big_n
        enough = n >= config.OVERLAY_LARGE_MIN_HITS
        if not enough.any():
            return np.zeros_like(enough)
        with np.errstate(divide="ignore", invalid="ignore"):
            sd_w = np.sqrt(np.maximum(st.big_wm2, 0.0) / np.maximum(n - 1.0, 1.0))
            sd_h = np.sqrt(np.maximum(st.big_hm2, 0.0) / np.maximum(n - 1.0, 1.0))
            cv_w = np.where(st.big_wm > 0, sd_w / np.maximum(st.big_wm, 1e-9), np.inf)
            cv_h = np.where(st.big_hm > 0, sd_h / np.maximum(st.big_hm, 1e-9), np.inf)
        return enough & (np.maximum(cv_w, cv_h) <= config.OVERLAY_LARGE_RIGIDITY)

    def _recompute(self, st: _SourceState) -> None:
        """Rebuild the mask from the persistence evidence, then apply the cap."""
        if st.frames < config.OVERLAY_WARMUP_FRAMES:
            st.mask = None
            st.big_mask = None
            st.masked_fraction = 0.0
            return

        if config.OVERLAY_LARGE_FILTER:
            big = self._rigid_cells(st)
            # same cap as the glyph channel
            st.big_mask = big if float(big.mean()) <= config.OVERLAY_MAX_FRACTION else None
        else:
            st.big_mask = None

        mask = (st.hits / max(st.frames, 1e-6)) >= config.OVERLAY_PERSISTENCE
        if config.OVERLAY_DILATE_CELLS > 0 and mask.any():
            # a glyph's box jitters between neighbouring cells, so grow the mask a bit
            # (a max filter, to avoid pulling in scipy)
            k = 2 * config.OVERLAY_DILATE_CELLS + 1
            padded = np.pad(mask, config.OVERLAY_DILATE_CELLS, constant_values=False)
            grown = np.zeros_like(mask)
            for dy in range(k):
                for dx in range(k):
                    grown |= padded[dy:dy + mask.shape[0], dx:dx + mask.shape[1]]
            mask = grown
        fraction = float(mask.mean())
        if fraction > config.OVERLAY_MAX_FRACTION:
            # too much of the frame, so the premise has failed. Fail open.
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
        if st is None or (st.mask is None and st.big_mask is None):
            return detections

        grid = config.OVERLAY_GRID
        kept = []
        for d in detections:
            # moving_object comes from the motion pass, so it is never a painted glyph
            if d.get("class_id", 0) == -1:
                kept.append(d)
                continue
            gy, gx = self._cell(d, grid)
            if not self._is_glyph_sized(d):
                if d.get("class_id", 0) in config.OVERLAY_LARGE_EXEMPT:
                    kept.append(d)
                    continue
                # above the glyph size only the rigidity channel can drop it
                if st.big_mask is None or not st.big_mask[gy, gx]:
                    kept.append(d)
                continue
            if st.mask is None or not st.mask[gy, gx]:
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
            "overlay_large_cells": int(st.big_mask.sum()) if st.big_mask is not None else 0,
        }


overlay_mask = OverlayMask()
