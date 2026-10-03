"""Motion detection that does not need to know what the object is.

Background subtraction finds what changed, then each blob is followed for a few
frames and kept only if it moves in a fairly straight line. Foliage jitters in
place, a person or a UAV does not.
"""
from collections import deque
from typing import Dict, List

import cv2
import numpy as np

from app import config


def iou_xyxy(a: tuple, b: tuple) -> float:
    """Intersection-over-union of two (x1, y1, x2, y2) boxes in matching units."""
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def merge_boxes(boxes: List[tuple], iou_threshold: float) -> List[tuple]:
    """Merge boxes that overlap more than iou_threshold, until none do."""
    merged = [list(b) for b in boxes]
    changed = True
    while changed:
        changed = False
        out: List[list] = []
        for b in merged:
            for o in out:
                if iou_xyxy(tuple(b), tuple(o)) > iou_threshold:
                    o[0], o[1] = min(o[0], b[0]), min(o[1], b[1])
                    o[2], o[3] = max(o[2], b[2]), max(o[3], b[3])
                    changed = True
                    break
            else:
                out.append(b)
        merged = out
    return [tuple(b) for b in merged]


def pad_box(box: tuple, frame_w: int, frame_h: int, padding: float,
            min_size: int) -> tuple:
    """Grow a blob box into a crop box, clipped to the frame. A blob is only the
    moving part of a target, so a tight crop would cut it in half."""
    x1, y1, x2, y2 = box
    w, h = max(1.0, x2 - x1), max(1.0, y2 - y1)
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    half_w = max(w * (1.0 + padding), min_size) / 2.0
    half_h = max(h * (1.0 + padding), min_size) / 2.0
    return (
        int(max(0, cx - half_w)), int(max(0, cy - half_h)),
        int(min(frame_w, cx + half_w)), int(min(frame_h, cy + half_h)),
    )


class _Track:
    __slots__ = ("id", "history", "bbox", "age")

    def __init__(self, track_id: int, centroid: tuple, bbox: tuple):
        self.id = track_id
        self.history = deque([centroid], maxlen=15)
        self.bbox = bbox
        self.age = 0


class MotionDetector:
    """Background subtraction and trajectory checks, with separate state per source."""

    def __init__(self):
        self._bg: Dict[str, cv2.BackgroundSubtractorMOG2] = {}
        self._tracks: Dict[str, List[_Track]] = {}
        self._next_id: Dict[str, int] = {}
        # previous grey frame per source, used to estimate the camera motion
        self._prev_gray: Dict[str, np.ndarray] = {}
        # whether the last frame's camera-motion cancelling can be trusted
        self._ego_reliable: Dict[str, bool] = {}
        # smoothed foreground fraction, so the check does not flicker on and off
        self._ego_residual_ema: Dict[str, float] = {}
        # smoothed camera shift, same reason
        self._shift_ema: Dict[str, float] = {}
        # frames still being suppressed after the chronic-blob check tripped
        self._chronic_cooldown: Dict[str, int] = {}
        # last frame's raw signals, for diagnosis only
        self._debug: Dict[str, dict] = {}

    def _subtractor(self, source_id: str):
        bg = self._bg.get(source_id)
        if bg is None:
            bg = cv2.createBackgroundSubtractorMOG2(history=200, varThreshold=25, detectShadows=True)
            self._bg[source_id] = bg
        return bg

    def reset(self, source_id: str):
        self._bg.pop(source_id, None)
        self._tracks.pop(source_id, None)
        self._next_id.pop(source_id, None)
        self._prev_gray.pop(source_id, None)
        self._ego_reliable.pop(source_id, None)
        self._ego_residual_ema.pop(source_id, None)
        self._shift_ema.pop(source_id, None)
        self._chronic_cooldown.pop(source_id, None)

    @staticmethod
    def _estimate_ego(prev_gray: np.ndarray, gray: np.ndarray):
        """Global frame-to-frame transform (2x3 affine), or None if the scene has
        too little to fit one. Tracked corners with RANSAC, so moving objects
        show up as outliers and don't pull the fit."""
        pts = cv2.goodFeaturesToTrack(prev_gray, maxCorners=200, qualityLevel=0.01,
                                      minDistance=8, blockSize=7)
        if pts is None or len(pts) < config.EGO_MIN_FEATURES:
            return None
        nxt, status, _ = cv2.calcOpticalFlowPyrLK(prev_gray, gray, pts, None)
        if nxt is None or status is None:
            return None
        ok = status.ravel() == 1
        if int(ok.sum()) < config.EGO_MIN_FEATURES:
            return None
        M, _ = cv2.estimateAffinePartial2D(pts[ok], nxt[ok], method=cv2.RANSAC,
                                           ransacReprojThreshold=3.0)
        return M

    @staticmethod
    def _compensated_mask(prev_gray: np.ndarray, gray: np.ndarray, M) -> np.ndarray:
        """Foreground mask with the camera's own motion cancelled. The previous frame
        is warped onto this one, so what is left is motion the camera does not
        explain. Borders are dropped, the warp pulls in pixels it never saw."""
        h, w = gray.shape[:2]
        warped = cv2.warpAffine(prev_gray, M, (w, h), flags=cv2.INTER_LINEAR,
                                borderMode=cv2.BORDER_REPLICATE)
        diff = cv2.absdiff(gray, warped)
        _, mask = cv2.threshold(diff, config.EGO_DIFF_THRESHOLD, 255, cv2.THRESH_BINARY)
        valid = cv2.warpAffine(np.full((h, w), 255, np.uint8), M, (w, h),
                               flags=cv2.INTER_NEAREST, borderValue=0)
        valid = cv2.erode(valid, np.ones((7, 7), np.uint8), iterations=2)
        return cv2.bitwise_and(mask, valid)

    @staticmethod
    def _restabilise(tracks: List["_Track"], M) -> None:
        """Move the stored track history into the current frame's coordinates, so a
        pan doesn't count as the object's own movement."""
        a, b, tx = float(M[0][0]), float(M[0][1]), float(M[0][2])
        c, d, ty = float(M[1][0]), float(M[1][1]), float(M[1][2])
        for t in tracks:
            t.history = deque(
                ((a * x + b * y + tx, c * x + d * y + ty) for x, y in t.history),
                maxlen=t.history.maxlen,
            )

    @staticmethod
    def _straightness(history: deque) -> float:
        """net displacement / total path length, in [0, 1]. Low = jitter (foliage),
        high = a consistent trajectory (a walking human, a flying UAV)."""
        pts = list(history)
        net = ((pts[-1][0] - pts[0][0]) ** 2 + (pts[-1][1] - pts[0][1]) ** 2) ** 0.5
        path = sum(
            ((pts[i][0] - pts[i - 1][0]) ** 2 + (pts[i][1] - pts[i - 1][1]) ** 2) ** 0.5
            for i in range(1, len(pts))
        )
        if path < config.MOTION_COHERENCE_MIN_PATH:
            return 0.0  # barely moved at all -- noise, not a track worth judging
        return net / path

    @staticmethod
    def _structure_score(prev_aligned, gray, box) -> float:
        """How much of the change under `box` is structure and not just light.

        Close to 1 means the content changed (something moved through). Close to 0
        means only the brightness changed (a flash, a headlight). Returns 1.0 when
        it can't judge, since a missed real contact is worse than a false one.
        """
        if prev_aligned is None or prev_aligned.shape != gray.shape:
            return 1.0
        x1, y1, x2, y2 = (int(v) for v in box)
        h, w = gray.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        if x2 - x1 < 3 or y2 - y1 < 3:
            return 1.0                       # too small to say anything
        a = prev_aligned[y1:y2, x1:x2].astype(np.float32)
        b = gray[y1:y2, x1:x2].astype(np.float32)
        sa, sb = float(a.std()), float(b.std())
        if sa < config.MOTION_STRUCTURE_MIN_STD or sb < config.MOTION_STRUCTURE_MIN_STD:
            return 1.0                       # featureless patch -- fail open

        # a uniform brightness shift moves every pixel by about the same amount
        d = b - a
        spread, level = float(d.std()), abs(float(d.mean()))
        if level > 1.0 and spread / level < config.MOTION_STRUCTURE_UNIFORM_RATIO:
            return 0.0

        # a brightness or contrast change keeps the structure, so the patches still correlate
        ncc = float((((a - a.mean()) / sa) * ((b - b.mean()) / sb)).mean())
        return 1.0 - max(0.0, min(1.0, ncc))

    @staticmethod
    def _coherence_threshold(view: str) -> float:
        """Drone and ground want different thresholds (see config). Any other view
        gets the ground one, the more permissive of the two."""
        return {
            "drone": config.MOTION_COHERENCE_THRESHOLD_DRONE,
            "ground": config.MOTION_COHERENCE_THRESHOLD_GROUND,
        }.get(view, config.MOTION_COHERENCE_THRESHOLD_GROUND)

    def detect(self, frame: np.ndarray, source_id: str, view: str = "ground") -> List[dict]:
        """Return the blobs that move coherently as [{x1,y1,x2,y2,coherence}] in the
        original frame's pixels. Runs on a downscaled copy and scales back up.
        `view` only changes the coherence threshold."""
        coherence_threshold = self._coherence_threshold(view)
        h0, w0 = frame.shape[:2]
        scale = config.MOTION_WORKING_WIDTH / w0 if w0 > config.MOTION_WORKING_WIDTH else 1.0
        small = cv2.resize(frame, (max(1, int(w0 * scale)), max(1, int(h0 * scale))),
                            interpolation=cv2.INTER_AREA) if scale < 1.0 else frame
        inv_scale = 1.0 / scale

        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY) if small.ndim == 3 else small
        prev_gray = self._prev_gray.get(source_id)
        self._prev_gray[source_id] = gray

        # MOG2 gets every frame so it stays current
        fg = self._subtractor(source_id).apply(small)
        # MOG2 labels shadow pixels 127; keep only confident foreground (255).
        _, fg = cv2.threshold(fg, 200, 255, cv2.THRESH_BINARY)

        ego = None
        if (config.EGO_COMPENSATION and prev_gray is not None
                and prev_gray.shape == gray.shape):
            ego = self._estimate_ego(prev_gray, gray)
        shift = (float(ego[0][2]) ** 2 + float(ego[1][2]) ** 2) ** 0.5 if ego is not None else 0.0
        # smoothed before deciding, or a slow pan flickers between the two paths
        if ego is not None:
            prev_shift_ema = self._shift_ema.get(source_id, shift)
            shift_ema = (config.EGO_SHIFT_EMA_ALPHA * shift
                         + (1.0 - config.EGO_SHIFT_EMA_ALPHA) * prev_shift_ema)
            self._shift_ema[source_id] = shift_ema
        else:
            shift_ema = 0.0
            self._shift_ema.pop(source_id, None)
        # A camera this still is better served by MOG2 than by differencing.
        moving_camera = ego is not None and shift_ema >= config.EGO_STATIC_SHIFT
        self._ego_reliable[source_id] = True
        degraded = False
        # the structure test needs the previous frame in this frame's coordinates,
        # so warp it once here and not once per blob
        prev_aligned = prev_gray
        if (config.MOTION_STRUCTURE_ENABLED and moving_camera
                and prev_gray is not None and prev_gray.shape == gray.shape):
            h_s, w_s = gray.shape[:2]
            prev_aligned = cv2.warpAffine(prev_gray, ego, (w_s, h_s),
                                          flags=cv2.INTER_LINEAR,
                                          borderMode=cv2.BORDER_REPLICATE)
        fg_fraction = float((fg > 0).mean())
        dbg = {"moving_camera": moving_camera, "shift": shift, "shift_ema": shift_ema,
               "pre_gate_fg_fraction": fg_fraction}
        if moving_camera:
            fg = self._compensated_mask(prev_gray, gray, ego)
            fg_fraction = float((fg > 0).mean())
            dbg["compensated_fg_fraction"] = fg_fraction
            # smoothed, because blur on a fast pan keeps this hovering around the limit
            prev_ema = self._ego_residual_ema.get(source_id, fg_fraction)
            ema = (config.EGO_RESIDUAL_EMA_ALPHA * fg_fraction
                   + (1.0 - config.EGO_RESIDUAL_EMA_ALPHA) * prev_ema)
            self._ego_residual_ema[source_id] = ema
            dbg["compensated_fg_fraction_ema"] = ema
            # still a lot of foreground after cancelling camera motion means one transform
            # doesn't fit the scene (parallax). Detector.track then classifies the whole frame
            if ema > config.EGO_MAX_RESIDUAL:
                self._ego_reliable[source_id] = False
                # only drop everything past twice the limit, in between run with stricter gates
                if ema > config.EGO_MAX_RESIDUAL * config.EGO_RESIDUAL_DEGRADED_FACTOR:
                    self._tracks[source_id] = []
                    dbg["dropped_reason"] = "ego_residual"
                    self._debug[source_id] = dbg
                    return []
                degraded = True
                dbg["degraded_reason"] = "ego_residual"
        else:
            # the camera stopped, so forget the old moving-camera noise level
            self._ego_residual_ema.pop(source_id, None)
            if fg_fraction > config.MOTION_MOG2_MAX_FRACTION:
                # MOG2's version of that check. A background that hasn't settled (foliage,
                # flicker) is mostly foreground and breaks into many blobs, so report nothing
                self._ego_reliable[source_id] = False
                self._tracks[source_id] = []
                dbg["dropped_reason"] = "mog2_fraction"
                self._debug[source_id] = dbg
                return []

        fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        fg = cv2.dilate(fg, np.ones((5, 5), np.uint8), iterations=1)

        contours, _ = cv2.findContours(fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        sized = []
        for c in contours:
            area = cv2.contourArea(c)
            if area < config.MOTION_MIN_BLOB_AREA:
                continue
            x, y, w, h = cv2.boundingRect(c)
            # shape check: a branch or a shadow edge is a sliver or a flat band
            if h <= 0:
                continue
            ratio = w / float(h)
            if ratio < config.MOTION_ASPECT_MIN or ratio > config.MOTION_ASPECT_MAX:
                continue
            sized.append((area, x, y, x + w, y + h, (x + w / 2.0, y + h / 2.0)))

        # fine texture (paving, gravel) can break into dozens of blobs while the
        # foreground fraction stays under its limit
        cooldown = self._chronic_cooldown.get(source_id, 0)
        chronic = len(sized) > config.MOTION_CHRONIC_BLOB_COUNT
        if chronic:
            cooldown = config.MOTION_CHRONIC_COOLDOWN
        elif cooldown > 0:
            cooldown -= 1
        self._chronic_cooldown[source_id] = cooldown

        if chronic or cooldown > 0:
            self._ego_reliable[source_id] = False
            self._tracks[source_id] = []
            dbg["dropped_reason"] = ("chronic_blob_count" if chronic
                                     else "chronic_cooldown")
            dbg["sized_blob_count"] = len(sized)
            dbg["chronic_cooldown"] = cooldown
            self._debug[source_id] = dbg
            return []

        # cap the blob count so a busy frame can't make this slow
        sized.sort(key=lambda s: s[0], reverse=True)
        blobs = [(x1, y1, x2, y2, c) for _, x1, y1, x2, y2, c in sized[:config.MOTION_MAX_BLOBS_PER_FRAME]]
        dbg["raw_contour_count"] = len(contours)
        dbg["sized_blob_count"] = len(sized)
        dbg["gated_blob_count"] = len(blobs)

        tracks = self._tracks.setdefault(source_id, [])
        if moving_camera:
            self._restabilise(tracks, ego)
        next_id = self._next_id.get(source_id, 0)
        matched_ids = set()
        results = []

        for x1, y1, x2, y2, centroid in blobs:
            best, best_dist = None, config.MOTION_MATCH_MAX_DIST
            for t in tracks:
                if t.id in matched_ids:
                    continue
                last = t.history[-1]
                d = ((last[0] - centroid[0]) ** 2 + (last[1] - centroid[1]) ** 2) ** 0.5
                if d < best_dist:
                    best, best_dist = t, d

            if best is None:
                best = _Track(next_id, centroid, (x1, y1, x2, y2))
                next_id += 1
                tracks.append(best)
            else:
                best.history.append(centroid)
                best.bbox = (x1, y1, x2, y2)
                best.age = 0
            matched_ids.add(best.id)

            n_pts = len(best.history)
            fast_ok = (config.MOTION_STRUCTURE_ENABLED and not degraded
                       and config.MOTION_COHERENCE_MIN_POINTS_FAST <= n_pts
                       < config.MOTION_COHERENCE_MIN_POINTS)
            if n_pts < config.MOTION_COHERENCE_MIN_POINTS and not fast_ok:
                continue

            # cheap checks first, the patch comparison only runs on blobs about to be reported
            structure = 1.0
            if config.MOTION_STRUCTURE_ENABLED:
                structure = self._structure_score(prev_aligned, gray, (x1, y1, x2, y2))

            if fast_ok:
                # brief appearance path: two points are always in a line, so straightness
                # means nothing here. Use real movement plus a real content change instead.
                if structure < config.MOTION_STRUCTURE_MIN_FAST:
                    continue
                p0, p1 = best.history[0], best.history[-1]
                moved = ((p1[0] - p0[0]) ** 2 + (p1[1] - p0[1]) ** 2) ** 0.5
                if moved < config.MOTION_COHERENCE_MIN_PATH:
                    continue
                coherence = 1.0
            else:
                threshold = coherence_threshold
                if degraded:
                    # stricter on both checks when camera motion isn't fully trusted
                    threshold = max(threshold, config.MOTION_COHERENCE_DEGRADED_MIN)
                    if structure < config.MOTION_STRUCTURE_MIN_FAST:
                        continue
                elif structure < config.MOTION_STRUCTURE_MIN:
                    continue
                coherence = self._straightness(best.history)
                if coherence < threshold:
                    continue

            results.append({
                "track_id": best.id,
                "x1": x1 * inv_scale, "y1": y1 * inv_scale,
                "x2": x2 * inv_scale, "y2": y2 * inv_scale,
                "coherence": round(coherence, 3),
                "structure": round(structure, 3),
                "fast": fast_ok,
            })

        for t in tracks:
            if t.id not in matched_ids:
                t.age += 1
        tracks = [t for t in tracks if t.age <= config.MOTION_TRACK_MAX_AGE]
        if len(tracks) > config.MOTION_MAX_TRACKS_PER_SOURCE:
            # keep the most recently matched tracks
            tracks.sort(key=lambda t: t.age)
            tracks = tracks[:config.MOTION_MAX_TRACKS_PER_SOURCE]
        self._tracks[source_id] = tracks
        self._next_id[source_id] = next_id

        dbg["result_count"] = len(results)
        self._debug[source_id] = dbg
        return results

    def debug_info(self, source_id: str) -> dict:
        """Last frame's raw motion signals for this source, for diagnosis only."""
        return self._debug.get(source_id, {})


    def ego_reliable(self, source_id: str) -> bool:
        """False if the last frame's ego compensation could not be trusted."""
        return self._ego_reliable.get(source_id, True)


motion_detector = MotionDetector()
