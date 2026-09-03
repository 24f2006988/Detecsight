"""Class-agnostic motion detection: background subtraction + trajectory
coherence.

The point is to tag "something moved" without needing to know what it is --
so this works independently of the trained 4-class model. Raw background
subtraction alone fires on anything that changed pixel-to-pixel, including
wind-blown leaves and branches, so every blob is tracked for a few frames and
scored on how STRAIGHT its path is: a person or UAV moves with a roughly
consistent trajectory over a handful of frames, while foliage jitters back
and forth in place. Low straightness gets filtered out before it ever reaches
the classifier.
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
    """Union boxes that overlap more than iou_threshold, repeatedly, until none
    do. Two motion blobs a few pixels apart would otherwise become two crops
    covering mostly the same pixels, and the whole point of stage 3 is not
    pushing the same pixels through the network twice."""
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
    """Grow a blob box outward into a crop box, clipped to the frame.

    Background subtraction fires on the moving *part* of a target -- swinging
    limbs, a rotor disc -- not its silhouette, so a crop tight to the blob can
    cut the object in half and the detector sees a fragment. Padding gives it
    the context back, and min_size stops a 10 px blob becoming a 10 px crop.
    """
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
    """Per-source background subtraction + blob-trajectory coherence.

    Each source_id gets its own background model and track list -- state
    from one feed must never leak into another, same reasoning as the
    per-feed trackers in Detector._bind_tracker.
    """

    def __init__(self):
        self._bg: Dict[str, cv2.BackgroundSubtractorMOG2] = {}
        self._tracks: Dict[str, List[_Track]] = {}
        self._next_id: Dict[str, int] = {}
        # Previous working-resolution grey frame, per source: the reference the
        # camera's own motion is estimated against.
        self._prev_gray: Dict[str, np.ndarray] = {}
        # Whether the last frame's ego compensation was trustworthy. False when
        # parallax defeated the single-global-transform assumption -- see
        # EGO_MAX_RESIDUAL in app/config.py.
        self._ego_reliable: Dict[str, bool] = {}
        # Exponential moving average of the compensated foreground fraction,
        # per source -- see EGO_RESIDUAL_EMA_ALPHA in app/config.py. Smooths
        # out frame-to-frame noise (motion blur on a fast pan) that otherwise
        # flickers the single-frame residual check above and below
        # EGO_MAX_RESIDUAL every frame instead of settling on one side of it.
        self._ego_residual_ema: Dict[str, float] = {}
        # Exponential moving average of the estimated camera shift, per
        # source -- see EGO_SHIFT_EMA_ALPHA in app/config.py. Same problem as
        # the residual EMA above, one step earlier: a slow, genuine pan whose
        # frame-to-frame shift hovers near EGO_STATIC_SHIFT flickers the
        # moving/static BRANCH CHOICE itself frame to frame (RANSAC noise in
        # the affine fit, not motion blur), routing some frames of a real pan
        # through plain MOG2 instead of ego-compensated diffing. MOG2 then
        # sees real pan-smeared structure -- not jitter -- as foreground,
        # which is coherent by construction and sails through the
        # trajectory-coherence gate the other chronic-noise gates were never
        # tuned to catch (see fix 13 in README.md).
        self._shift_ema: Dict[str, float] = {}
        # Frames still to suppress after MOTION_CHRONIC_BLOB_COUNT last
        # tripped, per source -- see MOTION_CHRONIC_COOLDOWN in app/config.py.
        self._chronic_cooldown: Dict[str, int] = {}
        # Last frame's raw signals per source, for diagnosis only (scripts/
        # diagnose_bursts.py) -- never read by detect() itself, so it cannot
        # change behaviour, only make it inspectable after the fact.
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
        """Global frame-to-frame transform (2x3 affine, rotation+scale+shift),
        or None if the scene gives too little to fit one confidently.

        Tracked corners rather than dense flow: a few hundred points is enough
        to pin down four parameters, and RANSAC lets the genuinely-moving
        objects fall out as outliers instead of dragging the fit toward them.
        """
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
        """Foreground mask with the camera's own motion cancelled out.

        The previous frame is warped into the current frame's coordinates, so
        static structure lands on itself and differences away to nothing. What
        survives is motion the camera does not explain -- an actual moving
        object. Border regions are dropped: the warp pulls in pixels the
        previous frame never saw, and those differ from everything.
        """
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
        """Move stored track history into the current frame's coordinates.

        Histories are in frame coords, so under a pan every track's path picks
        up the camera's motion on top of the object's own -- which is exactly
        what _straightness would then score. Warping the history by the same
        transform leaves the object's true trajectory behind.
        """
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
        """How much of the change under `box` is STRUCTURE rather than light.

        Returns a value in [0, 1]: high means the region genuinely changed
        content (an object moved into or through it), low means only the
        illumination level changed (muzzle flash, headlight, glare, a lamp).

        1.0 is also returned whenever the test cannot judge -- no previous
        frame, a degenerate box, or a patch too flat to carry structure. That
        is deliberate: this gate can only ever REJECT, and the standing
        judgement is that a suppressed real contact is the failure that
        matters, so an unjudgeable blob is passed through.
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

        # Signature 1: a uniform brightness shift moves every pixel by roughly
        # the same amount, so the difference image's mean dominates its spread.
        d = b - a
        spread, level = float(d.std()), abs(float(d.mean()))
        if level > 1.0 and spread / level < config.MOTION_STRUCTURE_UNIFORM_RATIO:
            return 0.0

        # Signature 2: brightness/contrast change preserves structure, so the
        # z-scored patches still correlate. Real content change does not.
        ncc = float((((a - a.mean()) / sa) * ((b - b.mean()) / sb)).mean())
        return 1.0 - max(0.0, min(1.0, ncc))

    @staticmethod
    def _coherence_threshold(view: str) -> float:
        """Drone and ground view want opposite things from this gate -- see
        the comment on MOTION_COHERENCE_THRESHOLD_DRONE/_GROUND in
        app/config.py. Any view without its own entry (there are currently
        only two) falls back to the ground value, since that's the more
        permissive of the two and matches this project's default view."""
        return {
            "drone": config.MOTION_COHERENCE_THRESHOLD_DRONE,
            "ground": config.MOTION_COHERENCE_THRESHOLD_GROUND,
        }.get(view, config.MOTION_COHERENCE_THRESHOLD_GROUND)

    def detect(self, frame: np.ndarray, source_id: str, view: str = "ground") -> List[dict]:
        """Return coherently-moving blobs as [{x1,y1,x2,y2,coherence}], in the
        ORIGINAL frame's pixel coords (internally this all runs on a
        downscaled copy -- see MOTION_WORKING_WIDTH -- and scales back up).

        `view` only affects the trajectory-coherence threshold below (see
        _coherence_threshold) -- background subtraction, ego compensation and
        every other gate in this method are view-independent."""
        coherence_threshold = self._coherence_threshold(view)
        h0, w0 = frame.shape[:2]
        scale = config.MOTION_WORKING_WIDTH / w0 if w0 > config.MOTION_WORKING_WIDTH else 1.0
        small = cv2.resize(frame, (max(1, int(w0 * scale)), max(1, int(h0 * scale))),
                            interpolation=cv2.INTER_AREA) if scale < 1.0 else frame
        inv_scale = 1.0 / scale

        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY) if small.ndim == 3 else small
        prev_gray = self._prev_gray.get(source_id)
        self._prev_gray[source_id] = gray

        # MOG2 is fed every frame regardless of which mask is used, so its model
        # stays current and the static path is ready the moment the camera settles.
        fg = self._subtractor(source_id).apply(small)
        # MOG2 labels shadow pixels 127; keep only confident foreground (255).
        _, fg = cv2.threshold(fg, 200, 255, cv2.THRESH_BINARY)

        ego = None
        if (config.EGO_COMPENSATION and prev_gray is not None
                and prev_gray.shape == gray.shape):
            ego = self._estimate_ego(prev_gray, gray)
        shift = (float(ego[0][2]) ** 2 + float(ego[1][2]) ** 2) ** 0.5 if ego is not None else 0.0
        # Smoothed before the branch decision, not read raw: a slow genuine
        # pan whose per-frame shift hovers near EGO_STATIC_SHIFT (RANSAC
        # noise in the affine fit, frame to frame) otherwise flickers which
        # branch runs -- a few frames misroute to plain MOG2, which then sees
        # real pan-smeared structure as foreground. That foreground is
        # coherent (it's real motion, not jitter), so it passes the
        # trajectory-coherence gate below untouched -- a burst of confident
        # moving_object boxes on content that never actually stopped moving.
        # Same fix shape as EGO_RESIDUAL_EMA_ALPHA (fix 9a), one step
        # earlier: dropped (not carried through a None reading) whenever ego
        # estimation itself fails, so a genuine loss of tracking doesn't drag
        # a stale average toward "static".
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
        # The structure test compares this frame against the previous one, so
        # under a pan the previous frame has to be brought into this frame's
        # coordinates first -- the same warp _compensated_mask uses, computed
        # once per frame rather than once per blob.
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
            # Smoothed with an EMA, not read raw: on a fast handheld pan,
            # motion blur keeps this fraction from settling clearly above or
            # below EGO_MAX_RESIDUAL -- it hovers across the line frame to
            # frame (measured on v6.mp4 frames 320-334: 0.017-0.028 against a
            # 0.02 threshold). Reading it raw made the reliable/unreliable
            # flag flip every frame, and the frames that landed "reliable" by
            # chance emitted a burst of moving_object boxes off noise that
            # never actually cleared. The EMA reads a stretch that's
            # consistently near the line as consistently over it.
            prev_ema = self._ego_residual_ema.get(source_id, fg_fraction)
            ema = (config.EGO_RESIDUAL_EMA_ALPHA * fg_fraction
                   + (1.0 - config.EGO_RESIDUAL_EMA_ALPHA) * prev_ema)
            self._ego_residual_ema[source_id] = ema
            dbg["compensated_fg_fraction_ema"] = ema
            # If this much of the frame still reads as foreground after
            # cancelling the camera's own motion, one global transform did not
            # describe the scene (parallax), and every blob below would be
            # static structure smeared by the pan. Report nothing rather than
            # a frame full of phantom contacts; Detector.track sees the flag
            # and falls back to classifying the whole frame.
            if ema > config.EGO_MAX_RESIDUAL:
                self._ego_reliable[source_id] = False
                # Measured 2026-09-02: dropping here unconditionally was
                # disabling the motion channel for 224 of 225 frames on v2 and
                # v9, and 535 of 1068 on v6 -- i.e. almost always, on exactly
                # the fast-handheld footage this system exists for. A brief
                # human appearance in any of those frames was unreportable by
                # construction. So only the clearly-hopeless band still drops;
                # the band just above the threshold runs DEGRADED instead of
                # blind (structure test required, no fast path, raised
                # coherence bar) -- see EGO_RESIDUAL_DEGRADED_FACTOR.
                if ema > config.EGO_MAX_RESIDUAL * config.EGO_RESIDUAL_DEGRADED_FACTOR:
                    self._tracks[source_id] = []
                    dbg["dropped_reason"] = "ego_residual"
                    self._debug[source_id] = dbg
                    return []
                degraded = True
                dbg["degraded_reason"] = "ego_residual"
        else:
            # Camera just stopped panning: drop the stale EMA rather than let
            # it anchor to a moving-camera noise level, so a genuine return to
            # a moving pan starts judging fresh instead of inheriting bias
            # from before the stop.
            self._ego_residual_ema.pop(source_id, None)
            if fg_fraction > config.MOTION_MOG2_MAX_FRACTION:
                # MOG2's counterpart to the residual check above: a background
                # model that hasn't settled (wind-blown foliage, a lighting
                # flicker) reads as a large, chronically "foreground" region
                # that fragments into many small blobs -- exactly what a real
                # target burst would look like to the coherence gate below,
                # just for the wrong reason. Report nothing this frame rather
                # than a swarm.
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
            # Shape gate: a swaying branch or a moving shadow edge projects to a
            # sliver or a flat band, nothing a person/vehicle/UAV can look like.
            # Costs one division and drops that noise before it ever gets a track.
            if h <= 0:
                continue
            ratio = w / float(h)
            if ratio < config.MOTION_ASPECT_MIN or ratio > config.MOTION_ASPECT_MAX:
                continue
            sized.append((area, x, y, x + w, y + h, (x + w / 2.0, y + h / 2.0)))

        # Chronic fine texture (brick paving, gravel, compression grain) can
        # fragment into dozens of small blobs while the raw foreground
        # FRACTION checked above stays comfortably under its threshold -- each
        # blob is individually small, there's just a lot of them, and the
        # fraction gates alone don't see that. Measured on v7.mp4 (a drone
        # hovering over brick paving): 30-58 blobs surviving the size/aspect
        # gate above for 30+ consecutive frames, while the foreground fraction
        # sat at 0.02-0.11 the whole time, under MOTION_MOG2_MAX_FRACTION's
        # 0.12 (that threshold was tuned only against v3.mp4's tree line,
        # which fragmented at a much higher 0.19-0.27 fraction -- it never
        # generalised to a scene that fragments finer but at lower coverage).
        # Quiet real frames on the same clip and on v6.mp4 sat at 0-16 blobs,
        # so 20 is a real gap, not a hair trigger. This is a density check
        # alongside the two fraction checks, not instead of either: whichever
        # branch produced `fg`, a real scene doesn't have this many
        # independently-moving things, so treat it the same as the other two
        # chronic-noise gates -- drop the frame's blobs rather than let the
        # coherence gate below score a fragment swarm as a burst of targets.
        # The gate itself, plus a cooldown. The bare threshold was not enough:
        # a scene that fragments to just UNDER the cutoff for several frames at
        # a time (v7.mp4's brick paving sits at 17-20 blobs against a cutoff of
        # 20, crossing it only occasionally) gets to build fresh coherent
        # tracks in the gaps between individual trips, and those tracks emit a
        # burst the moment enough of them reach MOTION_COHERENCE_MIN_POINTS.
        # Measured on v7.mp4 frame 261: 7 moving_object boxes against a local
        # baseline of 0.
        #
        # Note this is deliberately NOT the EMA treatment used for
        # EGO_SHIFT_EMA_ALPHA / EGO_RESIDUAL_EMA_ALPHA -- that was tried here
        # first and made the burst worse (7 -> 13 boxes, see the config
        # comment). Smoothing delays the trip, because the averaged value
        # climbs slower than the raw spike; this gate has to fire fast. What it
        # needs is not a slower decision but a LONGER one: once a scene is
        # established as fragmenting, stay suppressed for a few frames so the
        # sub-threshold gaps stop being usable as track-building windows.
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

        # A busy frame (e.g. camera panning, which reads as near-whole-frame
        # change to plain background subtraction) can produce far more
        # contours than any real scene has moving objects. Bound it here so
        # per-frame cost can't grow with scene busyness -- see config comment.
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

            # Cheap gates first: only pay for the patch comparison on a blob
            # that is otherwise about to be reported.
            structure = 1.0
            if config.MOTION_STRUCTURE_ENABLED:
                structure = self._structure_score(prev_aligned, gray, (x1, y1, x2, y2))

            if fast_ok:
                # The brief-appearance path. Two points make _straightness
                # degenerate (any two points are collinear), so it is NOT the
                # evidence here: real displacement plus a real content change
                # is. That is what makes a half-second appearance reportable
                # at all -- it can never accumulate MOTION_COHERENCE_MIN_POINTS.
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
                    # Paying for the missing ego reliability with a stricter
                    # bar on both axes rather than reporting nothing at all.
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
            # Keep the most recently matched (lowest age) tracks; a busy scene
            # sheds its stalest/noisiest tracks first rather than growing forever.
            tracks.sort(key=lambda t: t.age)
            tracks = tracks[:config.MOTION_MAX_TRACKS_PER_SOURCE]
        self._tracks[source_id] = tracks
        self._next_id[source_id] = next_id

        dbg["result_count"] = len(results)
        self._debug[source_id] = dbg
        return results

    def debug_info(self, source_id: str) -> dict:
        """Last frame's raw motion-pass signals for this source -- diagnosis
        only, see the note on self._debug above."""
        return self._debug.get(source_id, {})


    def ego_reliable(self, source_id: str) -> bool:
        """False if the last frame's ego compensation could not be trusted."""
        return self._ego_reliable.get(source_id, True)


motion_detector = MotionDetector()
