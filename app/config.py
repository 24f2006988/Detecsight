"""Central configuration. Every path and threshold lives here, not scattered
through the routers.

Every value below was measured rather than guessed, and each carries the result
that chose it plus a pointer into ENGINEERING_LOG.md, where the full sweep,
the alternatives and the rejected options live. The pointer is the contract:
if you are about to change a constant, read its section first -- most of these
have already been moved once and moved back.

Everything is overridable by environment variable, prefix BATTLESIGHT_.

    Model and precision ........ MODEL_PATH, DEVICE, QUANTIZE, IMGSZ, USE_TENSORRT
    Detection thresholds ....... CONF_THRESHOLD*, IOU_THRESHOLD, MAX_DET
    Far-field second pass ...... FARFIELD_*
    Tracking ................... TRACKER_CONFIG, MOTION_WINDOW, MAX_TRACKS_*
    Motion filter, stages 1-2 .. MOTION_MIN_BLOB_AREA, MOTION_COHERENCE_*
    Illumination discriminator . MOTION_STRUCTURE_*
    Motion gating (off) ........ MOTION_GATED, MOTION_CROP_*
    Ego compensation ........... EGO_*
    Chronic-noise gates ........ MOTION_CHRONIC_*
    Reference exclusion ........ EXCLUSION_*
    HUD overlay filter ......... OVERLAY_*
"""
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

# --- Model and precision ----------------------------------------------------
MODEL_PATH = os.getenv("BATTLESIGHT_MODEL", str(BASE_DIR / "weights" / "best.pt"))
# Optional per-view checkpoint, selected by the `view` query param. The
# VisDrone-only aerial specialist that used to live here was RETIRED: the
# blended checkpoint beat it on the drone view's own home domain, personnel
# mAP50 0.3162 -> 0.7064 (log 17). The mechanism stays -- drop a file here and
# ?view=drone uses it; Detector.load() falls back to MODEL_PATH when absent, so
# serving works with one weights file present.
DRONE_MODEL_PATH = os.getenv("BATTLESIGHT_DRONE_MODEL", str(BASE_DIR / "weights" / "drone_best.pt"))
DEVICE = os.getenv("BATTLESIGHT_DEVICE", "0")       # "0" = first GPU, "cpu" = CPU
# Inference precision, passed as `quantize=` to every predict()/track() call.
# FP16 costs <=0.002 mAP50 (noise) for a ~45% latency cut -- standard here.
# INT8 was measured and REJECTED: 4.23 ms against FP16's 6.95, but -4 pt mAP50
# and -5 pt recall. FP8 does not exist in this stack (checked the ultralytics
# exporter source: 32/16/8 and weight-only w8a16/w8a32, nothing else).
# None (FP32) on CPU, where half-precision is unsupported or slower. Log 11.
QUANTIZE = 16 if DEVICE != "cpu" else None
# 0.25, not the inherited 0.35, which was giving away a fifth of the achievable
# recall (mean F1 0.447 -> 0.499). Mean F1 actually peaks near 0.16, but this
# model puts confident boxes on out-of-distribution scenes -- it called a pencil
# case `light_vehicle` at 0.42 -- and an overlay full of phantom contacts is
# worse than a missed one. Log 3.
CONF_THRESHOLD = float(os.getenv("BATTLESIGHT_CONF", "0.25"))
# Ground-view-only, personnel-only override. Chosen on F2 rather than F1,
# because F1 weights precision and recall equally and this system's standing
# principle does not: recall 0.636 -> 0.725 for 5.7 points of precision.
# Ground view only -- on aerial imagery precision at 0.10 collapses to 0.431,
# so drone view keeps 0.25. Kept personnel-only to avoid re-inviting the
# low-confidence vehicle hallucination of log 14. Full sweep in log 17b.
CONF_THRESHOLD_PERSONNEL_GROUND = float(os.getenv("BATTLESIGHT_CONF_PERSONNEL_GROUND", "0.10"))

# The same floor, for the DRONE view. Added 2026-09-06 with the SARD-blended
# checkpoint (log 22h), which is more conservatively calibrated than its
# predecessor: on v10.mp4 it scores four genuine personnel at 0.117-0.256,
# every one of them under the global 0.25, while the checkpoint before it put
# the same four at 0.328-0.535. Nothing was lost in capability -- it finds all
# four -- but a fixed threshold turned a calibration shift into four missed
# people, which is why this floor now exists for both views rather than one.
#
# 0.10 is not inherited from the ground floor, it is measured. Sweeping SARD
# test (570 images, 732 hand-labelled boxes, UAV over terrain, i.e. the drone
# view domain) puts the F2 optimum at exactly 0.10 for BOTH checkpoints:
#
#   conf   recall  precision  F2
#   0.05   0.709   0.507      0.657
#   0.10   0.673   0.661      0.671   <- optimum
#   0.15   0.645   0.733      0.661
#   0.25   0.571   0.834      0.610
#
# F2 rather than F1 for the reason section 17b gives: a missed contact costs
# more than a spurious one. F1 would put this at 0.20 and cost 10 points of
# recall.
CONF_THRESHOLD_PERSONNEL_DRONE = float(os.getenv("BATTLESIGHT_CONF_PERSONNEL_DRONE", "0.10"))
# WARNING: a NO-OP for the model's own inference, and always has been. YOLO26's
# head reports `end2end: True` -- NMS-free one-to-one, so `iou=` is ignored.
# Sweeping 0.5/0.6/0.7/0.8 gave byte-identical results at every value.
# Do NOT reach for this to fix crowd or occlusion recall: there is no NMS to
# loosen, suppression is learned, and only training changes it.
# It IS still used, by this project's own IoU arithmetic only -- the far-field
# duplicate check, _claim_motion_blobs, overlay_mask, motion box merging. Log 19.
IOU_THRESHOLD = float(os.getenv("BATTLESIGHT_IOU", "0.5"))
# Ultralytics defaults to 300. VisDrone val frames hold up to 317 objects, so
# the default silently truncates the densest frames -- exactly the crowded
# scenes where the count matters. Raising it costs only NMS time. Log 4.
MAX_DET = int(os.getenv("BATTLESIGHT_MAX_DET", "500"))

# --- Far-field second pass (2026-09-02) -------------------------------------
# Recall collapses with target size (<16 px 0.187, >96 px 0.940), and 41% of
# all people are under 32 px -- so all the loss is in the far field. Tiling the
# WHOLE frame recovers some of it but costs 5-6x inference and craters
# precision. Instead the cheap full-frame pass CHOOSES where to spend one extra
# pass: the far field is wherever the small boxes already are. That buys 73% of
# full tiling's recall gain for 40% of its extra cost, precision preserved.
#
# DEFAULT OFF, and not for cost -- for JITTER. p90 48.5 ms against a 26.9 ms
# median makes an AR overlay stutter. It is the right tool OFFLINE
# (annotate_video.py, forensic review) where latency is free. Log 19, and
# docs/size_recall.png for the curve. BATTLESIGHT_FARFIELD=1 enables it.
FARFIELD_ENABLED = os.getenv("BATTLESIGHT_FARFIELD", "0") not in ("0", "false", "False")
# Run the second pass every Nth frame on the tracked path. The far field is a
# property of the scene geometry, not of the frame, so it barely moves between
# consecutive frames -- and a distant target persists across many of them.
# 1 = every frame (the measurement above), 3 keeps most of the gain at ~1.6x
# instead of ~2.8x total cost. Ignored by the stateless detect() path.
FARFIELD_STRIDE = int(os.getenv("BATTLESIGHT_FARFIELD_STRIDE", "3"))
# Boxes larger than this (px, full-frame scale) are discarded FROM THE TILE.
# Guards precision -- see the table above.
FARFIELD_MAX_BOX = float(os.getenv("BATTLESIGHT_FARFIELD_MAX_BOX", "48"))
# The far field is estimated from the smallest this-percentile of the full
# frame's boxes.
FARFIELD_SMALL_PCT = float(os.getenv("BATTLESIGHT_FARFIELD_SMALL_PCT", "40"))
# Below this many boxes there is nothing to estimate from, so fall back to the
# horizon prior below.
FARFIELD_MIN_BOXES = int(os.getenv("BATTLESIGHT_FARFIELD_MIN_BOXES", "3"))
FARFIELD_PAD = float(os.getenv("BATTLESIGHT_FARFIELD_PAD", "0.10"))
# Fallback region as (x1, y1, x2, y2) fractions: for a roughly level ground
# view the far field is the horizon band, upper-middle of frame. This is a
# PRIOR, used only when the frame gives us nothing better to go on.
FARFIELD_PRIOR = tuple(float(v) for v in os.getenv(
    "BATTLESIGHT_FARFIELD_PRIOR", "0.2,0.15,0.8,0.6").split(","))
# Never let the tile grow so large it is just the whole frame again (which
# would buy nothing and cost a full extra pass).
FARFIELD_MAX_FRACTION = float(os.getenv("BATTLESIGHT_FARFIELD_MAX_FRACTION", "0.55"))
# 1280, not the inherited 640. Targets here are tiny -- 75% of `personnel`
# boxes are under 16 px at 640, at the floor of what the stride-8 head can
# resolve. VisDrone val: mAP50 0.4331 -> 0.5046, personnel 0.382 -> 0.534.
# 1600 was measured and is strictly WORSE than 1280: above the trained size the
# model is off-distribution for its own scale priors. A single global size, not
# a per-source cap -- low-resolution sources lose nothing, and 1280 produced
# FEWER false positives than 640 on a 478x850 phone clip. Log 2 and log 19.
IMGSZ = int(os.getenv("BATTLESIGHT_IMGSZ", "1280"))
# Tile resolution for the far-field pass. MUST equal IMGSZ while serving a
# --static TensorRT engine, whose input shape is fixed: setting 640 crashes the
# far-field pass outright. Unfortunate, because the tile is a crop and a
# smaller one would be both cheaper and enough -- getting that needs a second
# engine built at the smaller size, which is not done here. Log 19.
FARFIELD_IMGSZ = int(os.getenv("BATTLESIGHT_FARFIELD_IMGSZ", str(IMGSZ)))

# Prefer a `.engine` next to MODEL_PATH (same stem) over the .pt -- same
# weights, compiled with layer fusion and FP16 kernels for this GPU, ~2x
# faster. Falls back to the .pt automatically when the engine is missing, fails
# to load (wrong TensorRT/driver version), or DEVICE is "cpu", since an engine
# only runs on the GPU family it was built on. Log 5b.
USE_TENSORRT = os.getenv("BATTLESIGHT_USE_TENSORRT", "1") not in ("0", "false", "False")

# Tracking
TRACKER_CONFIG = "bytetrack.yaml"
MOTION_WINDOW = 12          # frames of history kept per track
MOTION_THRESHOLD = 0.015    # normalised displacement that counts as "moving"
# Track ids only climb over the life of a feed, so the motion history is capped
# per source and the least recently seen track is evicted first.
MAX_TRACKS_PER_SOURCE = int(os.getenv("BATTLESIGHT_MAX_TRACKS", "512"))

# Training jobs
TRAIN_SCRIPT = BASE_DIR / "scripts" / "train.py"
RUNS_DIR = BASE_DIR / "runs" / "detect"
LOGS_DIR = BASE_DIR / "runs" / "logs"

CLASS_NAMES = ["personnel", "two_wheeler", "light_vehicle", "heavy_vehicle"]

# Class-agnostic motion detection (app/motion_filter.py). Tags anything that
# moves coherently -- not just the 4 trained classes -- while filtering out
# incoherent jitter like wind-blown foliage.
# 80, not 30. px^2 AT MOTION_WORKING_WIDTH resolution, not full frame. 30 px^2
# is a 5x6 pixel smudge -- nothing trustworthy can be said about a blob that
# small, and it was most of the phantom output. Measured on v3.mp4, phantom
# blobs had a median area of 284 px^2 (full-res) against 10,870 px^2 for a real
# personnel box: a 38x separation, so there is a lot of room here.
#
# TRADE-OFF, read before raising further: this is the floor on how small an
# unrecognised moving thing can be and still be reported, which is exactly the
# distant-UAV case `moving_object` exists for. At 1080p (working scale 0.25)
# 80 px^2 is a ~36x36 px object in the full frame. Raise it to cut false
# contacts on jittery ground feeds; lower it if you need smaller air targets.
MOTION_MIN_BLOB_AREA = int(os.getenv("BATTLESIGHT_MOTION_MIN_AREA", "80"))
MOTION_MATCH_MAX_DIST = int(os.getenv("BATTLESIGHT_MOTION_MAX_DIST", "60"))       # px, blob-to-track association
MOTION_TRACK_MAX_AGE = int(os.getenv("BATTLESIGHT_MOTION_TRACK_AGE", "5"))        # frames before a track is dropped
MOTION_COHERENCE_MIN_POINTS = int(os.getenv("BATTLESIGHT_MOTION_MIN_POINTS", "4"))  # history needed to judge
MOTION_COHERENCE_MIN_PATH = float(os.getenv("BATTLESIGHT_MOTION_MIN_PATH", "8"))    # px, below this is noise

# --- Illumination-vs-structure discriminator (2026-09-02) -------------------
# The trajectory gates above judge a blob by WHERE it went over several frames.
# That cannot separate two things this system confuses badly:
#   * a light -- muzzle flash, headlight, screen glare, a lamp switching on --
#     whose bright region drifts its centroid and reads as coherent travel;
#   * a person visible for half a second, who never survives long enough to
#     accumulate MOTION_COHERENCE_MIN_POINTS=4 frames of history and so cannot
#     be reported AT ALL, at any threshold setting.
# Both need a decision from appearance in ONE frame, not from path over many.
#
# The test: align the previous frame to this one (the ego transform already
# computed for the motion pass), then compare the two patches under the blob
# after z-scoring each. An illumination change preserves structure -- the same
# edges, scaled in brightness -- so the z-scored patches still correlate
# strongly. A real object moving into the region changes what is there, so they
# do not. Score is 1 - max(0, correlation): high means structural, low means
# "only the light level changed".
MOTION_STRUCTURE_ENABLED = os.getenv("BATTLESIGHT_MOTION_STRUCTURE", "1") not in ("0", "false", "False")
# Normal path: reject only on a clear illumination signature. Deliberately
# permissive -- this runs on blobs that ALREADY passed the trajectory gates,
# and the standing judgement is that a suppressed real contact is the worse
# failure. Raise it only with measurements on v3/v6 (whose moving_object
# counts are verified-real) in hand.
MOTION_STRUCTURE_MIN = float(os.getenv("BATTLESIGHT_MOTION_STRUCTURE_MIN", "0.25"))
# Fast path and degraded mode ask for more, because they are spending less
# trajectory evidence (or none) to make the same call.
MOTION_STRUCTURE_MIN_FAST = float(os.getenv("BATTLESIGHT_MOTION_STRUCTURE_MIN_FAST", "0.40"))
# A patch flatter than this (grayscale std) carries no structure to compare,
# so the test cannot judge it either way and FAILS OPEN -- same principle as
# overlay_mask.py's OVERLAY_MAX_FRACTION.
MOTION_STRUCTURE_MIN_STD = float(os.getenv("BATTLESIGHT_MOTION_STRUCTURE_MIN_STD", "4.0"))
# Second, independent illumination signature: a uniform brightness shift moves
# every pixel by about the same amount, so the difference image has a large
# mean relative to its spatial spread. Below this ratio the change is uniform
# (a light), not local (an object).
MOTION_STRUCTURE_UNIFORM_RATIO = float(os.getenv("BATTLESIGHT_MOTION_STRUCTURE_UNIFORM", "0.6"))
# Frames of history a blob needs on the FAST path. Two points make
# _straightness degenerate (any two points are collinear), so the fast path
# does not rely on it: it requires the structure test AND real displacement
# (>= MOTION_COHERENCE_MIN_PATH). Structure + travel in 2 frames is the
# evidence that replaces 4 frames of trajectory.
MOTION_COHERENCE_MIN_POINTS_FAST = int(os.getenv("BATTLESIGHT_MOTION_MIN_POINTS_FAST", "2"))
# Ego-residual degraded band. Above EGO_MAX_RESIDUAL the single global
# transform no longer describes the scene, and the old behaviour was to report
# NOTHING -- which on measurement was disabling the motion channel for 224 of
# 225 frames on v2/v9 and 535 of 1068 on v6, i.e. most of the time on exactly
# the fast-handheld footage this system is for. Between EGO_MAX_RESIDUAL and
# this multiple of it, run degraded instead of blind: structure test required,
# no fast path, and a raised coherence bar. Past it, drop the frame as before.
EGO_RESIDUAL_DEGRADED_FACTOR = float(os.getenv("BATTLESIGHT_EGO_RESIDUAL_DEGRADED_FACTOR", "2.0"))
# Coherence floor applied while degraded, if stricter than the view's own.
MOTION_COHERENCE_DEGRADED_MIN = float(os.getenv("BATTLESIGHT_MOTION_COHERENCE_DEGRADED", "0.85"))
# Per-view, not one global value: a straightness threshold is a
# recall/false-positive tradeoff, and the two views have opposite needs.
#
# DRONE: 0.85. Wind-blown foliage produces short-lived texture jitter scoring
# 0.55-0.97 straightness over the 4-frame window -- a brief consistent gust
# looks exactly like a real trajectory that briefly, and the other chronic-noise
# gates do not catch it (this is a "few blobs, briefly coherent" case, not a
# swarm). 0.85 is the first value in the sweep that fully clears the bursts.
#
# REAL COST, not free: it also cuts already-verified-real moving_object counts
# on ground clips by 47-63%, because this gate cannot tell "briefly coherent
# noise" from "a real target tracked for only a few frames" by straightness
# alone. Neither synthetic regression clip exercises it at any threshold, so it
# is NOT verified against a real slow or distant mover -- re-measure against
# real footage of that case before raising it further. Log 14 and 15.
MOTION_COHERENCE_THRESHOLD_DRONE = float(os.getenv("BATTLESIGHT_MOTION_COHERENCE_DRONE", "0.85"))
# GROUND: 0.5, the value logs 7-9 and 13 were tuned and verified against, and
# deliberately permissive. This view exists to catch personnel visible for only
# a short span; the drone view's tightening would suppress exactly those.
MOTION_COHERENCE_THRESHOLD_GROUND = float(os.getenv("BATTLESIGHT_MOTION_COHERENCE_GROUND", "0.5"))
# A panning camera makes nearly the whole frame register as "changed" every
# frame (background subtraction has no notion of camera ego-motion), which
# can spawn far more contours/tracks per frame than a static view ever would.
# Both caps below exist to keep that bounded rather than letting per-frame
# cost grow with scene busyness -- same reasoning as MAX_TRACKS_PER_SOURCE
# above, applied to the motion pass instead of the classifier's track history.
MOTION_MAX_BLOBS_PER_FRAME = int(os.getenv("BATTLESIGHT_MOTION_MAX_BLOBS", "40"))
MOTION_MAX_TRACKS_PER_SOURCE = int(os.getenv("BATTLESIGHT_MOTION_MAX_TRACKS", "150"))
# Background subtraction + contour finding runs on a downscaled copy of the
# frame; a near-fully-foreground frame (a panning camera reads as almost
# entirely "changed") makes cv2.findContours itself the actual cost at full
# 1080p resolution, well before any tracking logic runs at all. Blob
# coordinates are scaled back up to full-resolution pixel space afterward.
MOTION_WORKING_WIDTH = int(os.getenv("BATTLESIGHT_MOTION_WIDTH", "480"))
# Stage 2 shape gate: a blob whose width/height ratio is wildly outside what a
# person, vehicle or UAV can project to is structure, not a target -- a swaying
# branch reads as a long thin sliver, a lighting change as a wide flat band.
# Cheap to test and it runs before any trajectory bookkeeping.
MOTION_ASPECT_MIN = float(os.getenv("BATTLESIGHT_MOTION_ASPECT_MIN", "0.15"))   # w/h
MOTION_ASPECT_MAX = float(os.getenv("BATTLESIGHT_MOTION_ASPECT_MAX", "8.0"))
# Overlap needed for an existing YOLO detection to "claim" a motion blob
# (i.e. it's already classified, no generic detection needed for it too).
MOTION_CLAIM_IOU = float(os.getenv("BATTLESIGHT_MOTION_CLAIM_IOU", "0.1"))

# Stage 3: motion-gated inference -- run YOLO only on crops around blobs that
# survived stages 1-2, so a quiet frame costs zero GPU.
#
# DEFAULT OFF, and it must stay off. Gating means the classifier only ever sees
# what moved, so a stationary target is never detected at all -- and a parked
# truck and a standing sentry are exactly what an awareness overlay is for.
# Measured on a pan over a still scene: gated 0.0 detections/frame at 28.3 ms,
# ungated 26.5 detections/frame at 96.7 ms. Zero. The speed was real, but it
# was the speed of not looking.
#
# Do NOT try to buy the throughput back by lowering IMGSZ: the tracked path is
# dominated by fixed per-frame overhead, not the forward pass, so 1280 -> 640
# gives up a third of the detections to gain about 1 fps. Log 5.
MOTION_GATED = os.getenv("BATTLESIGHT_MOTION_GATED", "0") not in ("0", "false", "False")

# Run the CPU motion stage CONCURRENTLY with the GPU pass instead of before it.
# Profiled on v5.mp4 (warm, 120 frames): the motion filter alone is 23.6 ms
# median against a 52.2 ms total track() -- ~45% of the frame budget spent on
# the CPU while the GPU sat idle. With MOTION_GATED off nothing on the model
# path reads the blobs until _claim_motion_blobs at the very end, so the two
# are independent and the frame should cost max(CPU, GPU), not their sum.
# Both sides release the GIL (OpenCV; TensorRT), so the overlap is real.
# Only applies when MOTION_GATED is off -- the gated path consumes blobs
# immediately and cannot overlap.
MOTION_PARALLEL = os.getenv("BATTLESIGHT_MOTION_PARALLEL", "1") not in ("0", "false", "False")
# Blobs are the moving *part* of a target (a walking torso, not the whole
# person), so crops are padded outward to give the detector its context back.
MOTION_CROP_PADDING = float(os.getenv("BATTLESIGHT_MOTION_CROP_PAD", "0.6"))   # fraction of box side
MOTION_CROP_MIN_SIZE = int(os.getenv("BATTLESIGHT_MOTION_CROP_MIN", "128"))    # px, full-frame scale
# Crops that overlap this much are unioned into one, so the same pixels are
# never pushed through the network twice.
MOTION_CROP_MERGE_IOU = float(os.getenv("BATTLESIGHT_MOTION_CROP_MERGE_IOU", "0.2"))
# Past this many crops, one full-frame pass is cheaper than the batch -- a
# panning camera or a crowd should degrade to the old path, not to 40 forwards.
MOTION_MAX_CROPS_PER_FRAME = int(os.getenv("BATTLESIGHT_MOTION_MAX_CROPS", "6"))
# Crops are resized to this before inference. Smaller than IMGSZ on purpose:
# a 128-256 px crop upscaled to 640 buys nothing but latency.
MOTION_CROP_IMGSZ = int(os.getenv("BATTLESIGHT_MOTION_CROP_IMGSZ", "320"))

# Camera ego-motion compensation (app/motion_filter.py).
# Background subtraction has no notion of a moving camera: a handheld pan drags
# every static edge across the sensor, which reads as near-whole-frame change
# (measured on a 10 px/frame pan: 47% of the frame, peaking at 83%). Worse, the
# straightness gate above actively PREFERS it -- a pan drags static structure
# along a dead-straight line and scores ~1.0, so the filter built to reject
# jitter waves ego-motion through. The global frame-to-frame transform is
# therefore estimated and cancelled before anything is called motion.
# Set BATTLESIGHT_EGO_COMP=0 for the original uncompensated behaviour.
EGO_COMPENSATION = os.getenv("BATTLESIGHT_EGO_COMP", "1") not in ("0", "false", "False")
# Below this much estimated camera translation the camera counts as static and
# the MOG2 path is used unchanged -- MOG2 handles a fixed view better than
# frame differencing does (multi-modal backgrounds, gradual light changes).
# 0.3, not 1.0. Below this the camera counts as static and the MOG2 path runs
# unchanged -- but MOG2 has no tolerance for even sub-pixel movement, so a
# handheld camera drifting 0.5-1.0 px/frame was being classified as "static"
# and every high-contrast edge in the scene flickered as foreground. On
# v3.mp4 (handheld, railway platform) only 56 of 368 frames exceeded 1.0 px,
# so 85% of the clip went through MOG2 and produced the phantom explosion.
# Routing that jitter through compensation instead is what it is for.
EGO_STATIC_SHIFT = float(os.getenv("BATTLESIGHT_EGO_STATIC_SHIFT", "0.3"))   # px at MOTION_WORKING_WIDTH
# EGO_STATIC_SHIFT's own counterpart to EGO_RESIDUAL_EMA_ALPHA below, one
# step earlier in the pipeline: a slow, genuine pan whose per-frame shift
# hovers near EGO_STATIC_SHIFT (RANSAC noise in the affine fit, not motion
# blur this time) flickers the moving/static BRANCH CHOICE itself -- a few
# frames of a real pan get misrouted to plain MOG2 instead of ego-compensated
# diffing. Measured on v6.mp4, frames 956-962: shift bounced 0.20, 0.45,
# 0.32, 0.05, 0.26, 0.15, 0.21 around the 0.3 threshold while the camera was
# genuinely panning throughout (surrounding frames measured 0.49-1.66).  Each
# frame classified "static" ran plain MOG2 on real pan-smeared structure --
# not jitter -- which is coherent by construction and sails through the
# trajectory-coherence gate the other chronic-noise gates were never tuned to
# catch (they catch backgrounds that never settle, not real motion on the
# wrong branch): moving_object climbed 0 -> 2 -> 5 -> 7 over frames 960-962.
# Smoothing shift with an EMA before the threshold check, same shape as
# EGO_RESIDUAL_EMA_ALPHA, fixes it: a stretch that's consistently near the
# line reads as consistently over it instead of flickering. 0.25 matches
# EGO_RESIDUAL_EMA_ALPHA's tuning for the same class of RANSAC/estimation
# noise; dropped (not carried through) whenever ego estimation itself fails,
# so a genuine loss of tracking doesn't drag a stale average toward "static".
EGO_SHIFT_EMA_ALPHA = float(os.getenv("BATTLESIGHT_EGO_SHIFT_EMA_ALPHA", "0.25"))
EGO_MIN_FEATURES = int(os.getenv("BATTLESIGHT_EGO_MIN_FEATURES", "12"))      # too few to trust a fit
EGO_DIFF_THRESHOLD = int(os.getenv("BATTLESIGHT_EGO_DIFF_THRESH", "28"))     # grey levels, compensated diff
# Ego compensation cancels ONE global 2D transform. That holds for a distant,
# near-planar scene (an aerial feed, which is what it was tuned on) and breaks
# down under parallax: in a close-range handheld view the foreground and the
# background move across the sensor at different rates, so no single transform
# cancels both and static structure keeps registering as motion.
#
# The tell is how much of the frame still reads as foreground AFTER
# compensation. Measured per frame-pair:
#
#   tests/assets/static.mp4      0.0%   (static camera)
#   tests/assets/moving.mp4      0.6%   (synthetic pan)
#   tests/assets/drone_pan.mp4   0.8%   (aerial pan, real traffic)
#   v1.mp4                       3.5%   (handheld close-up: compensation fails)
#
# Above this fraction the motion pass is not trustworthy, so its blobs are
# dropped rather than reported as phantom contacts. 2% sits well clear of every
# good clip above. The detector falls back to full-frame classification for
# those frames -- see Detector.track -- so this suppresses false contacts
# without going blind. Set to 1.0 to disable the check entirely.
EGO_MAX_RESIDUAL = float(os.getenv("BATTLESIGHT_EGO_MAX_RESIDUAL", "0.02"))
# EGO_MAX_RESIDUAL's counterpart for the plain-MOG2 (static-camera) branch,
# which had no sanity check at all. On v3.mp4's tree-lined platform, wind-blown
# foliage never settles into MOG2's background model: raw foreground sits at a
# CHRONIC ~20-27% for the whole handheld-still segment (measured, not a single
# spike), and the morphology/contour step fragments that into dozens of tiny
# blobs whose count swings frame to frame. For a few consecutive frames enough
# fragments drift the same direction to pass MOTION_COHERENCE_THRESHOLD
# together, bursting from a ~2-4/frame baseline to 16 moving_object boxes in
# one frame -- scattered across the treeline and frame edges at 0.68-1.00
# confidence. 0.12 sits above every quiet frame measured on this clip (peaked
# ~0.10 around frames 6-8) and well below the ~0.20-0.27 foliage baseline.
MOTION_MOG2_MAX_FRACTION = float(os.getenv("BATTLESIGHT_MOTION_MOG2_MAX_FRAC", "0.12"))
# Motion blur on a fast handheld pan defeats a clean single-frame residual
# check (see EGO_MAX_RESIDUAL): the compensated-foreground fraction doesn't
# resolve clearly above or below the line, it HOVERS across it frame to frame.
# Measured on v6.mp4, frames 320-334: 0.017-0.028 against a 0.02 threshold --
# reading that raw flips the reliable/unreliable flag every frame, and the
# frames that land "reliable" by chance emit a burst of moving_object boxes
# off noise that never actually cleared. MotionDetector.detect() smooths the
# fraction with an exponential moving average (this is its weight on the
# newest frame) before comparing to EGO_MAX_RESIDUAL, so a stretch that's
# consistently near the line reads as consistently over it instead of
# flickering. Lower = smoother/slower to react to a genuine change; 1.0
# disables smoothing (raw per-frame value, the old behaviour). 0.25, not a
# faster 0.4: 0.4 still let one frame in the v6.mp4 burst (frame 329) dip
# under threshold by chance; 0.25 held the whole 320-334 stretch reliably
# unreliable, at the cost of reacting a little slower once a pan genuinely
# stabilises.
EGO_RESIDUAL_EMA_ALPHA = float(os.getenv("BATTLESIGHT_EGO_RESIDUAL_EMA_ALPHA", "0.25"))
# Chronic fine texture (brick paving, gravel, compression grain) fragments
# into dozens of small blobs while the raw foreground FRACTION (checked by
# EGO_MAX_RESIDUAL / MOTION_MOG2_MAX_FRACTION above) stays comfortably under
# threshold -- each blob is individually small, there's just a lot of them.
# Measured on v7.mp4 (a drone hovering over brick paving): 30-58 blobs surviving
# the size/aspect gate for 30+ consecutive frames, fraction 0.02-0.11 the whole
# time (well under MOTION_MOG2_MAX_FRACTION's 0.12, which was tuned only
# against v3.mp4's tree line -- a much higher fraction, 0.19-0.27, that never
# generalised to a scene fragmenting finer but at lower coverage). Quiet real
# frames on the same clip and on v6.mp4 sat at 0-16 blobs. A frame with more
# surviving blobs than this is treated as chronic noise -- same as the two
# fraction gates, its blobs are dropped rather than handed to the coherence
# gate, which would otherwise score a fragment swarm as a burst of targets.
MOTION_CHRONIC_BLOB_COUNT = int(os.getenv("BATTLESIGHT_MOTION_CHRONIC_BLOBS", "20"))
# TRIED AND REVERTED: EMA-smoothing this count the same way as
# EGO_SHIFT_EMA_ALPHA/EGO_RESIDUAL_EMA_ALPHA. Measured on v7.mp4 (drone
# hovering over brick paving), frames 258-265: raw blob count sits at 17-20,
# right under this gate, for several consecutive frames (one, 262, crosses to
# 23). Smoothing DELAYED the trip rather than preventing a false one -- the
# smoothed value rises slower than the raw spike, so the gate fired later and
# let MORE frames of coherent tracking accumulate first (burst went from 7 to
# 13 boxes, worse, not better). Unlike the shift/residual EMAs, this gate's
# job is to react fast to a spike, not to hold a decision steady across a
# value oscillating near the line -- smoothing was the wrong tool here. A
# real fix needs a different shape (e.g. a cooldown that keeps dropping blobs
# for a few frames after MOTION_CHRONIC_BLOB_COUNT trips once, so a scene
# hovering just under it doesn't get to build a fresh coherent track between
# trips) -- not attempted yet, see README "Still outstanding".

# Reference-image exclusion (app/exclusion.py)
EXCLUSION_STORE_PATH = BASE_DIR / "weights" / "exclusions.json"
EXCLUSION_SIMILARITY_THRESHOLD = float(os.getenv("BATTLESIGHT_EXCLUSION_SIM", "0.85"))

# Static HUD/OSD overlay rejection (app/overlay_mask.py).
# The real operational footage here is FPV/UAV video with a HUD burned in --
# reticles, telemetry text, emblems. Those glyphs are small, high-contrast and
# rectangular, which is what this model has learned a distant vehicle looks
# like from above. Measured on v11.mp4 BEFORE this filter: 54,658
# `light_vehicle` boxes over 1,746 frames (31.3/frame) on a clip with no
# vehicles in it; a spatial heatmap of those boxes reproduced the HUD exactly,
# one box per reticle dash and per telemetry character, median size 9x9 px.
# Prior sessions attributed this to "no training coverage of this camera
# angle" and recommended adding a driving dataset (BDD100K/KITTI) -- that
# would not have touched it, because the boxes are not on the scene at all.
# Set to 0 to disable and get the unfiltered behaviour back.
OVERLAY_FILTER = os.getenv("BATTLESIGHT_OVERLAY_FILTER", "1") not in ("0", "false", "False")
# Resolution of the persistence/stability grid over the normalised frame.
# 64x64 puts a cell at ~9x7 px on this 564x480 footage, about one HUD glyph --
# fine enough to mask a reticle dash without taking the terrain around it.
OVERLAY_GRID = int(os.getenv("BATTLESIGHT_OVERLAY_GRID", "64"))
# Qualifying (camera-moving) frames of evidence before any cell can be masked.
# Nothing is suppressed at all until this is met, so a feed's first couple of
# seconds are always unfiltered.
OVERLAY_WARMUP_FRAMES = float(os.getenv("BATTLESIGHT_OVERLAY_WARMUP", "60"))
# Fraction of camera-moving frames in which a cell must hold a small detection
# before it counts as persistent. A real object crossing the frame under a pan
# touches any given cell for a handful of frames; a painted glyph touches its
# own cell in nearly all of them. At 0.3 a real target would have to hold one
# ~9x7 px cell for 60 of the last 200 camera-moving frames to be masked.
#
# Swept offline against cached detections (so every row is the same inference,
# only the filter parameters varying). "v11 removed" is the share of that
# clip's 54,658 phantom `light_vehicle` boxes suppressed; the other columns are
# real detections lost on the regression clips:
#
#   persistence   dilate   v11 removed   v10   v2   v9   v6   cells masked
#      0.50         0         66.0%       0    0    0    0        28
#      0.50         1         75.5%       0    0    0    0       128
#      0.35         0         73.0%       0    0    0    0        36
#      0.35         1         82.7%       0    0    0    0       157
#      0.25         0         77.6%       0    0    0    0        46
#      0.25         1         86.0%       0    0    0    0       189
#
# Zero real detections lost on any regression clip at any setting -- the cost
# of loosening this is bounded by OVERLAY_MAX_FRACTION, not by recall.
OVERLAY_PERSISTENCE = float(os.getenv("BATTLESIGHT_OVERLAY_PERSISTENCE", "0.3"))
# Grow the mask by this many cells in each direction. A HUD glyph's detection
# box jitters frame to frame, so its centre lands in whichever of two or three
# adjacent cells depending on noise, and no single cell reaches the
# persistence threshold on its own -- the evidence gets split. Worth ~9 points
# of removal on the sweep above for a mask still well inside the safety cap.
OVERLAY_DILATE_CELLS = int(os.getenv("BATTLESIGHT_OVERLAY_DILATE", "1"))
# Exponential decay applied to that evidence each qualifying frame, so the
# mask tracks a HUD that changes layout (a mode switch, a new readout appearing)
# instead of being fixed by whatever the first seconds contained. ~200-frame
# memory at 0.995.
OVERLAY_DECAY = float(os.getenv("BATTLESIGHT_OVERLAY_DECAY", "0.995"))
# Second condition, and the safety property that matters most: only boxes
# below this fraction of the frame's area are ever learned from OR suppressed.
# A HUD glyph is tiny by nature (9x9 px measured on v11.mp4, ~0.0003 of that
# frame); 0.01 is ~56x48 px on the same clip, generously above any glyph and
# far below a real target worth looking at. This is what stops a drone that
# deliberately holds a genuine target centred in frame from ever having that
# target masked, however persistent it looks to condition 1.
#
# TRIED AND REJECTED in this slot first: requiring the cell's PIXELS to be
# temporally static, on the theory that a painted glyph doesn't change while
# a real target does. It carries no information on this footage -- a glyph
# sits on top of a CHANGING background and an analog FPV feed is noisy
# everywhere, so HUD cells are not pixel-static at all. Measured on v11.mp4
# over 900 frames: of the 40 cells with detection persistence >= 0.3,
# requiring cell std <= 0.35x the frame's median cell std kept only 11, while
# loosening it far enough to keep them (1.0x) admitted 2048 cells -- half the
# grid, discriminating nothing. Box size does that safety job on evidence
# that actually separates the two cases.
OVERLAY_MAX_BOX_AREA = float(os.getenv("BATTLESIGHT_OVERLAY_MAX_BOX_AREA", "0.01"))
# Safety cap. If the two conditions would mask more than this much of the
# frame, the premise has broken down (a genuinely static scene, a feed where
# everything persists) and the filter disables itself entirely rather than
# suppressing real contacts. Failing OPEN is the only acceptable direction
# here: a phantom contact is a nuisance, a suppressed real one is the failure
# this system must never have. Same reasoning as MOTION_GATED being off.
OVERLAY_MAX_FRACTION = float(os.getenv("BATTLESIGHT_OVERLAY_MAX_FRACTION", "0.10"))

# --- Large painted overlays (subtitle blocks, banners) -----------------------
# OVERLAY_MAX_BOX_AREA above deliberately excludes big boxes from the glyph
# channel, because a large persistent box is exactly what a real target held
# centred in frame looks like, and masking that is the one failure this system
# must not have. But v10.mp4 carries a burned-in subtitle block that the model
# reads as light_vehicle at ~137x137 px -- far above the glyph limit, and
# bursty (45 hits in frames 283-344), so it fails the persistence test too.
#
# This second channel keeps the safety argument by swapping the discriminator.
# A painted block is GEOMETRICALLY RIGID: measured on v10, the subtitle cell's
# box width varies by cv 0.0047 and height by cv 0.0092 across 40 detections,
# i.e. sub-pixel. Real personnel boxes in the same clip and the same frames
# vary 2.5-15x more (cv 0.0232 to 0.1357 for every cell with >= 10 samples),
# because the world slides under a moving camera and their apparent size
# changes. Evidence is still only gathered while the camera is established to
# be moving, which is what makes that true.
#
# DEFAULT OFF. The separation is real but measured on one clip, with 2.5x of
# margin at the closest point -- not enough to enable a suppression path by
# default when the standing rule is that gates fail open. Turn it on per-feed
# once a second clip with a large painted overlay confirms the threshold.
OVERLAY_LARGE_FILTER = os.getenv("BATTLESIGHT_OVERLAY_LARGE", "0") not in ("0", "false", "False")

# Detections needed in a cell before its geometry is trusted. At n=1 the
# variance is trivially zero, which would qualify every one-off box.
OVERLAY_LARGE_MIN_HITS = int(os.getenv("BATTLESIGHT_OVERLAY_LARGE_MIN_HITS", "10"))

# max(cv_width, cv_height) at or below which a cell counts as painted. 0.015
# sits between the measured 0.0092 (subtitle) and 0.0232 (closest real object).
OVERLAY_LARGE_RIGIDITY = float(os.getenv("BATTLESIGHT_OVERLAY_LARGE_RIGIDITY", "0.015"))

# Classes the rigidity channel may never suppress, and never learns geometry
# from. `moving_object` (-1) is exempt for the same reason it is in the glyph
# channel: the motion pass only fires on something moving relative to the
# world, so it cannot be painted.
#
# `personnel` (0) is exempt on measurement, not principle. Replaying v10 with
# the channel on: at conf 0.25 it removed 31 of 47 phantoms and cost zero
# personnel, but at conf 0.10 -- the floor the ground view actually runs
# personnel at, see CONF_THRESHOLD_PERSONNEL_GROUND -- it removed 13 real
# personnel and zero phantoms. The extra low-confidence boxes admitted at 0.10
# dilute a cell's geometry statistics until real detections start looking
# rigid. A filter that deletes people at the threshold people are detected at
# is the exact failure the fail-open rule is for, so personnel is held out of
# this channel entirely. The phantoms this exists for are vehicles.
OVERLAY_LARGE_EXEMPT = frozenset({-1, 0})

# Frames to keep suppressing blobs after MOTION_CHRONIC_BLOB_COUNT trips once.
# The bare threshold was not enough on its own: a scene fragmenting to just
# UNDER the cutoff, crossing it only occasionally, builds fresh coherent tracks
# in the gaps between trips and emits them all at once.
#
# This is the mechanism that worked after EMA-smoothing this gate was tried and
# made things WORSE. The distinction is the whole point: smoothing made the
# decision SLOWER to trip; what was needed was for it to LAST longer once made.
#
# 2 already clears both bursts on the clip it was tuned against; 3 is one frame
# of margin, costing ~7% of that clip's moving_object boxes. Past that it is
# only lost recall -- 5 and 8 give up 13% and 16% for nothing further. Set to
# the smallest value that does the job, not the safest-looking one. Log 13.
MOTION_CHRONIC_COOLDOWN = int(os.getenv("BATTLESIGHT_MOTION_CHRONIC_COOLDOWN", "3"))
