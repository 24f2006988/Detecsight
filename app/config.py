"""Central configuration. Every path and threshold lives here, not scattered
through the routers."""
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

# Model
MODEL_PATH = os.getenv("BATTLESIGHT_MODEL", str(BASE_DIR / "weights" / "best.pt"))
# Optional second checkpoint, specialised for top-down/aerial footage (trained
# on VisDrone alone -- data/battlesight.yaml -- rather than the VisDrone +
# WiderPerson mix that MODEL_PATH uses). WiderPerson is ground-level personnel
# imagery only, so mixing it in helps handheld/CCTV-angle sources but has
# nothing to teach a straight-down drone view; a model trained on aerial
# imagery only is not diluted by it. Selected per-request via the `view`
# query param ("ground", the default, or "drone"); Detector.load() loads this
# alongside MODEL_PATH only if the file exists, and every view falls back to
# MODEL_PATH when it doesn't -- so serving still works with only one weights
# file present. See README "Detection accuracy work" for the aerial-view split.
DRONE_MODEL_PATH = os.getenv("BATTLESIGHT_DRONE_MODEL", str(BASE_DIR / "weights" / "drone_best.pt"))
DEVICE = os.getenv("BATTLESIGHT_DEVICE", "0")       # "0" = first GPU, "cpu" = CPU
# Inference precision, passed as `quantize=` to every predict()/track() call
# (see app/detector.py). Measured on this GPU (RTX 4060 Laptop), both
# checkpoints, VisDrone-only val, imgsz 1280, batch 1, warmed up:
#
#              PyTorch FP32   PyTorch FP16   TensorRT FP16   TensorRT INT8
#   general
#     mAP50        0.5631        0.5616         0.5608          0.5189
#     mAP50-95     0.3160        0.3154         0.3172          0.2808
#     recall       0.5260        0.5270         0.5303          0.4789
#     inference     13.3ms        7.63ms         6.95ms          4.23ms
#   drone
#     mAP50        0.5822        0.5823         0.5813          0.5414
#     mAP50-95     0.3280        0.3278         0.3303          0.3024
#     recall       0.5450        0.5438         0.5468          0.5011
#     inference     13.5ms        7.63ms         7.29ms          4.25ms
#
# FP16 costs <=0.002 mAP50 either model (noise) for a ~45% latency cut --
# standard here. INT8 is NOT: a real 4-point mAP50 / 5-point recall loss on
# both models for another ~1.6x on top of FP16, so it stays available
# (weights/*_int8.engine exist) but is not the default -- that tradeoff is a
# deployment decision, not something to make silently. True FP8 does not
# exist in this stack: checked the ultralytics exporter source directly
# (`ultralytics/engine/exporter.py`) -- it supports quantize levels 32
# (FP32), 16 (FP16), 8 (INT8), and weight-only w8a16/w8a32, nothing else, on
# either the PyTorch or TensorRT export path.
#
# None (FP32) only makes sense on CPU: half-precision ops are unsupported or
# slower on most CPU kernels, so QUANTIZE is None whenever DEVICE == "cpu"
# (this is also what keeps tests/test_motion_gating.py, which runs on CPU
# with a stock checkpoint, unaffected).
QUANTIZE = 16 if DEVICE != "cpu" else None
# 0.25, not 0.35. Measured on the val set at imgsz=1280 (F1 curve, mean over
# the 4 classes) -- 0.35 was giving away a fifth of the achievable recall:
#
#   conf   meanF1      P       R
#   0.25    0.499   0.697   0.412
#   0.35    0.447   0.790   0.343     <- previous default
#   0.50    0.367   0.873   0.265
#
# Mean F1 actually peaks near 0.16, but this model puts confident boxes on
# out-of-distribution scenes (it called a pencil case `light_vehicle` at 0.42),
# and an AR overlay full of phantom contacts is worse than a missed one. 0.25
# takes most of the recall back without opening the floor that far.
CONF_THRESHOLD = float(os.getenv("BATTLESIGHT_CONF", "0.25"))
# Ground-view-only, personnel-class-only override, lower than CONF_THRESHOLD.
# Purpose: ground view is handheld/bodycam-style footage where the goal is
# catching a person visible for only a short span, so personnel recall is
# worth a small precision cost that the other 3 classes are NOT -- keeping
# this override personnel-only avoids re-inviting the vehicle-hallucination
# problem measured on v11.mp4 (README fix 14), which was specifically about
# `light_vehicle` false positives at low confidence on off-distribution
# content; personnel is a different, better-covered class (WiderPerson is
# personnel-only ground-level data) and a different failure mode.
#
# Measured (not assumed) with a class-specific P/R/F1 sweep, class 0 only.
#
# 2026-09-02 RE-MEASURED, and the earlier conclusion here was an artefact of
# the val set, not a property of the model. The previous sweep ran on the
# BLENDED val (VisDrone + WiderPerson) and found a flat curve -- "0.10 buys
# only ~2 points of recall". But VisDrone's aerial personnel dominate that
# blend by instance count, and their P/R curve genuinely is flat, which masked
# the ground-level headroom underneath. Split by domain, on the fpv checkpoint:
#
#   WiderPerson val (1,000 ground-level images -- the bodycam-like case):
#     conf     P       R      F1      F2
#     0.08   0.600   0.750   0.667   0.714   <- F2 optimum
#     0.10   0.657   0.725   0.689   0.710   <- ADOPTED (the knee)
#     0.12   0.701   0.703   0.702   0.703
#     0.15   0.753   0.674   0.712   0.689
#     0.20   0.814   0.636   0.714   0.665   <- previous floor, F1 optimum
#     0.25   0.853   0.606   0.709   0.644
#
#   VisDrone val (aerial) for contrast: at 0.10 precision collapses to 0.431,
#   which is why this override stays GROUND-VIEW ONLY. Drone view keeps 0.25.
#
# Why 0.10 and not the F1 optimum: F1 weights precision and recall equally,
# which is the wrong objective for this system. The standing judgement (README
# "a phantom contact is a nuisance; a suppressed real one is the failure this
# system must never have") is a recall-weighted one, so the floor is chosen on
# F2. F2 peaks at 0.08, but 0.10 scores within 0.004 of that peak while
# recovering 5.7 points of precision, so 0.10 is the knee, not 0.08.
#
# Effect: personnel recall on ground-level imagery 0.636 -> 0.725, i.e. ~18%
# more people found, at precision 0.814 -> 0.657. That is a deliberate trade:
# roughly one box in three is now wrong, against roughly one person in four
# being missed before.
#
# Secondary benefit for tracking: Detector._model_conf_floor() passes the
# lowest floor any class/view needs to the model itself, so lowering this also
# hands ByteTrack more low-confidence candidates for its second association
# pass -- which is exactly the mechanism that keeps a briefly-visible person on
# a track instead of dropping them.
#
# CAVEAT, unchanged and still the honest limit: WiderPerson is static, clean,
# well-lit street photography, not motion-blurred bodycam or rendered game
# footage. It does not represent that case, and this sweep cannot tell you how
# the model behaves there. If short-span personnel recall is still not good
# enough, the real fix is training-data coverage of dense, occluded,
# ground-level people (CrowdHuman is the closest public match) -- not a further
# threshold drop.
CONF_THRESHOLD_PERSONNEL_GROUND = float(os.getenv("BATTLESIGHT_CONF_PERSONNEL_GROUND", "0.10"))
# WARNING: this is a NO-OP for the model's own inference, and has been all
# along. YOLO26's head reports `end2end: True` -- it is an NMS-free one-to-one
# detector, so the `iou=` argument passed to predict()/track() is ignored
# entirely. Measured 2026-09-02: sweeping it over 0.5/0.6/0.7/0.8 on 150
# WiderPerson val images produced byte-identical results (TP=3364, preds=5210,
# R=0.698, P=0.646 at every value). The exported engine's fixed
# `output0 (1, 300, 6)` shape is the same story from the other side.
#
# Do not reach for this to fix crowd/occlusion recall -- there is no NMS to
# loosen. Duplicate suppression is learned by the one-to-one head, so the only
# way to change that behaviour is training.
#
# It IS still used, but only by this project's own IoU arithmetic:
# _far_field_pass's duplicate check, _claim_motion_blobs, overlay_mask and the
# motion filter's box merging all call iou_xyxy() with it. Changing it moves
# those and nothing else.
IOU_THRESHOLD = float(os.getenv("BATTLESIGHT_IOU", "0.5"))
# Ultralytics defaults max_det to 300. VisDrone val frames hold up to 317
# objects, so the default silently truncates the densest frames -- exactly the
# crowded scenes where the count matters. Raising it costs only NMS time.
MAX_DET = int(os.getenv("BATTLESIGHT_MAX_DET", "500"))

# --- Far-field second pass (2026-09-02) -------------------------------------
# Recall is not uniform across target size -- it collapses with distance.
# Measured on 300 WiderPerson val images (ground-level), personnel, imgsz 1280:
#
#   size(px)     GT   recall
#      <16     1427    0.187     <- 41% of all people are under 32 px
#     16-32    2214    0.622        and we find fewer than half of them
#     32-48    1426    0.805
#     48-64    1128    0.855
#     64-96    1481    0.937
#      >96     1207    0.940     <- near targets are effectively solved
#
# So all the loss is in the far field. Tiling the WHOLE frame recovers it
# (<16 -> 0.306) but costs 5-6x inference and craters precision 0.662 -> 0.478
# on duplicate/edge boxes -- not affordable at this project's ~10 fps.
#
# Instead the cheap full-frame pass CHOOSES where to spend one extra pass: the
# far field is wherever the small boxes already are. Measured, same 300 images:
#
#                       found   precision   ms/img
#   full frame only      6279     0.662       ~40
#   + far-field tile     6611     0.601       115
#   + size filter        6555     0.644       111    <- adopted
#
# i.e. 73% of the full-tiling recall gain for 40% of its extra cost, with
# precision essentially preserved. The size filter is what preserves it: the
# tile exists to find SMALL targets, so any large box it returns is a duplicate
# of one the full frame already had.
# DEFAULT OFF. Measured end-to-end on video it costs 80-94 ms/frame against a
# 25-28 ms full-frame baseline -- 2-3x over the 40 ms serving budget. It stays
# in the tree because it is the right tool OFFLINE (annotate_video.py, forensic
# review of a recorded clip) where latency does not matter and the small-target
# recall is worth 3x the time. Set BATTLESIGHT_FARFIELD=1 to enable.
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
# Resolution for the TILE pass. MUST equal IMGSZ while serving a --static
# TensorRT engine: the engine has a FIXED input shape, and anything else dies
# with "input size (1,3,640,640) not equal to max model size (1,3,1280,1280)".
# Measured, not assumed -- setting 640 here crashed the far-field pass outright.
#
# This is unfortunate, because the tile is a crop and does not need IMGSZ to
# out-resolve the full-frame pass: a region covering fraction f of the frame at
# size S has effective resolution (S/f)/IMGSZ times the full pass, so at f~0.4
# even 640 would still be a ~1.25x gain for a quarter of the cost. That matters
# because the strided frame is a LATENCY SPIKE, not an averaged cost: measured
# p90 48.5 ms against a 26.9 ms median, on a 40 ms budget.
#
# To actually get the cheap tile you need a second engine built at the smaller
# size (scripts/export_engine.py --model weights/best.pt --imgsz 640 --static,
# to its own path) and to point the far-field model at it, or to let the
# far-field model load the .pt instead of an engine. Neither is done here.
# 1280, not 640. Targets here are tiny: the median VisDrone val object is 11 px
# across at imgsz=640 and 75% of `personnel` boxes are under 16 px, which is at
# the floor of what the stride-8 head can resolve. Measured on the 548-image val
# set with weights/best.pt (see scripts/diagnose.py, runs/diagnose.json):
#
#   imgsz   mAP50   mAP50-95   recall   personnel mAP50
#     640   0.4331    0.2308   0.4182            0.382
#    1280   0.5046    0.2772   0.4855            0.534
#
# On real 1080p drone footage that is 33.5 vs 20.2 detections per frame -- 66%
# more -- for 35.6 ms vs 9.3 ms (28 fps, still realtime for a feed).
# 1600 was measured too and is strictly worse than 1280 (30.3 dets, 38.9 ms).
# Low-resolution sources lose nothing by this: on a 478x850 phone clip the
# only boxes at any size were false positives (a pencil case as light_vehicle),
# and 1280 produced FEWER of them than 640. So this is a single global size,
# not a per-source cap.
IMGSZ = int(os.getenv("BATTLESIGHT_IMGSZ", "1280"))
# Tile resolution for the far-field pass -- see the FARFIELD_IMGSZ note above.
# Defined here, after IMGSZ, because it defaults to it.
FARFIELD_IMGSZ = int(os.getenv("BATTLESIGHT_FARFIELD_IMGSZ", str(IMGSZ)))

# TensorRT engine acceleration (scripts/export_engine.py). PyTorch inference
# through ultralytics is the single biggest chunk of glass-to-overlay latency
# on the tracked path. If a `.engine` file sits next to MODEL_PATH (same stem,
# built by the export script), Detector.load() prefers it over the .pt -- same
# weights, same classes, compiled with layer fusion and FP16 kernels for this
# GPU. Falls back to the .pt automatically if the engine is missing, fails to
# load (wrong TensorRT/driver version, corrupted build), or DEVICE is "cpu"
# (an engine only runs on the GPU family it was built on). Set to 0 to always
# use the .pt checkpoint even when a matching engine exists.
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
# Per-view, not one global value -- see MotionDetector.detect()'s `view`
# param in app/motion_filter.py. A straightness threshold is fundamentally a
# recall/false-positive tradeoff (see the measured table below), and the two
# views have opposite needs: drone view looks down at terrain/foliage, where
# brief wind-blown texture jitter is the main noise source and losing a few
# frames of genuine short-lived motion is an acceptable trade; ground view is
# handheld/bodycam-style, where the whole point is catching a person who's
# only on screen a handful of frames, so the same tightening would work
# directly against the goal. `view` defaults to "ground" -- see
# MotionDetector._coherence_threshold below for the lookup and its fallback.
#
# DRONE: 0.85, not the original 0.5. Raised after v10.mp4 (wind-blown
# grass/foliage, off-domain nature footage that reads like aerial/top-down
# terrain) showed short-lived coherent texture jitter can score 0.55-0.97
# straightness over just MOTION_COHERENCE_MIN_POINTS=4 frames -- a brief
# consistent gust looks exactly like a real trajectory over that short a
# window, which no amount of the OTHER chronic-noise gates catches (v10's
# fg_fraction ~0.04 and blob count ~9-10 sit far under both, this is a "few
# blobs, briefly coherent" case unlike the "many blobs" swarms those target).
# Measured with a full sweep, not assumed (v10.mp4/v3.mp4/v6.mp4, this
# machine, all other settings default):
#
#   threshold   v10 moving_object   v10 burst frames   v3 moving_object   v6 moving_object
#     0.50            465                  7                  34                348
#     0.65            363                  5                  29                243
#     0.75            266                  3                  24                205
#     0.85            205                  0                  18                128
#     0.92            127                  0                   7                 64
#
# 0.85 is the first value that fully clears v10's bursts (frames with 6+
# moving_object boxes). REAL COST, not free: it also cuts v3's and v6's
# already-verified-real moving_object counts by ~47% and ~63% respectively --
# this gate cannot distinguish "briefly coherent noise" from "a real target
# only tracked for a few frames" by straightness alone, so tightening it
# costs genuine short-lived detections too. Neither of this project's
# synthetic regression clips (tests/assets/moving.mp4, drone_pan.mp4)
# exercises this gate at ANY threshold (0 moving_object throughout), so this
# was NOT verified against a real slow/distant-target case -- if a genuine
# UAV or slow mover ever needs to be caught only by a handful of coherent
# frames, re-measure against real footage of that before raising further.
# Applied ONLY to drone view -- v3/v6 are ground-view clips and were not
# meant to lose that recall, they were just the regression check.
MOTION_COHERENCE_THRESHOLD_DRONE = float(os.getenv("BATTLESIGHT_MOTION_COHERENCE_DRONE", "0.85"))
# GROUND: reverted to the original 0.5 -- the value fixes 7-9 and 13 were
# tuned and verified against (v3.mp4/v6.mp4 are ground-view clips). Kept
# deliberately permissive: ground view's stated purpose is catching personnel
# visible for only a short span (handheld/bodycam-style footage), which is
# the opposite need from drone view's terrain-noise problem above -- a
# stricter threshold there would suppress exactly the brief, real detections
# this view exists to catch.
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

# Stage 3: motion-gated inference. Instead of running YOLO over every pixel of
# every frame, run it only on crops around the blobs that survived stages 1-2.
# A quiet frame costs zero GPU; a busy one costs a few small crops instead of
# one full frame. Set BATTLESIGHT_MOTION_GATED=0 for the original behaviour
# (full-frame YOLO every frame, motion used only to add moving_object boxes).
# DEFAULT FLIPPED TO OFF. Motion gating means the classifier only ever sees
# what moved, so a target that is standing still is never detected at all --
# for situational awareness a parked truck and a stationary sentry are exactly
# what you want on the overlay. Measured on tests/assets/drone_pan.mp4 through
# the tracked path:
#
#   MOTION_GATED=1    0.0 detections/frame    28.3 ms   (35 fps)
#   MOTION_GATED=0   26.5 detections/frame    96.7 ms   (10 fps)
#
# Zero. The clip is a pan over a still scene, so nothing in it moves relative
# to the ground and the gate discards all 26.5 real targets per frame -- 267
# light_vehicle, 195 two_wheeler, 62 personnel, 6 heavy_vehicle over 20 frames.
# The speed was real but it was the speed of not looking.
#
# Cost of the fix is throughput: ~85 ms/frame at IMGSZ=1280 on 1080p, i.e.
# ~10 fps end-to-end over the WebSocket.
#
# Do NOT try to buy that back by lowering IMGSZ. Measured in one warm process,
# same frames, only imgsz varying:
#
#   imgsz   ms/frame   fps   detections/frame
#     640       81     12.4       17.7
#     960       82     12.2       25.5
#    1280       88     11.4       26.5
#
# The tracked path is dominated by fixed per-frame overhead -- the motion pass
# (~27 ms), ByteTrack, exclusion, postprocessing -- not by the forward pass.
# Going 1280 -> 640 gives up a third of the detections to gain about 1 fps.
# If throughput really has to improve, cut the motion pass instead: on this
# path it only supplies the moving/static flag and the class-agnostic contacts.
# Set BATTLESIGHT_MOTION_GATED=1 to restore the old behaviour.
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

# Frames to keep suppressing blobs after MOTION_CHRONIC_BLOB_COUNT trips once.
# The bare threshold above was not sufficient on its own: a scene that
# fragments to just UNDER the cutoff for several consecutive frames, crossing
# it only occasionally, gets to build fresh coherent tracks in the gaps
# between trips and then emits them all at once. Measured on v7.mp4 (drone
# over brick paving), which sits at 17-20 surviving blobs against the cutoff
# of 20: frame 261 produced 7 moving_object boxes against a local baseline of
# 0, and frame 104 produced 6.
#
# This is the "different mechanism" the earlier note called for after
# EMA-smoothing this gate was tried and made things worse -- smoothing DELAYS
# the trip (the averaged value climbs slower than the raw spike), where what
# is actually needed is for the decision to LAST longer once made.
#
# Swept with the reconstructed scripts/diagnose_bursts.py on v7.mp4 (ground
# view, motion-only, all other settings default). "bursts" counts frames far
# above their own local rolling median, which is the shape of the bug:
#
#   cooldown   moving_object   max/frame   bursts   extra frames dropped
#      0            242            7          2              0     <- the bug
#      2            228            5          0              7
#      3            224            5          0             13     <- adopted
#      5            210            5          0             22
#      8            204            5          0             32
#
# 2 already clears both bursts; 3 is one frame of margin for a scene
# fragmenting slightly differently, at a total cost of ~7% of this clip's
# moving_object boxes. Past that the curve is just lost recall -- 5 and 8 give
# up 13% and 16% for nothing further. Like every gate in this file this trades
# real short-lived detections against phantom ones, so it is set to the
# smallest value that does the job rather than the safest-looking one.
MOTION_CHRONIC_COOLDOWN = int(os.getenv("BATTLESIGHT_MOTION_CHRONIC_COOLDOWN", "3"))
