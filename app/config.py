"""Central configuration. Every path and threshold lives here.

All values were measured rather than guessed. The one-line reason next to each
is the short version; ENGINEERING_LOG.md has the sweep, the alternatives and
the things that were tried and rejected. Read the log section before changing a
constant, because several of these have been moved once and moved back.

Everything is overridable by environment variable, prefix BATTLESIGHT_.
"""
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

# --- Model and precision ----------------------------------------------------
MODEL_PATH = os.getenv("BATTLESIGHT_MODEL", str(BASE_DIR / "weights" / "best.pt"))
# Optional per-view checkpoint. The VisDrone-only specialist that lived here was
# retired (it lost on its own home domain, log 17); drop a file back in and
# ?view=drone picks it up, otherwise this falls back to MODEL_PATH.
DRONE_MODEL_PATH = os.getenv("BATTLESIGHT_DRONE_MODEL", str(BASE_DIR / "weights" / "drone_best.pt"))
DEVICE = os.getenv("BATTLESIGHT_DEVICE", "0")       # "0" = first GPU, "cpu" = CPU
# FP16 costs <=0.002 mAP50 for ~45% less latency. INT8 was measured and rejected
# at -4 mAP50 / -5 recall. Log 11.
QUANTIZE = 16 if DEVICE != "cpu" else None
# 0.25, not the inherited 0.35 (mean F1 0.447 -> 0.499). F1 peaks nearer 0.16,
# but this model puts confident boxes on off-distribution scenes. Log 3.
CONF_THRESHOLD = float(os.getenv("BATTLESIGHT_CONF", "0.25"))
# personnel-only floors, chosen on F2 rather than F1 because a missed person
# costs more than a false box. Ground: recall 0.636 -> 0.725 (log 17b).
# Drone: the SARD-blended checkpoint scores real targets at 0.117-0.256, and
# 0.10 is the measured F2 optimum on SARD test (log 22h/22i).
CONF_THRESHOLD_PERSONNEL_GROUND = float(os.getenv("BATTLESIGHT_CONF_PERSONNEL_GROUND", "0.10"))
CONF_THRESHOLD_PERSONNEL_DRONE = float(os.getenv("BATTLESIGHT_CONF_PERSONNEL_DRONE", "0.10"))
# NO-OP for model inference: YOLO26's head is end2end (NMS-free), so `iou=` is
# ignored and sweeping it gives byte-identical results. Do not reach for this to
# fix crowd recall. Still used by this repo's own IoU arithmetic. Log 19.
IOU_THRESHOLD = float(os.getenv("BATTLESIGHT_IOU", "0.5"))
# Ultralytics defaults to 300; VisDrone val frames hold up to 317. Log 4.
MAX_DET = int(os.getenv("BATTLESIGHT_MAX_DET", "500"))

# --- Far-field second pass --------------------------------------------------
# Recall collapses with target size, so one extra pass is spent wherever the
# small boxes already are. Off by default for jitter, not cost: p90 48.5 ms
# against a 26.9 ms median stutters an AR overlay. Good for offline work. Log 19.
FARFIELD_ENABLED = os.getenv("BATTLESIGHT_FARFIELD", "0") not in ("0", "false", "False")
FARFIELD_STRIDE = int(os.getenv("BATTLESIGHT_FARFIELD_STRIDE", "3"))     # every Nth tracked frame
FARFIELD_MAX_BOX = float(os.getenv("BATTLESIGHT_FARFIELD_MAX_BOX", "48"))  # px, discard larger from the tile
FARFIELD_SMALL_PCT = float(os.getenv("BATTLESIGHT_FARFIELD_SMALL_PCT", "40"))
FARFIELD_MIN_BOXES = int(os.getenv("BATTLESIGHT_FARFIELD_MIN_BOXES", "3"))
FARFIELD_PAD = float(os.getenv("BATTLESIGHT_FARFIELD_PAD", "0.10"))
# Fallback (x1, y1, x2, y2) fractions: the horizon band, used only when the
# frame gives nothing better to go on.
FARFIELD_PRIOR = tuple(float(v) for v in os.getenv(
    "BATTLESIGHT_FARFIELD_PRIOR", "0.2,0.15,0.8,0.6").split(","))
FARFIELD_MAX_FRACTION = float(os.getenv("BATTLESIGHT_FARFIELD_MAX_FRACTION", "0.55"))
# 1280, not the inherited 640: at 640 three quarters of personnel boxes are
# under 16 px. 1600 is strictly worse, above the trained size. Logs 2 and 19.
IMGSZ = int(os.getenv("BATTLESIGHT_IMGSZ", "1280"))
# Must equal IMGSZ while serving a --static engine, whose input shape is fixed.
FARFIELD_IMGSZ = int(os.getenv("BATTLESIGHT_FARFIELD_IMGSZ", str(IMGSZ)))
# Prefer a .engine next to MODEL_PATH; falls back to the .pt when it is missing,
# fails to load, or DEVICE is "cpu". Log 5b.
USE_TENSORRT = os.getenv("BATTLESIGHT_USE_TENSORRT", "1") not in ("0", "false", "False")

# --- Tracking ---------------------------------------------------------------
TRACKER_CONFIG = "bytetrack.yaml"
MOTION_WINDOW = 12          # frames of history kept per track
MOTION_THRESHOLD = 0.015    # normalised displacement that counts as "moving"
MAX_TRACKS_PER_SOURCE = int(os.getenv("BATTLESIGHT_MAX_TRACKS", "512"))  # LRU cap, ids only climb

# --- Training jobs ----------------------------------------------------------
TRAIN_SCRIPT = BASE_DIR / "scripts" / "train.py"
RUNS_DIR = BASE_DIR / "runs" / "detect"
LOGS_DIR = BASE_DIR / "runs" / "logs"

CLASS_NAMES = ["personnel", "two_wheeler", "light_vehicle", "heavy_vehicle"]

# --- Motion filter, stages 1-2 (app/motion_filter.py) -----------------------
# Tags anything moving coherently, not just the four trained classes.
# 80 px^2 at MOTION_WORKING_WIDTH, not 30: a 5x6 smudge was most of the phantom
# output. This is also the floor on how small an unrecognised moving thing can
# be and still be reported, so raising it costs distant air targets.
MOTION_MIN_BLOB_AREA = int(os.getenv("BATTLESIGHT_MOTION_MIN_AREA", "80"))
MOTION_MATCH_MAX_DIST = int(os.getenv("BATTLESIGHT_MOTION_MAX_DIST", "60"))       # px, blob-to-track association
MOTION_TRACK_MAX_AGE = int(os.getenv("BATTLESIGHT_MOTION_TRACK_AGE", "5"))        # frames before a track is dropped
MOTION_COHERENCE_MIN_POINTS = int(os.getenv("BATTLESIGHT_MOTION_MIN_POINTS", "4"))  # history needed to judge
MOTION_COHERENCE_MIN_PATH = float(os.getenv("BATTLESIGHT_MOTION_MIN_PATH", "8"))    # px, below this is noise

# --- Illumination-vs-structure discriminator --------------------------------
# Trajectory gates judge a blob by where it went. That cannot separate a light
# (whose bright region drifts and reads as coherent travel) from an object, and
# cannot report anyone visible for fewer than 4 frames at all. This decides from
# one frame: align the previous frame, then test whether the change is a uniform
# brightness shift or preserves structure under z-scoring. Fails open.
MOTION_STRUCTURE_ENABLED = os.getenv("BATTLESIGHT_MOTION_STRUCTURE", "1") not in ("0", "false", "False")
# Permissive on purpose: these blobs already passed the trajectory gates.
MOTION_STRUCTURE_MIN = float(os.getenv("BATTLESIGHT_MOTION_STRUCTURE_MIN", "0.25"))
# Stricter, because the fast path spends less trajectory evidence for the same call.
MOTION_STRUCTURE_MIN_FAST = float(os.getenv("BATTLESIGHT_MOTION_STRUCTURE_MIN_FAST", "0.40"))
MOTION_STRUCTURE_MIN_STD = float(os.getenv("BATTLESIGHT_MOTION_STRUCTURE_MIN_STD", "4.0"))  # flatter = cannot judge
MOTION_STRUCTURE_UNIFORM_RATIO = float(os.getenv("BATTLESIGHT_MOTION_STRUCTURE_UNIFORM", "0.6"))
# Two points make _straightness degenerate, so the fast path requires the
# structure test plus real displacement instead of relying on it.
MOTION_COHERENCE_MIN_POINTS_FAST = int(os.getenv("BATTLESIGHT_MOTION_MIN_POINTS_FAST", "2"))
# Past EGO_MAX_RESIDUAL the old behaviour was to report nothing, which blinded
# the channel on 535 of 1068 frames of v6. Run degraded up to this multiple.
EGO_RESIDUAL_DEGRADED_FACTOR = float(os.getenv("BATTLESIGHT_EGO_RESIDUAL_DEGRADED_FACTOR", "2.0"))
MOTION_COHERENCE_DEGRADED_MIN = float(os.getenv("BATTLESIGHT_MOTION_COHERENCE_DEGRADED", "0.85"))
# Per-view, because the two views want opposite trades. Drone 0.85 clears the
# foliage-jitter bursts but costs 47-63% of verified-real counts on ground
# clips; ground stays permissive at 0.5 to keep briefly-visible people. Logs 14, 15.
MOTION_COHERENCE_THRESHOLD_DRONE = float(os.getenv("BATTLESIGHT_MOTION_COHERENCE_DRONE", "0.85"))
MOTION_COHERENCE_THRESHOLD_GROUND = float(os.getenv("BATTLESIGHT_MOTION_COHERENCE_GROUND", "0.5"))
# A panning camera reads as near-whole-frame change, so both caps keep per-frame
# cost bounded instead of growing with scene busyness.
MOTION_MAX_BLOBS_PER_FRAME = int(os.getenv("BATTLESIGHT_MOTION_MAX_BLOBS", "40"))
MOTION_MAX_TRACKS_PER_SOURCE = int(os.getenv("BATTLESIGHT_MOTION_MAX_TRACKS", "150"))
# Subtraction and contour finding run downscaled; at 1080p findContours itself
# is the cost. Blob coordinates are scaled back to full resolution afterwards.
MOTION_WORKING_WIDTH = int(os.getenv("BATTLESIGHT_MOTION_WIDTH", "480"))
# Shape gate: a swaying branch reads as a long thin sliver, a lighting change as
# a wide flat band. Cheap, and runs before any trajectory bookkeeping.
MOTION_ASPECT_MIN = float(os.getenv("BATTLESIGHT_MOTION_ASPECT_MIN", "0.15"))   # w/h
MOTION_ASPECT_MAX = float(os.getenv("BATTLESIGHT_MOTION_ASPECT_MAX", "8.0"))
MOTION_CLAIM_IOU = float(os.getenv("BATTLESIGHT_MOTION_CLAIM_IOU", "0.1"))  # YOLO box claims a blob

# --- Motion gating: off, and it must stay off -------------------------------
# Gating means the classifier only sees what moved, so a stationary target is
# never detected. Measured on a pan over a still scene: 0.0 detections/frame at
# 28.3 ms against 26.5 at 96.7 ms. The speed was the speed of not looking.
# Do not try to win the throughput back by lowering IMGSZ either. Log 5.
MOTION_GATED = os.getenv("BATTLESIGHT_MOTION_GATED", "0") not in ("0", "false", "False")
# Run the CPU motion stage beside the GPU pass rather than before it. Nothing on
# the model path reads the blobs until the end, and both sides release the GIL,
# so the frame costs max(CPU, GPU) instead of the sum: 46.4 -> 24.6 ms median.
MOTION_PARALLEL = os.getenv("BATTLESIGHT_MOTION_PARALLEL", "1") not in ("0", "false", "False")
# Blobs are the moving part of a target, not the whole of it, so crops are
# padded outward to give the detector its context back.
MOTION_CROP_PADDING = float(os.getenv("BATTLESIGHT_MOTION_CROP_PAD", "0.6"))   # fraction of box side
MOTION_CROP_MIN_SIZE = int(os.getenv("BATTLESIGHT_MOTION_CROP_MIN", "128"))    # px, full-frame scale
MOTION_CROP_MERGE_IOU = float(os.getenv("BATTLESIGHT_MOTION_CROP_MERGE_IOU", "0.2"))
MOTION_MAX_CROPS_PER_FRAME = int(os.getenv("BATTLESIGHT_MOTION_MAX_CROPS", "6"))  # past this, one full pass is cheaper
MOTION_CROP_IMGSZ = int(os.getenv("BATTLESIGHT_MOTION_CROP_IMGSZ", "320"))

# --- Camera ego-motion compensation -----------------------------------------
# Background subtraction has no notion of a moving camera, and the straightness
# gate actively prefers a pan: dragged static structure scores ~1.0. So the
# global frame-to-frame transform is estimated and cancelled first.
EGO_COMPENSATION = os.getenv("BATTLESIGHT_EGO_COMP", "1") not in ("0", "false", "False")
# 0.3, not 1.0. MOG2 has no tolerance for sub-pixel drift, so a handheld camera
# at 0.5-1.0 px/frame was being called static and every edge flickered.
EGO_STATIC_SHIFT = float(os.getenv("BATTLESIGHT_EGO_STATIC_SHIFT", "0.3"))   # px at MOTION_WORKING_WIDTH
# Smooths the static/moving branch choice, which otherwise flickers on a slow pan.
EGO_SHIFT_EMA_ALPHA = float(os.getenv("BATTLESIGHT_EGO_SHIFT_EMA_ALPHA", "0.25"))
EGO_MIN_FEATURES = int(os.getenv("BATTLESIGHT_EGO_MIN_FEATURES", "12"))      # too few to trust a fit
EGO_DIFF_THRESHOLD = int(os.getenv("BATTLESIGHT_EGO_DIFF_THRESH", "28"))     # grey levels, compensated diff
# Above this the single global transform no longer describes the scene
# (parallax, rolling shutter), so the frame is degraded or dropped.
EGO_MAX_RESIDUAL = float(os.getenv("BATTLESIGHT_EGO_MAX_RESIDUAL", "0.02"))
EGO_RESIDUAL_EMA_ALPHA = float(os.getenv("BATTLESIGHT_EGO_RESIDUAL_EMA_ALPHA", "0.25"))
# The same sanity check for the static-camera MOG2 branch, which had none. On
# v3's tree-lined platform, foliage never settles into the background model and
# raw foreground sits chronically at 20-27%. 0.12 is above every quiet frame
# measured there and well below that baseline.
MOTION_MOG2_MAX_FRACTION = float(os.getenv("BATTLESIGHT_MOTION_MOG2_MAX_FRAC", "0.12"))

# --- Chronic-noise gates ----------------------------------------------------
# A scene fragmenting into this many surviving blobs is texture, not targets.
MOTION_CHRONIC_BLOB_COUNT = int(os.getenv("BATTLESIGHT_MOTION_CHRONIC_BLOBS", "20"))

# --- Reference-image exclusion ----------------------------------------------
EXCLUSION_STORE_PATH = BASE_DIR / "weights" / "exclusions.json"
EXCLUSION_SIMILARITY_THRESHOLD = float(os.getenv("BATTLESIGHT_EXCLUSION_SIM", "0.85"))

# --- HUD overlay filter (app/overlay_mask.py) -------------------------------
# A painted glyph holds image position while the world slides under it; a real
# object is attached to the world. So evidence is only gathered while the camera
# is independently established to be moving. Cut v11's phantom light_vehicle
# boxes from 54,658 to 6,713 with the control clip byte-identical. Log 16a/16b.
OVERLAY_FILTER = os.getenv("BATTLESIGHT_OVERLAY_FILTER", "1") not in ("0", "false", "False")
OVERLAY_GRID = int(os.getenv("BATTLESIGHT_OVERLAY_GRID", "64"))
OVERLAY_WARMUP_FRAMES = float(os.getenv("BATTLESIGHT_OVERLAY_WARMUP", "60"))
OVERLAY_PERSISTENCE = float(os.getenv("BATTLESIGHT_OVERLAY_PERSISTENCE", "0.3"))
OVERLAY_DILATE_CELLS = int(os.getenv("BATTLESIGHT_OVERLAY_DILATE", "1"))
OVERLAY_DECAY = float(os.getenv("BATTLESIGHT_OVERLAY_DECAY", "0.995"))
# Large boxes are excluded on purpose: a big persistent box is what a real
# target held centred in frame looks like.
OVERLAY_MAX_BOX_AREA = float(os.getenv("BATTLESIGHT_OVERLAY_MAX_BOX_AREA", "0.01"))
# Past this the filter disables itself rather than risk blinding the detector.
# A phantom contact is a nuisance; a suppressed real one is the failure this
# system must never have. Same reasoning as MOTION_GATED being off.
OVERLAY_MAX_FRACTION = float(os.getenv("BATTLESIGHT_OVERLAY_MAX_FRACTION", "0.10"))

# --- Large painted overlays (subtitle blocks, banners) ----------------------
# Same safety argument, different discriminator: a painted block is
# geometrically rigid (cv 0.0092 on v10's subtitle) where real boxes vary
# 2.5-15x more as the world slides under a moving camera.
# Ships disabled: one clip, 2.5x of margin, not enough to enable suppression by
# default. personnel is exempt on measurement, see ENGINEERING_LOG.md 22g.
OVERLAY_LARGE_FILTER = os.getenv("BATTLESIGHT_OVERLAY_LARGE", "0") not in ("0", "false", "False")
OVERLAY_LARGE_MIN_HITS = int(os.getenv("BATTLESIGHT_OVERLAY_LARGE_MIN_HITS", "10"))
OVERLAY_LARGE_RIGIDITY = float(os.getenv("BATTLESIGHT_OVERLAY_LARGE_RIGIDITY", "0.015"))
OVERLAY_LARGE_EXEMPT = frozenset({-1, 0})   # moving_object, personnel

# Frames to keep suppressing after MOTION_CHRONIC_BLOB_COUNT trips once. The
# bare threshold was not enough: a scene hovering just under the cutoff rebuilds
# coherent tracks between trips. Fixed v7 frame 261.
MOTION_CHRONIC_COOLDOWN = int(os.getenv("BATTLESIGHT_MOTION_CHRONIC_COOLDOWN", "3"))
