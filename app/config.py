"""Every path and threshold lives here. The values were measured, and the short
reason is next to each one. ENGINEERING_LOG.md has the full numbers, so read it
before changing a constant, because a few have been moved and moved back.

Everything can be overridden with an environment variable starting BATTLESIGHT_.
"""
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

# --- Model and precision ----------------------------------------------------
MODEL_PATH = os.getenv("BATTLESIGHT_MODEL", str(BASE_DIR / "weights" / "best.pt"))
# optional separate checkpoint for the drone view. The old one was retired (log 17),
# and without a file here the drone view uses MODEL_PATH
DRONE_MODEL_PATH = os.getenv("BATTLESIGHT_DRONE_MODEL", str(BASE_DIR / "weights" / "drone_best.pt"))
DEVICE = os.getenv("BATTLESIGHT_DEVICE", "0")       # "0" = first GPU, "cpu" = CPU
# FP16 costs 0.002 mAP50 or less and cuts latency by about 45%. INT8 lost 4 points (log 11)
QUANTIZE = 16 if DEVICE != "cpu" else None
# 0.25 and not 0.35 (mean F1 0.447 to 0.499). F1 peaks nearer 0.16, but the model
# puts confident boxes on things it has never seen (log 3)
CONF_THRESHOLD = float(os.getenv("BATTLESIGHT_CONF", "0.25"))
# personnel floors, picked on F2 because a missed person is worse than a false box.
# The SARD checkpoint scores real people at 0.117 to 0.256, and 0.10 is the F2 best (log 22)
CONF_THRESHOLD_PERSONNEL_GROUND = float(os.getenv("BATTLESIGHT_CONF_PERSONNEL_GROUND", "0.10"))
CONF_THRESHOLD_PERSONNEL_DRONE = float(os.getenv("BATTLESIGHT_CONF_PERSONNEL_DRONE", "0.10"))
# does nothing for the model: YOLO26 has no NMS step and ignores iou= (log 19).
# Only the IoU checks in this repo use it
IOU_THRESHOLD = float(os.getenv("BATTLESIGHT_IOU", "0.5"))
# ultralytics defaults to 300, but a VisDrone val frame has up to 317 (log 4)
MAX_DET = int(os.getenv("BATTLESIGHT_MAX_DET", "500"))

# --- Far-field second pass --------------------------------------------------
# a second pass over wherever the small boxes are, since recall falls with size.
# Off by default because latency jumps (p90 48.5 ms against 26.9 median). Fine offline (log 19)
FARFIELD_ENABLED = os.getenv("BATTLESIGHT_FARFIELD", "0") not in ("0", "false", "False")
FARFIELD_STRIDE = int(os.getenv("BATTLESIGHT_FARFIELD_STRIDE", "3"))     # every Nth tracked frame
FARFIELD_MAX_BOX = float(os.getenv("BATTLESIGHT_FARFIELD_MAX_BOX", "48"))  # px, discard larger from the tile
FARFIELD_SMALL_PCT = float(os.getenv("BATTLESIGHT_FARFIELD_SMALL_PCT", "40"))
FARFIELD_MIN_BOXES = int(os.getenv("BATTLESIGHT_FARFIELD_MIN_BOXES", "3"))
FARFIELD_PAD = float(os.getenv("BATTLESIGHT_FARFIELD_PAD", "0.10"))
# fallback (x1, y1, x2, y2) fractions, a horizon band, for when nothing else is known
FARFIELD_PRIOR = tuple(float(v) for v in os.getenv(
    "BATTLESIGHT_FARFIELD_PRIOR", "0.2,0.15,0.8,0.6").split(","))
FARFIELD_MAX_FRACTION = float(os.getenv("BATTLESIGHT_FARFIELD_MAX_FRACTION", "0.55"))
# 1280 and not 640, because at 640 three quarters of people are under 16 px.
# Going above 1280 is worse (logs 2 and 19)
IMGSZ = int(os.getenv("BATTLESIGHT_IMGSZ", "1280"))
# has to equal IMGSZ with a --static engine, its input shape is fixed
FARFIELD_IMGSZ = int(os.getenv("BATTLESIGHT_FARFIELD_IMGSZ", str(IMGSZ)))
# use a .engine next to MODEL_PATH if there is one, otherwise the .pt (log 5b)
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

# --- Hosting ----------------------------------------------------------------
# Turn this on for anything reachable from the internet. It drops /train and
# /exclude, because anyone could otherwise start a fine-tune on your machine or
# change what the detector hides.
PUBLIC_DEMO = os.getenv("BATTLESIGHT_PUBLIC_DEMO", "0") not in ("0", "false", "False")
# Comma separated. "*" is fine on a laptop, set real origins when you host it.
CORS_ORIGINS = [o.strip() for o in os.getenv("BATTLESIGHT_CORS_ORIGINS", "*").split(",") if o.strip()]
# Uploads bigger than this get a 413 instead of being decoded.
MAX_UPLOAD_MB = int(os.getenv("BATTLESIGHT_MAX_UPLOAD_MB", "10"))

# --- Motion filter, stages 1-2 (app/motion_filter.py) -----------------------
# tags anything that moves coherently, not just the four classes.
# 80 px^2 and not 30, because 5x6 smudges were most of the false output. It is also
# the smallest unknown mover that gets reported, so raising it loses far-off aircraft
MOTION_MIN_BLOB_AREA = int(os.getenv("BATTLESIGHT_MOTION_MIN_AREA", "80"))
MOTION_MATCH_MAX_DIST = int(os.getenv("BATTLESIGHT_MOTION_MAX_DIST", "60"))       # px, blob-to-track association
MOTION_TRACK_MAX_AGE = int(os.getenv("BATTLESIGHT_MOTION_TRACK_AGE", "5"))        # frames before a track is dropped
MOTION_COHERENCE_MIN_POINTS = int(os.getenv("BATTLESIGHT_MOTION_MIN_POINTS", "4"))  # history needed to judge
MOTION_COHERENCE_MIN_PATH = float(os.getenv("BATTLESIGHT_MOTION_MIN_PATH", "8"))    # px, below this is noise

# --- Illumination-vs-structure discriminator --------------------------------
# the trajectory checks can't tell a light from an object, and can't report anyone
# seen for fewer than 4 frames. This decides from one frame: align the previous
# frame, then test if the change is a uniform brightness shift. Fails open
MOTION_STRUCTURE_ENABLED = os.getenv("BATTLESIGHT_MOTION_STRUCTURE", "1") not in ("0", "false", "False")
# loose on purpose, these blobs already passed the trajectory checks
MOTION_STRUCTURE_MIN = float(os.getenv("BATTLESIGHT_MOTION_STRUCTURE_MIN", "0.25"))
# stricter, because the fast path has less trajectory evidence
MOTION_STRUCTURE_MIN_FAST = float(os.getenv("BATTLESIGHT_MOTION_STRUCTURE_MIN_FAST", "0.40"))
MOTION_STRUCTURE_MIN_STD = float(os.getenv("BATTLESIGHT_MOTION_STRUCTURE_MIN_STD", "4.0"))  # flatter = cannot judge
MOTION_STRUCTURE_UNIFORM_RATIO = float(os.getenv("BATTLESIGHT_MOTION_STRUCTURE_UNIFORM", "0.6"))
# two points are always in a line, so the fast path needs the structure test
# plus real movement instead
MOTION_COHERENCE_MIN_POINTS_FAST = int(os.getenv("BATTLESIGHT_MOTION_MIN_POINTS_FAST", "2"))
# past EGO_MAX_RESIDUAL it used to report nothing, which blinded the channel on
# 535 of 1068 frames of v6. Now it runs degraded up to this multiple
EGO_RESIDUAL_DEGRADED_FACTOR = float(os.getenv("BATTLESIGHT_EGO_RESIDUAL_DEGRADED_FACTOR", "2.0"))
MOTION_COHERENCE_DEGRADED_MIN = float(os.getenv("BATTLESIGHT_MOTION_COHERENCE_DEGRADED", "0.85"))
# per view, because they want opposite things. 0.85 clears foliage jitter but loses
# 47 to 63% of real contacts on ground clips, so ground stays at 0.5 (logs 14 and 15)
MOTION_COHERENCE_THRESHOLD_DRONE = float(os.getenv("BATTLESIGHT_MOTION_COHERENCE_DRONE", "0.85"))
MOTION_COHERENCE_THRESHOLD_GROUND = float(os.getenv("BATTLESIGHT_MOTION_COHERENCE_GROUND", "0.5"))
# a panning camera looks like the whole frame changing, so cap the work per frame
MOTION_MAX_BLOBS_PER_FRAME = int(os.getenv("BATTLESIGHT_MOTION_MAX_BLOBS", "40"))
MOTION_MAX_TRACKS_PER_SOURCE = int(os.getenv("BATTLESIGHT_MOTION_MAX_TRACKS", "150"))
# subtraction and contours run downscaled, since findContours is the cost at 1080p.
# Blob coordinates are scaled back up afterwards
MOTION_WORKING_WIDTH = int(os.getenv("BATTLESIGHT_MOTION_WIDTH", "480"))
# shape check: a swaying branch is a thin sliver, a lighting change is a flat band
MOTION_ASPECT_MIN = float(os.getenv("BATTLESIGHT_MOTION_ASPECT_MIN", "0.15"))   # w/h
MOTION_ASPECT_MAX = float(os.getenv("BATTLESIGHT_MOTION_ASPECT_MAX", "8.0"))
MOTION_CLAIM_IOU = float(os.getenv("BATTLESIGHT_MOTION_CLAIM_IOU", "0.1"))  # YOLO box claims a blob

# --- Motion gating: off, and it must stay off -------------------------------
# gating means the classifier only sees what moved, so a target that isn't moving is
# never detected. On a pan over a still scene it found 0.0 a frame at 28.3 ms, against
# 26.5 at 96.7 ms. The speed came from not looking (log 5)
MOTION_GATED = os.getenv("BATTLESIGHT_MOTION_GATED", "0") not in ("0", "false", "False")
# run the CPU motion stage beside the GPU pass and not before it, so a frame costs
# the longer of the two and not the sum: 46.4 to 24.6 ms median
MOTION_PARALLEL = os.getenv("BATTLESIGHT_MOTION_PARALLEL", "1") not in ("0", "false", "False")
# a blob is only the moving part of a target, so pad the crops
MOTION_CROP_PADDING = float(os.getenv("BATTLESIGHT_MOTION_CROP_PAD", "0.6"))   # fraction of box side
MOTION_CROP_MIN_SIZE = int(os.getenv("BATTLESIGHT_MOTION_CROP_MIN", "128"))    # px, full-frame scale
MOTION_CROP_MERGE_IOU = float(os.getenv("BATTLESIGHT_MOTION_CROP_MERGE_IOU", "0.2"))
MOTION_MAX_CROPS_PER_FRAME = int(os.getenv("BATTLESIGHT_MOTION_MAX_CROPS", "6"))  # past this, one full pass is cheaper
MOTION_CROP_IMGSZ = int(os.getenv("BATTLESIGHT_MOTION_CROP_IMGSZ", "320"))

# --- Camera ego-motion compensation -----------------------------------------
# background subtraction doesn't know the camera moves, and a pan makes static
# things look like straight paths. So the camera motion is estimated and cancelled first
EGO_COMPENSATION = os.getenv("BATTLESIGHT_EGO_COMP", "1") not in ("0", "false", "False")
# 0.3 and not 1.0, because a handheld camera at 0.5 to 1.0 px a frame was treated
# as still and every edge flickered
EGO_STATIC_SHIFT = float(os.getenv("BATTLESIGHT_EGO_STATIC_SHIFT", "0.3"))   # px at MOTION_WORKING_WIDTH
# smooths the still/moving choice, which flickers on a slow pan
EGO_SHIFT_EMA_ALPHA = float(os.getenv("BATTLESIGHT_EGO_SHIFT_EMA_ALPHA", "0.25"))
EGO_MIN_FEATURES = int(os.getenv("BATTLESIGHT_EGO_MIN_FEATURES", "12"))      # too few to trust a fit
EGO_DIFF_THRESHOLD = int(os.getenv("BATTLESIGHT_EGO_DIFF_THRESH", "28"))     # grey levels, compensated diff
# above this one transform no longer fits the scene (parallax, rolling shutter)
EGO_MAX_RESIDUAL = float(os.getenv("BATTLESIGHT_EGO_MAX_RESIDUAL", "0.02"))
EGO_RESIDUAL_EMA_ALPHA = float(os.getenv("BATTLESIGHT_EGO_RESIDUAL_EMA_ALPHA", "0.25"))
# the same check for the still-camera branch. On v3's tree-lined platform the
# foreground sat at 20 to 27%. 0.12 is above every quiet frame and well below that
MOTION_MOG2_MAX_FRACTION = float(os.getenv("BATTLESIGHT_MOTION_MOG2_MAX_FRAC", "0.12"))

# --- Chronic-noise gates ----------------------------------------------------
# this many surviving blobs means texture, not targets
MOTION_CHRONIC_BLOB_COUNT = int(os.getenv("BATTLESIGHT_MOTION_CHRONIC_BLOBS", "20"))

# --- Reference-image exclusion ----------------------------------------------
EXCLUSION_STORE_PATH = BASE_DIR / "weights" / "exclusions.json"
EXCLUSION_SIMILARITY_THRESHOLD = float(os.getenv("BATTLESIGHT_EXCLUSION_SIM", "0.85"))

# --- HUD overlay filter (app/overlay_mask.py) -------------------------------
# a painted glyph stays put while the world slides under it, and a real object moves
# with the world. So evidence only counts while the camera is moving. Cut v11's
# phantom light_vehicle boxes from 54,658 to 6,713 (log 16)
OVERLAY_FILTER = os.getenv("BATTLESIGHT_OVERLAY_FILTER", "1") not in ("0", "false", "False")
OVERLAY_GRID = int(os.getenv("BATTLESIGHT_OVERLAY_GRID", "64"))
OVERLAY_WARMUP_FRAMES = float(os.getenv("BATTLESIGHT_OVERLAY_WARMUP", "60"))
OVERLAY_PERSISTENCE = float(os.getenv("BATTLESIGHT_OVERLAY_PERSISTENCE", "0.3"))
OVERLAY_DILATE_CELLS = int(os.getenv("BATTLESIGHT_OVERLAY_DILATE", "1"))
OVERLAY_DECAY = float(os.getenv("BATTLESIGHT_OVERLAY_DECAY", "0.995"))
# big boxes are left alone on purpose, a big box that stays put looks like a real
# target held in the middle of the frame
OVERLAY_MAX_BOX_AREA = float(os.getenv("BATTLESIGHT_OVERLAY_MAX_BOX_AREA", "0.01"))
# past this the filter turns itself off. A false box is a nuisance, and a hidden
# real one is the failure this must never have
OVERLAY_MAX_FRACTION = float(os.getenv("BATTLESIGHT_OVERLAY_MAX_FRACTION", "0.10"))

# --- Large painted overlays (subtitle blocks, banners) ----------------------
# same idea for big banners: a painted block keeps an identical shape (cv 0.0092 on
# v10's subtitle), real boxes vary 2.5 to 15x more under a moving camera.
# Off by default, since it rests on one clip. Personnel is exempt (log 22g)
OVERLAY_LARGE_FILTER = os.getenv("BATTLESIGHT_OVERLAY_LARGE", "0") not in ("0", "false", "False")
OVERLAY_LARGE_MIN_HITS = int(os.getenv("BATTLESIGHT_OVERLAY_LARGE_MIN_HITS", "10"))
OVERLAY_LARGE_RIGIDITY = float(os.getenv("BATTLESIGHT_OVERLAY_LARGE_RIGIDITY", "0.015"))
OVERLAY_LARGE_EXEMPT = frozenset({-1, 0})   # moving_object, personnel

# frames to keep suppressing after the blob count trips, because a scene just
# under the cutoff rebuilds tracks in between (fixed v7 frame 261)
MOTION_CHRONIC_COOLDOWN = int(os.getenv("BATTLESIGHT_MOTION_CHRONIC_COOLDOWN", "3"))
