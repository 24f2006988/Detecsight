# DetecSight — Engineering Log

The fix-by-fix record behind [`README.md`](README.md): what broke, how it was
measured, what was changed, and — where it matters most — what was tried and
rejected. It is kept in chronological order rather than tidied into a narrative,
so the sections near the end supersede earlier ones on the same subject.

**Read it with that in mind.** Section 16 supersedes sections 14 and 15: those
two were tuned against a mistaken description of the `v10`/`v11` test clips, and
their diagnosis was wrong. Where a bullet is struck through, the strikethrough is
the correction and the original text is kept underneath deliberately — a log that
silently deletes its wrong turns is not evidence of anything.

Measured throughout on: Windows 11, RTX 4060 Laptop (8 GB), Python 3.12.10,
torch 2.13.0+cu126, ultralytics 8.4.135, fastapi 0.141.1. Every latency number
here is from that machine; see §19c before comparing any of them.

---

## What was tried and rejected

The shortest useful summary of this file. Each row is a change that looked
reasonable, was measured, and was **not** kept — with the number that killed it.
Full reasoning is in the linked section.

| rejected | why, measured | where |
|---|---|---|
| **INT8 quantisation** | 4.23 ms vs FP16's 6.95 ms, but −4 pt mAP50 and −5 pt recall. At ~10 fps the per-frame fixed overhead dominates, so 2.7 ms buys nothing visible while 5 points of recall is visible. | §11 |
| **Tuning `IOU_THRESHOLD` for crowds** | A silent no-op, and always had been. YOLO26's head reports `end2end: True` — it is NMS-free and the `iou=` argument is ignored. Sweeping 0.5/0.6/0.7/0.8 gave byte-identical results at every value. Crowd behaviour is *learned*; only training changes it. | §19 |
| **Raising imgsz above 1280** | 1280→1536 buys +0.4% detections for +46% cost, and the *large* buckets get worse — above the trained size the model is off-distribution for its own scale priors. | §19 |
| **`MOTION_GATED`** | Goes blind to stationary targets. It is a real latency win and a real safety regression; the standing decision is that it stays off. | §5 |
| **A drone-specialised checkpoint** | Retired after the blended `fpv` checkpoint beat it on the drone view's own home domain: VisDrone val personnel mAP50 0.3162 → 0.7064. Two weight files were kept in GPU memory to serve identical detections. | §17 |
| **Far-field second pass on by default** | Not rejected for cost but for jitter: p90 48.5 ms against a 26.9 ms median, which makes an AR overlay stutter. Kept, and off by default; good for offline annotation where latency is free. | §19 |
| **EMA smoothing for the v7 burst** | Made it worse. Smoothing made the decision *slower*; the gate needed the decision to *last* longer, which is what `MOTION_CHRONIC_COOLDOWN` does. | §13 |
| **`predict()` between `track()` calls** | Cost ~60% of detections, silently. Bisected by running the far-field pass with every box discarded: v5 personnel 1560 → 562 from the bare `predict()` call. Restoring `predictor.trackers` did not fix it. Fixed with a dedicated second model instance. | §19 |
| **Raising `OVERLAY_MAX_BOX_AREA` for the v10 subtitle** | It is the module's only safety property: the size limit is what stops a drone holding a real target centred in frame from having it masked. The phantom box is 137x137 px at frame centre, which is what such a target looks like. Fixed with a rigidity test instead. | §22f |
| **A confidence floor for the v10 phantoms** | Phantom confidence reaches 0.461; real `personnel` in the same clip reaches 0.535. The distributions overlap, and §20 already showed raising the vehicle floor is the wrong direction. | §22f |
| **Applying the rigidity filter to `personnel`** | At conf 0.10, the floor the ground view runs personnel at, it removed 13 real personnel and zero phantoms. Low-confidence boxes dilute a cell's geometry until real detections look rigid. `personnel` is exempt. | §22g |
| **Adding BDD100K/KITTI for the `v11` phantoms** | The diagnosis it rested on was wrong — the boxes were never on the scene, they were on the HUD. No training data would have fixed it. | §14 → §16 |

---

## Contents

| section | subject |
|---|---|
| [Setup](#setup) · [Dataset](#dataset) · [Training](#training) · [API surface](#api-surface-in-full) | orientation |
| [Deviations from the original setup notes](#deviations-from-the-original-setup-notes) | what the notes got wrong |
| §1–§4 | checkpoint, resolution, confidence, `max_det` |
| §5, §5b | motion gating; TensorRT + WebRTC latency |
| §6–§9 | ego compensation, `moving_object` explosions, chronic noise |
| §10 | drone-view specialised model *(retired — see §17)* |
| §11 | FP16 standard, INT8 rejected |
| §12, §13 | screen capture; branch-selection flicker |
| §14, §15 | **superseded by §16** — tuned against a wrong description of the clips |
| §16 | the clips were misidentified; the HUD was the dominant error source |
| §17 | the `fpv` run: completed, measured, promoted |
| §18 | lights vs. people; the split-second target |
| §19 | the far field and the 40 ms budget |
| §20 | dense ground-level crowds, and a confirmed vehicle-label contamination |
| §21 | SARD: pose confirmed; fine-tuning on it alone destroyed the model |
| §22 | SARD blended: 4.7x target recall, flat on the val, a rigidity filter, and why the last epoch is not the best one |
| [Still outstanding](#still-outstanding) · [Known limitations](#known-limitations) | open problems |

## Setup

Installation, serving and the API contract are in [`README.md`](README.md) —
this file does not restate them. Two environment notes that cost real time and
are not obvious from the README:

- `requirements.txt` carries an `--extra-index-url` for PyTorch's CUDA 12.6
  wheel index, so a single `pip install -r requirements.txt` resolves the pinned
  `torch==2.13.0+cu126` / `torchvision==0.28.0+cu126` GPU builds directly, with
  no separate pre-install step. If `scripts/check_gpu.py` reports
  `CUDA available: False`, something resolved a CPU wheel instead — reinstall
  with `pip install --force-reinstall torch==2.13.0+cu126
  torchvision==0.28.0+cu126 --extra-index-url https://download.pytorch.org/whl/cu126`.
- The datasets are not vendored. Point ultralytics at wherever they live —
  `yolo settings datasets_dir="<your datasets dir>"` — and all three yamls in
  `data/` resolve against it. None of them contains an absolute path: each
  anchors at VisDrone (a name that does not exist relative to cwd, which is what
  makes ultralytics consult `datasets_dir`) and reaches sibling datasets with
  `../`. `path: .` would not work — "." always exists, so it is kept as cwd and
  the setting is never consulted.

## Dataset

```powershell
python -c "from ultralytics.data.utils import check_det_dataset; check_det_dataset('VisDrone.yaml')"
python scripts\remap_visdrone.py
```

The remap backs originals up to `labels_visdrone_raw/` on first run and always
remaps *from* that backup, so editing `CLASS_MAP` and re-running is safe.

| new id | name | from VisDrone | train boxes |
|--------|------|---------------|-------------|
| 0 | `personnel` | pedestrian, people | 106,396 |
| 1 | `two_wheeler` | bicycle, tricycle, awning-tricycle, motor | 48,185 |
| 2 | `light_vehicle` | car, van | 169,823 |
| 3 | `heavy_vehicle` | truck, bus | 18,801 |

## Training

```powershell
python scripts\train.py --model yolo26s.pt --epochs 60 --batch 8 --imgsz 640 --name battlesight_v1
```

Measured on this machine: **159 s/epoch**, peak **5539 MiB / 8188 MiB** at
`batch=8 imgsz=640 amp=True`. A 60-epoch run is therefore ~2.7 hours, not
overnight. Batch 8 has real headroom; batch 12 would likely still fit.

Promote a finished run:

```powershell
yolo val model=runs\detect\battlesight_v1\weights\best.pt data=data\battlesight.yaml imgsz=640 device=0
copy runs\detect\battlesight_v1\weights\best.pt weights\best.pt
```

## API surface, in full

[`README.md`](README.md) lists the endpoints that matter to a client; this is the
complete set, including the state-management and training-control routes it
leaves out.

| endpoint | purpose |
|---|---|
| `GET /health` | model loaded, device, classes, training active |
| `POST /detect` | single frame, stateless, no track IDs |
| `POST /detect/tracked` | frame-in-stream over HTTP, track IDs + motion |
| `DELETE /detect/state/{source_id}` | drop a feed's tracker and motion history |
| `WS /ws/track/{source_id}` | live feed: send JPEG bytes, receive JSON per frame |
| `POST /webrtc/offer/{source_id}` | live feed over WebRTC: POST an SDP offer, get an SDP answer; detections arrive on a `"detections"` data channel, same JSON as the WebSocket path |
| `POST /train` | start a run (202), 409 if one is already running |
| `GET /train`, `GET /train/{id}` | job list / status with live epoch + log tail |
| `POST /train/{id}/cancel` | terminate a running job |

Boxes are returned as **normalised** `x1,y1,x2,y2` in 0–1 so the AR client can
scale to its own viewport without knowing the source resolution.

## Deviations from the original setup notes

These are changes made because the code as specified did not work on this stack.

1. **`project="runs/detect"` removed from `scripts/train.py`.** Ultralytics
   resolves a *relative* `project` **under** `runs/<task>/`, so this produced
   `runs/detect/runs/detect/<name>` and `trainer.py` could never find
   `best.pt`. Omitting `project` gives the intended `runs/detect/<name>`.

2. **`_is_moving()` returns `bool(...)`.** The comparison yields `numpy.bool_`,
   which `json.dumps` rejects. Pydantic coerced it on the HTTP path, so only the
   WebSocket broke — and only from the 3rd frame, once the history window filled.

3. **Per-feed trackers (`_bind_tracker` / `_adopt_tracker` in `detector.py`).**
   Ultralytics hangs a *single* tracker off the predictor and reuses it for every
   `track()` call. Keying `_history` by `source_id` was not enough: two feeds
   shared one ID space and contaminated each other's `track_id`s and motion
   state. Each `source_id` now gets its own tracker.

4. **`subprocess.Popen(..., encoding="utf-8", errors="replace")` in `trainer.py`.**
   Windows text pipes decode as cp1252; ultralytics progress bars emit UTF-8
   box-drawing bytes that cp1252 cannot map. The reader thread died on
   `UnicodeDecodeError`, nothing drained the pipe, and the child blocked forever
   on a full stdout buffer — every API-launched training hung at `running`
   permanently. The reader body is now also wrapped in `try/except` so a reader
   failure kills the child instead of wedging the job.

5. **`-u` / `PYTHONUNBUFFERED=1` on the training subprocess.** `bufsize=1` only
   affects the parent's read side; the child block-buffered stdout into the pipe,
   so `log_tail` and `current_epoch` never updated until the run ended.

6. **`ANSI_RE` strips escape codes before epoch scraping.** Ultralytics prefixes
   progress rows with `ESC[K`, so `EPOCH_RE`'s `^\s*` never matched and
   `current_epoch` stayed 0.

7. **`_class_name()` prefers the checkpoint's own label map.** Indexing
   `config.CLASS_NAMES` directly raises `IndexError` whenever the service is
   pointed at a stock 80-class COCO checkpoint.

8. **Blocking inference moved off the event loop** (`run_in_threadpool` in both
   routers). The handlers were `async def` but called straight into YOLO, so a
   single streaming feed blocked every other request — `/health` included — for
   the duration of each frame. Measured: `/health` now answers in ~1.2 ms while a
   feed runs at ~25 ms/frame.

9. **`Detector` is lock-guarded.** Once inference runs in worker threads, two
   feeds could interleave the predictor's tracker swap mid-inference and mix
   their ID spaces. `bind -> infer -> adopt` is now atomic.

10. **Motion history is a bounded LRU** (`MAX_TRACKS_PER_SOURCE`, default 512).
    Track ids only climb over a feed's life, so the old unbounded dict grew for
    as long as a drone stayed connected — a slow leak on exactly the long-running
    feeds this is built for.

11. **`trainer.start()` holds the lock across check-and-spawn.** Two POSTs
    arriving together both passed the bare `is_training()` check and put two
    trainings on one 8 GB GPU. Verified: 4 simultaneous requests now yield
    exactly one 202 and three 409s.

12. **Cancel kills the whole process tree** (`taskkill /F /T` on Windows).
    `proc.terminate()` signalled only the launcher, orphaning ultralytics'
    dataloader workers, which kept holding VRAM. Measured: 16 python processes
    mid-training drop to 2 after cancel, GPU 4212 MiB -> 1551 MiB.

13. **`best_weights` comes from the run's real `save_dir`,** scraped from the
    ultralytics args dump. With `exist_ok=False` a reused name increments to
    `<name>-2`, and the old code reported a stale path pointing at the *previous*
    run's weights. Verified: re-running `race_v1` reports `race_v1-2`.

14. **`is_training()` checks the process is actually alive,** so a job whose
    bookkeeping got stuck can no longer block the queue permanently.

15. **WebSocket hardening.** It never checked whether the model was loaded (first
    frame died on an `AttributeError` and the client just saw the socket drop),
    and only cleaned up feed state on a clean `WebSocketDisconnect` — a crashed
    feed leaked its tracker. Cleanup now runs in `finally`.

16. **Clear error when the model file is missing,** instead of failing deep
    inside `YOLO()`. `scripts/train.py --resume` likewise reports a missing
    checkpoint plainly.

## Detection accuracy work

Detection quality was subpar. Four causes, found by measurement
(`scripts/diagnose.py`, `scripts/bench_imgsz.py`; raw numbers in
`runs/diagnose.json`). Three were fixed without retraining.

### 1. The deployed checkpoint was a 1-epoch probe

`weights/best.pt` was the output of a `battlesight_probe` run with
`epochs=1`, while the real 13-epoch `battlesight_v1` sat unused in
`runs/detect/battlesight_v1/weights/`. Promoted; the probe is kept at
`weights/best_probe_1epoch.pt.bak`.

### 2. Inference resolution was far too low for the target size

The median VisDrone val object is **11 px** across at `imgsz=640`, and 75% of
`personnel` boxes are under 16 px — at the floor of what the stride-8 head
resolves. `IMGSZ` is now 1280.

| checkpoint | imgsz | mAP50 | mAP50-95 | recall | personnel mAP50 |
|---|---|---|---|---|---|
| probe (was deployed) | 640 | 0.2475 | 0.1009 | 0.2922 | 0.170 |
| probe | 1280 | 0.2638 | 0.1115 | 0.3136 | 0.200 |
| battlesight_v1 | 640 | 0.4331 | 0.2308 | 0.4182 | 0.382 |
| battlesight_v1 @ epoch 13 | 1280 | 0.5046 | 0.2772 | 0.4855 | 0.534 |
| **battlesight_v1 @ epoch 30 (deployed)** | **1280** | **0.5623** | **0.3159** | **0.5262** | **0.567** |

Together that is **mAP50 0.2475 -> 0.5046** and **personnel mAP50 0.170 ->
0.534**, with no retraining. On 1080p drone footage it is 33.5 vs 20.2
detections per frame at 20 ms/frame — still ~50 fps for one feed.

1600 was measured and is worse than 1280 (30.3 dets, 38.9 ms), so 1280 is the
top of the useful range.

### 3. Confidence threshold was past the useful point

`CONF_THRESHOLD` 0.35 -> 0.25. Mean F1 over the 4 classes on val at 1280:

| conf | mean F1 | P | R |
|---|---|---|---|
| 0.25 | 0.499 | 0.697 | 0.412 |
| 0.35 | 0.447 | 0.790 | 0.343 |
| 0.50 | 0.367 | 0.873 | 0.265 |

F1 actually peaks near 0.16, but this model puts confident boxes on
out-of-distribution scenes — on a close-up desk video it labelled a pencil case
`light_vehicle` at 0.42 — and phantom contacts on an AR overlay are worse than
a missed one. 0.25 recovers most of the recall without opening the floor.

### 4. `max_det` truncated dense frames

Ultralytics defaults `max_det=300`; VisDrone val frames hold up to 317 objects,
so the densest frames — the ones where the count matters — were silently
capped. `MAX_DET` is now 500.

### 5. Motion gating made the tracked path blind to stationary targets

`MOTION_GATED` defaulted to on, so the classifier only ever saw pixels that
moved. On `tests/assets/drone_pan.mp4` through `Detector.track`:

| | detections/frame | ms/frame |
|---|---|---|
| `MOTION_GATED=1` *(was default)* | **0.0** | 28.3 (35 fps) |
| `MOTION_GATED=0` *(now default)* | **26.5** | 96.7 (10 fps) |

Zero. The clip pans over a still scene, so nothing moves relative to the
ground and the gate discarded all 26.5 real targets per frame (267
`light_vehicle`, 195 `two_wheeler`, 62 `personnel`, 6 `heavy_vehicle` across 20
frames). For situational awareness a parked truck and a standing sentry are
exactly what belongs on the overlay, so the default is now off.

The cost is throughput: ~85 ms/frame at `IMGSZ=1280`, ~10 fps end-to-end over
the WebSocket. Lowering `IMGSZ` does **not** buy that back — measured in one
warm process with only `imgsz` varying, 640 costs a third of the detections to
gain about 1 fps, because the tracked path is dominated by fixed overhead (the
~27 ms motion pass, ByteTrack, exclusion, postprocessing) rather than the
forward pass:

| imgsz | ms/frame | fps | detections/frame |
|---|---|---|---|
| 640 | 81 | 12.4 | 17.7 |
| 960 | 82 | 12.2 | 25.5 |
| 1280 | 88 | 11.4 | 26.5 |

`BATTLESIGHT_MOTION_GATED=1` restores the old behaviour.

### Live feed latency (WebSocket, 1080p)

Measured end to end from a client over `WS /ws/track/{id}`, 240 frames in one
connection:

| | |
|---|---|
| steady-state round trip | **~100 ms (≈10 fps)** |
| of which server inference | 50–85 ms |
| of which client JPEG encode | 7–10 ms |
| first ~2 s of a feed | up to ~200 ms |

The slow start is the laptop GPU clocking up from idle (270 MHz / 3.6 W between
bursts), not a leak — latency is flat from frame ~80 through 240. Short
benchmark runs that finish before the clocks boost will report roughly double
the real steady-state cost; warm up before trusting a number.

**Those numbers are loopback (127.0.0.1) and exclude the network.** For a real
second device, add wire time. The traffic is wildly asymmetric — the frame going
up is everything, the boxes coming back are free:

| direction | payload | time @50 Mbps |
|---|---|---|
| uplink, 1080p JPEG q85 | 604 KB | ~97 ms |
| uplink, 720p JPEG q70 | 197 KB | ~32 ms |
| downlink, JSON boxes (27 dets) | 4.3 KB | ~1 ms |

Glass-to-overlay budget, device -> server -> device:

| upload | link | encode | uplink | inference | total | detections kept |
|---|---|---|---|---|---|---|
| 1080p q85 | 25 Mbps | 8 ms | 193 ms | 85 ms | **~290 ms** | 100% |
| 1080p q85 | 50 Mbps | 8 ms | 97 ms | 85 ms | **~190 ms** | 100% |
| 1080p q85 | 100 Mbps | 8 ms | 48 ms | 85 ms | **~145 ms** | 100% |
| 720p q70 | 50 Mbps | 5 ms | 32 ms | 85 ms | **~125 ms** | 74% |
| 540p q70 | 50 Mbps | 4 ms | 20 ms | 85 ms | **~110 ms** | 59% |

So downscaling the *upload* is the main latency lever, and it is not free:
720p q70 cuts the payload 3x and keeps 74% of detections, 540p keeps 59%.

**Viability.** At ~125–190 ms this is usable for situational awareness — marking
vehicles and people on a drone or vehicle feed, where targets move slowly in
frame. It is *not* fast enough for head-locked AR reticles: boxes will visibly
trail quick head motion. The client already receives `track_id` and a
moving/static flag, so the honest fix there is client-side extrapolation from
track velocity between replies rather than chasing server latency.

**One GPU serialises every feed.** Two devices roughly halve each other's frame
rate; this is a hard ceiling, not a tuning problem.

### 5b. Latency reduction: TensorRT engine + WebRTC transport

Two of the three latency levers from the numbers above, added without changing
`app/detector.py`'s cascade logic or dropping the WebSocket path:

**Engine.** `scripts/export_engine.py` exports a checkpoint to an FP16
TensorRT engine (`yolo export ... format=engine quantize=16`).
`Detector.load()` (`app/detector.py`) prefers `weights/*.engine` over the
matching `.pt` automatically whenever it exists and `BATTLESIGHT_USE_TENSORRT`
isn't `0`, and falls back to the `.pt` checkpoint if the engine fails to load
(wrong driver/TensorRT version) or `DEVICE=cpu`. Needs the `tensorrt` package
installed separately, matched to your CUDA/driver — not something to
auto-install, since a mismatched build fails silently different ways (see
"Building an engine on this machine" below for what that took here).

**Built and benchmarked on this machine as of 2026-09-01** — both
`weights/best.engine` and `weights/drone_best.engine` exist and are the
checkpoints actually deployed (see §11 for the numbers). One correction to the
reasoning above: the script's `--static` flag was originally framed as unsafe
because this codebase runs inference at two sizes (`IMGSZ` for the
full-frame/tracked path, `MOTION_CROP_IMGSZ` for the motion-gated crop path),
and a static engine only serves the size it was built for. **That concern
doesn't apply here** — `MOTION_GATED` is confirmed off and staying off (see
§5, and the explicit decision recorded in §11), so `MOTION_CROP_IMGSZ` is
never actually invoked, and every real call in this deployment is batch=1 at
`IMGSZ`. The dynamic-shape default was tried first and **failed**: ultralytics
sizes a dynamic engine's memory pool for its max batch (16 by default), which
requested 6.4 GB of workspace on top of everything else already resident and
hit `OutOfMemory` on this 8 GB laptop GPU. `--static` (fixed batch=1, single
shape) builds cleanly in ~4 minutes and is the right choice for how this
service actually calls the model. If `MOTION_GATED` is ever turned back on,
rebuild without `--static` first — the script's own runtime warning covers
this.

**Building an engine on this machine, if you need to repeat it:** the
`tensorrt` package needed the CUDA-12-specific wheels
(`tensorrt-cu12`/`-libs`/`-bindings`, not the bare `tensorrt` meta-package,
which resolved to a `cu13` build mismatched with this venv's `torch==...+cu126`).
The `tensorrt-cu12-libs` wheel is ~2.25 GB and this network's connection to
`pypi.nvidia.com` was unstable enough that it took several `pip install
--extra-index-url https://pypi.nvidia.com/ tensorrt-cu12 tensorrt-cu12-libs
tensorrt-cu12-bindings --retries 10` attempts, resuming from pip's cache each
time, before one completed. Separately, `onnxruntime-gpu` (used only for the
ONNX-export accuracy check below, not for serving) needs pinning below 1.29 —
that version and later require CUDA 13 (`cublasLt64_13.dll`), which silently
fails to load and falls back to CPU on a CUDA 12 machine, crashing on a
device-mismatch error rather than failing loudly; `onnxruntime-gpu==1.20.2`
matches CUDA 12. The first `export_engine.py` run also triggers ultralytics'
own `pip install nvidia-modelopt[onnx]` (~500 MB across its dependencies,
needed for the FP16-mixed-precision ONNX conversion step TensorRT export goes
through) — budget ~25-30 minutes for that on a slow connection; it is a
one-time cost; every export after it is fast.

**Transport.** `app/routers/webrtc.py` adds `POST /webrtc/offer/{source_id}`
alongside (not instead of) `/ws/track/{source_id}`. WebRTC carries frames over
UDP/SRTP instead of JSON-over-TCP: a dropped frame is skipped rather than
stalling the connection waiting on retransmission, and the client hands over
already-decoded video instead of paying a base64/JPEG round trip. Detections
come back over an RTCDataChannel in the exact schema as the WebSocket path —
`app/detector.py` and `app/schemas.py` are untouched, only how frames arrive
and results leave. Verified end-to-end in-process (synthetic video track in,
detection JSON out, clean teardown on disconnect) — see the note below on what
that test did and didn't cover.

`source_id` behaves identically to the WebSocket path: it's the same key into
Detector's per-feed history/tracker/motion state, so multiple devices stay
isolated from each other exactly as before. One GPU still serialises every
feed — that ceiling from the section above is unchanged by either of these.

**Deliberately not done:** client-side decoupled tracking (local
Kalman/optical-flow extrapolation between server replies, described as the
third lever in the original latency proposal). Server-side per-frame detection
stays authoritative on every frame; nothing here changes the client contract.

### 6. Ego compensation fails under parallax, flooding the overlay

Ego compensation cancels **one global 2D transform**, which is valid for a
distant near-planar scene (aerial — where it was tuned) but not for a
close-range handheld view, where foreground and background cross the sensor at
different rates. On `v1.mp4` the estimator never failed (0/76 frame pairs) and
correctly flagged all 76 as a moving camera — and static desk furniture still
registered as motion, giving 7.4 phantom blobs/frame and 348 track IDs in 77
frames.

The tell is the foreground fraction remaining *after* compensation:

| clip | residual |
|---|---|
| `static.mp4` (static camera) | 0.0% |
| `moving.mp4` (synthetic pan) | 0.6% |
| `drone_pan.mp4` (aerial pan) | 0.8% |
| `v1.mp4` (handheld close-up) | **3.5%** |

Above `EGO_MAX_RESIDUAL` (2%) the motion pass now reports nothing for that
frame and `Detector.track` classifies the whole frame instead, so false
contacts are suppressed without going blind. On `v1.mp4` that took the tracked
path from 597 detections (94% phantom `moving_object`) to 86.

### 7. `moving_object` explosions on ground-level handheld feeds

`v3.mp4` (handheld, railway platform) produced **1,954 `moving_object` boxes
over 368 frames — 5.3/frame against 3.0/frame of real people**, swarming the
overlay. Three independent causes, all fixed:

**a. The claim test used IoU instead of containment.** A walking person's
swinging leg or bag is a small blob sitting *entirely inside* a large
`personnel` box. Its IoU with that box is `blob_area / person_area`, far below
`MOTION_CLAIM_IOU`, so the blob escaped the claim and was re-emitted as a
separate phantom stacked on a person who was already correctly detected.
**425 boxes — 22% of all output.** What matters is whether a blob is already
accounted for, which is containment, not IoU. `_claim_motion_blobs` now scores
`max(intersection/blob_area, IoU)`.

**b. Handheld jitter was routed to MOG2.** `EGO_STATIC_SHIFT` was 1.0 px, so a
camera drifting 0.5–1.0 px/frame counted as "static" and went to background
subtraction, which has no tolerance for even sub-pixel movement — every
high-contrast edge in the scene flickered as foreground. Only **56 of 368
frames** on `v3.mp4` exceeded 1.0 px, so 85% of the clip took the MOG2 path.
Now 0.3 px, so that jitter goes through ego compensation instead.

**c. The blob floor was meaningless.** `MOTION_MIN_BLOB_AREA` was 30 px² — a
5×6 pixel smudge. Phantom blobs had a median area of 284 px² against 10,870 px²
for a real `personnel` box, a 38× separation with room to spare. Now 80 px².

Result on `v3.mp4`: **1,954 → 279 `moving_object` (5.3 → 0.8 per frame, −86%)**
with `personnel` unchanged at exactly 1,121 detections and track IDs preserved.
`drone_pan.mp4` is unchanged at 26.5 real detections/frame; `moving.mp4` and
`static.mp4` emit zero phantoms.

**If it still fires too much on a ground feed**, raise
`BATTLESIGHT_MOTION_MIN_AREA` — but note that is the floor on how small an
unrecognised moving thing can be and still be reported, which is exactly the
distant-UAV case `moving_object` exists for. At 1080p, 80 px² is a ~36×36 px
object in the full frame. Raise it for cleaner ground feeds, lower it for
smaller air targets.

### 8. `moving_object` bursts on a chronically-noisy background (the "vertex explosion")

Even after fix 7, `v3_annotated.mp4` still showed a burst — frames 108-113
jumped from a ~2-4/frame baseline to **16 `moving_object` boxes in one
frame**, scattered across the tree line and frame edges at 0.68-1.00
confidence: a swarm of small magenta boxes flickering in and out over half a
second, alongside the two correctly-boxed `personnel`.

Cause: frames 30-149 of `v3.mp4` hold the camera on a wind-blown tree line
while nearly static (estimated shake stays under `EGO_STATIC_SHIFT`), which
routes them to plain MOG2 background subtraction rather than ego-compensated
diffing. The foliage never settles into MOG2's background model, so raw
foreground sits at a **chronic 19-24% median** in that stretch (vs 2-9%
elsewhere in the same clip) and fragments into dozens of tiny blobs whose
count swings every frame. For a few consecutive frames enough fragments drift
the same direction to pass `MOTION_COHERENCE_THRESHOLD` together — a burst,
not a single bad frame.

The ego-compensated branch already had a sanity check for exactly this shape
of failure (`EGO_MAX_RESIDUAL`, §6) — the plain-MOG2 branch had none. Added
`MOTION_MOG2_MAX_FRACTION` (0.12) as its counterpart in
`MotionDetector.detect()`: when raw MOG2 foreground exceeds it, that frame's
blobs are dropped rather than fed to the coherence gate. 0.12 sits above every
quiet-frame fraction measured on this clip (peaks ~0.10) and below the
~0.19-0.24 foliage baseline, so it cuts the bad stretch without touching the
clean 80% of the clip.

Result on `v3.mp4`: **279 → 33 `moving_object` boxes (0.8 → 0.09/frame)**,
peak-frame count 16 → 3, with `personnel`/vehicle detections on the same
frames unchanged — `MOTION_GATED=0` means classification never depended on
the motion pass; this only removes the phantom class-agnostic contacts and
their `MOVING` tag during the noisy stretch.

### 9. Periodic `moving_object` explosions beyond v3 (two more chronic-noise causes)

The user reported the "explosion" recurring across the wider set of demo
clips even after fix 8. Re-running the same per-frame burst diagnostic
(`scripts/diagnose_bursts.py` -- rolling-median outlier detection, not
present before this pass) against all seven `vN.mp4` sources found it was
real, but from two mechanisms neither of the two existing chronic-noise
gates (`EGO_MAX_RESIDUAL`, `MOTION_MOG2_MAX_FRACTION`) caught, because both
were tuned against v3.mp4 specifically:

**a. Ego-residual threshold flicker on a fast, motion-blurred pan.**
`v6.mp4` (handheld, running/panning at night) bursts to 6-9 `moving_object`
boxes at frames 327-329. Instrumenting `MotionDetector.detect()` showed the
compensated foreground fraction there was **0.017-0.028 against the 0.02
`EGO_MAX_RESIDUAL` threshold** -- not clearly above or below it, hovering
across the line frame to frame because motion blur keeps the affine fit from
resolving cleanly. Reading that raw flipped the reliable/unreliable flag
every frame, and the frames that landed "reliable" by chance emitted a
burst off noise that never actually cleared. Fix: smooth the fraction with
an EMA (`EGO_RESIDUAL_EMA_ALPHA`, 0.25) before comparing to the threshold,
so a stretch that's consistently near the line reads as consistently over
it. Reset when the camera stops panning, so a genuine resumed pan judges
fresh rather than inheriting bias from before the stop.

**b. Chronic fine-texture fragmentation under the fraction threshold.**
`v7.mp4` (drone hovering over brick paving) burst to 8-22 `moving_object`
boxes for **30+ consecutive frames** (253-284). The raw foreground fraction
there was 0.02-0.11 -- comfortably under `MOTION_MOG2_MAX_FRACTION` (0.12) --
so the existing chronic-noise gate never fired. But the brick pattern
fragmented into **30-58 small blobs per frame** (vs. 0-16 on quiet real
frames on the same clip and on v6.mp4): a density problem the fraction
check can't see, because plenty of individually-small blobs can add up to a
modest total area. Fix: a new gate, `MOTION_CHRONIC_BLOB_COUNT` (20),
applied to both the ego and MOG2 branches right after the size/aspect
filter -- more surviving blobs than that in one frame is treated as texture
noise, same as the two fraction gates, and its blobs are dropped rather than
scored by the trajectory-coherence gate.

Both fixes are in `app/motion_filter.py` / `app/config.py`. Result, full
seven-clip re-run (`scripts/diagnose_bursts.py`), classified detections
(`personnel`, vehicles) unchanged throughout:

| clip | burst frames before | burst frames after |
|---|---|---|
| v6.mp4 | 5: 327-329 (6-9 boxes, `moving_object`), 517, 735 | 2: 517, 735 only |
| v7.mp4 | 8: 40, 253-259, 280 (peak 22 boxes, `moving_object`) | 1: 40, milder (10 vs. 22) |

v7's `moving_object` total fell 516 -> 182 (-65%) with the 30-frame chronic
swarm (253-284) gone entirely. v6's `moving_object` burst at 327-329 is also
fully gone (visually confirmed on the regenerated `v6_annotated.mp4`, frame
328: one real `light_vehicle` box, no magenta swarm); its total
`moving_object` count moved from 329 to 359 rather than dropping, because
removing one gate's false trigger elsewhere in the clip let a few more
genuinely-coherent distant blobs through -- not a regression, just not the
single clean number the v7 case gives.

v6's remaining two burst frames (517, 735) are **not** motion-filter
phantoms -- they're low-confidence (0.25-0.53), non-overlapping,
spread-across-the-frame `light_vehicle` boxes from the classifier itself, on
a heavily motion-blurred night pass over a dense row of parked motorbikes.
Visually inspected: consistent with the known weak `two_wheeler` class (mAP50
0.34-0.46, well behind `light_vehicle`'s 0.79-0.86) being confused for
`light_vehicle` under blur and low light, not a filter bug. Left as a known
edge case rather than chased further -- see "Still outstanding" below.

`v2.mp4`'s burst (frames 171-180, `personnel` climbing to 8-9 from a
baseline of 1-2) was checked too and is real content -- the camera pans
across a genuinely crowded street/bus-stop scene, visually confirmed frame by
frame. Left untouched.

### 10. Drone-view specialized model

The general-purpose model (`weights/best.pt`, trained on VisDrone + WiderPerson)
mixes in WiderPerson specifically to fix ground-level personnel false
positives (fix 7/8's `v3.mp4` numbers). That mix has nothing to teach a
straight-down drone view -- WiderPerson is ground-level-only imagery -- so a
model trained on VisDrone alone, without that dilution, was worth testing as
a second, view-specific checkpoint.

Frame samples confirmed `v5.mp4` and `v7.mp4` are genuine top-down aerial
footage (a drone hovering over a street/courtyard), distinct from the
handheld/elevated-balcony angles of the other five clips.

Fine-tuned `weights/best_e12_visdrone.pt` (the pre-WiderPerson 13-epoch
VisDrone-only checkpoint, previously an intermediate artifact only) for 25
more epochs on `data/battlesight.yaml` (VisDrone only, `flipud=0.5` --
correct here, unlike for the mixed set) as run `battlesight_drone`: 0.95 h,
name kept separate from `battlesight_v1`. Validated against the deployed
`weights/best.pt` with `scripts/eval_rubric.py --data data/battlesight.yaml`
(VisDrone-only val, imgsz 1280 -- the fair aerial-only comparison, not the
mixed val set the rubric defaults to):

| | `weights/best.pt` (general) | `weights/drone_best.pt` (specialized) |
|---|---|---|
| mAP50 | 0.5631 | **0.5822** (+0.019) |
| mAP50-95 | 0.3160 | **0.3280** |
| recall | 0.5260 | **0.5450** |
| precision | 0.6430 | **0.6580** |
| `two_wheeler` mAP50 | 0.405 | **0.459** |
| `heavy_vehicle` mAP50 | 0.414 | **0.443** |
| `light_vehicle` mAP50 | 0.789 | **0.860** |
| `personnel` mAP50 | 0.567 | 0.567 (flat) |

All four rubric criteria passed; promoted to `weights/drone_best.pt`.

Wired into serving as a `view` parameter, not a config swap, so both
checkpoints are loaded simultaneously and a caller picks per feed:
`Detector.load()` loads `DRONE_MODEL_PATH` alongside the default model if the
file exists (silently falling back to the default model for `?view=drone` if
it doesn't, so serving still works with only one weights file present).
`view` ("ground", default, or "drone") is a query param on `POST /detect`,
`POST /detect/tracked`, and `ws://.../ws/track/{source_id}?view=drone`, and a
`--view` flag on `scripts/annotate_video.py`. Per-feed tracker state is keyed
by `(view, source_id)` internally, since each loaded YOLO model owns its own
ByteTrack predictor -- switching a feed's view mid-stream is not a supported
use case (one feed is one camera angle), but the key prevents two feeds using
different views from ever sharing track IDs by accident. `GET /health` now
reports `views_loaded` so a caller can check whether `?view=drone` is really
a separate checkpoint before relying on it.

`v5_annotated.mp4` and `v7_annotated.mp4` regenerated with
`--view drone`; the rest with the default model and fix 9's changes.

**Known quirk, not chased further:** with two YOLO/torch models loaded,
`scripts/annotate_video.py` now reliably segfaults (exit 139) a moment
*after* printing its final "Wrote N frames" / "Average inference" summary --
i.e. after all real work is done, during Python/CUDA context teardown at
interpreter exit. Output files are complete and correct in every case
observed. Treat a nonzero exit from that script as informational, not a sign
the annotation failed -- check for the "Wrote N frames" line instead. Not
investigated further since it doesn't affect the FastAPI server (which
doesn't exit per-request) or the correctness of any output.

### 11. Inference precision: FP16 made standard, INT8 measured and rejected, FP8 doesn't exist here

Checked whether the deployed system was actually using FP16 anywhere: it
wasn't. Neither `app/detector.py`'s `model.predict()`/`.track()` calls nor
`scripts/eval_rubric.py`'s `model.val()` calls ever passed `half`/`quantize`,
so every prior number in this document (including §10's rubric table) is
FP32, confirmed by reading `ultralytics/engine/{validator,predictor}.py`
directly (`fp16=self.args.quantize == 16`, and `quantize` defaults to unset).

Measured FP16 and INT8 for real -- not from documentation, from
`model.val()` against `data/battlesight.yaml` (VisDrone-only val, the
aerial-relevant set), imgsz 1280, batch 1 (matching how this service actually
calls the model), warmed up. Warming matters more than it sounds: this GPU
idles at 270 MHz, so a cold first call runs about 2x a steady-state one and a
short benchmark that skips the warm-up reports roughly double the true latency.
Both the raw PyTorch checkpoint and a built TensorRT engine (§5b) were
measured per precision:

|  | PyTorch FP32 | PyTorch FP16 | TensorRT FP16 | TensorRT INT8 |
|---|---|---|---|---|
| **general** mAP50 | 0.5631 | 0.5616 | 0.5608 | 0.5189 |
| mAP50-95 | 0.3160 | 0.3154 | 0.3172 | 0.2808 |
| recall | 0.5260 | 0.5270 | 0.5303 | 0.4789 |
| inference | 13.3ms | 7.63ms | 6.95ms | 4.23ms |
| **drone** mAP50 | 0.5822 | 0.5823 | 0.5813 | 0.5414 |
| mAP50-95 | 0.3280 | 0.3278 | 0.3303 | 0.3024 |
| recall | 0.5450 | 0.5438 | 0.5468 | 0.5011 |
| inference | 13.5ms | 7.63ms | 7.29ms | 4.25ms |

**FP16 costs <=0.002 mAP50 on either model** -- noise, not a real
regression -- for a ~45% inference cut (13.3-13.5ms -> 6.95-7.63ms). Made
standard: `config.QUANTIZE = 16 if DEVICE != "cpu" else None`, threaded
through every `model.predict()`/`.track()` call in `app/detector.py`
(including the warmup pass in `_load_one`). `None` on CPU deliberately --
half-precision ops are unsupported or slower on most CPU kernels, and this is
also what keeps `tests/test_motion_gating.py` (CPU, stock checkpoint)
unaffected. Both TensorRT engines (`weights/best.engine`,
`weights/drone_best.engine`, §5b) were built at FP16 and are what
`Detector.load()` now actually loads, so production inference is
TensorRT-FP16, not PyTorch-FP16 -- the PyTorch-FP16 column above exists only
to isolate the precision effect from the engine effect.

**INT8 was measured, not assumed, and rejected.** A real 4-point mAP50 drop
(0.561->0.519 general, 0.581->0.541 drone) and 5-point recall drop
(0.530->0.479 / 0.547->0.501) for another ~1.6x on top of FP16's speed --
that is a real accuracy/speed tradeoff, not a free lunch, so it is not the
default. `weights/best_int8.engine` and `weights/drone_best_int8.engine`
exist (calibrated on the VisDrone val images via
`model.export(format="engine", quantize=8, data=...)`) if that tradeoff is
ever wanted for a throughput-critical deployment; nothing in `app/` loads
them automatically.

**FP8 does not exist in this stack.** Checked
`ultralytics/engine/exporter.py` directly rather than assume: `quantize`
only accepts 32 (FP32), 16 (FP16), 8 (INT8), and the weight-only
`w8a16`/`w8a32` schemes, on every export target this version supports,
PyTorch or TensorRT. There is no FP8 path to measure, hence no FP8 row
above -- INT8 is the actual lower-than-FP16 option this hardware/software
combination offers, measured honestly instead of substituted silently.

Regenerated all seven `vN_annotated.mp4` under the new FP16/TensorRT
default and re-ran the full `scripts/diagnose_bursts.py` sweep from fix 9 to
confirm the explosion fix is precision-independent (it operates on raw
background-subtraction pixels, never touches the classifier) -- v3 stayed
clean, v6's frame 327-329 burst and v7's chronic 253-284 swarm stayed fixed;
only sub-frame confidence-threshold noise shifted a few individual box
counts by 1-2, consistent with FP16 nudging some detections a hair either
side of `CONF_THRESHOLD`. Also re-ran `tests/test_history_cap.py`,
`test_motion_coherence.py`, and `test_motion_gating.py` (the last confirms
the CPU/`QUANTIZE=None` path specifically) -- all pass.

**Motion gating was checked, not changed.** Asked directly whether
`MOTION_GATED` should be flipped on for the throughput win, given it was
already off; kept off. The measured reason in §5 stands: the gated path
sees only what moved (0.0 detections/frame on a static-camera scene vs 26.5
ungated), so it would go blind to a parked vehicle or a stationary sentry --
not something to trade for speed silently.

### 12. Live screen capture (`scripts/annotate_screen.py`)

For testing against video that's already playing somewhere on screen (a
YouTube video in a browser, a video call, another app's own preview window)
without downloading a file first or wiring up a camera. Sibling of
`scripts/annotate_video.py` -- same `Detector`, same drawing code (imported
from `annotate_video`, not duplicated), same `--view`/`--weights`/`--output`/
`--show` conventions -- the only difference is where frames come from: `mss`
grabs a screen region each iteration instead of `cv2.VideoCapture` reading a
file/camera/stream URL.

```powershell
python scripts\annotate_screen.py                     # primary monitor, preview only
python scripts\annotate_screen.py --window chrome      # just that window, tracks it if it moves/resizes
python scripts\annotate_screen.py --region 100,100,1280,720
python scripts\annotate_screen.py --window chrome --output capture.mp4
python scripts\annotate_screen.py --view drone         # aerial-view checkpoint
python scripts\annotate_screen.py --max-frames 200     # stop automatically instead of 'q'/Ctrl+C
```

`--window TEXT` matches the first visible window whose title contains `TEXT`
(case-insensitive) via `pygetwindow`, and re-reads that window's live
position/size every frame (not a one-time snapshot), so it keeps following
the window if it's dragged or resized -- a browser showing a video is the
expected common case, hence matching by title rather than requiring a fixed
pixel region. `--region x,y,w,h` is available for a fixed area instead
(e.g. one monitor in a multi-monitor setup, or a sub-region of a window). If
neither is given, `--monitor N` (default 1, the primary monitor) captures
the whole thing.

Needs `mss` (screen capture) and `pygetwindow` (window lookup, `--window`
only) -- both lightweight, unlike TensorRT's install in §5b/§11, and now in
`requirements.txt`.

Screen capture has no native frame rate, so -- like the webcam/stream path in
`annotate_video.py` -- this reads-infers-draws as fast as inference allows
and does not try to pace itself to the source video's playback speed;
`--fps` only labels the *output file's* header, it doesn't control capture
speed. A window that's minimised, fully occluded, or resized mid-capture is
handled by skipping that frame (empty grab, or a shape that no longer
matches the output writer's) rather than crashing.

Verified end-to-end on this machine: `--region` against a fixed area and
`--window` against a real browser window (title `'... - Brave'`) both
captured actual on-screen content (confirmed by inspecting a decoded output
frame, not just a successful exit) and ran detection/tracking/drawing on it
exactly as `annotate_video.py` does on a file.

### 13. `moving_object` branch-selection flicker (a 4th chronic-noise cause)

Re-audited the explosion fixes from 7-9 against the live TensorRT/FP16
production path (weights/*.engine, not the .pt files those sections were
measured against) with a standalone frame-by-frame script, since
`scripts/diagnose_bursts.py` -- referenced throughout sections 8, 9 and
11 as the tool that verified them -- is not actually present in `scripts/`
(missing from this checkout; not root-caused, but every number attributed to
it in this document should be treated as unverified against the current tree
until it's recreated or found). Re-running v3/v6/v7/v9 directly through
`Detector.track()` confirmed fixes 7-9 are real and still holding on the
engine path (v3.mp4 moving_object max 16 -> 3, `mog2_fraction` gate still
firing as designed) -- but turned up a burst neither of them covers.

`v6.mp4` frame 962: 7 `moving_object` boxes, baseline 0, not one of the
frames (327-329, 517, 735) documented in fix 9. Cause: `MotionDetector.detect()`
picks the ego-compensated vs. plain-MOG2 branch from the RAW per-frame
estimated shift (`moving_camera = shift >= EGO_STATIC_SHIFT`), with no
smoothing on that decision -- unlike the *residual fraction* fix 9a already
smooths, which only helps once a frame is already on the ego branch. A slow,
genuine pan whose shift hovers near `EGO_STATIC_SHIFT` (0.3) flickers the
BRANCH CHOICE itself frame to frame (RANSAC noise in the affine fit, not
motion blur this time): measured shift 0.20, 0.45, 0.32, 0.05, 0.26, 0.15,
0.21 across frames 956-962 while the camera was genuinely panning the whole
time (surrounding frames measured 0.49-1.66). Every frame that misclassified
as "static" ran plain MOG2 on real pan-smeared structure instead of
ego-compensated diffing -- and that structure is coherent by construction
(it's real motion, not jitter), so it sailed through the trajectory-coherence
gate untouched, plus past both existing chronic-noise gates (fraction stayed
~0.06, blob count 16-18, both comfortably under their thresholds -- those
gates catch backgrounds that never settle, not real motion routed to the
wrong branch). `moving_object` climbed 0 -> 2 -> 5 -> 7 over frames 960-962.

Fix: `EGO_SHIFT_EMA_ALPHA` (0.25, same value and shape as
`EGO_RESIDUAL_EMA_ALPHA`) in `app/config.py` -- the estimated shift is now
smoothed before the `EGO_STATIC_SHIFT` comparison, in `app/motion_filter.py`.
Verified on v6.mp4 frames 955-968 directly: `moving_camera` now stays `True`
throughout instead of flickering to `False` at 956/959-962, the frame-962
burst is gone (0 `moving_object` at 960-962, vs. 2/5/7 before), and the full
clip re-run confirms it (max `moving_object` 7 -> 5, no burst frames, `kept`
frames unchanged at 533). v3.mp4/v9.mp4 re-run unaffected (noise-level only).

**Tried and reverted:** the same EMA shape applied to
`MOTION_CHRONIC_BLOB_COUNT`, since `v7.mp4` frame 261 showed the identical
boundary-flicker pattern (surviving blob count 17-20, right under the gate,
for several consecutive frames). Smoothing made it WORSE (burst grew 7 -> 13
boxes): that gate's job is to react fast to a spike, and averaging delays the
trip rather than preventing a false one. Reverted; see the comment on
`MOTION_CHRONIC_BLOB_COUNT` in `app/config.py`. v7.mp4 frame 261 (7
`moving_object` boxes, baseline 0) is therefore still open -- added below.

### 14. `v10`/`v11`: off-domain content, two different diagnoses, one config change

> **SUPERSEDED by section 16.** The premise of this section is wrong: the
> clips are not off-domain wildlife footage. The diagnosis and the fix below
> were both derived from that mistaken description. Kept because the reasoning
> is what section 16 had to undo, not because it is correct.

The user supplied two more clips outside the tactical/aerial/handheld domain
this project targets -- `v10.mp4` (a wildlife/nature clip with burned-in
captions, wind-blown grass) and `v11.mp4` (a forward-facing highway dashcam,
open desert road) -- both apparently captured via `scripts/annotate_screen.py`
against generic video (README §12). Diagnosed separately because they turned
out to be two unrelated problems:

**`v11.mp4`: pure classifier hallucination, not a filter bug.** Total
detections averaged 41/frame, peaking at 60. Isolated the cause by running
`Detector.detect()` (stateless, classifier only -- no tracker, no motion
filter at all) on a burst frame: 45 of the frame's 55 boxes were already
present with the motion pass completely out of the loop -- dozens of
overlapping, near-duplicate `light_vehicle` boxes (conf 0.25-0.83) stacked on
empty sky and empty road shoulder. This is the same "confident false
positives off-distribution" failure already logged below (the
pencil-case/highlighter case), just far more severe here: forward-facing
highway dashcam footage has nothing in common with VisDrone (aerial-down
traffic) or WiderPerson (ground-level pedestrian crowds). No threshold, IoU,
or motion-filter change fixes this -- these are independent detections the
network is genuinely confident about, not one real box NMS failed to merge.
The only real fix is training-data coverage: see the updated outstanding item
below.

**`v10.mp4`: a genuinely new motion-filter gap, distinct from fixes 8/9/13.**
All boxes on the burst frames were `moving_object` (zero classifier
detections) -- wind-blown grass/foliage. Unlike the tree-line/brick-texture
chronic swarms already fixed, `fg_fraction` (~0.04) and blob count (~9-10)
both sat far under the existing chronic-noise gates (0.12 / 20) -- not a
"many blobs" problem. Printed the blobs' own `coherence` score (what
`MOTION_COHERENCE_THRESHOLD` is checked against): 0.55-0.97. A brief,
consistent gust makes wind-blown texture look exactly as straight-line as a
real trajectory over the `MOTION_COHERENCE_MIN_POINTS=4`-frame judging
window -- no existing gate targets "few blobs, briefly coherent," only "many
blobs" or "large area."

Measured a full threshold sweep (not assumed) across the problem clip and
both already-fixed regression clips, all other settings default:

| `MOTION_COHERENCE_THRESHOLD` | v10 `moving_object` | v10 burst frames | v3 `moving_object` | v6 `moving_object` |
|---|---|---|---|---|
| 0.50 (previous default) | 465 | 7 | 34 | 348 |
| 0.65 | 363 | 5 | 29 | 243 |
| 0.75 | 266 | 3 | 24 | 205 |
| **0.85 (new default)** | **205** | **0** | 18 | 128 |
| 0.92 | 127 | 0 | 7 | 64 |

0.85 is the first value that fully clears v10's bursts. **Real cost, not
free**: it also cuts v3's and v6's already-verified-real `moving_object`
counts by ~47% and ~63% respectively -- straightness alone can't tell "briefly
coherent noise" from "a real target only tracked a few frames," so tightening
it costs genuine short-lived detections too. Neither of this project's
synthetic regression clips (`tests/assets/moving.mp4`, `drone_pan.mp4`)
exercises this gate at ANY threshold tested (0 `moving_object` throughout at
every value) -- so this change was **not** verified against a real
slow/distant-target case. If a genuine UAV or slow mover ever needs to be
caught from only a handful of coherent frames, re-measure against real
footage of that specifically before relying on 0.85, and consider a
longer-window fix (require more than 4 points before trusting straightness,
since real sustained motion outlasts a wind gust) instead of a flat
threshold if recall on that case turns out to matter.

Adopted at the user's explicit request, prioritizing cutting false-positive
clutter over raw recall for this project's stated purpose (fast-moving-
personnel situational awareness). `v10_annotated.mp4` regenerated;
`v3_annotated.mp4`/`v6_annotated.mp4` not regenerated for this change (spot-
verified via the sweep above instead, to keep the session moving).

### 15. Per-view motion coherence, and a personnel recall floor for ground view

> **SUPERSEDED by section 16** for everything concerning `v10`/`v11`. The
> per-view coherence thresholds themselves survive and are still deployed; the
> reasoning about what they do for `v10` personnel recall does not -- section
> 16e measures that effect at zero.

Fix 14's 0.85 `MOTION_COHERENCE_THRESHOLD` was global, so it also tightened
ground view -- the handheld/bodycam-style view whose whole purpose is
catching a person visible for only a short span, exactly the kind of
short-lived detection that threshold suppresses. User confirmed the intent:
drone view should keep the stricter value (aerial/terrain footage, where
wind-blown texture is the noise source), ground view should stay permissive
and additionally get more personnel recall specifically.

**Per-view split.** `MOTION_COHERENCE_THRESHOLD` became
`MOTION_COHERENCE_THRESHOLD_DRONE` (0.85) and `MOTION_COHERENCE_THRESHOLD_GROUND`
(reverted to the original 0.5, i.e. what fixes 7-9/13 were tuned against).
`MotionDetector.detect()` now takes a `view` argument and looks up the right
one (`_coherence_threshold`); `Detector.track()` passes its own `view`
through. Every other gate in the motion pass (ego compensation, the two
chronic-noise fraction/count gates, fix 13's shift EMA) stays view-independent
-- only the final straightness comparison differs. Re-verified: `v10.mp4`
(ground) shows its grass bursts again (the deliberate, accepted tradeoff --
see fix 14, this content was never the target of the "short span" ask
anyway); `v10.mp4 --view drone` stays clean; `v6.mp4` unaffected.
`v3_annotated.mp4`/`v6_annotated.mp4` regenerated; `v10_annotated.mp4` left
as-is from fix 14 (drone-suppressed) since `annotate_video.py` defaults to
ground view and regenerating it would just show the reverted, expected
grass noise on content that was never the point.

**Personnel confidence floor, ground view only.** Measured (not guessed) a
class-specific P/R/F1 sweep for `personnel` alone on the ground-view val set,
at several confidence floors:

| conf | personnel P | R | F1 |
|---|---|---|---|
| 0.10 | 0.777 | 0.568 | 0.656 |
| 0.15 | 0.777 | 0.568 | 0.656 |
| 0.20 | 0.762 | **0.579** | 0.658 |
| 0.25 (`CONF_THRESHOLD`, previous floor) | 0.803 | 0.549 | 0.652 |

Flatter than expected -- dropping from 0.25 to 0.10 buys only ~2 points of
recall for a real precision cost, so there isn't much room in this lever on
this val set. Adopted 0.20 (`CONF_THRESHOLD_PERSONNEL_GROUND` in
`app/config.py`) as the best P/R balance measured, applied ONLY to
`class_name == "personnel"` on `view == "ground"` -- deliberately not a
blanket drop, to avoid reopening the vehicle-hallucination problem from fix
14 (`v11.mp4`), which was specifically about `light_vehicle` at low
confidence on off-distribution content; personnel is a different class with
better training coverage (WiderPerson) and a different failure mode.

Mechanically: `Detector._model_conf_floor(view)` passes the lowest threshold
any class/view needs straight to `model.predict()`/`.track()` (so ultralytics
never strips a box before it can be evaluated), then
`Detector._to_detections()` applies the real per-class/view floor
(`_conf_threshold_for`) afterward -- threaded through `detect()`,
`_full_frame_track()`, `_gated_detections()`'s fallback, and
`_detect_on_crops()` for consistency, though the last two are unreachable
with `MOTION_GATED` at its default (off).

**Important caveat, not yet resolved:** the val set this was measured against
(VisDrone + WiderPerson) is mostly clean, static, well-lit imagery, not
motion-blurred video frames -- it does not represent genuine fast-handheld/
bodycam footage well, so this measurement likely UNDERSTATES how much
confidence drops on a real blurry, fast-moving target. If short-span
personnel recall still isn't good enough after this in practice, the more
impactful fix is training-data coverage (motion-blurred / fast-handheld
personnel imagery) -- the same conclusion as fix 14's `v11.mp4` item and the
pre-existing off-distribution item below -- not a further threshold drop,
which this sweep shows has limited room left to give.

### 16. The clips were misidentified, and the dominant error source was the HUD

This section corrects sections 14 and 15 on a point of fact, so read it before
relying on either. `v10.mp4` and `v11.mp4` are **not** "a wildlife/nature clip"
and "a forward-facing highway dashcam". Both are UAV/FPV combat footage:

- `v10.mp4` — a drone observing **personnel moving through vegetation**, in a
  false-colour EO/IR-style palette (the grass renders magenta), with a HUD
  (crosshair, `425`, unit emblem, telemetry) and burned-in subtitles.
- `v11.mp4` — an **FPV drone** flying low over open terrain, 564x480 analog
  video, fisheye, heavy compression, with a HUD (two dotted reticle columns,
  a `UEX10 002587` telemetry string, a range readout, an emblem).

Everything in sections 14/15 was tuned against a description of these clips
that does not match their content. The thresholds those sections set are not
all wrong, but their stated justifications are, and section 14's recommendation
(add BDD100K/KITTI to teach "empty road" from a forward-facing vehicle camera)
would not have addressed the actual failure at all.

#### 16a. What v11's 54,658 phantom boxes actually were

Section 14 attributed them to the classifier having no notion of this camera
angle. Re-measured with the reconstructed `scripts/diagnose_bursts.py`
(16c below), the deployed system emits **54,658 `light_vehicle` boxes across
1,746 frames — 31.3 per frame** on a clip containing no vehicles.

Rolling-median burst detection reports *zero* bursts on that series, which is
itself the clue: this is not an explosion on some frames, it is a constant
error on nearly every frame.

Accumulating those boxes into a spatial heatmap reproduces the HUD exactly —
one box per dash of each dotted reticle column, one per character of the
telemetry string, one per character of the range readout. Median false box:
**9x9 px**, i.e. glyph-sized. Total footprint: 1.6% of the frame.

The cause is not the camera angle and not the terrain. Small, high-contrast,
rectangular synthetic glyphs are what a VisDrone-trained model has learned a
distant vehicle looks like from above. No amount of driving-scene training data
would have changed that, because the boxes are not on the scene.

#### 16b. Fix: static-overlay rejection (`app/overlay_mask.py`)

The distinguishing property of a HUD element is not appearance but
**attachment**: a glyph is painted onto the sensor output and holds its
image-space position while the world slides underneath it. A real object,
moving or parked, is attached to the world and traverses the frame as the
camera pans.

So the filter accumulates, per source, which grid cells keep producing small
detections **while the camera is established to be moving** — reusing the
motion pass's existing ego-motion estimate rather than computing a second one.
Three conditions, each covering a different failure of the others:

1. Detection persistence, on camera-moving frames only. A still camera makes
   every parked car look persistent, so those frames contribute no evidence.
2. Small boxes only (`OVERLAY_MAX_BOX_AREA`). This is what stops a drone
   deliberately holding a real target centred in frame from ever having that
   target masked, however persistent it looks.
3. A hard cap (`OVERLAY_MAX_FRACTION`): past that, the premise has broken down
   and the filter disables itself. Failing **open** is the only acceptable
   direction here — same reasoning as `MOTION_GATED` being off.

Parameters swept offline against cached detections, so every row is the same
inference with only the filter varying. "v11 removed" is the share of that
clip's phantom `light_vehicle` boxes suppressed; the rest are real detections
lost on regression clips:

| persistence | dilate | v11 removed | v10 | v2 | v9 | v6 | cells masked |
|---|---|---|---|---|---|---|---|
| 0.50 | 0 | 66.0% | 0 | 0 | 0 | 0 | 28 |
| 0.50 | 1 | 75.5% | 0 | 0 | 0 | 0 | 128 |
| 0.35 | 0 | 73.0% | 0 | 0 | 0 | 0 | 36 |
| 0.35 | 1 | 82.7% | 0 | 0 | 0 | 0 | 157 |
| 0.25 | 0 | 77.6% | 0 | 0 | 0 | 0 | 46 |
| 0.25 | 1 | 86.0% | 0 | 0 | 0 | 0 | 189 |

Dilation matters because a glyph's box jitters, so its centre lands in one of
two or three adjacent cells and the evidence gets split; no single cell reaches
threshold alone. Adopted `OVERLAY_PERSISTENCE=0.3`, `OVERLAY_DILATE_CELLS=1`.

Verified live through `Detector.track()` on the TensorRT path:

| | before | after |
|---|---|---|
| `v11` `light_vehicle` total | 54,658 | **6,713** (−87.7%) |
| `v11` `light_vehicle` per frame | 31.3 | **3.84** |
| `v11` `heavy_vehicle` | 303 | 90 |
| `v11` `two_wheeler` | 627 | 512 |
| `v10` (every class) | — | **unchanged, exactly** |

`v10`'s counts are byte-identical before and after, which is the result to
want: the filter engages only where there is evidence for it.

**TRIED AND REJECTED** as a second condition: requiring the cell's pixels to be
temporally static. It carries no information on this footage — a glyph sits on
a *changing* background and an analog FPV feed is noisy everywhere. Of the 40
cells with persistence >= 0.3 on `v11`, requiring cell std <= 0.35x the frame's
median kept only 11; loosening it enough to keep them (1.0x) admitted 2048
cells, half the grid. Box size does that safety job on evidence that actually
separates the two cases.

#### 16c. `scripts/diagnose_bursts.py` reconstructed

Listed as missing under "Still outstanding". Rebuilt: per-frame class counts
across every `vN.mp4`, with bursts found as frames far above their own local
rolling median (a global threshold flags genuinely busy stretches as bugs).
`--motion-only` exercises `MotionDetector` alone, which is CPU and therefore
usable while the GPU is training.

It independently reproduced the open `v7.mp4` frame-261 burst at exactly the
documented magnitude (7 boxes against a baseline of 0), which is the evidence
that this reconstruction measures the same thing the original did.

Two caveats on its numbers. `--motion-only` counts are strictly higher than the
served path, because `_claim_motion_blobs` removes blobs already covered by a
classifier box. And burst *counts* can rise when a fix lowers totals — the
rolling-median baseline drops with them, so smaller spikes clear the relative
threshold. Compare totals and max/frame alongside the burst count.

#### 16d. `v7.mp4` frame 261 fixed: a cooldown, not a smoother

The open item called for "a different mechanism, e.g. a short cooldown", after
EMA-smoothing `MOTION_CHRONIC_BLOB_COUNT` was tried and made the burst worse.
Implemented as `MOTION_CHRONIC_COOLDOWN`: once the gate trips, keep dropping
blobs for a few frames, so a scene sitting just under the cutoff (v7's brick
paving runs 17-20 blobs against a cutoff of 20) can't use the sub-threshold
gaps as windows to rebuild coherent tracks.

The distinction that matters: smoothing made the decision *slower*, and this
gate has to fire fast. What it needed was for the decision to *last* longer.

| cooldown | `moving_object` | max/frame | bursts | extra frames dropped |
|---|---|---|---|---|
| 0 | 242 | 7 | 2 | 0 |
| 2 | 228 | 5 | 0 | 7 |
| **3** | **224** | **5** | **0** | **13** |
| 5 | 210 | 5 | 0 | 22 |
| 8 | 204 | 5 | 0 | 32 |

Adopted 3. Full-sweep regression (ground view, motion-only), cooldown 0 vs 3:
`v2`/`v3`/`v4`/`v5`/`v6`/`v9` **identical**, `v7` bursts 2 -> 0, `v1` improved
(10 -> 4 boxes, max 5 -> 1). No clip lost anything.

#### 16e. The real personnel problem, measured

On `v10.mp4` the deployed system puts `personnel` boxes on **vegetation** while
**missing every actual person in frame** — at frame 300, two people are plainly
visible and it returns zero detections. This is far worse than the clean-val
`personnel` mAP50 of 0.567 suggests, because VisDrone and WiderPerson do not
contain this case at all.

Three things were ruled out by measurement rather than assumed:

- **Not resolution.** The people in these frames are 50-60 px across, not
  tiny.
- **Not our fine-tuning.** A stock COCO-pretrained `yolo26s.pt` at conf 0.15
  finds 1 person across the same four frames. The generic model fails too.
- **Not the motion channel picking up the slack.** At *either* drone coherence
  threshold (0.85 or the pre-15 value of 0.5) the motion filter puts boxes only
  on the burned-in subtitles, never on the people — they crawl too slowly to
  clear `MOTION_COHERENCE_MIN_PATH` over a 4-frame window. Section 15's choice
  between those thresholds does not affect this clip's personnel recall either
  way, contrary to the reasoning recorded there.

What is left is pose, viewpoint and palette: prone/crawling people seen from an
oblique overhead angle in a false-colour feed. That is a training-data gap, and
it is the one thing in this document that no threshold can close.

#### 16f. Training prepared (not run)

- **`albumentations` was not installed in this venv at all.** Ultralytics skips
  its entire Albumentations stage silently when the import fails (info-level
  log only), so *every checkpoint in this project was trained with zero blur,
  noise, compression or colour augmentation* — not merely with the library
  defaults, which is what the absence of any note about it would suggest.
- `scripts/fpv_augment.py` adds an `fpv` profile modelling the real input
  chain in capture order — optics (motion blur/defocus), sensor resolution
  (downscale), sensor noise, codec (compression), exposure, then palette.
  Ultralytics 8.4 takes a custom transform list through the first-class
  `augmentations=` argument, so no library patching is involved. Deliberately
  moderate: degradation strong enough to look dramatic erases an 11 px target
  and teaches the model to fit noise against a label with nothing under it.
  The palette transforms (wide hue rotation, occasional channel shuffle) exist
  specifically because of 16e — ultralytics' own `hsv_h` default is a ±1.5%
  rotation, nowhere near enough to span a false-colour feed.
- **`data/battlesight_fpv.yaml`** adds VisDrone's **test-dev split to train**:
  all 1,610 images carry ground truth and were used by neither train nor val, a
  free ~25% increase in VisDrone training data costing no held-out set.
- **AerialPerson** (Zenodo 7740081, CC-BY-4.0) — UAV frames over campus and
  Civil Defense exercises, 3,136 images, already YOLO format, single class
  `people` which is already id 0 == `personnel`. Median box ~11 px at imgsz
  1280. It is the only source here with a person seen small from altitude
  against natural terrain. `scripts/convert_aerialperson.py` converts it.
- `scripts/prepare_training.py` checks what is on disk, converts the archive if
  it has finished downloading, writes a yaml naming **only paths that exist**
  (ultralytics fails hard otherwise), and prints the launch command. It never
  starts training itself.
- The deployed checkpoints were trained at **imgsz 640 but are served at 1280**
  — a train/serve mismatch on exactly the tiny targets this system cares about.
  The prepared command trains at 1280.


#### 16g. AerialPerson labels people only — vehicles had to be pseudo-labelled

Caught before training, not after. AerialPerson is single-class: it annotates
people and nothing else. Its imagery is top-down aerial over a university
campus, i.e. large parking lots — and none of those cars carry a label. That is
the **same viewpoint VisDrone teaches vehicles from**, so mixing the two raw
presents every one of those cars to the trainer as a confirmed negative for
`light_vehicle`.

Measured with `weights/drone_best.pt` over 40 random AerialPerson train images
at conf 0.35:

| class | per image |
|---|---|
| `personnel` | 11.6 |
| `light_vehicle` | **97.3 — none labelled** |
| `two_wheeler` | 0.8 |
| `heavy_vehicle` | 0.5 |

Across 2,613 train images that extrapolates to roughly **258,000 unlabelled
vehicles**, which is more negative vehicle evidence than VisDrone supplies
positive. This would not have diluted the vehicle classes, it would have
destroyed them — and the failure would have surfaced only after a 7-hour run,
as an unexplained `light_vehicle` collapse in the rubric.

`scripts/pseudo_label_vehicles.py` labels the vehicles with the existing
drone-view checkpoint and merges them into the label files. Guardrails, since
pseudo-labelling is easy to get wrong:

- Ground-truth person boxes are never modified or removed.
- A pseudo-box overlapping a ground-truth person (IoU > 0.3) is dropped — the
  human label wins; we do not relabel a person as a vehicle.
- Vehicle classes only. `personnel` is what this dataset already annotates
  properly, and the model's personnel predictions are the thing being fixed.
- Confidence floor 0.5, double serving's 0.25: a false positive here becomes a
  permanent wrong label, so precision beats recall.
- Originals backed up to `labels/<split>.orig/`, `--restore` undoes everything.

`scripts/prepare_training.py` warns loudly if this has not been run.

**This applies to WiderPerson too**, which has been in the training mix since
`battlesight_multi.yaml`: ground-level street scenes, personnel-only labels,
unlabelled traffic. It is weaker there because ground-level cars look different
from VisDrone's aerial ones, so the contradiction is less direct — but it is a
pre-existing, previously undocumented source of vehicle-class damage and a
prime suspect if vehicle metrics ever look inexplicably poor.


### 17. The `fpv` run: completed, measured, promoted (2026-09-02)

`battlesight_fpv` — the run prepared in section 16f — was executed, evaluated
and promoted. It is the first checkpoint in this project trained with any
blur/noise/compression/colour augmentation at all (albumentations was simply
not installed before; ultralytics skips that stage silently when the import
fails, so every earlier checkpoint had none).

**The run.** 8 epochs at imgsz 1280, batch 4, on `battlesight_fpv.yaml`
(18,694 train / 1,548 val). Interrupted after epoch 3 and resumed with
`python scripts/train.py --resume --name battlesight_fpv`. The resume restored
the optimizer state, the epoch counter and all 10 `fpv` transforms without any
flags being re-passed, exactly as section 16f predicted — that path is now
proven against a real interruption rather than a deliberate test kill.

Per-epoch val mAP50: 0.573, 0.580, 0.592, 0.607, 0.626, 0.626, 0.634, 0.638.
Monotonic, still climbing at epoch 8 — the run was budget-limited, not
converged.

**The rubric** (`eval_rubric.py`, candidate vs. the then-deployed
`weights/best.pt`, same val, imgsz 1280):

| | baseline | candidate | delta |
|---|---|---|---|
| overall mAP50 | 0.5910 | **0.6274** | +0.0364 |
| personnel mAP50 | 0.6785 | **0.7064** | +0.0279 |
| recall | 0.540 | **0.570** | +0.030 |
| precision | 0.669 | **0.716** | +0.047 |

Per-class mAP50: personnel 0.706, two_wheeler 0.494, light_vehicle 0.860,
heavy_vehicle 0.449. All four criteria PASS. `no_collapsed_class` passing is
the one that matters most: it is the direct check on the AerialPerson
pseudo-labelling trap in section 16f, and `light_vehicle` at 0.860 shows the
263,700 pseudo-labels did not poison the class they were added to protect.

**FP16 engine parity — a check the rubric does not perform.** `eval_rubric.py`
measures the `.pt`, but `Detector.load()` serves `weights/best.engine`. The
rebuilt engine was validated separately:

| | `.pt` | `.engine` (served) |
|---|---|---|
| mAP50 | 0.6274 | 0.6252 |
| mAP50-95 | 0.384 | 0.384 |
| recall | 0.570 | 0.571 |
| inference | 28.8 ms | **14.8 ms** |

Every class within 0.002 mAP50-95 — FP16 rounding, not degradation, in clear
contrast to the INT8 result in section 11. The engine is also roughly twice as
fast as the checkpoint, which is the other reason the export step is not
optional: skipping it costs both correctness *and* half the throughput.

**Promoted**, with rollback kept as `weights/best_pre_fpv.pt` **and**
`weights/best_pre_fpv.engine` — keeping the old engine as well as the old
checkpoint makes a rollback a file copy instead of a 3-minute rebuild.
`weights/best_int8.engine` is now stale (old checkpoint); it is not loaded by
default, but rebuild or delete it before revisiting INT8.

**Two documentation bugs found by executing the notes rather than reading
them.** The promotion command recorded in `start.txt` and `context.md`,
`export_engine.py weightsest.pt --static`, fails twice over: `--model` is a
flag rather than a positional (so argparse rejects the path), and the script
needs `PYTHONPATH=.` like the annotate scripts (`ModuleNotFoundError: No module
named 'app'`). Both files now carry the working form.

**Drone view promoted as well, on a separate measurement.** `drone_best.pt` is a
different lineage (VisDrone-only, trained at 640), so it got its own rubric run
rather than being replaced on the strength of the ground result:

| | `drone_best.pt` | candidate |
|---|---|---|
| overall mAP50 | 0.5036 | **0.6274** |
| personnel mAP50 | 0.3162 | **0.7064** |

PASS, and by a far wider margin than the ground comparison — personnel more than
doubled on VisDrone val, drone view's own home domain. Part of that gap is the
640-vs-1280 train/serve mismatch this run existed to close, but 1280 is the
serving resolution, so it is the deployment-realistic comparison.

Consequence worth recording: `weights/best.pt` and `weights/drone_best.pt` are
now byte-identical, with separately built engines, and the drone specialisation
of section 10 is retired. The views still differ, but only through config
(section 15's per-view coherence thresholds and personnel confidence floor), not
through weights. Both engines are loaded at once, so the same model now occupies
GPU memory twice (~380 MB of avoidable overhead); deleting `weights/drone_best.*`
would fall back to the default model with identical detections, at the cost of
being able to let the views diverge again. Left in place deliberately.

**The v10 question is now answered: NO, and the answer is more interesting than
a failure.** Measured with stateless `Detector.detect(view="drone")` over all 374
frames, old drone checkpoint vs new:

| v10.mp4 | old `drone_best` | new |
|---|---|---|
| `personnel` | 245 (in 132 frames, 35.3%) | **0** |
| `light_vehicle` | 385 | **58** (−85%) |
| `heavy_vehicle` | 20 | **0** |

Those 245 old `personnel` boxes were **all false positives on vegetation**.
Rendered and inspected directly: frame 186 has a person plainly visible, prone in
the open, and the old model put 8 small `personnel` boxes (conf 0.26–0.53)
scattered across the magenta foliage with **not one on the person**. The new
model returns zero detections on that frame — it stopped hallucinating people in
the grass, but it still does not find the person.

So the run **removed ~592 false positives on this clip and gained no true
positives**. That is a real improvement (a phantom contact is exactly what §9
says must not be normalised), but it is not what the run was for.

This is precisely what §7 predicted before the run: *"nothing in AerialPerson
shows a prone or crawling person, which is the specific v10 failure. Expect
partial help."* The prediction was right, and the remaining gap is now narrower
and better characterised — it is **pose**, not palette, viewpoint or resolution.
The new model is trained on false-colour EO/IR-like augmentation and on aerial
personnel, and it still misses a prone body from above. Closing this needs
training data containing prone/crawling people seen from a UAV; no dataset
currently in the mix has it.

### 17b. Personnel confidence floor 0.20 → 0.10 on ground view, chosen on F2 (2026-09-02)

The note justifying the old 0.20 floor claimed that dropping it "buys only ~2
points of recall". **That was an artefact of the validation set, not a property
of the model.** The sweep behind it ran on the blended val, where VisDrone's
aerial personnel dominate by instance count and their P/R curve genuinely is
flat — which masked the ground-level headroom underneath it.

Re-measured with a class-specific sweep, class 0 only, split by domain, on the
`fpv` checkpoint. WiderPerson val, 1,000 ground-level images — the bodycam-like
case:

| conf | P | R | F1 | F2 | |
|---|---|---|---|---|---|
| 0.08 | 0.600 | 0.750 | 0.667 | **0.714** | F2 optimum |
| **0.10** | 0.657 | 0.725 | 0.689 | 0.710 | **adopted** — the knee |
| 0.12 | 0.701 | 0.703 | 0.702 | 0.703 | |
| 0.15 | 0.753 | 0.674 | 0.712 | 0.689 | |
| 0.20 | 0.814 | 0.636 | **0.714** | 0.665 | previous floor, F1 optimum |
| 0.25 | 0.853 | 0.606 | 0.709 | 0.644 | |

**Chosen on F2, not F1.** F1 weights precision and recall equally, which is the
wrong objective for a system whose standing principle is that a phantom contact
is a nuisance and a suppressed real one is not acceptable. F2 peaks at 0.08, but
0.10 scores within 0.004 of that peak while recovering 5.7 points of precision —
so 0.10 is the knee, and 0.08 is not worth the precision.

Effect: personnel recall on ground-level imagery **0.636 → 0.725**, about 18%
more people found, at precision 0.814 → 0.657. A deliberate trade — roughly one
box in three now wrong, against roughly one person in four previously missed.

**Ground view only.** On VisDrone val (aerial) precision at 0.10 collapses to
0.431, so drone view keeps 0.25. Ground-level personnel mAP50 is 0.757, better
than the blended 0.706 and better than aerial's 0.605 — the model is stronger at
ground-level people than the headline number suggests.

Secondary effect on tracking: `Detector._model_conf_floor()` passes the lowest
floor any class or view needs down to the model, so lowering this also hands
ByteTrack more low-confidence candidates for its second association pass. That
is the mechanism that keeps a briefly-visible person on a track rather than
dropping them.

**The honest limit**, unchanged: WiderPerson is static, clean, well-lit street
photography, not motion-blurred bodycam or rendered game footage. This sweep
cannot say how the model behaves there. If short-span personnel recall is still
not good enough, the fix is training coverage of dense, occluded, ground-level
people — not a further threshold drop. (See §20: it is also the dataset behind
the ground-level vehicle collapse.)

### 18. Lights vs. people, and the split-second target (2026-09-02)

Two failures the trajectory gates cannot separate, because both are judged by
WHERE a blob went over several frames:

* a **light** -- muzzle flash, headlight, sodium glare, a lamp switching on --
  whose bright region drifts its centroid and reads as coherent travel;
* a **person visible for half a second**, who never accumulates
  `MOTION_COHERENCE_MIN_POINTS` = 4 frames of history and so cannot be
  reported at all, at any threshold setting.

The second point is the important one: this was not a tuning problem. The
evidence the gates required took longer to accumulate than the event lasted.

**The discriminator** (`MotionDetector._structure_score`) decides from
appearance in one frame instead. It aligns the previous frame to the current
one (the ego transform the motion pass already computed, warped once per frame
rather than per blob) and tests two independent illumination signatures:
a uniform brightness shift moves every pixel by about the same amount, so the
difference image's mean dominates its spatial spread; and a brightness or
contrast change preserves structure, so the two patches still correlate after
z-scoring. Real content arriving in the region does neither. Synthetic
measurement: uniform brightening 0.000, contrast-only 0.000, new content 0.547.

It **fails open** -- 1.0 on no previous frame, a box under 3 px, a patch flatter
than `MOTION_STRUCTURE_MIN_STD`, or a shape mismatch. The gate can only reject,
and a suppressed real contact is the failure that matters.

**The fast path** spends that structure evidence to buy back the brief-
appearance case: a blob with only `MOTION_COHERENCE_MIN_POINTS_FAST` = 2 points
is reported if it clears a stricter structure bar (0.40) **and** has genuinely
travelled `MOTION_COHERENCE_MIN_PATH`. It deliberately does **not** consult
`_straightness` -- any two points are collinear, so straightness at 2 points is
1.0 by construction, and trusting it is exactly how the burst problem would
reopen.

**The ego-residual degraded band** is the change with the widest reach. Any
residual above `EGO_MAX_RESIDUAL` previously returned nothing, which measurement
showed was blinding the motion channel on **535 of 1068 frames of v6** -- half
the clip -- and on 224/225 of v2 and v9. Now only residuals past
`EGO_RESIDUAL_DEGRADED_FACTOR` x the threshold drop the frame; the band between
runs degraded (structure test required, no fast path, coherence raised to 0.85)
rather than blind.

Measured across the clip sweep, `--view ground`, served path:

| clip | `moving_object` | bursts | frames blind |
|---|---|---|---|
| v1 | 4 → 3 | 0 → 0 | 60 → 47 |
| v3 | 27 → 44 | 0 → 0 | 196 → 167 |
| v4 | 63 → 6 | 0 → 0 | 25 → 3 |
| v5 | 16 → **0** | 0 → 0 | 0 → 0 |
| v6 | 344 → 326 | 0 → 1 | **535 → 70** |
| v7 | 99 → 29 | 0 → 0 | 36 → 29 |
| v10 | **449 → 34** | **6 → 1** | 97 → 69 |
| v11 | **2707 → 2105** | **65 → 43** | 508 → 445 |

**v5's 16 → 0 was checked visually rather than assumed**, because a clip losing
its entire motion channel is exactly what a bad gate looks like. It is a
sodium-lit night courtyard; the rejected blobs sit on empty pavement and shadow
edges beside the walking pair, never on a person (frames 44 and 131 rendered and
inspected). They were lighting artefacts, and the real people in that clip are
carried by the classifier -- personnel 1,560. v6's verified-real count is
preserved (344 → 326) while its blind frames collapse, which is the outcome that
matters most: more coverage, same real detections, far less noise.

v3 rising 27 → 44 is recovered coverage, not new noise -- it comes with 29 fewer
blind frames and no bursts.

**The fast path fires on real footage**, and these are detections the old system
could not produce at all:

| clip | motion detections | via fast path |
|---|---|---|
| v3 | 348 | 33 (9.5%) |
| v6 | 331 | 48 (14.5%) |
| v7 | 68 | 10 (14.7%) |
| v10 | 35 | 10 (28.6%) |
| v11 | 2199 | **679 (30.9%)** |

(Motion-only counts, so higher than the served path -- `_claim_motion_blobs`
removes YOLO-covered blobs. The ratio is the point.)

**Honest limitation.** v2 and v9 barely moved (224 → 221, 224 → 223). Their
residual sits far above 2x `EGO_MAX_RESIDUAL`, so the degraded band never
reaches them and they still go blind. Raising the factor would reach them at the
cost of admitting genuinely parallax-broken frames; that trade has not been
measured. The static-camera `mog2_fraction` drop (122 frames on v3) has the same
all-or-nothing shape and could take the same treatment -- deliberately left
alone to keep this change's blast radius to the moving-camera path.

`tests/test_motion_structure.py` covers all of it: 12 assertions over the
light-vs-object separation, all four fail-open paths, and a 2-frame appearance
actually being reported. `BATTLESIGHT_MOTION_STRUCTURE=0` disables the
discriminator entirely.

### 19. The far field, and the 40 ms budget (2026-09-02)

Started as "improve accuracy at distance", ended as a profiling result. The
useful lesson is the order: **the accuracy work was being measured against a
latency budget that was already blown, and nobody had profiled the pipeline.**

**Recall is not uniform across target size -- it collapses with distance.**
300 WiderPerson val images, personnel, imgsz 1280:

| size (px) | GT | recall |
|---|---|---|
| <16 | 1,427 | **0.187** |
| 16-32 | 2,214 | **0.622** |
| 32-48 | 1,426 | 0.805 |
| 48-64 | 1,128 | 0.855 |
| 64-96 | 1,481 | 0.937 |
| >96 | 1,207 | 0.940 |

41% of all people are under 32 px and fewer than half of them are found. Near
targets are effectively solved.

**Two obvious levers are dead, and both are worth recording so nobody spends a
day rediscovering them.**

*Resolution does not help.* 1280 -> 1536 gains +14 detections of 3,364 (+0.4%)
for +46% cost; 1920 costs 7x for nothing. The <16 bucket improves slightly while
16-32 and everything larger DEGRADE. The model was trained at 1280, so running
above it is off-distribution for its learned scale priors. More pixels do not
help a model that has not learned what a 12 px person looks like.

*NMS tuning is impossible -- there is no NMS.* Sweeping `iou` over
0.5/0.6/0.7/0.8 produced byte-identical results (TP=3364, preds=5210, R=0.698,
P=0.646 at every value). YOLO26's head reports `end2end: True`: it is an
NMS-free one-to-one detector and the `iou=` argument is silently ignored. See
the note on IOU_THRESHOLD in config.py. Duplicate suppression is LEARNED, so the
only way to change crowd/occlusion behaviour is training.

**The far-field pass.** Tiling the whole frame recovers the loss (<16 ->
0.306) but costs 5-6x and craters precision 0.662 -> 0.478 on duplicate and edge
boxes. Instead the cheap full-frame pass CHOOSES where to spend one extra pass:
the smallest boxes ARE the distant ones, so their bounding region is the far
field (horizon-band prior when the frame offers nothing).

| | found | precision | ms/img |
|---|---|---|---|
| full frame only | 6,279 | 0.662 | ~40 |
| 2x2 everything | 6,736 | 0.478 | 182 |
| far-field tile | 6,611 | 0.601 | 115 |
| **+ size filter** | **6,555** | **0.644** | 111 |

73% of the full-tiling gain for 40% of its extra cost. The size filter is what
preserves precision: the tile exists to find SMALL targets, so any large box it
returns is a duplicate the full frame already had.

**It is OFF by default** -- not for cost (see below) but for jitter: strided, its
p90 is 48.5 ms against a 26.9 ms median, so an AR overlay gets a periodic
stutter. Running the tile at a smaller `imgsz` would fix that, except the
`--static` TensorRT engine has a FIXED input shape and anything but 1280 dies
with `input size (1,3,640,640) not equal to max model size`. Getting the cheap
tile needs a second engine built at the smaller size. Not done.

#### 19a. A latent trap: predict() between track() calls

Enabling the far-field pass initially made things WORSE -- v5.mp4 personnel fell
1560 -> 775. Since the pass only ever APPENDS boxes, a fall in the total proved
the damage was upstream, in the full-frame pass.

Two plausible diagnoses were both wrong on measurement: saving and restoring
`predictor.trackers` around the call changed nothing, and disabling
`overlay_mask` (which learns small clustered boxes as HUD, and far-field boxes
are small and clustered by construction) changed nothing either. What isolated
it was a bisect -- run the pass but DISCARD every tile box, leaving only the
`predict()` call: v5 fell to 562, worse still.

**Interleaving `predict()` with `track()` on the same YOLO object corrupts that
object's predictor state, and the damage lands on the tracked pass.** It costs
~60% of detections, silently. `_gated_detections` and `_detect_on_crops` both
call `predict()` and would both hit this the moment `MOTION_GATED` is switched
on. The fix here is a dedicated second model instance for the far-field pass
(lazily loaded, ~256 MB, returns None on failure so an enhancement can never
take serving down).

#### 19b. The actual win: overlap the CPU stage with the GPU pass

Per-stage profile of `track()` on v5.mp4 (warm, 120 frames):

| stage | median | p90 |
|---|---|---|
| motion filter (CPU) | 23.6 ms | 27.6 |
| total track() | 52.2 ms | 61.7 |

**~45% of the frame budget was the CPU motion stage running in sequence with the
GPU pass while the GPU sat idle.** With `MOTION_GATED` off -- the standing
default -- nothing on the model path reads `blobs` until `_claim_motion_blobs`
at the very end, so the two stages are independent. Both sides release the GIL
(OpenCV; TensorRT), so overlapping them costs max() instead of sum.

| | median | p90 | v2 | v5 |
|---|---|---|---|---|
| sequential | 46.4 ms | 50.4 | 67 ms/f | 60 ms/f |
| **parallel** | **24.6 ms** | **28.2** | **34 ms/f** | **22 ms/f** |

Roughly 2x, and *more stable* -- max() does not accumulate the variance of two
noisy stages the way summing does.

**This is a pure latency change and was verified as one.** Detection counts are
byte-identical (v2 914/914, v5 1560/1560) and the full 10-clip burst sweep is
byte-identical in every column. `BATTLESIGHT_MOTION_PARALLEL=0` reverts it.

The gated branches read `blobs` immediately and take the sequential path
unchanged. A defensive collect before stage 4 makes sure a future is never left
dangling -- the motion filter is stateful and order-dependent, so an uncollected
one would desynchronise the next frame.

#### 19c. Health warning on every latency figure here

Absolute timings on this machine varied ~3x on UNCHANGED code across one
session (the same far-field-off path measured 25, 46, 48.9, 52.2, 67 and 74
ms). An env-var ablation of today's changes produced a 3 ms spread in which
turning the structure test OFF measured SLOWER than leaving it on -- physically
impossible, and proof the noise floor exceeded the effect being measured.

Trust WITHIN-RUN comparisons (parallel on vs off, back to back, warmed) and
treat absolutes as indicative. README's "Measured on this machine" convention
assumes a warm GPU; these numbers warm for 40 frames first, which is the only
reason the parallel result is trustworthy at all.

### 20. Dense ground-level crowds, and a confirmed vehicle-label contamination (2026-09-05)

A pedestrian crossing at rush hour, 1280×720, 3,212 frames — hundreds of people
per frame at every scale from 10 px to full height, with vehicles crossing
between phases. This is the densest ground-level footage the system has been run
on, and it was chosen to test the case `CrowdHuman` was registered for.

**Tracked path, whole clip:**

| | |
|---|---|
| `personnel` | 63,133 total, 19.66/frame |
| per-frame spread | min 1, median 19, p75 27, p90 34, **max 78** |
| frames with zero personnel | 0 of 3,212 |
| unique `personnel` track ids | 2,281 |
| `light_vehicle` / `heavy_vehicle` / `moving_object` | 402 / 48 / 380 |
| served-path latency | median 46.1 ms, p90 58.5 ms |
| inference alone (`annotate_video.py`) | 18.7 ms/frame |

**a. The system tracks the near field well and is effectively blind past
mid-field.** Detections cluster in the lower third of the frame; the standing
crowd behind it, several hundred people at 10–25 px, produces almost nothing.
The size histogram of what it *reports* is the mirror image of what is *there*:

| detected box size | share of detections |
|---|---|
| <16 px | 5 boxes, 0.0% |
| 16–32 px | 12.7% |
| 32–48 px | 8.1% |
| 48–64 px | 7.4% |
| 64–96 px | 19.4% |
| >96 px | **52.4%** |

Median detected box is 98.6 px. In a scene dominated by small people, over half
of what comes back is large. This is §19's size-recall curve reproduced on real
footage rather than on a validation split, and it is consistent with it —
nothing new is wrong, but the shape of the limit is now visible outside the lab.

**b. Latency scales with detection count, which the sparse clips hid.** 46.1 ms
median here against the 24.6 ms measured on `v5` in §19c. The extra cost is
per-detection work — tracking association, motion claiming, exclusion embedding
— not model inference, which stayed at 18.7 ms. The 40 ms budget in §19 holds on
sparse footage and does not hold on a crowd. Worth stating plainly because §19's
number was measured on clips averaging a tenth of this density.

**c. Vehicle recall on ground-level urban footage has collapsed. The vehicles
are found and then thrown away at the confidence floor.**

Over frames 500–559, which contain a taxi, a white van, a box truck and several
cars continuously and unambiguously in frame, the stateless path returned
**4 `light_vehicle` detections in 60 frames — 0.07/frame**, against roughly six
vehicles visibly present per frame. Rendered and inspected directly: at frame
514 not one of the six is boxed, while 78 `personnel` are.

> **This subsection was rewritten after measuring.** It first attributed the
> collapse to WiderPerson's unlabelled vehicles, on the strength of the standing
> prediction that WiderPerson would one day show the AerialPerson trap. That
> attribution was asserted before it was checked, and the checks below do not
> support it as the main cause. The original claim is not preserved because it
> was wrong on the magnitude, not merely incomplete.

A stock COCO `yolo26n` — which has never seen this project's training mix, and
so carries none of its biases — was run on the same frames as a control:

| on frames 500–559 | vehicles found |
|---|---|
| stock COCO `yolo26n`, conf 0.35 | **5.80/frame** (40 car, 17 truck, 1 bus) |
| `weights/best.pt`, conf 0.25 (deployed) | 0.10/frame |
| `weights/best.pt`, conf 0.10 | 1.20/frame |
| `weights/best.pt`, conf 0.05 | 2.00/frame |
| `weights/best.pt`, conf 0.02 | **4.60/frame** |

The control confirms roughly six vehicles are genuinely present and trivially
detectable. **The deployed model localises them correctly and assigns them
near-zero confidence.** That is a different failure from not having learned
them: a model that had never seen ground-level vehicles would return nothing at
any threshold, or return nonsense. This one returns the right boxes at 0.02.

That signature — correct localisation, suppressed confidence — is what negative
evidence produces. So the mechanism of the contamination hypothesis is right.
The magnitude attributed to WiderPerson was not. Measured with the same stock
control over 300 training images per dataset:

| dataset | unlabelled vehicles found | images with any | labelled in the dataset |
|---|---|---|---|
| AerialPerson (after pseudo-labelling) | 1.49/image | 22% | 29,559 ✓ |
| WiderPerson | **0.23/image** | 8% | **0** |
| CrowdHuman | **0.06/image** | 3% | 0 |

WiderPerson contributes on the order of ~1,800 unlabelled vehicles across its
8,000 images — real contamination, and it should still be fixed, but two orders
of magnitude short of AerialPerson's ~258,000 and too small to be the whole
cause on its own.

**The larger cause is an asymmetry in training coverage that was never
noticed.** `personnel` has both aerial data (VisDrone, AerialPerson) *and*
ground-level data (WiderPerson). The three vehicle classes have **aerial data
only** — VisDrone is the sole source, and AerialPerson's pseudo-labels are also
aerial. The model has effectively learned "a vehicle is a small object seen from
above," so a large, close, horizontal-view car is off-distribution for the
vehicle classes specifically while being perfectly in-distribution for
`personnel`. The weak WiderPerson negatives then push the confidence of those
already-uncertain detections below the floor.

Consequences, revised:

- **Do NOT pseudo-label CrowdHuman.** Measured twice, with two independent
  detectors that agree: 0.047/image with `weights/best.pt`, 0.06/image with the
  stock control, and only 3% of images contain a vehicle at all. The plan's
  decision rule was to skip below ~1/image, and this is far below it. CrowdHuman
  is dense human crowds — stadiums, gatherings, indoor scenes — not street
  traffic, so the AerialPerson trap simply does not apply to it. The conversion
  is sound: 339,565 person boxes over 15,000 images, ~22.6/image as expected.
- **Pseudo-label WiderPerson anyway**, but expect it to be a small correction
  rather than the fix.
- **The actual fix is ground-level vehicle training data**, which nothing in
  the current mix provides. Until then the blended validation set cannot even
  see the problem: its vehicle instances are overwhelmingly aerial, which is
  exactly why `light_vehicle` scores mAP50 0.860 there while returning
  0.10/frame on a street. **A ground-level vehicle val split is a prerequisite
  for measuring any fix at all.**
- **Interim mitigation available today, unmeasured:** a per-class vehicle
  confidence floor for ground view, mirroring what
  `CONF_THRESHOLD_PERSONNEL_GROUND` does for people (§17b). The detections
  exist at 0.05–0.10. This would trade precision for them and has not been
  swept — do not adopt it without the sweep and a look at annotated frames,
  since low-confidence vehicle boxes are exactly the failure of §14.

**d. Duplicate boxes on single targets in dense crowds.** Several people carry
two or three concentric boxes. Since the head is `end2end` and NMS-free (§19),
there is no suppression parameter to tighten — this is learned behaviour and, like
the crowd recall it accompanies, only training data changes it.

### 21. SARD: the pose hypothesis is confirmed, and fine-tuning on it alone destroys the model (2026-09-05)

The longest-standing open failure here is personnel missed in vegetation from a
UAV. Resolution, palette, viewpoint and augmentation were each ruled out by
direct experiment; what remained was POSE, and nothing in the training mix
contained a prone or crawling person seen from above. SARD is that missing
distribution -- 1,980 source frames of actors simulating exhausted and injured
people over grass, forest shade and quarries, filmed from a drone.

**a. The baseline was the most informative measurement of the exercise.**
Deployed `weights/best.pt` on SARD's held-out test split, 570 images, 732
boxes, imgsz 1280, conf 0.10:

| size (px) | GT | recall | WiderPerson val |
|---|---|---|---|
| <16 | 8 | 0.000 | 0.187 |
| 16-32 | 108 | 0.037 | 0.622 |
| 32-48 | 202 | 0.149 | 0.805 |
| 48-64 | 134 | 0.164 | 0.855 |
| 64-96 | 173 | 0.179 | 0.937 |
| >96 | 107 | 0.168 | 0.940 |
| **overall** | **732** | **0.143** | 0.707 |

Recall is **flat across every size bucket**. Read the >96 px row twice: a person
larger than 96 px is trivially detectable, and this model finds 94% of them in
street imagery and 17% of them here. That rules out scale as the explanation
and establishes the remaining gap as pose, background and viewpoint -- across
732 boxes rather than the single v10 frame the diagnosis had rested on.

It also separates two problems that are easy to conflate: this is the POSE gap.
The <32 px far-field gap (0.187) is a different problem with a different fix.

**b. The fine-tune worked, spectacularly, on its own domain.** 12 epochs from
`weights/best.pt` on SARD train only, imgsz 1280, batch 4, lr0 0.002, `fpv`
augmentation profile, 1.29 hours:

| | before | after |
|---|---|---|
| SARD test recall | 0.143 | **0.870** |
| SARD test precision | 0.356 | **0.755** |

The flat ~0.17 line became a normal curve rising to 0.981 at >96 px.

**c. It transferred to v10, partially, and the detections are real.** Measured
over all 374 frames, stateless:

| | deployed | SARD fine-tune |
|---|---|---|
| `personnel` @ conf 0.25 | **0** | **72**, in 70 frames |
| `personnel` @ conf 0.10 | 22 | 205 |
| phantom `light_vehicle` @ 0.25 | 26 | 0 |

Rendered and inspected rather than counted: the boxes sit on a genuine
crouching person in the vegetation, at 0.55-0.61 confidence across 70
consecutive frames, which the deployed model never found at any point in the
clip.

**But frames 186 and 300 are still NOT solved.** Frame 186's prone figure:
both models return zero. Frame 300's two visible people: the deployed model
returns two detections and both are phantoms on HUD elements, the new model
returns none. Pose alone was not sufficient for the hardest frames; pose AND
false-colour palette AND heavy clutter is a harder conjunction than one
4,041-image dataset teaches.

**d. And it FAILED the promotion gate on all four criteria, catastrophically.**
`scripts/evaluate.py` against the blended val:

| class | baseline mAP50 | candidate | |
|---|---|---|---|
| ALL | 0.627 | **0.065** | -89% |
| personnel | 0.706 | **0.221** | -69% |
| two_wheeler | 0.494 | 0.005 | collapsed |
| light_vehicle | 0.860 | 0.034 | collapsed |
| heavy_vehicle | 0.449 | **0.000** | gone |

Overall recall 0.570 -> 0.048.

The vehicle classes losing ground was expected and would have been accepted for
a search-and-rescue specialist. **`personnel` losing 69% was not.** This is
catastrophic forgetting: 12 epochs at lr0 0.002 on a single-domain, single-class
set of 4,041 images overwrote the general model. The checkpoint is excellent on
SARD and close to useless everywhere else.

Worth stating plainly, because it is the argument for having the gate at all:
this model looks outstanding on its own evaluation and is 89% worse in
production. Nothing but the rubric would have caught that.

**e. Conclusion, and what to do instead.** The hypothesis is confirmed and the
method is rejected. SARD must be BLENDED into the existing mix -- the approach
the `fpv` run used for AerialPerson -- not trained on alone. 4,041 images
against the existing ~18,700 is a healthy ratio that will not dominate, and the
other three classes keep receiving positive examples throughout. Lower `lr0` to
0.001, since this data is now known to pull hard, and watch `personnel` mAP50
on the blended val per epoch rather than only at the end.

SARD's unlabelled-vehicle rate was measured first, with a stock COCO control
rather than this project's own checkpoint: 0.060/image over 300 tiles, far
below the ~1/image threshold. It is merged as-is and must not be
pseudo-labelled.

### 22. SARD blended: a 4.7x recall gain the validation set cannot see, and a final epoch that scored better by losing it (2026-09-06)

Section 21 confirmed the pose hypothesis and rejected the method: a SARD-only
fine-tune reached 0.870 recall on its own domain and destroyed the general model
(blended mAP50 0.627 -> 0.065). Its conclusion was to blend SARD into the
existing mix instead. This is that run.

**a. Configuration.** `battlesight_blend`, resumed from the deployed
`weights/best.pt`, 6 epochs on `data/battlesight_fpv.yaml`: VisDrone train
(6,471) + VisDrone test-dev (1,610) + WiderPerson (8,000) + AerialPerson
(2,613) + **SARD (4,041)** = 22,735 images. SARD is 17.8% of the mix. imgsz
1280, batch 4, `close_mosaic=1`. Val is unchanged from `battlesight_multi.yaml`
(1,548 images, 66,112 boxes), so the numbers stay comparable to section 17.

Note that `optimizer=auto` overrode the configured `lr0=0.002` and selected
AdamW at **lr 0.00125**, decaying to 0.000425. Section 21e asked for 0.001; what
ran was 0.00125, chosen by ultralytics rather than set.

CrowdHuman was dropped from this mix first, which made SARD the single variable
against the deployed checkpoint.

**Sections b through g measure the EPOCH 5 checkpoint**, kept at
`weights/best_e5_backup.pt`. Epoch 6 finished later, took `best.pt` on
fitness, and is a materially different model; see 22h.

**b. The promotion gate says nothing happened.** `scripts/evaluate.py`, both
checkpoints measured in one process:

| | baseline | candidate |
|---|---|---|
| mAP50 | 0.6274 | 0.6266 |
| recall | 0.5700 | 0.5741 |
| precision | 0.7160 | 0.7090 |
| personnel mAP50 | 0.7064 | 0.7030 |
| two_wheeler | 0.4940 | 0.4940 |
| light_vehicle | 0.8600 | 0.8590 |
| heavy_vehicle | 0.4490 | 0.4500 |

All four criteria PASS. Overall mAP50 moved -0.0008. On this evidence alone the
run is indistinguishable from `battlesight_crowd`, which was written off as flat.

**A trap worth recording.** The per-epoch `metrics/mAP50(B)` in `results.csv`
read 0.6373 for the promoted epoch, which looks like +0.010 over the baseline.
It is not comparable: the trainer validates at `max_det=902` (this run's
`args.yaml`) on EMA weights, while the rubric path uses `max_det=300`, raised to
317 on this val. Every recorded baseline in this file came through the rubric
path. Do not compare a `results.csv` row to a number in this document.

**c. The target-domain measurement is where the result is.** SARD test split,
570 images, 732 boxes, imgsz 1280, conf 0.10, identical to section 21a:

| size (px) | GT | deployed | blend | SARD-only specialist |
|---|---|---|---|---|
| <16 | 8 | 0.000 | 0.000 | |
| 16-32 | 108 | 0.037 | 0.287 | |
| 32-48 | 202 | 0.149 | 0.574 | |
| 48-64 | 134 | 0.164 | 0.776 | |
| 64-96 | 173 | 0.179 | 0.827 | |
| >96 | 107 | 0.168 | 0.907 | 0.981 |
| **overall** | **732** | **0.143** | **0.671** | 0.870 |

Precision on the same split: 0.356 -> 0.503. The deployed checkpoint reproduced
section 21a bucket for bucket, which is the control that makes the rest of this
trustworthy.

The shape matters more than the total. Section 21a's signature failure was that
recall was FLAT at ~0.17 across every bucket, including boxes over 96 px that
the same model finds 94% of in street imagery. That flat line is now a normal
curve. The model did not get better at small objects. It learned the pose.

**d. It transfers to the real UAV footage.** `v10.mp4`, all 374 frames,
stateless, no cascade:

| | personnel @0.25 | personnel @0.10 | phantom light_vehicle @0.25 |
|---|---|---|---|
| deployed | 0 (0 frames) | 22 (20 frames) | 26 |
| blend | 71 (50 frames) | 189 (131 frames) | 47 |
| specialist (21c) | 72 (70 frames) | 205 | 0 |

The deployed row reproduces 21c exactly. At the deployed threshold the system
goes from finding nobody in this clip to finding people in 50 of 374 frames,
reaching the specialist's detection count without the specialist's collapse.

**e. `battlesight_crowd`, measured properly for the first time.** It was
declared flat on blended mAP50 alone, the same metric that just declared this
run flat. Measured on the same two target-domain tests:

| checkpoint | blended mAP50 | SARD recall | v10 personnel @0.25 |
|---|---|---|---|
| deployed | 0.627 | 0.143 | 0 |
| `battlesight_crowd` | 0.627 | 0.175 | 2 |
| `battlesight_blend` | 0.627 | 0.671 | 71 |

The original verdict was right, and is now evidence rather than assumption:
CrowdHuman is dense standing crowds at ground level and does not teach prone
figures seen from altitude. But three checkpoints with identical blended mAP50
span 0.143 to 0.671 on the failure this project exists to fix. **The promotion
val cannot rank these runs.** It is a regression guard, not a measure of
progress, and any future personnel run must report SARD test recall beside it.

**f. One number moved the wrong way: v10 phantoms 26 -> 47.** These are not new
in kind. Both checkpoints put `light_vehicle` boxes on the same fixed spot: on a
64x64 grid, cell (54,32) holds 23 of the deployed model's 26 and 40 of the
blend's 47. That is x~0.50, y~0.85, bottom centre, held across the whole clip.
It is the burned-in subtitle block already listed in Still Outstanding as a
`moving_object` false positive, now also read as a vehicle.

`app/overlay_mask.py` removes none of them, and is right not to, for two
independent reasons:

- **Size.** Median phantom box is 0.0203 of the frame, about 137x137 px on
  1280x720. `OVERLAY_MAX_BOX_AREA` is 0.01, so 46 of 47 fail `_is_glyph_sized`
  and are neither learned from nor suppressed.
- **Persistence.** They fire in one burst, frames 283-344: 45 hits against 169.3
  decayed camera-moving frames, a ratio of ~0.27 against the required 0.30. Even
  with the size limit lifted they would not qualify.

Raising `OVERLAY_MAX_BOX_AREA` is the obvious fix and is rejected. That
threshold is the module's only safety property. It is what stops a drone holding
a real target centred in frame from having that target masked, and a 137x137 px
box at frame centre is exactly what such a target looks like.

A confidence threshold is also rejected: phantom confidence reaches 0.461 while
real `personnel` in the same clip reaches 0.535, and section 20 already
established that raising the vehicle floor is the wrong direction.

**g. The fix: a second evidence channel keyed on geometric rigidity.**
Persistence cannot separate a painted block from a held target, because both sit
still in image space. Box GEOMETRY can. A world-attached object seen from a
moving camera changes apparent size; a painted one does not. Measured on v10,
over cells with at least 10 detections:

| cell | n | cv width | cv height |
|---|---|---|---|
| (54,32) subtitle | 40 | 0.0047 | 0.0092 |
| (28,32) personnel | 20 | 0.1143 | 0.1357 |
| (29,32) personnel | 14 | 0.0324 | 0.0447 |
| (27,32) personnel | 13 | 0.0128 | 0.0436 |
| (36,26) personnel | 10 | 0.0181 | 0.0232 |

`OVERLAY_LARGE_*` in `app/config.py` adds a per-cell Welford accumulator for
above-glyph-size boxes and masks a cell at `max(cv_w, cv_h) <= 0.015` with at
least 10 hits, still only on camera-moving frames and still under the
`OVERLAY_MAX_FRACTION` fail-open cap. Replayed over v10:

| | phantoms | personnel |
|---|---|---|
| off (default) @0.25 | 47 -> 47 | 71 -> 71 |
| on @0.25 | **47 -> 16** | 71 -> 71 |
| on @0.10 | 137 -> 137 | 190 -> 190 |

**`personnel` is exempt from this channel, on measurement rather than
principle.** Before the exemption, at conf 0.10, the floor the ground view
actually runs personnel at, the channel removed **13 real personnel and zero
phantoms**. The extra low-confidence boxes admitted at 0.10 dilute a cell's
geometry until real detections start looking rigid. A filter that deletes people
at the threshold people are detected at is the failure the fail-open rule exists
for.

**It ships disabled.** `BATTLESIGHT_OVERLAY_LARGE=0`. The separation is real but
rests on one clip with 2.5x of margin at the closest point (0.0092 against
0.0232), and it does nothing at conf 0.10 even for vehicles. That is not enough
to enable a new suppression path by default. A second clip carrying a large
painted overlay is what would justify turning it on.

**h. The final epoch improved every dataset metric and lost the target.**
Epoch 6 is the mosaic-off polish pass and it took `best.pt` on fitness. Measured
against epoch 5, which is preserved at
`runs/detect/battlesight_blend/weights/best_e5_backup.pt`:

| | epoch 5 | epoch 6 |
|---|---|---|
| gate mAP50 vs 0.6274 baseline | 0.6266 (-0.0008) | **0.6337 (+0.0063)** |
| gate recall | 0.5741 | **0.5771** |
| SARD test recall | 0.671 | **0.673** |
| SARD test precision | 0.503 | **0.664** |
| v10 personnel @0.25 | **71**, in 50 frames | 15, in 15 frames |
| v10 personnel @0.10 | **189** | 62 |
| v10 phantom light_vehicle @0.25 | 47 | 6 |

Epoch 6 wins every dataset metric. It is the only epoch of this run that clears
the gate on merit rather than on tolerance. On the real UAV footage it finds
79% fewer people.

**Resolved by looking at the frames, not the counts.** Detections-per-frame is
not a quality proxy here, and epoch 5 has the worse SARD precision, so the
obvious reading is that its extra 56 boxes are false positives on vegetation.
Rendered side by side, they are not:

- frames 139 and 174: an unmistakable crouching figure in vegetation, boxed by
  epoch 5 at 0.30 and 0.26, and returned by epoch 6 as nothing at all.
- frame 246: an upright, high-contrast figure crossing open ground. Both
  checkpoints find it.
- the 15 frames epoch 6 fires on are a STRICT SUBSET of epoch 5's 50. Epoch 6
  finds nothing that epoch 5 misses.

The pattern is that epoch 6 keeps the easy targets and drops the hard ones. The
crouching-in-vegetation detections sit at 0.26-0.30, just above the deployed
0.25 floor, and the final epoch pushed them under it. That is the exact failure
this run exists to fix, so the run improved its own scores by giving back most
of its own result.

The phantom drop from 47 to 6 is the same conservatism, not an independent win.

**Epoch 5 is the checkpoint to promote**, on the standing rule that a phantom
contact is a nuisance and a suppressed real contact is the failure this system
must never have. Section 22b-g measures epoch 5 throughout and stands as
written.

**i. That conclusion was wrong, and the error was the comparison itself.**
Epoch 6 is deployed. What 22h does above is compare two differently-calibrated
checkpoints at one fixed threshold, which is not a comparison of the models --
it is a comparison of where each one happens to sit relative to 0.25. The
detection counts that argument rests on (71 against 15) are also fewer distinct
targets than they sound: the 35 disagreement frames are about four people
tracked across consecutive frames.

Compared instead at each checkpoint's own F2 optimum on SARD test, epoch 6 wins
outright:

| at conf 0.10 | recall | precision | F2 |
|---|---|---|---|
| epoch 5 | 0.672 | 0.500 | 0.629 |
| **epoch 6** | 0.673 | **0.661** | **0.671** |

Epoch 6 does see all four v10 targets. It scores them at 0.117-0.256, just
under the global floor, which is a calibration difference and not a capability
loss -- and it is what `CONF_THRESHOLD_PERSONNEL_DRONE = 0.10` now exists for.
With that floor the four targets return at 23, 1, 12 and 3 detections.

So the rule is not "prefer the epoch with more boxes". It is: **never compare
two checkpoints at one fixed threshold.** Compare each at its own operating
point, then choose the threshold separately. 22h reached the right suspicion
from the wrong evidence, and the frames were what settled it in both
directions.

Two things follow for future runs. `close_mosaic` is not free on this project:
it sharpens the model onto the training distribution and costs off-distribution
recall, and `v10.mp4` is currently the only asset that can detect that. And a
run must not be judged on its last epoch by default -- the fitness that selects
`best.pt` is blended-val mAP, which section 22e already established cannot see
this failure.

### Still outstanding

- **Personnel are missed on the real UAV footage** (section 16e). On `v10.mp4`
  the system boxes vegetation while missing every visible person; at frame 300
  two people are plainly visible and it returns zero detections. Measured, not
  assumed: it is not resolution (they are 50-60 px), not our fine-tuning (a
  stock COCO `yolo26s.pt` finds 1 of ~5), and not recoverable from the motion
  channel (at either coherence threshold it tags only the burned-in subtitles).
  Pose, viewpoint and false-colour palette are the gap. The training run that
  targets this **has now been run and promoted** (section 17) — but whether it
  actually put boxes on the people in `v10` is still unverified, and the val
  metrics cannot answer it.
  - **Substantially improved 2026-09-06 (section 22).** Blending SARD into the
    training mix took SARD test recall 0.143 -> 0.671 and v10 `personnel` at
    conf 0.25 from 0 to 71, across 50 of 374 frames, with the promotion gate
    flat. Frames 186 and 300 specifically are still unverified, and the
    blended val remains unable to measure any of this.
- ~~The AerialPerson download may still be in flight.~~ **Landed 2026-09-02** and is in the promoted run's train split (2,613 images). Original bullet: Zenodo throttles this
  network to ~125 KB/s. `scripts/prepare_training.py` writes a yaml naming only
  datasets that exist, so training can start without it and be rerun later.
- **Burned-in subtitles are tagged as `moving_object`** on `v10.mp4`. They are
  genuinely changing pixels fixed in image space, so neither the overlay mask
  (which exempts `moving_object` by design) nor the coherence gate rejects
  them. Not chased; the same attachment argument that drives `overlay_mask.py`
  would apply if it becomes a problem.
  - **2026-09-06:** section 22f traces the v10 `light_vehicle` phantoms to the
    same subtitle block: 40 of 47 boxes land in one 64x64 grid cell at bottom
    centre. Section 22g adds an opt-in rigidity channel that removes 31 of
    them. It is disabled by default and exempts `personnel`.
- ~~Training never finished.~~ **Done 2026-08-31.** `battlesight_v1` resumed
  from epoch 13 and completed 30/30 in 2.58 h, on `battlesight_multi.yaml`
  (VisDrone + WiderPerson) rather than VisDrone alone. Promoted after
  `eval_rubric.py` passed all four criteria; the epoch-13 checkpoint is kept at
  `weights/best_e12_visdrone.pt`. On the same VisDrone val at 1280, mAP50
  0.5046 -> 0.5623, mAP50-95 0.2772 -> 0.3159, recall 0.4855 -> 0.5262, and
  every class improved (`two_wheeler` mAP50 +25%, `heavy_vehicle` +21%).
  Adding WiderPerson also cut ground-level false positives sharply: on
  `v3.mp4`, spurious `light_vehicle` boxes fell from 40 to 5.
  - Note the resume inherited `flipud=0.5` from the original run's `args.yaml`.
    That was deliberate for aerial-only VisDrone but is wrong for
    ground-level WiderPerson imagery (upside-down people), and `--flipud 0`
    on the resume command line does not override a restored arg. Worth
    retraining without it to see if `personnel` improves further.
- **Drones-as-targets are still not a class and cannot be detected.** VisDrone
  is footage taken *from* drones, not *of* them; it contains no UAV
  annotations, and neither does WiderPerson. No threshold or resolution
  change can fix this — it needs UAV-labelled data and a 5th class. **Do not
  confuse this with fix 10's drone-view model**, which is a checkpoint
  specialized for footage shot *from* a drone (a different, already-solved
  problem) — it still only detects the same 4 ground/vehicle classes, just
  better, on that camera angle.
- **Weak `two_wheeler` confused for `light_vehicle` under blur/low light.**
  Two residual burst frames remain on `v6.mp4` (517, 735) after fix 9: a
  cluster of low-confidence (0.25-0.53), non-overlapping `light_vehicle`
  boxes over a row of parked motorbikes during a heavily motion-blurred night
  pass. Visually consistent with `two_wheeler`'s weak mAP50 (0.34-0.46 vs.
  `light_vehicle`'s 0.79-0.86) rather than a motion-filter phantom — see fix
  9. Same underlying cause as the existing "confident off-distribution false
  positives" item below, not a new bug; would need blur/low-light training
  imagery or a `two_wheeler`-specific accuracy pass to fix properly.
- **Buses are diluted.** `bus` is merged with `truck` into `heavy_vehicle`,
  which is both the rarest class (1001 val boxes vs 16039 for `light_vehicle`)
  and the weakest (mAP50 0.334, R 0.329). If buses matter specifically, split
  them into their own class in `scripts/remap_visdrone.py`.
- **Confident false positives off-distribution.** On a close-up desk clip the
  model still puts `light_vehicle` at 0.32-0.43 on a highlighter. Nothing in
  training resembles indoor close-range scenes; fix it with negative/indoor
  imagery in the training set, not with the threshold. **SUPERSEDED for `v11.mp4` — see section 16a/16b.** That clip's boxes were
  landing on the HUD (reticle dashes and telemetry glyphs), not on the scene,
  and are now 87.7% removed by `app/overlay_mask.py` with no training change.
  The driving-dataset recommendation below would not have addressed it. The
  general off-distribution item stands on its own for other content. Original
  note follows: **Same root cause, much worse severity, confirmed on
  `v11.mp4`** (fix 14): forward-facing
  highway dashcam footage produces 40+ overlapping `light_vehicle` boxes per
  frame on empty road/sky, verified independent of the motion filter. VisDrone
  is aerial-down, WiderPerson is ground-level pedestrians -- neither covers a
  forward-facing vehicle-mounted camera angle at all, so the model has no
  learned notion of "empty road" for that viewpoint. Recommended fix, not yet
  done: add a driving-scene dataset shot from this camera angle (BDD100K,
  KITTI, or Mapillary Vistas are the standard public options) remapped to the
  4-class taxonomy the same way `scripts/remap_visdrone.py` /
  `convert_widerperson.py` do it -- this both teaches genuine vehicle
  appearance from a forward-facing view AND supplies real negative (empty
  road) frames, which a threshold or motion-filter change cannot substitute
  for. A cheaper partial mitigation if a full dataset add isn't worth it: pull
  a handful of empty-road frames from clips like `v11.mp4` itself, label them
  with zero boxes, and add them as hard negatives in a short fine-tune pass --
  smaller effort, narrower fix, won't generalize past this camera angle the
  way a real driving dataset would.
- **Tracked throughput is ~10 fps** at `IMGSZ=1280` ungated, down from the
  16-20 fps the gated path managed while detecting nothing. If that is too
  slow, the 27 ms motion pass only supplies the moving/static flag and the
  class-agnostic contacts on this path, so it is the first thing to cut.
- ~~**`scripts/diagnose_bursts.py` does not exist in this checkout.**~~ **Recreated 2026-09-01 (section 16c).** It reproduced the open `v7.mp4` frame-261 burst at exactly the documented magnitude, which is the evidence it measures the same thing the original did. Read 16c's two caveats before comparing its numbers to older ones. Original note follows:
- **`scripts/diagnose_bursts.py` did not exist in this checkout.** Sections
  8, 9 and 11 all cite it as the tool that verified those fixes; it is not in
  `scripts/` and no file matching `*burst*` exists anywhere in the repo
  outside `.venv`. The fixes themselves check out (re-verified directly
  against the live TensorRT engines for fix 13 above), but the script should
  be recreated so future motion-filter changes get an automated regression
  sweep across all `vN.mp4` clips instead of relying on manual spot checks —
  which is exactly how fix 13's `v6.mp4` frame 962 and the still-open
  `v7.mp4` frame 261 (below) went undetected until now.
- ~~**`v7.mp4` frame 261: 7 `moving_object` boxes, baseline 0**~~ **Fixed 2026-09-01 (section 16d)** with `MOTION_CHRONIC_COOLDOWN=3` — the cooldown the note below called for. Both of that clip's bursts are gone; `v2`/`v3`/`v4`/`v5`/`v6`/`v9` are identical and `v1` improved. Original note follows:
- **`v7.mp4` frame 261: 7 `moving_object` boxes, baseline 0** — same
  boundary-flicker shape as fix 13, but on `MOTION_CHRONIC_BLOB_COUNT`
  (surviving blob count 17-20 for several consecutive frames, right under the
  20 cutoff) rather than `EGO_STATIC_SHIFT`. EMA-smoothing this gate the same
  way was tried and made it worse (see fix 13) — the gate needs to react
  fast, not average. Needs a different mechanism, e.g. a short cooldown that
  keeps dropping blobs for a few frames after the gate trips once, so a scene
  hovering just under the cutoff can't rebuild a coherent track between
  individual trips. Not implemented.
- **`v9.mp4` was never added to the documented clip sweep** (sections 8/9
  only cover v1-v7). Directly re-run: it has no `moving_object` explosion,
  but its ego-compensation residual check drops 224/225 frames as unreliable
  (parallax-heavy footage), so the class-agnostic motion tag is effectively
  blind on this clip — a different, non-urgent gap (under-detection, not a
  burst), noted here so it isn't mistaken for an untested unknown next time.

## Known limitations

- **One GPU, two consumers.** The inference model holds ~1.5 GB while training
  wants ~5.5 GB. Both fit on 8 GB at `batch=8`, but it is tight. Set
  `BATTLESIGHT_DEVICE=cpu` before starting uvicorn if you need to train and serve
  at once, or export to TensorRT/ONNX for serving.
- **GPU work still serialises** — deliberately, since there is one GPU. The
  threadpool keeps the event loop responsive; it does not make inference
  parallel. Two feeds at 25 ms/frame share ~40 fps between them.
- **Job state is in-memory.** Restarting the API loses job history; a running
  training subprocess survives but is orphaned.
- VisDrone-DET has no consecutive frames, so the motion filter is verified
  against synthetic pans (`tests/make_clips.py`), not real drone video. Tune
  `MOTION_THRESHOLD` against a real feed before trusting it.
- `allow_origins=["*"]` — tighten before this leaves the laptop.

## Tests

The current suite and how to run it are in [`README.md`](README.md). Two notes
that belong here rather than there:

- `tests/test_feed_isolation.py` and `scripts/test_stream.py` need the API
  already running; every other test is standalone and CPU-only.
- `tests/make_clips.py` regenerates the synthetic static/pan clips the motion
  tests use, so they are not committed. VisDrone-DET has no consecutive frames,
  which is why the motion filter is verified against synthetic pans in the first
  place — see the limitation below.

## Measured on this machine

| | |
|---|---|
| epoch time (`yolo26s`, batch 8, 640) | 149–162 s |
| peak VRAM training | 5539 MiB / 8188 MiB |
| mAP50 after 1 / 2 / 3 epochs | 0.309 / 0.364 / 0.409 |
| inference, 1920x1080 frame | 24–57 ms |
| `/health` under an active feed | ~1.2 ms |
