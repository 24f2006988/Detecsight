# DetecSight

Real-time person and vehicle detection for drone and helmet-camera video, served
over FastAPI. Built for an AR situational-awareness overlay: the system tells a
human operator what is in frame and what is moving.

![Motion-filter false positives before and after the chronic-noise gates](docs/motion_before_after.jpg)

*Same three frames, before and after the motion filter's chronic-noise gates.
Purple `moving_object` boxes are the class-agnostic motion channel; green are
classified `personnel`. The phantom purple boxes on foliage and platform edge
are gone while the classified detections carry through untouched — the track IDs
(`#91`, `#64`, `#62`, `#45`, `#157`) are the same before and after. That is the
property the filter is held to: it may only remove what it can prove is noise,
and it fails open when it cannot prove it.*

**Scope:** operator situational awareness only — not fire control, not automated
targeting, not combatant classification. The model detects four generic object
classes and flags coherent motion; every decision stays with the person wearing
the display.

**Provenance:** this work was originally built as part of a Smart India
Hackathon project, published under the team lead's account as
[Fusion-Sight](https://github.com/soumik15630m/Fusion-Sight). The engineering in
`app/`, `scripts/` and `tests/` is mine; this repository is that work under my
own name, with a fresh history and a rewritten README. The two repositories have
diverged since — this one is where development continues.

---

## What it does

A YOLO26 detector fine-tuned to four classes — `personnel`, `two_wheeler`,
`light_vehicle`, `heavy_vehicle` — plus a class-agnostic `moving_object`
(`class_id -1`) produced by an independent motion channel, so something that
moves coherently is still reported even when the classifier has no name for it.

One checkpoint — `weights/best.pt`, trained on VisDrone + VisDrone test-dev +
WiderPerson + AerialPerson at imgsz 1280 — serves both of the `?view=` values:

| view | classifier | differs by |
|---|---|---|
| `ground` (default) | `weights/best.pt` | personnel confidence floor 0.10, ground motion-coherence threshold |
| `drone` | same checkpoint | personnel floor 0.25, stricter coherence threshold |

**The views differ through config, not through weights.** There *was* a separate
VisDrone-only checkpoint specialised for top-down footage; it was retired when
this one beat it on the drone view's own home domain — VisDrone val personnel
mAP50 0.3162 → 0.7064 ([`ENGINEERING_LOG.md`](ENGINEERING_LOG.md) §17). The
mechanism is still there: drop a `weights/drone_best.pt` in and `?view=drone`
picks it up, and `Detector.load()` falls back to the default model for any view
without its own file. `/health` reports `views_loaded` so a client can tell which
case it is rather than guessing.

Served as a TensorRT FP16 engine when a matching `.engine` exists, with automatic
fallback to the `.pt` if the engine fails to load.

## The detection cascade

`Detector.track()` is five stages, and each one exists because of a measured
failure rather than a guess:

1. **Motion channel** (`app/motion_filter.py`) — background subtraction with
   ego-motion compensation. A frame-to-frame affine transform is fit by
   Shi-Tomasi corners + Lucas-Kanade flow + RANSAC, the previous frame is warped
   into the current one so static structure differences away, and stored track
   histories are warped by the same transform so trajectory *straightness*
   measures the object rather than the camera. Four independent gates suppress
   chronic noise (parallax residual, MOG2 foreground fraction, blob-density with
   a cooldown, branch-selection flicker).
2. **Illumination-vs-structure discriminator** — separates a light (muzzle
   flash, headlight, glare) from a real object in a single frame by testing
   whether the change is a uniform brightness shift or preserves structure under
   z-scoring. A fast path reports targets visible for only two frames, which the
   four-frame coherence requirement could never do.
3. **Classification** — full-frame inference, plus a strided far-field second
   pass that re-examines wherever the small boxes are, since recall collapses
   with target size.
4. **Static overlay rejection** (`app/overlay_mask.py`) — removes burned-in
   HUD/OSD glyphs. See below.
5. **Reference-image exclusion** (`app/exclusion.py`) — upload one photo of an
   object and matching detections are dropped, using a MobileNetV3 embedding and
   cosine similarity. No retraining, no new class.

Every stage **fails open**. For a situational-awareness system a phantom contact
is a nuisance and a suppressed real one is unacceptable, so any gate that cannot
judge passes the detection through.

## Results

**Inference resolution was the first real fix.** The deployed checkpoint was
running at imgsz 640 with a 0.35 confidence floor — both inherited defaults,
neither measured. VisDrone targets are routinely under 32 px, which 640 destroys
before the head ever sees them.

![Aerial detection at imgsz 640/conf 0.35 versus 1280/conf 0.25](docs/aerial_before_after.jpg)

*Same frame, imgsz 640 → 1280 and conf 0.35 → 0.25. The parked two-wheelers
under the awnings and the pedestrians on the near pavement are recovered rather
than invented — they are visible in the source frame. Note the false positive on
the blue roof at bottom-left: the lower floor is not free. On VisDrone val this
moved mAP50 0.5046 → 0.5623 and recall 0.4855 → 0.5262.*

**TensorRT FP16 adopted, INT8 measured and rejected** (VisDrone val, imgsz 1280,
batch 1, warmed up, ground checkpoint):

| | PyTorch FP32 | TensorRT FP16 | TensorRT INT8 |
|---|---|---|---|
| mAP50 | 0.5631 | 0.5608 | 0.5189 |
| recall | 0.5260 | 0.5303 | 0.4789 |
| inference | 13.3 ms | **6.95 ms** | 4.23 ms |

INT8 is nearly 2× faster again and costs ~4 points of mAP50 and 5 of recall —
not a trade this system can make.

**Burned-in HUD glyphs were the dominant error source on real FPV footage.** One
clip produced 54,658 `light_vehicle` boxes across 1,746 frames — 31.3 per frame,
on footage containing no vehicles at all. A spatial heatmap of those boxes
reproduced the HUD exactly: one box per dash of each dotted reticle line, one per
character of the telemetry string, median box 9×9 px.

The fix is not appearance-based but *attachment*-based: a painted glyph holds its
image-space position while the world slides underneath it, whereas a real object
is attached to the world and travels across the frame as the camera pans. So the
filter learns which grid cells keep producing **small** detections **while the
camera is established to be moving**, reusing the motion pass's existing ego
estimate.

| | before | after |
|---|---|---|
| phantom `light_vehicle` | 54,658 (31.3/frame) | **6,713 (3.84/frame)** — −87.7% |
| unaffected clip | — | byte-identical output |

**Confidence floor retuned on F2, not F1.** The blended validation set hid
ground-level headroom because aerial instances dominate by count. Split by
domain, dropping the ground-view personnel floor from 0.20 to 0.10 moved recall
0.636 → 0.725 for 5.7 points of precision — the right side of the trade when a
missed contact is the failure that matters.

**Deployed ground checkpoint:** mAP50 0.638, mAP50-95 0.389 on the blended
validation set.

**Live feed:** ~100 ms round trip (≈10 fps) end to end at 1080p over WebSocket on
loopback, of which 50–85 ms is server inference. `/health` answers in ~1.2 ms
while a feed is running, because inference is dispatched off the event loop.

Full fix-by-fix history, including the approaches that were tried and rejected,
is in [`ENGINEERING_LOG.md`](ENGINEERING_LOG.md).

## API

| endpoint | purpose |
|---|---|
| `GET /health` | model/device/class state, per-feed tracker sizes |
| `POST /detect` | single frame, stateless |
| `POST /detect/tracked` | frame-in-stream over HTTP, track IDs + motion flags |
| `WS /ws/track/{source_id}` | live feed: send JPEG bytes, receive JSON per frame |
| `POST /webrtc/offer/{source_id}` | live feed over WebRTC, detections on a data channel |
| `POST /train`, `GET /train/{id}` | launch and monitor a fine-tuning run |
| `POST /exclude` | upload a reference image to suppress |

Boxes are returned **normalised** to 0–1 so a client can scale to its own
viewport without knowing the source resolution.

Each `source_id` gets its own tracker, background model, motion history and
overlay mask — state never leaks between feeds. Ultralytics hangs a single
tracker off the predictor and reuses it for every call, so this had to be
managed explicitly.

## Setup

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
bash scripts/fetch_weights.sh        # weights/best.pt from the latest release
python scripts\check_gpu.py          # must print CUDA available: True
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

The checkpoint is a [release](../../releases) asset rather than a tracked file —
20 MB of binary that changes wholesale on every retrain is what git stores worst.
`fetch_weights.sh` verifies it against the published sha256, because a truncated
checkpoint otherwise fails deep inside `torch.load` with an unhelpful error.

Swagger UI at `/docs`. Do not use `--reload` — it reloads the model onto the GPU
on every file change.

Environment variables use a `BATTLESIGHT_` prefix (the project's original name):
`BATTLESIGHT_MODEL`, `BATTLESIGHT_DEVICE`, `BATTLESIGHT_USE_TENSORRT`. Every
threshold in `app/config.py` is overridable the same way, and each one is
commented with the measurement that justifies its value.

## Tests

```powershell
$env:PYTHONPATH="."
python tests\test_overlay_mask.py      # masks glyphs, never masks large boxes, fails open
python tests\test_motion_structure.py  # light vs. object, brief-appearance fast path
python tests\test_motion_coherence.py  # coherent walk survives, foliage jitter does not
python tests\test_history_cap.py       # bounded LRU history, per-feed isolation
python tests\test_exclusion.py         # reference-image matching
python tests\test_motion_gating.py     # gated path skips the GPU on a still scene
```

They assert safety properties, not just happy paths.

## Known limitations

- **Personnel in vegetation from a UAV are still missed.** Resolution, palette,
  viewpoint and augmentation have each been ruled out by direct experiment; what
  remains is pose — nothing in the training mix contains prone or crawling
  people seen from above.
- **No UAV-as-target class.** VisDrone is footage taken *from* drones, not *of*
  them. This needs UAV-labelled data and a fifth class; no threshold change can
  substitute.
- **Confident false positives off-distribution.** Nothing in VisDrone or
  WiderPerson resembles an indoor close-range scene, and the model has no
  learned notion of "not a vehicle" for that viewpoint:

  ![light_vehicle false positives on a pencil case and a highlighter](docs/offdistribution_false_positives.jpg)

  *`light_vehicle` at 0.39–0.42 on a pencil case and a highlighter, across three
  inference resolutions. Raising resolution removes one of the two boxes and
  tightens the other; it does not remove the failure. This is a training-data
  gap — indoor and hard-negative imagery — not a threshold to tune, and it is
  the reason the confidence floor sits at 0.25 rather than the 0.16 where mean
  F1 actually peaks.*

- **~10 fps tracked throughput** at imgsz 1280. Fixed per-frame overhead
  dominates, so lowering imgsz does not help much.
- **One GPU, serialised.** The threadpool keeps the event loop responsive; it
  does not make inference parallel. Multiple feeds share the throughput.
- Training job state is in-memory, and CORS is `*` — both fine for a laptop
  deployment, neither fine for a real one.

## Layout

```
app/          config, detector cascade, motion filter, overlay mask,
              exclusion store, FastAPI routers, training manager
scripts/      dataset conversion/remapping, training, TensorRT export,
              annotation, benchmarking, burst diagnosis, promotion rubric
tests/        standalone property tests
data/         4-class dataset configs
weights/      checkpoint lands here; fetched from a release, not tracked
docs/         figures used above
```

Datasets, training runs and captured footage are not tracked — see
`scripts/prepare_training.py` and `scripts/remap_visdrone.py` for how the
training data is assembled. The dataset yamls carry no absolute paths: they
anchor at VisDrone and reach its siblings with `../`, so they resolve against
whatever you set once with `yolo settings datasets_dir="<path>"`.

## License

MIT — see [`LICENSE`](LICENSE). The datasets it is trained on carry their own
terms: VisDrone, WiderPerson and AerialPerson are each licensed for research use
by their respective authors, and the checkpoints in `weights/` inherit those
terms.
