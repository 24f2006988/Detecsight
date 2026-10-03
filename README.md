# DetecSight

Real-time person detection for video from drones and helmet cameras.

I built this for a Smart India Hackathon project and kept working on it after the hackathon was over. The main goal is finding people. It also detects two-wheelers and vehicles, but those are weaker and I say where below.

It is YOLO26 with a motion filter and a HUD-overlay filter around it, served over FastAPI. On my laptop GPU it runs as a TensorRT FP16 engine at about 10.5 ms a frame.

[![CI](https://github.com/24f2006988/Detecsight/actions/workflows/ci.yml/badge.svg)](https://github.com/24f2006988/Detecsight/actions/workflows/ci.yml)
[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

## Try it

```bash
docker build -t detecsight . && docker run -p 8000:8000 detecsight
```

Open `http://localhost:8000`, upload a frame and it draws the boxes. The Docker image is CPU only, so it takes about 700 ms a frame. The API docs are at `/docs`.

To put it online, see `docs/DEPLOY.md`.

## What I tried

- trained on a blend of VisDrone, WiderPerson, AerialPerson and SARD, 22,735 images, at 1280 px
- ran every checkpoint through `scripts/evaluate.py` before keeping it, instead of trusting mAP
- added a motion filter that cancels camera motion first, so a pan does not look like everything moving
- added a filter for HUD overlays burned into the video
- made every filter fail open, so if it cannot decide the detection goes through
- tracked each feed separately, with its own tracker and motion history
- let you upload a reference image and drop detections that look like it, with no retraining

## Best result

Personnel is the class I spent nearly all the time on.

| | value |
|---|---|
| personnel mAP50 | `0.706` |
| ground-level personnel mAP50 | `0.757` |
| recall on people over 96 px | `0.940` |
| recall on people under 16 px | `0.187` |

The last row is the real result. People who are close are mostly solved. People who are far away are not, and there are a lot of them: 41% of the people in my ground-level validation set are under 32 px and fewer than half get found.

![Recall against target size, over the instance histogram](docs/size_recall.png)

Going from 1280 to 1536 px gave 0.4% more detections for 46% more compute, and the large boxes got worse. So more pixels is not the answer. More data might be.

![Tracked personnel with IDs through a crowded pedestrian crossing](docs/crowd_tracking.gif)

*A crossing at rush hour. Median 19 tracked people a frame, peaking at 78, with IDs held through occlusion. The people close to the camera are fine. The crowd behind them, several hundred people at 10 to 25 px, is mostly missed, which is what the curve says.*

## The HUD bug

One FPV clip gave me 54,658 vehicle boxes over 1,746 frames and there were no vehicles in it.

I assumed it was missing training data, because the camera angle was unlike anything in my datasets, and I was about to download 100 GB of driving footage on a metered connection. Before doing that I plotted where the boxes actually were. They spelled out the HUD, one box per dash of the reticle and one per character of the telemetry text. They were never on the scene.

The fix does not look at what is inside a box. It looks at what the box is stuck to. A painted glyph stays in the same place while the world slides under it, and a real object moves across the frame when the camera pans. The filter watches which grid cells keep producing small detections while the camera is known to be moving. It gets the camera motion from the motion filter that was already running.

That took 54,658 boxes down to 6,713, and the control clip came out byte-identical.

![Phantom detections on HUD-overlaid FPV footage, before and after the overlay filter](docs/hud_overlay_before_after.gif)

It never masks a big box, because a big box that stays put is what a real person held in the centre of the frame looks like. If it would mask more than a set fraction of the frame it turns itself off. Both are tested in `tests/test_overlay_mask.py`.

## What worked

- measuring before downloading anything, which is how the HUD bug got found
- choosing thresholds on F2 and not F1, because a missed person is worse than a false box
- comparing checkpoints at their own best threshold. I used to compare at one fixed threshold, and that mostly compares the thresholds
- adding SARD, which took recall on people lying down seen from a drone from `0.143` to `0.673`
- tests that check safety properties (never masks a big box, fails open, feeds keep separate state) and not just the happy path. They import nothing heavier than OpenCV, so CI runs them in a couple of seconds

## What did not work so well

| I tried | and dropped it because |
|---|---|
| a drone-only checkpoint | it lost on its own home domain, personnel `0.3162` against `0.7064` for the general model |
| fine-tuning on SARD alone | `0.870` on the target domain, but everything else fell apart. Personnel went `0.706` to `0.221` and two classes disappeared |
| INT8 quantisation | 2.7 ms faster but 4 points of mAP50 and 5 points of recall worse. At about 10 fps the speed gain is not visible and the recall loss is |
| imgsz above 1280 | see above |
| tuning `IOU_THRESHOLD` for crowds | it does nothing. YOLO26 has no NMS step and ignores `iou=`, and sweeping 0.5 to 0.8 gave identical results every time |
| CrowdHuman, 15,000 images | most of a day of training and it moved nothing |

The full list with the workings is in `ENGINEERING_LOG.md`.

Three problems are still open:

- **People lying down, seen from a drone.** Better, not fixed. It took weeks. I ruled out resolution, the infrared palette, the viewpoint and my augmentation one at a time, and the answer was pose, because nothing in training had anyone lying down. It only finds them if I drop the confidence floor to `0.10`, because it scores them at `0.117` to `0.256`. It finds nothing under 16 px, and all of this rests on one clip and 570 test images.
- **Vehicles at eye level.** On a street with six cars it reports `0.27` a frame and stock COCO gets `5.60`. The vehicle classes only ever saw aerial data, so the model learned that a vehicle is a small thing seen from above. At a threshold of `0.02` mine finds `6.35`, so it sees them and then will not commit. `heavy_vehicle` scores `0.025` at ground level, which is a class that does not work. BDD100K is converted and blended in as the fix, but I have not run that training yet.
- **Confident boxes on things it has never seen.** It puts `light_vehicle` at `0.39` to `0.42` on a pencil case and a highlighter. That is a data gap and not a threshold problem, and it is why the floor is `0.25` and not the `0.16` where F1 peaks.

![light_vehicle false positives on a pencil case and a highlighter](docs/offdistribution_false_positives.jpg)

Smaller things: about 10 fps tracked at 1280 px, no drone-as-a-target class, buses and trucks share one class, and training state is kept in memory.

## What I would improve next

- run the BDD100K training and check it against a ground-level vehicle validation split. The blended validation set cannot see this failure at all
- find or make more data of people far away and people lying down
- add a drone class
- keep the training job state on disk and not in memory

## How it works

There are four classes, plus a `moving_object` tag from a separate motion channel. That tag means something moved in a coherent way and the classifier had no name for it. `Detector.track()` runs five stages:

1. motion detection with camera-motion compensation
2. a check that tells a light from an object in a single frame, so someone visible for only two frames can still be reported
3. the detector
4. the HUD filter
5. reference-image exclusion

| endpoint | what it does |
|---|---|
| `GET /` | the demo page |
| `POST /detect` | one frame, no state |
| `POST /detect/tracked` | a frame from a stream, with track IDs |
| `WS /ws/track/{id}` | live feed, JPEG in and JSON out |
| `POST /webrtc/offer/{id}` | live feed over WebRTC |
| `POST /train`, `GET /train/{id}` | start and watch a fine-tune |
| `POST /exclude` | upload a reference image to suppress |

Boxes come back normalised from 0 to 1. Each feed keeps its own tracker and motion history. I had to do that by hand, because ultralytics reuses one tracker for everything.

## Files

- `app/`: the service. `detector.py`, `motion_filter.py` and `overlay_mask.py` are the core
- `scripts/`: dataset converters, training, evaluation and the annotate tools
- `tests/`: pytest, 22 of them run without a GPU or a checkpoint
- `data/`: dataset configs
- `docs/REPRODUCE.md`: the command behind every number above
- `docs/DEPLOY.md`: putting it online
- `ENGINEERING_LOG.md`: everything I tried, including what failed

## Run

```bash
pip install -e ".[serve]"
bash scripts/fetch_weights.sh v1.0.0
uvicorn app.main:app --port 8000     # no --reload, it reloads the model onto the GPU every time
pytest
```

The environment variables start with `BATTLESIGHT_`, which is what the project was called first. The dataset configs use relative paths, so set the dataset folder once with `yolo settings datasets_dir="<path>"`. The datasets themselves are not in the repo.

## Scope

This tells a person what is in the frame and what is moving, and they decide what it means. It finds four generic object types and flags motion. It is not for targeting and it does not classify anyone. The same thing is useful for search and rescue and crowd safety, and the hardest problem in here, finding someone lying still in vegetation from the air, is a search and rescue problem.

MIT licensed. VisDrone, WiderPerson, AerialPerson, SARD, CrowdHuman and BDD100K are each licensed for research use by their authors.

This started as a Smart India Hackathon team project, published under our team lead's account as [Fusion-Sight](https://github.com/soumik15630m/Fusion-Sight). That repo is the whole team's work. The detection and tracking side in `app/`, `scripts/` and `tests/` is mine, and this is that part, carried on by me afterwards.
