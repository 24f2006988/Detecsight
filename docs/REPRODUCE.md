# Reproducing the numbers

Every figure quoted in [`README.md`](../README.md) and
[`ENGINEERING_LOG.md`](../ENGINEERING_LOG.md), with the command that produced
it. A number without a command beside it is an assertion; this file is what
makes them checkable.

**Before anything else:**

```bash
pip install -e ".[serve,gpu]"
bash scripts/fetch_weights.sh v1.0.0
yolo settings datasets_dir="<where your datasets live>"
```

The dataset yamls anchor at VisDrone and reach its siblings with `../`, so that
one setting resolves all four. Footage referred to as `v1.mp4`–`v11.mp4` is not
redistributable and is not in this repository; substitute your own clips.

---

## Accuracy

| number | command |
|---|---|
| Deployed checkpoint: **mAP50 0.638, mAP50-95 0.389** | `python scripts/evaluate.py` |
| Promotion gate for a candidate (four criteria, PASS/FAIL, exit status) | `python scripts/evaluate.py runs/detect/<run>/weights/best.pt` |
| Size-stratified recall: **<16 px 0.187 · 16–32 0.622 · >96 0.940** | `python scripts/eval_size_recall.py --weights weights/best.pt --images datasets/WiderPerson/images/val --limit 300 --imgsz 1280 --conf 0.10` |
| The chart in `docs/size_recall.png` | `python scripts/plot_size_recall.py` |
| Rubric against a named baseline (used for the drone-view retirement, §17) | `python scripts/eval_rubric.py <candidate> --baseline weights/drone_best.pt --data data/battlesight_fpv.yaml --imgsz 1280` |

`evaluate.py` defaults to `data/battlesight_multi.yaml` at imgsz 1280. Both
matter: the val split there is byte-identical to the `fpv` yaml's, which is what
keeps every mAP figure in the log comparable, and 1280 is what production serves
at — measuring at 640 produces numbers that do not describe the deployment.

## Latency

| number | command |
|---|---|
| **FP32 13.3 ms / FP16 6.95 ms / INT8 4.23 ms** per frame | `python scripts/bench_imgsz.py --weights weights/best.pt --imgsz 1280 --frames 40 --warmup 8 --source v5.mp4` against each engine in turn |
| Building the FP16 engine those numbers come from | `python scripts/export_engine.py --model weights/best.pt --static` |
| Parallel motion stage: **46.4 → 24.6 ms median** | run any clip through `annotate_video.py` with `BATTLESIGHT_MOTION_PARALLEL=0` and then `=1` |
| CPU path, ~**709 ms**/frame at 1280 | `BATTLESIGHT_DEVICE=cpu python scripts/annotate_video.py <clip>` |

**Latency figures on this machine vary about 3× run to run** — the same
unchanged code path has measured 25, 46 and 74 ms in one session. Only warmed,
back-to-back, within-run comparisons mean anything. `--warmup 8` is not
optional: the GPU idles at 270 MHz and a cold first call reports roughly double
the true latency. `--static` is not optional either — dynamic shapes size the
memory pool for batch 16 and OOM on an 8 GB card.

## Filters and regressions

| number | command |
|---|---|
| HUD glyphs: **54,658 → 6,713 phantom `light_vehicle`** (−87.7%) | `python scripts/diagnose_bursts.py v11.mp4 --view drone` with `BATTLESIGHT_OVERLAY_FILTER=0` and then `=1` |
| Control clip byte-identical under the same change | `python scripts/diagnose_bursts.py v10.mp4 --view drone --json before.json` then diff against `after.json` |
| Per-clip burst regression across the whole set | `python scripts/diagnose_bursts.py --view ground` |
| Motion-filter changes without touching the GPU (safe during training) | `python scripts/diagnose_bursts.py --motion-only` |
| Dense-crowd behaviour (§20) | `python scripts/annotate_video.py <crossing clip> --view ground` |

Two cautions on burst counts. `--motion-only` reports higher totals than the
served path, because `_claim_motion_blobs` removes blobs a YOLO box already
covers. And burst *counts* can rise when a fix lowers totals, because the
rolling-median baseline falls with them — read totals and max/frame alongside.

Rolling-median burst detection reports **zero** bursts on the HUD failure,
because a constant per-frame error is not a spike. Burst counts alone will not
find that class of bug.

## Properties, not numbers

```bash
pytest                 # 22 CPU property tests, no GPU, no checkpoint, ~0.5 s
pytest -m model        # needs a checkpoint and the torch stack
pytest -m server       # needs uvicorn on 127.0.0.1:8000
pytest -m ""           # everything
ruff check .
```

## What cannot be reproduced from this repository

Stated so the gaps are explicit rather than discovered:

- **The training runs.** VisDrone, WiderPerson, AerialPerson and CrowdHuman are
  fetched and converted by `scripts/`, not vendored — see
  `scripts/prepare_training.py`, which re-checks what is on disk and prints the
  exact training command. A full run is ~7 hours on this hardware.
- **The clips.** `v1.mp4`–`v11.mp4` and the crossing footage in §20 are not
  redistributable. The clip-referenced numbers are therefore reproducible in
  method but not on the identical input.
- **INT8 figures.** `weights/best_int8.engine` was built from a superseded
  checkpoint and is stale. Rebuild before re-measuring; the conclusion (−4 pt
  mAP50) is not expected to change, since it is a property of the quantisation
  rather than of that checkpoint.
