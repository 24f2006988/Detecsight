"""Loads the model once and serves many requests. Also has the moving-target logic."""
import threading
import time
from collections import OrderedDict, deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from ultralytics import YOLO

from app import config
from app.exclusion import exclusion_store
from app.motion_filter import iou_xyxy, merge_boxes, motion_detector, pad_box
from app.overlay_mask import overlay_mask


class Detector:
    def __init__(self):
        self.model: Optional[YOLO] = None
        # extra models per view, only "drone" for now (config.DRONE_MODEL_PATH)
        self._view_models: Dict[str, YOLO] = {}
        # source_id -> {track_id -> deque of (cx, cy)}, kept as an LRU
        self._history: Dict[str, "OrderedDict[int, deque]"] = {}
        # (view, source_id) -> that feed's own ultralytics trackers
        self._trackers: Dict[tuple, list] = {}
        # frame counter per source for the far-field pass
        self._farfield_tick: Dict[str, int] = {}
        self._farfield_models: Dict[str, Optional[YOLO]] = {}
        # requests run in worker threads, so the model and the histories need a lock
        self._lock = threading.Lock()
        # one worker, because MotionDetector keeps state per source
        self._pool = ThreadPoolExecutor(max_workers=1,
                                        thread_name_prefix='motion')

    def _load_one(self, path: str) -> YOLO:
        """Load a checkpoint, using the matching TensorRT engine if there is one."""
        p = Path(path)
        engine_path = p.with_suffix(".engine")
        if config.USE_TENSORRT and config.DEVICE != "cpu" and engine_path.exists():
            try:
                return self._load_and_warm(engine_path)
            except Exception as e:  # noqa: BLE001
                # an engine only works on the GPU, driver and TensorRT it was built with
                print(f"TensorRT engine at {engine_path} unusable ({e!r}); "
                      f"falling back to {p}")
        return self._load_and_warm(p)

    def _load_and_warm(self, load_path: Path) -> YOLO:
        """Load one model and run a dummy frame, so the slow first inference
        happens at startup and not on the first real frame."""
        print(f"Loading model from {load_path} on device {config.DEVICE}")
        model = YOLO(str(load_path))
        # don't pass imgsz here, a static engine rejects anything but its own shape
        dummy = np.zeros((config.IMGSZ, config.IMGSZ, 3), dtype=np.uint8)
        model.predict(dummy, device=config.DEVICE, quantize=config.QUANTIZE, verbose=False)
        # a static engine is built for one shape, so predict() must never override it
        backend = getattr(model.predictor, "model", None)
        model._static_engine = load_path.suffix == ".engine" and getattr(backend, "dynamic", True) is False
        return model

    @staticmethod
    def _imgsz(model: YOLO, size: int) -> dict:
        """imgsz for predict(), left out for a static engine so it uses its own shape."""
        return {} if getattr(model, "_static_engine", False) else {"imgsz": size}

    def load(self):
        p = Path(config.MODEL_PATH)
        # a bare name like "yolo26s.pt" is a stock checkpoint that ultralytics downloads
        if p.parent != Path(".") and not p.exists():
            raise FileNotFoundError(
                f"Model not found at {p}. Train one with scripts/train.py and copy "
                f"its best.pt to weights/best.pt, or point BATTLESIGHT_MODEL at a "
                f"stock checkpoint name such as yolo26s.pt."
            )
        self.model = self._load_one(config.MODEL_PATH)
        print("Model loaded and warmed up.")

        # the drone model is optional, without it every view uses self.model
        drone_path = Path(config.DRONE_MODEL_PATH)
        if drone_path.exists():
            try:
                self._view_models["drone"] = self._load_one(config.DRONE_MODEL_PATH)
                print("Drone-view model loaded and warmed up.")
            except Exception as e:  # noqa: BLE001
                print(f"Drone-view model at {drone_path} failed to load ({e!r}); "
                      f"'drone' view will fall back to the default model.")
        else:
            print(f"No drone-view model at {drone_path}; 'drone' view will use the default model.")

    def unload(self):
        with self._lock:
            self.model = None
            self._view_models.clear()
            self._history.clear()
            self._trackers.clear()
            self._farfield_tick.clear()
            self._farfield_models.clear()

    def _model_for(self, view: str) -> YOLO:
        """The model for a view, or the default one if that view has none."""
        return self._view_models.get(view, self.model)

    def _bind_tracker(self, model: YOLO, view: str, source_id: str):
        """Give this feed its own tracker. Ultralytics keeps one tracker on the
        predictor, so without this two feeds would share track IDs."""
        pred = getattr(model, "predictor", None)
        if pred is None or not hasattr(pred, "trackers"):
            return  # first call, ultralytics builds one and we pick it up after
        key = (view, source_id)
        if key in self._trackers:
            pred.trackers = self._trackers[key]
        else:
            from ultralytics.trackers.track import on_predict_start
            on_predict_start(pred, persist=False)
            self._trackers[key] = pred.trackers

    def _adopt_tracker(self, model: YOLO, view: str, source_id: str):
        """Save whatever tracker the predictor ended up with for this feed."""
        pred = getattr(model, "predictor", None)
        if pred is not None and hasattr(pred, "trackers"):
            self._trackers[(view, source_id)] = pred.trackers

    def _track_history(self, source_id: str, track_id: int) -> deque:
        """Centroid history for one track. Track IDs only go up, so the oldest
        ones are dropped to keep a long feed from growing forever."""
        feed = self._history.setdefault(source_id, OrderedDict())
        hist = feed.get(track_id)
        if hist is None:
            hist = deque(maxlen=config.MOTION_WINDOW)
            feed[track_id] = hist
        feed.move_to_end(track_id)
        while len(feed) > config.MAX_TRACKS_PER_SOURCE:
            feed.popitem(last=False)
        return hist

    def _is_moving(self, source_id: str, track_id: int, cx: float, cy: float) -> bool:
        """Moving means the centroid drifted more than MOTION_THRESHOLD over the window."""
        hist = self._track_history(source_id, track_id)
        hist.append((cx, cy))
        if len(hist) < 3:
            return False
        x0, y0 = hist[0]
        x1, y1 = hist[-1]
        displacement = ((x1 - x0) ** 2 + (y1 - y0) ** 2) ** 0.5
        # bool() because numpy.bool_ breaks json.dumps on the WebSocket path
        return bool(displacement > config.MOTION_THRESHOLD)

    @staticmethod
    def _class_name(result, class_id: int) -> str:
        """Use the checkpoint's own label map, so a stock COCO model reports honestly."""
        names = getattr(result, "names", None) or {}
        if class_id in names:
            return str(names[class_id])
        if 0 <= class_id < len(config.CLASS_NAMES):
            return config.CLASS_NAMES[class_id]
        return str(class_id)

    @staticmethod
    def _conf_threshold_for(view: str, class_name: str) -> float:
        """Confidence floor for a class. Personnel has its own in each view."""
        if class_name == "personnel":
            if view == "ground":
                return config.CONF_THRESHOLD_PERSONNEL_GROUND
            return config.CONF_THRESHOLD_PERSONNEL_DRONE
        return config.CONF_THRESHOLD

    @classmethod
    def _model_conf_floor(cls, view: str) -> float:
        """Lowest floor of any class in this view. The model gets this one, so it
        doesn't drop a box that _conf_threshold_for would still keep."""
        floor = config.CONF_THRESHOLD
        if view == "ground":
            floor = min(floor, config.CONF_THRESHOLD_PERSONNEL_GROUND)
        else:
            floor = min(floor, config.CONF_THRESHOLD_PERSONNEL_DRONE)
        return floor

    def _to_detections(self, result, source_id: str, tracking: bool, view: str = "ground") -> List[dict]:
        h, w = result.orig_shape
        boxes = result.boxes
        out = []
        if boxes is None or len(boxes) == 0:
            return out

        xyxy = boxes.xyxy.cpu().numpy()
        confs = boxes.conf.cpu().numpy()
        clss = boxes.cls.cpu().numpy().astype(int)
        ids = boxes.id.cpu().numpy().astype(int) if boxes.id is not None else None

        for i in range(len(xyxy)):
            cid = int(clss[i])
            cname = self._class_name(result, cid)
            conf = float(confs[i])
            # the model only applied the lowest floor, so apply the real one here
            if conf < self._conf_threshold_for(view, cname):
                continue

            x1, y1, x2, y2 = xyxy[i]
            nx1, ny1, nx2, ny2 = x1 / w, y1 / h, x2 / w, y2 / h
            track_id = int(ids[i]) if ids is not None else None

            moving = None
            if tracking and track_id is not None:
                cx, cy = (nx1 + nx2) / 2, (ny1 + ny2) / 2
                moving = self._is_moving(source_id, track_id, cx, cy)

            out.append({
                "class_id": cid,
                "class_name": cname,
                "confidence": conf,
                "x1": float(nx1), "y1": float(ny1),
                "x2": float(nx2), "y2": float(ny2),
                "track_id": track_id,
                "moving": moving,
            })
        return out

    def detect(self, frame: np.ndarray, source_id: str = "single", view: str = "ground") -> dict:
        """Detect on one image. No track IDs."""
        with self._lock:
            if self.model is None:
                raise RuntimeError("Model not loaded")
            model = self._model_for(view)
            t0 = time.perf_counter()
            results = model.predict(
                frame,
                **self._imgsz(model, config.IMGSZ),
                conf=self._model_conf_floor(view),
                iou=config.IOU_THRESHOLD,
                max_det=config.MAX_DET,
                device=config.DEVICE,
                quantize=config.QUANTIZE,
                verbose=False,
            )
            r = results[0]
            h, w = r.orig_shape
            detections = self._to_detections(r, source_id, tracking=False, view=view)
            if config.FARFIELD_ENABLED:
                # no frame counter here, so it always runs the second pass
                detections = self._far_field_pass(frame, detections, view, w, h)
            elapsed = (time.perf_counter() - t0) * 1000
            return {
                "source_id": source_id,
                "frame_width": w,
                "frame_height": h,
                "inference_ms": round(elapsed, 2),
                "detections": detections,
            }

    def _farfield_model_for(self, view: str) -> Optional[YOLO]:
        """A second model instance just for the far-field pass. It is loaded
        late because it costs about 256 MB of GPU memory, and returns None if
        it can't load, since the far-field pass is optional."""
        cached = self._farfield_models.get(view)
        if cached is not None:
            return cached
        if view in self._farfield_models:
            return None                       # failed before, don't retry
        path = config.DRONE_MODEL_PATH if view == "drone" else config.MODEL_PATH
        if view == "drone" and not Path(path).exists():
            path = config.MODEL_PATH
        try:
            model = self._load_one(path)
        except Exception as e:  # noqa: BLE001
            print(f"Far-field model failed to load ({e!r}); far-field pass disabled.")
            self._farfield_models[view] = None
            return None
        self._farfield_models[view] = model
        return model

    @staticmethod
    def _far_field_region(detections: List[dict], w: int, h: int):
        """Pick the part of the frame for a second pass: where the small boxes
        are, since small boxes are the far ones. Falls back to a horizon band.
        Returns (x1, y1, x2, y2) in pixels, or None if it's not worth it."""
        boxes = [d for d in detections if d.get("class_id", -1) >= 0]
        region = None
        if len(boxes) >= config.FARFIELD_MIN_BOXES:
            sizes = []
            for d in boxes:
                bw = (d["x2"] - d["x1"]) * w
                bh = (d["y2"] - d["y1"]) * h
                sizes.append((bw * bh) ** 0.5)
            cut = float(np.percentile(sizes, config.FARFIELD_SMALL_PCT))
            small = [d for d, s in zip(boxes, sizes, strict=True) if s <= cut]
            if len(small) >= config.FARFIELD_MIN_BOXES:
                x1 = min(d["x1"] for d in small) * w
                y1 = min(d["y1"] for d in small) * h
                x2 = max(d["x2"] for d in small) * w
                y2 = max(d["y2"] for d in small) * h
                pw, ph = (x2 - x1) * config.FARFIELD_PAD, (y2 - y1) * config.FARFIELD_PAD
                region = (x1 - pw, y1 - ph, x2 + pw, y2 + ph)
        if region is None:
            px1, py1, px2, py2 = config.FARFIELD_PRIOR
            region = (px1 * w, py1 * h, px2 * w, py2 * h)

        x1 = max(0, int(region[0])); y1 = max(0, int(region[1]))  # noqa: E702
        x2 = min(w, int(region[2])); y2 = min(h, int(region[3]))  # noqa: E702
        if x2 - x1 < 32 or y2 - y1 < 32:
            return None
        if ((x2 - x1) * (y2 - y1)) / float(max(1, w * h)) > config.FARFIELD_MAX_FRACTION:
            return None                      # that's the whole frame again
        return (x1, y1, x2, y2)

    def _far_field_pass(self, frame: np.ndarray, detections: List[dict],
                        view: str, w: int, h: int) -> List[dict]:
        """Run the detector again on the far field and add what it finds."""
        region = self._far_field_region(detections, w, h)
        if region is None:
            return detections
        x1, y1, x2, y2 = region
        tile = frame[y1:y2, x1:x2]
        if tile.size == 0:
            return detections
        # this has to be its own model. Calling predict() on the serving model
        # between track() calls broke its tracker state and cost about 60% of
        # detections (see the log, section 19)
        model = self._farfield_model_for(view)
        if model is None:
            return detections
        results = model.predict(
            tile, **self._imgsz(model, config.FARFIELD_IMGSZ), conf=self._model_conf_floor(view),
            iou=config.IOU_THRESHOLD, max_det=config.MAX_DET,
            device=config.DEVICE, quantize=config.QUANTIZE, verbose=False,
        )
        extra = self._to_detections(results[0], "far_field", tracking=False, view=view)
        tw, th = x2 - x1, y2 - y1
        added = []
        for d in extra:
            # tile coordinates back to full frame
            fx1 = (x1 + d["x1"] * tw) / w
            fy1 = (y1 + d["y1"] * th) / h
            fx2 = (x1 + d["x2"] * tw) / w
            fy2 = (y1 + d["y2"] * th) / h
            side = (((fx2 - fx1) * w) * ((fy2 - fy1) * h)) ** 0.5
            # a big box here is a copy of one the full frame already found
            if side > config.FARFIELD_MAX_BOX:
                continue
            box = (fx1, fy1, fx2, fy2)
            if any(iou_xyxy(box, (o["x1"], o["y1"], o["x2"], o["y2"])) > config.IOU_THRESHOLD
                   for o in detections):
                continue
            d.update({"x1": fx1, "y1": fy1, "x2": fx2, "y2": fy2,
                      "track_id": None, "far_field": True})
            added.append(d)
        return detections + added

    def _claim_motion_blobs(self, detections: List[dict], blobs: List[dict],
                             w: int, h: int, assign_ids: bool = False) -> List[dict]:
        """Anything moving that no detection overlaps becomes a 'moving_object'
        (class_id -1). That is how a UAV, or anything outside the four classes,
        still gets reported. With assign_ids the blob also gives its track id to
        the detection that claimed it."""
        extra = []
        for blob in blobs:
            blob_box = (blob["x1"], blob["y1"], blob["x2"], blob["y2"])
            blob_area = max(1e-6, (blob_box[2] - blob_box[0]) * (blob_box[3] - blob_box[1]))
            claimer, best_score = None, config.MOTION_CLAIM_IOU
            for d in detections:
                det_box = (d["x1"] * w, d["y1"] * h, d["x2"] * w, d["y2"] * h)
                # containment and not just IoU, because a swinging arm is a small
                # blob inside a big person box and has a tiny IoU with it
                ix = max(0.0, min(blob_box[2], det_box[2]) - max(blob_box[0], det_box[0]))
                iy = max(0.0, min(blob_box[3], det_box[3]) - max(blob_box[1], det_box[1]))
                overlap = max(ix * iy / blob_area, iou_xyxy(blob_box, det_box))
                if overlap >= best_score:
                    claimer, best_score = d, overlap

            if claimer is not None:
                if assign_ids:
                    claimer["track_id"] = blob["track_id"]
                    claimer["moving"] = True
                continue

            extra.append({
                "class_id": -1,
                "class_name": "moving_object",
                "confidence": blob["coherence"],
                "x1": blob["x1"] / w, "y1": blob["y1"] / h,
                "x2": blob["x2"] / w, "y2": blob["y2"] / h,
                "track_id": blob["track_id"] if assign_ids else None,
                "moving": True,
            })
        return detections + extra

    def _detect_on_crops(self, frame: np.ndarray, crop_boxes: List[tuple],
                          w: int, h: int, model: YOLO, view: str = "ground") -> List[dict]:
        """Detect on the crops as one batch and map the boxes back to the frame."""
        usable = [(b, frame[b[1]:b[3], b[0]:b[2]]) for b in crop_boxes]
        usable = [(b, c) for b, c in usable if c.size > 0]
        if not usable:
            return []

        results = model.predict(
            [c for _, c in usable],
            **self._imgsz(model, config.MOTION_CROP_IMGSZ),
            conf=self._model_conf_floor(view),
            iou=config.IOU_THRESHOLD,
            device=config.DEVICE,
            quantize=config.QUANTIZE,
            verbose=False,
        )

        out = []
        for (ox, oy, _, _), r in zip([b for b, _ in usable], results, strict=True):
            boxes = r.boxes
            if boxes is None or len(boxes) == 0:
                continue
            xyxy = boxes.xyxy.cpu().numpy()
            confs = boxes.conf.cpu().numpy()
            clss = boxes.cls.cpu().numpy().astype(int)
            for i in range(len(xyxy)):
                cid = int(clss[i])
                cname = self._class_name(r, cid)
                conf = float(confs[i])
                if conf < self._conf_threshold_for(view, cname):
                    continue
                x1, y1, x2, y2 = xyxy[i]
                out.append({
                    "class_id": cid,
                    "class_name": cname,
                    "confidence": conf,
                    "x1": float((x1 + ox) / w), "y1": float((y1 + oy) / h),
                    "x2": float((x2 + ox) / w), "y2": float((y2 + oy) / h),
                    "track_id": None,
                    # _claim_motion_blobs sets this to True for what a blob claims
                    "moving": False,
                })
        return out

    @staticmethod
    def _dedupe(detections: List[dict], iou_threshold: float) -> List[dict]:
        """NMS across crops, for an object that sits on the edge of two."""
        kept: List[dict] = []
        for d in sorted(detections, key=lambda d: d["confidence"], reverse=True):
            box = (d["x1"], d["y1"], d["x2"], d["y2"])
            if any(k["class_id"] == d["class_id"]
                   and iou_xyxy(box, (k["x1"], k["y1"], k["x2"], k["y2"])) > iou_threshold
                   for k in kept):
                continue
            kept.append(d)
        return kept

    def _gated_detections(self, frame: np.ndarray, blobs: List[dict],
                           source_id: str, w: int, h: int, model: YOLO,
                           view: str = "ground") -> List[dict]:
        """Stage 3: only classify what the motion stages let through."""
        if not blobs:
            # nothing moved, so the GPU isn't used at all
            return []

        crop_boxes = merge_boxes(
            [pad_box((b["x1"], b["y1"], b["x2"], b["y2"]), w, h,
                     config.MOTION_CROP_PADDING, config.MOTION_CROP_MIN_SIZE)
             for b in blobs],
            config.MOTION_CROP_MERGE_IOU,
        )

        if len(crop_boxes) > config.MOTION_MAX_CROPS_PER_FRAME:
            # too much is moving, one full pass is cheaper than all the crops
            results = model.predict(
                frame, **self._imgsz(model, config.IMGSZ), conf=self._model_conf_floor(view),
                iou=config.IOU_THRESHOLD, max_det=config.MAX_DET,
                device=config.DEVICE, quantize=config.QUANTIZE, verbose=False,
            )
            return self._to_detections(results[0], source_id, tracking=False, view=view)

        return self._dedupe(self._detect_on_crops(frame, crop_boxes, w, h, model, view),
                            config.IOU_THRESHOLD)

    def _filter_excluded(self, frame: np.ndarray, detections: List[dict], w: int, h: int) -> List[dict]:
        """Drop detections that look like an uploaded reference image."""
        if not detections:
            return detections
        crops = []
        for d in detections:
            x1 = max(0, int(d["x1"] * w))
            y1 = max(0, int(d["y1"] * h))
            x2 = max(x1, int(d["x2"] * w))
            y2 = max(y1, int(d["y2"] * h))
            crops.append(frame[y1:y2, x1:x2])
        # one batched pass for all the crops
        flags = exclusion_store.are_excluded(crops)
        return [d for d, excluded in zip(detections, flags, strict=True) if not excluded]

    def _full_frame_track(self, frame: np.ndarray, source_id: str, view: str, model: YOLO) -> List[dict]:
        """Track over the whole frame every frame. This is the normal path, and
        the only one that sees targets that aren't moving."""
        # bind, infer and adopt have to happen together, or two feeds mix their IDs
        self._bind_tracker(model, view, source_id)
        results = model.track(
            frame,
            **self._imgsz(model, config.IMGSZ),
            conf=self._model_conf_floor(view),
            iou=config.IOU_THRESHOLD,
            max_det=config.MAX_DET,
            device=config.DEVICE,
            quantize=config.QUANTIZE,
            tracker=config.TRACKER_CONFIG,
            persist=True,
            verbose=False,
        )
        self._adopt_tracker(model, view, source_id)
        return self._to_detections(results[0], source_id, tracking=True, view=view)

    def track(self, frame: np.ndarray, source_id: str, view: str = "ground") -> dict:
        """Detect on a frame from a stream, keeping state between frames.

        Returns track IDs and a moving flag for each target, plus
        'moving_object' entries for anything moving that the four classes
        don't cover, with reference-image exclusions applied at the end.
        """
        with self._lock:
            if self.model is None:
                raise RuntimeError("Model not loaded")
            model = self._model_for(view)
            t0 = time.perf_counter()
            h, w = frame.shape[:2]

            # stages 1 and 2 (motion_filter.py): background subtraction, then the
            # size, shape and trajectory checks. Foliage jitter dies here.
            blobs_future = None
            if config.MOTION_PARALLEL and not config.MOTION_GATED:
                blobs_future = self._pool.submit(
                    motion_detector.detect, frame, source_id, view)
                blobs = None
            else:
                blobs = motion_detector.detect(frame, source_id, view)

            if config.MOTION_GATED and not motion_detector.ego_reliable(source_id):
                # camera motion couldn't be cancelled, so there are no usable
                # blobs. Classify the whole frame so it doesn't go blind.
                detections = self._full_frame_track(frame, source_id, view, model)
            elif config.MOTION_GATED:
                # stage 3: classify only what moved
                detections = self._gated_detections(frame, blobs, source_id, w, h, model, view)
                # the blob's track id becomes the detection's here, because
                # nothing else tracks on this path
                detections = self._claim_motion_blobs(detections, blobs, w, h,
                                                       assign_ids=True)
            else:
                detections = self._full_frame_track(frame, source_id, view, model)
                if blobs_future is not None:
                    # the GPU pass is done, collect the motion stage that ran beside it
                    blobs = blobs_future.result()
                    blobs_future = None
                # far field goes before the blobs are claimed, so a distant
                # target it names doesn't also come out as a moving_object
                if config.FARFIELD_ENABLED and config.FARFIELD_STRIDE > 0:
                    n = self._farfield_tick.get(source_id, 0)
                    self._farfield_tick[source_id] = n + 1
                    if n % config.FARFIELD_STRIDE == 0:
                        detections = self._far_field_pass(frame, detections, view, w, h)
                detections = self._claim_motion_blobs(detections, blobs, w, h)

            if blobs_future is not None:
                # always collect it, or the next frame's motion state falls out of order
                blobs = blobs_future.result()

            # stage 4: HUD overlay filter (overlay_mask.py), before the exclusion
            # pass because it is cheaper and removes most of the boxes on FPV footage
            camera_moving = bool(
                motion_detector.debug_info(source_id).get("moving_camera", False))
            overlay_mask.observe(frame, detections, source_id, camera_moving)
            detections = overlay_mask.filter(detections, source_id)

            # stage 5: reference-image exclusion
            detections = self._filter_excluded(frame, detections, w, h)

            elapsed = (time.perf_counter() - t0) * 1000
            return {
                "source_id": source_id,
                "frame_width": w,
                "frame_height": h,
                "inference_ms": round(elapsed, 2),
                "detections": detections,
            }

    def reset_source(self, source_id: str):
        with self._lock:
            self._history.pop(source_id, None)
            for key in [k for k in self._trackers if k[1] == source_id]:
                self._trackers.pop(key, None)
            self._farfield_tick.pop(source_id, None)
            motion_detector.reset(source_id)
            overlay_mask.reset(source_id)

    def stats(self) -> dict:
        """How many tracks each feed is holding, for /health."""
        with self._lock:
            return {sid: len(h) for sid, h in self._history.items()}


detector = Detector()
