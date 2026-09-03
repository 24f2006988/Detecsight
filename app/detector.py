"""Model wrapper: loads once, serves many. Also holds the moving-target logic."""
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from collections import OrderedDict, deque
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
        # Extra view-specific models, keyed by view name (currently just
        # "drone" -- see config.DRONE_MODEL_PATH). Absent unless that weights
        # file exists, in which case _model_for() falls back to self.model.
        self._view_models: Dict[str, YOLO] = {}
        # source_id -> LRU of {track_id -> deque of (cx, cy) normalised centroids}
        self._history: Dict[str, "OrderedDict[int, deque]"] = {}
        # (view, source_id) -> that feed's own list of ultralytics tracker
        # objects. Keyed by view too: each YOLO model instance owns its own
        # predictor, so a feed switching view is a different tracker slot, not
        # a shared one.
        self._trackers: Dict[tuple, list] = {}
        # per-source frame counter for the strided far-field pass
        self._farfield_tick: Dict[str, int] = {}
        # dedicated far-field model instances (see _farfield_model_for)
        self._farfield_models: Dict[str, Optional[YOLO]] = {}
        # Inference runs in a worker thread (see the routers), so the model, the
        # predictor's tracker slot and the motion history all need guarding.
        # One GPU means serialising the work anyway; this just makes it explicit.
        self._lock = threading.Lock()
        # One worker, reused: the CPU motion stage overlaps the GPU pass.
        # Single-threaded on purpose -- MotionDetector is stateful per
        # source and the lock already serialises frames for one feed.
        self._pool = ThreadPoolExecutor(max_workers=1,
                                        thread_name_prefix='motion')

    def _load_one(self, path: str) -> YOLO:
        """Load one checkpoint (preferring a matching TensorRT engine) and
        warm it up. Shared by the default model and any view-specific ones so
        they get identical treatment."""
        p = Path(path)
        engine_path = p.with_suffix(".engine")
        load_path = p
        if config.USE_TENSORRT and config.DEVICE != "cpu" and engine_path.exists():
            load_path = engine_path

        print(f"Loading model from {load_path} on device {config.DEVICE}")
        try:
            model = YOLO(str(load_path))
        except Exception as e:  # noqa: BLE001
            if load_path == p:
                raise
            # The engine is tied to the exact GPU/driver/TensorRT version it
            # was built with -- one of those having moved on is not a reason
            # to take the whole service down when the .pt still works fine.
            print(f"TensorRT engine at {load_path} failed to load ({e!r}); "
                  f"falling back to {p}")
            model = YOLO(str(p))

        # Warm up: the first inference triggers CUDA kernel compilation (or,
        # for a dynamic-shape engine, context setup for this input shape) and
        # is 10-20x slower than steady state. Burn that cost at startup, not
        # on the first soldier's frame.
        dummy = np.zeros((config.IMGSZ, config.IMGSZ, 3), dtype=np.uint8)
        model.predict(dummy, device=config.DEVICE, quantize=config.QUANTIZE, verbose=False)
        return model

    def load(self):
        p = Path(config.MODEL_PATH)
        # A bare name like "yolo26s.pt" is a stock checkpoint ultralytics will
        # fetch. Anything with a directory in it is meant to be a real file, and
        # a missing one should say so plainly rather than fail deep inside YOLO().
        if p.parent != Path(".") and not p.exists():
            raise FileNotFoundError(
                f"Model not found at {p}. Train one with scripts/train.py and copy "
                f"its best.pt to weights/best.pt, or point BATTLESIGHT_MODEL at a "
                f"stock checkpoint name such as yolo26s.pt."
            )
        self.model = self._load_one(config.MODEL_PATH)
        print("Model loaded and warmed up.")

        # Drone-view specialisation is optional: only load it if it's actually
        # there, and never let its absence or a bad file take down serving --
        # every view falls back to self.model in that case (see _model_for).
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
        """Resolve a view name ("ground", "drone", ...) to the model that
        serves it, falling back to the default model for any view without its
        own checkpoint loaded -- including "ground" itself."""
        return self._view_models.get(view, self.model)

    def _bind_tracker(self, model: YOLO, view: str, source_id: str):
        """Give this feed its own tracker on this model.

        Ultralytics hangs a single tracker off the predictor and reuses it for
        every track() call, so without this two feeds would share one ID space
        and contaminate each other's track_ids.
        """
        pred = getattr(model, "predictor", None)
        if pred is None or not hasattr(pred, "trackers"):
            return  # first ever call: ultralytics builds one, we adopt it below
        key = (view, source_id)
        if key in self._trackers:
            pred.trackers = self._trackers[key]
        else:
            from ultralytics.trackers.track import on_predict_start
            on_predict_start(pred, persist=False)  # build a fresh tracker list
            self._trackers[key] = pred.trackers

    def _adopt_tracker(self, model: YOLO, view: str, source_id: str):
        """Record whatever tracker the predictor ended up holding for this feed."""
        pred = getattr(model, "predictor", None)
        if pred is not None and hasattr(pred, "trackers"):
            self._trackers[(view, source_id)] = pred.trackers

    def _track_history(self, source_id: str, track_id: int) -> deque:
        """Centroid history for one track, as an LRU.

        Track IDs only ever increase over the life of a feed, so an unbounded
        dict here would grow for as long as the drone stays connected. Evicting
        the least recently seen track keeps a long-running feed flat in memory.
        """
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
        """A track is 'moving' if its centroid has drifted more than
        MOTION_THRESHOLD across the retained history window."""
        hist = self._track_history(source_id, track_id)
        hist.append((cx, cy))
        if len(hist) < 3:
            return False
        x0, y0 = hist[0]
        x1, y1 = hist[-1]
        displacement = ((x1 - x0) ** 2 + (y1 - y0) ** 2) ** 0.5
        # bool(): the comparison yields numpy.bool_, which json.dumps rejects
        # on the WebSocket path (Pydantic would coerce it, raw json.dumps will not).
        return bool(displacement > config.MOTION_THRESHOLD)

    @staticmethod
    def _class_name(result, class_id: int) -> str:
        """Prefer the checkpoint's own label map so the API stays honest when
        it is pointed at a stock COCO model instead of a fine-tuned one."""
        names = getattr(result, "names", None) or {}
        if class_id in names:
            return str(names[class_id])
        if 0 <= class_id < len(config.CLASS_NAMES):
            return config.CLASS_NAMES[class_id]
        return str(class_id)

    @staticmethod
    def _conf_threshold_for(view: str, class_name: str) -> float:
        """Per-class/per-view confidence floor. Currently only one override
        exists (see CONF_THRESHOLD_PERSONNEL_GROUND in app/config.py) --
        everything else uses the global CONF_THRESHOLD."""
        if view == "ground" and class_name == "personnel":
            return config.CONF_THRESHOLD_PERSONNEL_GROUND
        return config.CONF_THRESHOLD

    @classmethod
    def _model_conf_floor(cls, view: str) -> float:
        """Lowest conf needed across all classes for this view, passed to
        model.predict()/track() itself so ultralytics doesn't strip a box
        before _to_detections gets a chance to apply the real per-class/view
        threshold via _conf_threshold_for."""
        floor = config.CONF_THRESHOLD
        if view == "ground":
            floor = min(floor, config.CONF_THRESHOLD_PERSONNEL_GROUND)
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
            # The model call itself only enforces the LOWEST threshold any
            # class/view needs (_model_conf_floor) so a lower personnel floor
            # doesn't get stripped before it reaches here; apply the real,
            # per-class/view floor now so the other classes aren't loosened
            # by accident.
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
        """Stateless single-image detection. No track IDs."""
        with self._lock:
            if self.model is None:
                raise RuntimeError("Model not loaded")
            model = self._model_for(view)
            t0 = time.perf_counter()
            results = model.predict(
                frame,
                imgsz=config.IMGSZ,
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
                # Stateless path has no frame counter to stride on, so it always
                # pays for the second pass -- callers here are single images.
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
        """A SECOND model instance, used only by the far-field pass.

        Loaded lazily and only when the feature is switched on, because it
        costs another engine's worth of GPU memory (~256 MB) on a card that is
        already tight. Returns None if it cannot be loaded -- the far-field
        pass is an enhancement and must never take serving down with it.
        """
        cached = self._farfield_models.get(view)
        if cached is not None:
            return cached
        if view in self._farfield_models:
            return None                       # previously failed; don't retry
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
        """Where to spend a second inference pass: wherever the SMALL boxes are.

        Recall collapses with target size (see FARFIELD_ENABLED in config for
        the measured curve), so the far field is the only part of the frame
        worth re-examining at higher effective resolution. Rather than assume
        where it is, read it off the cheap full-frame pass that just ran: the
        smallest boxes ARE the distant ones. Falls back to a horizon prior when
        the frame gives nothing to estimate from.

        Returns (x1, y1, x2, y2) in pixels, or None if the region would be so
        large that the second pass buys nothing over the first.
        """
        boxes = [d for d in detections if d.get("class_id", -1) >= 0]
        region = None
        if len(boxes) >= config.FARFIELD_MIN_BOXES:
            sizes = []
            for d in boxes:
                bw = (d["x2"] - d["x1"]) * w
                bh = (d["y2"] - d["y1"]) * h
                sizes.append((bw * bh) ** 0.5)
            cut = float(np.percentile(sizes, config.FARFIELD_SMALL_PCT))
            small = [d for d, s in zip(boxes, sizes) if s <= cut]
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

        x1 = max(0, int(region[0])); y1 = max(0, int(region[1]))
        x2 = min(w, int(region[2])); y2 = min(h, int(region[3]))
        if x2 - x1 < 32 or y2 - y1 < 32:
            return None
        if ((x2 - x1) * (y2 - y1)) / float(max(1, w * h)) > config.FARFIELD_MAX_FRACTION:
            return None                      # basically the whole frame again
        return (x1, y1, x2, y2)

    def _far_field_pass(self, frame: np.ndarray, detections: List[dict],
                        view: str, w: int, h: int) -> List[dict]:
        """Re-run detection on the far field and merge in what only it found."""
        region = self._far_field_region(detections, w, h)
        if region is None:
            return detections
        x1, y1, x2, y2 = region
        tile = frame[y1:y2, x1:x2]
        if tile.size == 0:
            return detections
        # MUST NOT be the serving model. A predict() call interleaved with the
        # track() calls on the same YOLO object wrecks that object's predictor
        # state, and the damage is to the FULL-FRAME pass, not to this one.
        # Bisected by running this pass with every tile box discarded, so only
        # the predict() call remained: v5.mp4 personnel fell 1560 -> 562. Saving
        # and restoring predictor.trackers around the call was NOT enough
        # (measured: no change at all), so the interference is broader than the
        # tracker list. A dedicated instance is the only clean answer.
        model = self._farfield_model_for(view)
        if model is None:
            return detections
        results = model.predict(
            tile, imgsz=config.FARFIELD_IMGSZ, conf=self._model_conf_floor(view),
            iou=config.IOU_THRESHOLD, max_det=config.MAX_DET,
            device=config.DEVICE, quantize=config.QUANTIZE, verbose=False,
        )
        extra = self._to_detections(results[0], "far_field", tracking=False, view=view)
        tw, th = x2 - x1, y2 - y1
        added = []
        for d in extra:
            # tile-normalised -> full-frame-normalised
            fx1 = (x1 + d["x1"] * tw) / w
            fy1 = (y1 + d["y1"] * th) / h
            fx2 = (x1 + d["x2"] * tw) / w
            fy2 = (y1 + d["y2"] * th) / h
            side = (((fx2 - fx1) * w) * ((fy2 - fy1) * h)) ** 0.5
            # The tile exists to find SMALL targets. A large box in it is a
            # duplicate of one the full frame already has, and keeping those is
            # what craters precision (0.644 -> 0.601 measured).
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
        """Anything moving that no classified detection overlaps becomes a
        generic 'moving_object' detection (class_id -1). This is how a UAV --
        or anything else outside the 4 trained classes -- still gets tagged:
        the classifier didn't recognize it, but the motion pass saw it move
        coherently, which is all this system is meant to require.

        With assign_ids the blob also lends its track id to the detection that
        claimed it. The gated path needs that: it runs predict() on crops, not
        track() on frames, so the motion tracker is the only thing carrying
        identity across frames there.
        """
        extra = []
        for blob in blobs:
            blob_box = (blob["x1"], blob["y1"], blob["x2"], blob["y2"])
            blob_area = max(1e-6, (blob_box[2] - blob_box[0]) * (blob_box[3] - blob_box[1]))
            claimer, best_score = None, config.MOTION_CLAIM_IOU
            for d in detections:
                det_box = (d["x1"] * w, d["y1"] * h, d["x2"] * w, d["y2"] * h)
                # Containment (intersection / blob area), NOT IoU. A walking
                # person's swinging leg or bag is a small blob sitting entirely
                # inside a large personnel box; its IoU with that box is
                # blob_area/person_area, which is far below any sane threshold,
                # so on IoU the blob escaped and was re-reported as a separate
                # phantom 'moving_object' stacked on a person already detected.
                # Measured on v3.mp4: 425 such boxes, 22% of all moving_object
                # output. What matters is whether the blob is already accounted
                # for by a detection, and that is containment.
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
        """Run the detector on crop_boxes only, as ONE batch, and map the boxes
        back into full-frame normalised coords.

        MOTION_CROP_IMGSZ rather than IMGSZ: these are already small regions,
        so letterboxing them up to 640 spends latency on padding. The batch is
        bounded by MOTION_MAX_CROPS_PER_FRAME, so it cannot outgrow the VRAM
        the full-frame path would have used anyway.
        """
        usable = [(b, frame[b[1]:b[3], b[0]:b[2]]) for b in crop_boxes]
        usable = [(b, c) for b, c in usable if c.size > 0]
        if not usable:
            return []

        results = model.predict(
            [c for _, c in usable],
            imgsz=config.MOTION_CROP_IMGSZ,
            conf=self._model_conf_floor(view),
            iou=config.IOU_THRESHOLD,
            device=config.DEVICE,
            quantize=config.QUANTIZE,
            verbose=False,
        )

        out = []
        for (ox, oy, _, _), r in zip([b for b, _ in usable], results):
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
                    # _claim_motion_blobs flips this for whatever a coherent
                    # blob claims. Anything else found in the crop is a static
                    # object that happened to sit beside something that moved.
                    "moving": False,
                })
        return out

    @staticmethod
    def _dedupe(detections: List[dict], iou_threshold: float) -> List[dict]:
        """Cross-crop NMS. Merged crops can still abut, and an object lying on
        the seam gets found once in each -- ordinary per-image NMS never sees
        that, because it happens between two separate inferences. IoU is
        invariant to scaling, so comparing normalised boxes is correct here."""
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
        """Stage 3: the classifier runs only on what stages 1-2 let through."""
        if not blobs:
            # Nothing moved coherently, so there is nothing worth classifying
            # and the GPU is never touched at all. This is where the latency
            # goes: a quiet feed costs background subtraction and nothing more.
            return []

        crop_boxes = merge_boxes(
            [pad_box((b["x1"], b["y1"], b["x2"], b["y2"]), w, h,
                     config.MOTION_CROP_PADDING, config.MOTION_CROP_MIN_SIZE)
             for b in blobs],
            config.MOTION_CROP_MERGE_IOU,
        )

        if len(crop_boxes) > config.MOTION_MAX_CROPS_PER_FRAME:
            # Too much of the frame is moving (panning camera, a crowd) for
            # crops to be the cheap option -- one full-frame pass covers all of
            # it for less. Bounding this is what stops the gate from ever being
            # SLOWER than the path it replaced.
            results = model.predict(
                frame, imgsz=config.IMGSZ, conf=self._model_conf_floor(view),
                iou=config.IOU_THRESHOLD, max_det=config.MAX_DET,
                device=config.DEVICE, quantize=config.QUANTIZE, verbose=False,
            )
            return self._to_detections(results[0], source_id, tracking=False, view=view)

        return self._dedupe(self._detect_on_crops(frame, crop_boxes, w, h, model, view),
                            config.IOU_THRESHOLD)

    def _filter_excluded(self, frame: np.ndarray, detections: List[dict], w: int, h: int) -> List[dict]:
        """Drop any detection whose crop visually matches an uploaded
        reference image (app/exclusion.py), regardless of what it was
        otherwise classified -- or not classified -- as."""
        if not detections:
            return detections
        crops = []
        for d in detections:
            x1 = max(0, int(d["x1"] * w))
            y1 = max(0, int(d["y1"] * h))
            x2 = max(x1, int(d["x2"] * w))
            y2 = max(y1, int(d["y2"] * h))
            crops.append(frame[y1:y2, x1:x2])
        # One batched forward pass for the whole frame's crops instead of one
        # per detection, which is what actually cost anything in this stage.
        flags = exclusion_store.are_excluded(crops)
        return [d for d, excluded in zip(detections, flags) if not excluded]

    def _full_frame_track(self, frame: np.ndarray, source_id: str, view: str, model: YOLO) -> List[dict]:
        """Ungated path: ultralytics tracking over the whole frame, every
        frame. Kept as the fallback behind MOTION_GATED because it is the one
        that gives real ByteTrack IDs and sees static targets the motion pass
        by definition never reports."""
        # bind -> infer -> adopt must be atomic: another feed swapping the
        # predictor's tracker mid-inference would mix the two ID spaces.
        self._bind_tracker(model, view, source_id)
        results = model.track(
            frame,
            imgsz=config.IMGSZ,
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
        """Stateful frame-in-a-stream detection, as a four-stage cascade.

        `view` selects which loaded checkpoint classifies this frame --
        "ground" (default) or "drone", see config.DRONE_MODEL_PATH -- and is
        otherwise orthogonal to the cascade below: the motion pass and track
        history are unaffected by which classifier is running.

        Returns track IDs and a moving/static flag per classified target, plus
        class-agnostic 'moving_object' detections for anything moving that the
        trained classes don't recognize, with reference-image exclusions
        applied to the final result.
        """
        with self._lock:
            if self.model is None:
                raise RuntimeError("Model not loaded")
            model = self._model_for(view)
            t0 = time.perf_counter()
            h, w = frame.shape[:2]

            # Stage 1+2 (app/motion_filter.py): background subtraction, then
            # size/aspect/trajectory-coherence gates. Cheap, CPU, downscaled --
            # incoherent jitter (wind-blown foliage) dies here and never costs
            # GPU time at all.
            # With MOTION_GATED off (the standing default) nothing on the model
            # path reads `blobs` until _claim_motion_blobs runs at the end, so
            # this CPU stage and the GPU pass are independent -- and running
            # them in sequence costs their SUM for no reason. Profiled on
            # v5.mp4: motion filter 23.6 ms median against a 52.2 ms total, so
            # this stage was ~45% of the frame budget while the GPU idled.
            # Both sides release the GIL (OpenCV, TensorRT), so overlapping
            # them costs max() instead of sum. The gated branches below DO read
            # blobs, so they wait for the result immediately.
            blobs_future = None
            if config.MOTION_PARALLEL and not config.MOTION_GATED:
                blobs_future = self._pool.submit(
                    motion_detector.detect, frame, source_id, view)
                blobs = None
            else:
                blobs = motion_detector.detect(frame, source_id, view)

            if config.MOTION_GATED and not motion_detector.ego_reliable(source_id):
                # Parallax beat the ego compensation, so the motion pass has
                # nothing trustworthy to gate on. An empty blob list would make
                # the gated path skip the GPU entirely and report nothing, i.e.
                # go blind exactly when the camera is moving. Classify the whole
                # frame instead and emit no class-agnostic motion contacts.
                detections = self._full_frame_track(frame, source_id, view, model)
            elif config.MOTION_GATED:
                # Stage 3: classify only the survivors. On a quiet frame this
                # is zero GPU work; on a normal one it is a few small crops
                # instead of a full 640x640 pass.
                detections = self._gated_detections(frame, blobs, source_id, w, h, model, view)
                # Blobs the classifier could not name are still reported, and
                # the blob's own track id becomes the detection's -- nothing
                # else is tracking identity on this path.
                detections = self._claim_motion_blobs(detections, blobs, w, h,
                                                       assign_ids=True)
            else:
                detections = self._full_frame_track(frame, source_id, view, model)
                if blobs_future is not None:
                    # GPU pass is done; collect the CPU stage that ran beside it.
                    blobs = blobs_future.result()
                    blobs_future = None
                # Far field BEFORE motion blobs are claimed: a distant target
                # the second pass names is a classified contact, and claiming
                # should see it so it doesn't also emit a `moving_object` on
                # top of the same thing. Strided -- the far field is scene
                # geometry, it does not move frame to frame.
                if config.FARFIELD_ENABLED and config.FARFIELD_STRIDE > 0:
                    n = self._farfield_tick.get(source_id, 0)
                    self._farfield_tick[source_id] = n + 1
                    if n % config.FARFIELD_STRIDE == 0:
                        detections = self._far_field_pass(frame, detections, view, w, h)
                detections = self._claim_motion_blobs(detections, blobs, w, h)

            if blobs_future is not None:
                # Defensive: a branch above that never collected it. Leaving a
                # future dangling would desynchronise the next frame's motion
                # state, which is stateful and order-dependent.
                blobs = blobs_future.result()

            # Stage 4: static HUD/OSD overlay rejection (app/overlay_mask.py).
            # Runs before the exclusion pass because it is the cheaper test and
            # drops the bulk of the boxes on FPV/UAV footage -- 31 per frame of
            # reticle dashes and telemetry glyphs on v11.mp4 -- so the exclusion
            # stage's per-crop embedding work is spent only on plausible ones.
            # `observe` is fed the UNFILTERED list, since a cell only becomes
            # known-overlay by repeatedly producing detections; filtering first
            # would erase the evidence the mask is built from. Camera state
            # comes from the motion pass's existing ego estimate rather than a
            # second one of our own.
            camera_moving = bool(
                motion_detector.debug_info(source_id).get("moving_camera", False))
            overlay_mask.observe(frame, detections, source_id, camera_moving)
            detections = overlay_mask.filter(detections, source_id)

            # Stage 5: reference-image exclusion on the final detection set.
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
        """Per-feed tracker/history sizes, for /health."""
        with self._lock:
            return {sid: len(h) for sid, h in self._history.items()}


detector = Detector()
