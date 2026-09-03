"""Reference-image exclusion: upload a picture of an object and detections
that visually match it get dropped, without needing that object to be one
of the trained classes and without any retraining.

Uses a pretrained ImageNet backbone purely as a feature extractor (its
classification head is discarded) to embed both the reference image and
each detection crop, then compares them by cosine similarity. This is why
it works with a single example image: it doesn't learn "headphone", it just
measures "does this crop look like that reference photo".
"""
import json
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small

from app import config


class ExclusionStore:
    def __init__(self, path: Path = None, device: str = "cpu"):
        # CPU by design: embeddings only run on a handful of surviving crops
        # per frame, not the whole image, and this keeps GPU memory free for
        # the detector, which is the thing that actually needs it.
        self.path = path or config.EXCLUSION_STORE_PATH
        self.device = device
        self._model = None
        self._transform = None
        self._entries: dict = {}  # name -> embedding (list[float])
        self._load()

    def _ensure_model(self):
        if self._model is not None:
            return
        weights = MobileNet_V3_Small_Weights.IMAGENET1K_V1
        model = mobilenet_v3_small(weights=weights)
        model.classifier = torch.nn.Identity()  # embedding, not classification
        model.eval().to(self.device)
        self._model = model
        self._transform = weights.transforms()

    def _embed_batch(self, images_bgr: List[np.ndarray]) -> np.ndarray:
        """One forward pass for all crops. Per-crop passes dominated this stage
        on a busy frame -- the transform and the batch dimension are the cost,
        not the network."""
        self._ensure_model()
        tensors = []
        for img in images_bgr:
            if img is None or img.size == 0:
                raise ValueError("empty image")
            rgb = img[:, :, ::-1]  # OpenCV BGR -> RGB
            pil = Image.fromarray(np.ascontiguousarray(rgb))
            tensors.append(self._transform(pil))
        batch = torch.stack(tensors).to(self.device)
        with torch.no_grad():
            embs = self._model(batch).cpu().numpy()
        norms = np.linalg.norm(embs, axis=1, keepdims=True)
        return embs / np.where(norms > 1e-8, norms, 1.0)

    def _embed(self, image_bgr: np.ndarray) -> np.ndarray:
        if image_bgr is None or image_bgr.size == 0:
            raise ValueError("empty image")
        return self._embed_batch([image_bgr])[0]

    def _load(self):
        if self.path.exists():
            self._entries = json.loads(self.path.read_text())

    def _save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self._entries))

    def add(self, name: str, image_bgr: np.ndarray):
        emb = self._embed(image_bgr)
        self._entries[name] = emb.tolist()
        self._save()

    def remove(self, name: str) -> bool:
        if name in self._entries:
            del self._entries[name]
            self._save()
            return True
        return False

    def list(self) -> List[str]:
        return list(self._entries.keys())

    def is_excluded(self, crop_bgr: np.ndarray,
                     threshold: float = None) -> Tuple[bool, Optional[str], float]:
        """Returns (excluded, matched_name, similarity). Cheap no-op when the
        store is empty, so this costs nothing until someone actually uploads
        a reference image."""
        if not self._entries or crop_bgr is None or crop_bgr.size == 0:
            return False, None, 0.0
        threshold = config.EXCLUSION_SIMILARITY_THRESHOLD if threshold is None else threshold
        emb = self._embed(crop_bgr)
        best_name, best_sim = None, -1.0
        for name, ref in self._entries.items():
            sim = float(np.dot(emb, np.asarray(ref)))
            if sim > best_sim:
                best_name, best_sim = name, sim
        return best_sim >= threshold, best_name, best_sim

    def are_excluded(self, crops: List[np.ndarray],
                      threshold: float = None) -> List[bool]:
        """Batched is_excluded over several crops from one frame. Same no-op
        shortcut when nothing has been uploaded, so this costs nothing until
        someone actually adds a reference image."""
        if not self._entries or not crops:
            return [False] * len(crops)
        threshold = config.EXCLUSION_SIMILARITY_THRESHOLD if threshold is None else threshold

        # An empty crop (a zero-area box after clipping) can't be embedded, but
        # it also can't match anything -- keep it out of the batch and mark it
        # not-excluded rather than failing the whole frame.
        usable = [(i, c) for i, c in enumerate(crops) if c is not None and c.size > 0]
        out = [False] * len(crops)
        if not usable:
            return out

        embs = self._embed_batch([c for _, c in usable])
        refs = np.asarray(list(self._entries.values()))
        sims = embs @ refs.T                      # (n_crops, n_refs) cosine sims
        best = sims.max(axis=1)
        for (idx, _), sim in zip(usable, best):
            out[idx] = bool(sim >= threshold)
        return out


exclusion_store = ExclusionStore()
