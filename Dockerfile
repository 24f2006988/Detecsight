# CPU-only image: the point of it is that someone without an 8 GB NVIDIA card
# can still see the service work. Expect seconds per frame at imgsz 1280, not
# the 6.95 ms the TensorRT FP16 engine does on the GPU -- see the README.
FROM python:3.12-slim

# libgl1/libglib2.0-0 are what opencv-python links against and a slim image
# does not ship. The alternative -- swapping in opencv-python-headless, which
# needs neither -- is deliberately not taken: both packages unpack into the
# same cv2/ directory and whichever installs last wins, so keeping a single
# OpenCV story in every environment is worth ~50 MB here.
# curl and ca-certificates are for fetching the checkpoint at first start.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
        curl \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# CPU torch first, from PyTorch's CPU index. Without this the plain PyPI wheel
# pulled in by `.[serve]` drags in ~2.5 GB of CUDA libraries that this image can
# never call. Its own version is what the unpinned `torch>=2.6` in
# pyproject.toml then resolves against, so the install below is a no-op for it.
RUN pip install --no-cache-dir \
        --index-url https://download.pytorch.org/whl/cpu \
        torch torchvision

COPY pyproject.toml README.md LICENSE docker-entrypoint.sh ./
COPY app/ ./app/
COPY scripts/ ./scripts/
COPY data/ ./data/

# Editable, and after the source copy. The tempting version of this -- copying
# pyproject.toml alone, installing against a stub app/ to cache the dependency
# layer, then copying the real source over it -- leaves an `app` package in
# site-packages that shadows the real one depending on how the process is
# started. Losing that cache layer costs about half a minute; torch, which is
# the expensive part, is already cached above and is unaffected by source edits.
RUN pip install --no-cache-dir -e ".[serve]" \
    && chmod +x docker-entrypoint.sh scripts/*.sh

# No GPU here, so no FP16 and no TensorRT: config already turns both off when
# DEVICE == "cpu", and Detector.load() falls back to the .pt checkpoint.
ENV BATTLESIGHT_DEVICE=cpu \
    PYTHONUNBUFFERED=1 \
    DETECSIGHT_WEIGHTS_TAG=v1.0.0

EXPOSE 8000

# The checkpoint is a release asset rather than a tracked file, so it is
# fetched on first start rather than baked in -- that keeps the image free of a
# 20 MB binary that changes wholesale on every retrain, and lets a newer
# checkpoint be used by overriding DETECSIGHT_WEIGHTS_TAG. Mount a volume at
# /app/weights to keep it across container restarts.
ENTRYPOINT ["./docker-entrypoint.sh"]
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
