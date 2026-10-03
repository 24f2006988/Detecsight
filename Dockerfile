# CPU-only image, so anyone without an NVIDIA card can run it. Expect a few
# hundred ms to a couple of seconds a frame at imgsz 1280, not the GPU numbers
# from the README.
FROM python:3.12-slim

# opencv-python needs libgl1 and libglib2.0-0, which the slim image leaves out.
# curl is for downloading the checkpoint on first start.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
        curl \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Hugging Face Spaces runs containers as uid 1000, so do the same everywhere.
RUN useradd -m -u 1000 app
WORKDIR /app

# CPU torch first. The normal PyPI wheel drags in about 2.5 GB of CUDA libraries
# that this image can never use.
RUN pip install --no-cache-dir \
        --index-url https://download.pytorch.org/whl/cpu \
        torch torchvision

COPY pyproject.toml README.md LICENSE docker-entrypoint.sh ./
COPY app/ ./app/
COPY scripts/ ./scripts/
COPY data/ ./data/

# Install after copying the source. Installing against an empty app/ to cache
# the layer leaves a second copy of the package in site-packages.
RUN pip install --no-cache-dir -e ".[serve]" \
    && chmod +x docker-entrypoint.sh scripts/*.sh \
    && mkdir -p weights \
    && chown -R app:app /app

USER app

# No GPU, so no FP16 or TensorRT. The config turns both off when DEVICE is cpu.
# PUBLIC_DEMO leaves out /train and /exclude, which anyone could call otherwise.
ENV BATTLESIGHT_DEVICE=cpu \
    BATTLESIGHT_PUBLIC_DEMO=1 \
    PYTHONUNBUFFERED=1 \
    DETECSIGHT_WEIGHTS_TAG=v1.1.0

EXPOSE 8000

# The checkpoint is a release asset, so it is downloaded on first start instead
# of being baked in. Mount a volume at /app/weights to keep it between restarts.
# Hosts like Render and Hugging Face set PORT, so use it when it is there.
ENTRYPOINT ["./docker-entrypoint.sh"]
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
