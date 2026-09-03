import cv2
import numpy as np
from fastapi import APIRouter, File, HTTPException, Query, UploadFile
from starlette.concurrency import run_in_threadpool

from app.detector import detector
from app.schemas import DetectionResponse

router = APIRouter(prefix="/detect", tags=["detection"])


async def _decode(file: UploadFile) -> np.ndarray:
    raw = await file.read()
    # np.frombuffer wraps the uploaded bytes with no copy, and imdecode hands
    # back a BGR array ready for YOLO. No disk round-trip on the latency path.
    frame = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_COLOR)
    if frame is None:
        raise HTTPException(400, "Could not decode image: is it a valid JPEG/PNG?")
    return frame


@router.post("", response_model=DetectionResponse)
async def detect_image(
    file: UploadFile = File(..., description="JPEG or PNG frame"),
    source_id: str = Query("single", description="Feed identifier, e.g. drone-01"),
    view: str = Query("ground", description="Which checkpoint to classify with: "
                       "'ground' (default, VisDrone+WiderPerson) or 'drone' "
                       "(VisDrone-only, specialised for top-down aerial footage -- "
                       "falls back to the default model if no drone checkpoint is loaded)"),
):
    """Single-frame detection. Stateless: no track IDs, no motion flag."""
    if detector.model is None:
        raise HTTPException(503, "Model not loaded")

    frame = await _decode(file)
    # Inference is blocking and CPU/GPU bound. Off the event loop, or one slow
    # frame stalls every other request including /health.
    return await run_in_threadpool(detector.detect, frame, source_id, view)


@router.post("/tracked", response_model=DetectionResponse)
async def detect_tracked(
    file: UploadFile = File(...),
    source_id: str = Query(..., description="Feed identifier, keeps track state separate per feed"),
    view: str = Query("ground", description="Which checkpoint to classify with: "
                       "'ground' (default) or 'drone' -- see /detect"),
):
    """Frame-in-a-stream detection over HTTP. Returns track IDs and motion flags.
    Use the WebSocket endpoint instead for real feeds: this is for testing."""
    if detector.model is None:
        raise HTTPException(503, "Model not loaded")

    frame = await _decode(file)
    return await run_in_threadpool(detector.track, frame, source_id, view)


@router.delete("/state/{source_id}")
async def reset_source(source_id: str):
    """Clear tracking history for a feed: call when a drone reconnects."""
    detector.reset_source(source_id)
    return {"reset": source_id}
