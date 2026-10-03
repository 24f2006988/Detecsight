import json

import cv2
import numpy as np
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from starlette.concurrency import run_in_threadpool

from app.detector import detector

router = APIRouter(tags=["streaming"])

# Close codes (RFC 6455): 1011 = server hit an unexpected condition.
WS_INTERNAL_ERROR = 1011


@router.websocket("/ws/track/{source_id}")
async def ws_track(websocket: WebSocket, source_id: str, view: str = "ground"):
    """Live feed. The client sends one JPEG per binary message and gets one JSON
    result back. view is a query param (?view=drone) and stays fixed for the connection."""
    await websocket.accept()

    if detector.model is None:
        # otherwise the first frame fails inside the detector and the client just sees the socket drop
        await websocket.send_text(json.dumps({"error": "model_not_loaded"}))
        await websocket.close(code=WS_INTERNAL_ERROR)
        return

    print(f"[ws] feed connected: {source_id}")

    try:
        while True:
            raw = await websocket.receive_bytes()
            frame = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)

            if frame is None:
                await websocket.send_text(json.dumps({"error": "decode_failed"}))
                continue

            # keep blocking inference off the event loop
            result = await run_in_threadpool(detector.track, frame, source_id, view)
            await websocket.send_text(json.dumps(result))

    except WebSocketDisconnect:
        print(f"[ws] feed disconnected: {source_id}")
    except Exception as e:  # noqa: BLE001
        # A decode/inference failure must not leak this feed's tracker state.
        print(f"[ws] feed {source_id} failed: {e!r}")
        try:
            await websocket.close(code=WS_INTERNAL_ERROR)
        except RuntimeError:
            pass  # already closed
    finally:
        # Runs on every exit path, so a crashed feed frees its state too.
        detector.reset_source(source_id)
