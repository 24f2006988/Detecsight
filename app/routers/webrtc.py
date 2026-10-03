"""WebRTC transport for live feeds, an alternative to /ws/track/{source_id}.

WebRTC runs over UDP, so a late or lost frame is skipped and not waited for, which
suits a live overlay. Signaling is one HTTP POST (offer in, answer out). Results go
over a data channel labelled "detections", with the same JSON as the WebSocket path.
"""
import asyncio
import json

from aiortc import RTCPeerConnection, RTCSessionDescription
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from starlette.concurrency import run_in_threadpool

from app.detector import detector

router = APIRouter(prefix="/webrtc", tags=["streaming"])

# kept so shutdown in main.py can close every open connection
_peer_connections: set[RTCPeerConnection] = set()


class SessionDescription(BaseModel):
    sdp: str
    type: str


def _send(channel, payload: dict):
    # the channel can close while sending if the client drops, which is normal
    if channel is not None and channel.readyState == "open":
        try:
            channel.send(json.dumps(payload))
        except Exception:  # noqa: BLE001
            pass


@router.post("/offer/{source_id}")
async def offer(source_id: str, body: SessionDescription):
    """POST an SDP offer, get an SDP answer. source_id works like on /ws/track/{source_id}."""
    if detector.model is None:
        raise HTTPException(503, "Model not loaded")

    pc = RTCPeerConnection()
    _peer_connections.add(pc)
    state = {"channel": pc.createDataChannel("detections")}

    @pc.on("datachannel")
    def on_datachannel(channel):
        # a channel the client opens replaces ours
        if channel.label == "detections":
            state["channel"] = channel

    @pc.on("connectionstatechange")
    async def on_connectionstatechange():
        if pc.connectionState in ("failed", "closed", "disconnected"):
            detector.reset_source(source_id)
            _peer_connections.discard(pc)
            await pc.close()

    @pc.on("track")
    def on_track(track):
        if track.kind != "video":
            return

        async def consume():
            try:
                while True:
                    frame = await track.recv()
                    # bgr24 is what cv2 and ultralytics expect
                    img = frame.to_ndarray(format="bgr24")
                    # same call as the WebSocket path
                    result = await run_in_threadpool(detector.track, img, source_id)
                    _send(state["channel"], result)
            except Exception as e:  # noqa: BLE001
                # aiortc raises this when the client stops sending, which is normal
                print(f"[webrtc] feed {source_id} track ended: {e!r}")

        asyncio.ensure_future(consume())

    await pc.setRemoteDescription(RTCSessionDescription(sdp=body.sdp, type=body.type))
    answer = await pc.createAnswer()
    await pc.setLocalDescription(answer)

    return {"sdp": pc.localDescription.sdp, "type": pc.localDescription.type}


async def close_all():
    """Called from app/main.py's shutdown handler."""
    for pc in list(_peer_connections):
        await pc.close()
    _peer_connections.clear()
