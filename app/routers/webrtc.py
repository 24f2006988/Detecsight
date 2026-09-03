"""WebRTC transport for live feeds -- the low-latency alternative to
/ws/track/{source_id}.

WebSockets run over TCP: one dropped packet stalls the whole connection until
it is retransmitted (head-of-line blocking), and every frame pays a
base64/JPEG encode-decode round trip. WebRTC carries video over UDP/SRTP --
a lost or late frame is simply skipped, which is exactly what a live AR
overlay wants (recency over completeness), and it lets the client
hardware-encode instead of shipping raw JPEGs.

Signaling here is a single HTTP POST (offer in, answer out); there is no
separate signaling server to run. The client should open its own data channel
labelled "detections" before creating its offer -- this endpoint also opens
one itself, so a client that skips that step still gets results on whichever
channel comes up. Detection JSON on that channel has the exact same shape as
the WebSocket path (app/schemas.py DetectionResponse): nothing about
app/detector.py changes, only how frames arrive and results leave.

/ws/track/{source_id} is untouched and keeps working -- this is an additional
transport, not a replacement, so existing clients need no changes.
"""
import asyncio
import json

from aiortc import RTCPeerConnection, RTCSessionDescription
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from starlette.concurrency import run_in_threadpool

from app.detector import detector

router = APIRouter(prefix="/webrtc", tags=["streaming"])

# Tracked so app/main.py's shutdown can close every open connection instead of
# leaving UDP sockets and their per-frame inference loops running past exit.
_peer_connections: set[RTCPeerConnection] = set()


class SessionDescription(BaseModel):
    sdp: str
    type: str


def _send(channel, payload: dict):
    # The channel can close between this check and the send if the client
    # drops mid-frame -- a normal disconnect, not a server error.
    if channel is not None and channel.readyState == "open":
        try:
            channel.send(json.dumps(payload))
        except Exception:  # noqa: BLE001
            pass


@router.post("/offer/{source_id}")
async def offer(source_id: str, body: SessionDescription):
    """Standard WebRTC HTTP signaling: POST an SDP offer, get an SDP answer.

    source_id works exactly like on /ws/track/{source_id} -- it is the key
    into Detector's per-feed history/tracker/motion state, so multiple
    devices stay isolated from each other the same way they already do.
    """
    if detector.model is None:
        raise HTTPException(503, "Model not loaded")

    pc = RTCPeerConnection()
    _peer_connections.add(pc)
    state = {"channel": pc.createDataChannel("detections")}

    @pc.on("datachannel")
    def on_datachannel(channel):
        # A client-opened channel takes over from the server-opened one so
        # results end up wherever the client is actually listening.
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
                    # bgr24: what cv2/ultralytics expect everywhere else in
                    # this codebase (app/routers/track.py, app/routers/detect.py).
                    img = frame.to_ndarray(format="bgr24")
                    # Same blocking call, same lock, same four-stage cascade
                    # as the WebSocket path -- only the transport differs.
                    result = await run_in_threadpool(detector.track, img, source_id)
                    _send(state["channel"], result)
            except Exception as e:  # noqa: BLE001
                # Raised by aiortc as the normal way this loop ends when the
                # client stops sending (track ended) -- not worth more than a
                # log line.
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
