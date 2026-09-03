"""Stream a video, camera, or network stream to a running server over WebRTC,
drawing the returned detections as a live overlay.

This is the real live-feed path added on top of app/routers/webrtc.py: this
script stands in for an AR client's camera + WebRTC sender, and the overlay
drawn here is exactly what a real client would build itself from the same
detection JSON (app/schemas.py DetectionResponse) -- nothing about the wire
format differs between this script and a real device.

Needs a running server (unlike scripts/annotate_video.py, which calls the
Detector directly with no networking):

    uvicorn app.main:app --host 0.0.0.0 --port 8000

Then, in another terminal:

    python scripts/webrtc_stream.py v1.mp4                    # file, live overlay window
    python scripts/webrtc_stream.py v1.mp4 --output out.mp4   # ...and record the overlay
    python scripts/webrtc_stream.py 0                          # webcam, live overlay
    python scripts/webrtc_stream.py --url http://192.168.1.5:8000 v1.mp4  # remote server

The overlay you see is deliberately not synced frame-exact to what was sent:
frames are captured and displayed continuously while detections arrive
asynchronously and get drawn as soon as they land, using whatever the latest
result is -- that is the actual behaviour a live AR client sees, not an
artifact of this script.
"""
import argparse
import asyncio
import sys
from pathlib import Path

import cv2
import httpx
from av import VideoFrame
from aiortc import RTCPeerConnection, RTCSessionDescription
from aiortc.mediastreams import VideoStreamTrack

sys.path.insert(0, str(Path(__file__).resolve().parent))
from annotate_video import draw_detections  # same overlay as the offline tool


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("video",
                   help="Video file, camera index (0, 1, ...), or stream URL "
                        "(rtsp://, http://)")
    p.add_argument("--url", default="http://127.0.0.1:8000",
                   help="Base URL of the running server (default: local)")
    p.add_argument("--source-id", default="webrtc-demo",
                   help="Feed identifier for server-side tracking state")
    p.add_argument("--output", default=None,
                   help="Write the overlaid video here. Defaults to "
                        "<video>_webrtc_annotated.mp4 for a file source; a live "
                        "source records nothing unless this is passed.")
    p.add_argument("--show", action="store_true",
                   help="Open a live preview window (implied if --output is not given)")
    return p.parse_args()


class QueueTrack(VideoStreamTrack):
    """Feeds frames pushed onto an asyncio.Queue as a WebRTC video track,
    decoupling capture pace from aiortc's own recv() pacing."""

    def __init__(self, queue: asyncio.Queue):
        super().__init__()
        self.queue = queue

    async def recv(self):
        pts, time_base = await self.next_timestamp()
        frame = await self.queue.get()
        vframe = VideoFrame.from_ndarray(frame, format="bgr24")
        vframe.pts = pts
        vframe.time_base = time_base
        return vframe


async def main():
    args = parse_args()

    is_camera = args.video.isdigit()
    is_stream = "://" in args.video
    live = is_camera or is_stream
    source = int(args.video) if is_camera else args.video
    label = f"camera {source}" if is_camera else args.video

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise SystemExit(f"Could not open {label}")

    if live:
        out_path = Path(args.output) if args.output else None
        show = args.show or out_path is None
    else:
        video_path = Path(args.video)
        out_path = Path(args.output) if args.output else video_path.with_name(
            f"{video_path.stem}_webrtc_annotated.mp4"
        )
        show = args.show

    writer = None
    if out_path is not None:
        fps = cap.get(cv2.CAP_PROP_FPS) or 25
        if not 1 <= fps <= 120:
            fps = 25
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"avc1"), fps, (w, h))
        if not writer.isOpened():
            print("Codec 'avc1' unavailable, falling back to mp4v")
            writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
        if not writer.isOpened():
            raise SystemExit(f"Could not open a video writer for {out_path}")

    frame_queue: asyncio.Queue = asyncio.Queue(maxsize=2)
    latest = {"detections": [], "inference_ms": None}

    pc = RTCPeerConnection()
    channel = pc.createDataChannel("detections")

    @channel.on("message")
    def on_message(msg):
        import json
        payload = json.loads(msg)
        latest["detections"] = payload["detections"]
        latest["inference_ms"] = payload["inference_ms"]

    pc.addTrack(QueueTrack(frame_queue))
    offer = await pc.createOffer()
    await pc.setLocalDescription(offer)

    print(f"Connecting to {args.url}/webrtc/offer/{args.source_id} ...")
    async with httpx.AsyncClient() as client:
        r = await client.post(
            f"{args.url}/webrtc/offer/{args.source_id}",
            json={"sdp": pc.localDescription.sdp, "type": pc.localDescription.type},
            timeout=10,
        )
        r.raise_for_status()
        answer = r.json()
    await pc.setRemoteDescription(RTCSessionDescription(sdp=answer["sdp"], type=answer["type"]))
    print("Connected. Streaming...")

    if show:
        window = f"BattleSight WebRTC preview - {label} (q to quit)"
        cv2.namedWindow(window, cv2.WINDOW_NORMAL)

    frame_no, consecutive_failures = 0, 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                if not live:
                    break
                consecutive_failures += 1
                if consecutive_failures > 30:
                    print("Live source stopped delivering frames; giving up.")
                    break
                continue
            consecutive_failures = 0

            # Non-blocking push: if the WebRTC track hasn't drained the last
            # frame yet, drop this one rather than build a backlog -- the
            # same "recency over completeness" behaviour the transport itself
            # gives you on a lossy network.
            if frame_queue.full():
                try:
                    frame_queue.get_nowait()
                except asyncio.QueueEmpty:
                    pass
            await frame_queue.put(frame)

            display = frame.copy()
            draw_detections(display, latest["detections"])
            if latest["inference_ms"] is not None:
                cv2.putText(display, f'frame {frame_no}  server: {latest["inference_ms"]:.1f} ms',
                            (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
            if writer is not None:
                writer.write(display)
            frame_no += 1

            if show:
                cv2.imshow(window, display)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    print(f"Stopped early at frame {frame_no}")
                    break
            else:
                await asyncio.sleep(0)  # yield to the aiortc send loop
    finally:
        cap.release()
        if writer is not None:
            writer.release()
        if show:
            cv2.destroyAllWindows()
        await pc.close()

    if out_path is not None:
        print(f"Wrote {frame_no} frames to {out_path}")
    else:
        print(f"Processed {frame_no} frames from {label} (not recorded)")


if __name__ == "__main__":
    asyncio.run(main())
