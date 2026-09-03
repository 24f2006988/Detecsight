"""Simulate a drone feed against the WebSocket endpoint."""
import asyncio
import json
import sys

import cv2
import websockets

VIDEO = sys.argv[1] if len(sys.argv) > 1 else "test_drone_clip.mp4"
URL = "ws://localhost:8000/ws/track/drone-01"


async def stream():
    cap = cv2.VideoCapture(VIDEO)
    if not cap.isOpened():
        print(f"Could not open {VIDEO}")
        return

    async with websockets.connect(URL, max_size=8 * 1024 * 1024) as ws:
        frame_no = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            if not ok:
                continue

            await ws.send(buf.tobytes())
            reply = json.loads(await ws.recv())

            moving = [d for d in reply["detections"] if d.get("moving")]
            print(f"frame {frame_no:>4}  {reply['inference_ms']:>6.1f} ms  "
                  f"{len(reply['detections'])} targets, {len(moving)} moving")
            frame_no += 1

    cap.release()


if __name__ == "__main__":
    asyncio.run(stream())
