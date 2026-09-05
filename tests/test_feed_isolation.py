"""Interleave two feeds through the tracked endpoint and prove their track_id
spaces and motion state stay independent.

Needs a live service:  uvicorn app.main:app --host 0.0.0.0 --port 8000
    pytest -m server tests/test_feed_isolation.py
"""
import cv2
import numpy as np
import pytest

BASE = "http://127.0.0.1:8000"
FEEDS = ("drone-01", "helmet-A")


@pytest.fixture
def api(bus_image):
    """Skip rather than fail when no server is listening -- absence of a
    running instance is not a regression in the code under test."""
    requests = pytest.importorskip("requests")
    try:
        requests.get(f"{BASE}/health", timeout=2).raise_for_status()
    except Exception:
        pytest.skip(f"no service on {BASE}")

    for feed in FEEDS:
        requests.delete(f"{BASE}/detect/state/{feed}")

    def post(feed, frame):
        ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        assert ok
        r = requests.post(f"{BASE}/detect/tracked", params={"source_id": feed},
                          files={"file": ("f.jpg", buf.tobytes(), "image/jpeg")})
        r.raise_for_status()
        return r.json()["detections"]

    return post


@pytest.mark.server
def test_two_feeds_keep_independent_state(api, bus_image):
    img = bus_image
    h, w = img.shape[:2]

    a = b = []
    for i in range(8):
        M = np.float32([[1, 0, i * 8], [0, 1, 0]])
        panned = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REPLICATE)

        a = api("drone-01", panned)     # this feed pans
        b = api("helmet-A", img)        # this feed never changes

    assert not any(d["moving"] for d in b), "a static feed must report nothing moving"
    assert any(d["moving"] for d in a), "a panning feed must report movement"
