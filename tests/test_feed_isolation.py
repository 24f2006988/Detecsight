"""Interleave two feeds through the tracked endpoint and prove their track_id
spaces and motion state stay independent."""
import cv2
import requests

BASE = "http://127.0.0.1:8000"

img = cv2.imread("tests/assets/bus.jpg")
h, w = img.shape[:2]

for feed in ("drone-01", "helmet-A"):
    requests.delete(f"{BASE}/detect/state/{feed}")


def post(feed, frame):
    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
    r = requests.post(f"{BASE}/detect/tracked", params={"source_id": feed},
                      files={"file": ("f.jpg", buf.tobytes(), "image/jpeg")})
    r.raise_for_status()
    return r.json()["detections"]


import numpy as np

print(f"{'frame':<6}{'drone-01 (panning)':<44}{'helmet-A (static)'}")
for i in range(8):
    M = np.float32([[1, 0, i * 8], [0, 1, 0]])
    panned = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REPLICATE)

    a = post("drone-01", panned)     # this feed pans
    b = post("helmet-A", img)        # this feed never changes

    fa = sorted((d["track_id"], d["moving"]) for d in a)
    fb = sorted((d["track_id"], d["moving"]) for d in b)
    print(f"{i:<6}{str(fa):<44}{fb}")

ids_a = {d["track_id"] for d in a}
ids_b = {d["track_id"] for d in b}
mov_b = [d["moving"] for d in b]
print()
print("drone-01 final ids :", sorted(ids_a))
print("helmet-A final ids :", sorted(ids_b))
print("helmet-A any moving:", any(mov_b), "(expected False)")
print("drone-01 any moving:", any(d["moving"] for d in a), "(expected True)")
