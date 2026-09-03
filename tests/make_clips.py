"""Generate two synthetic drone-style clips to exercise the motion filter.

  static.mp4  - the same frame repeated; every track should stay moving=False
  moving.mp4  - the scene pans a few px per frame; tracks should flip to True
"""
import cv2
import numpy as np

SRC = "tests/assets/bus.jpg"
N_FRAMES = 20

img = cv2.imread(SRC)
h, w = img.shape[:2]
fourcc = cv2.VideoWriter_fourcc(*"mp4v")

# --- static clip -------------------------------------------------------
out = cv2.VideoWriter("tests/assets/static.mp4", fourcc, 10, (w, h))
for _ in range(N_FRAMES):
    out.write(img)
out.release()

# --- moving clip: translate the scene 6 px right per frame -------------
out = cv2.VideoWriter("tests/assets/moving.mp4", fourcc, 10, (w, h))
for i in range(N_FRAMES):
    M = np.float32([[1, 0, i * 6], [0, 1, 0]])
    out.write(cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REPLICATE))
out.release()

print(f"wrote 2 clips, {N_FRAMES} frames each, {w}x{h}")
print("per-frame shift on moving.mp4:", round(6 / w, 4), "normalised (threshold 0.015)")
