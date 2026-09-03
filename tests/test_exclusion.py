"""Verify the reference-image exclusion store: an uploaded reference image
should match itself, but not an unrelated image. Uses a throwaway store path
so this never touches the real weights/exclusions.json.
"""
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.exclusion import ExclusionStore

ASSETS = Path(__file__).parent / "assets"
TEST_STORE = Path("weights/exclusions_test.json")


def main():
    TEST_STORE.unlink(missing_ok=True)
    store = ExclusionStore(path=TEST_STORE)

    ref = cv2.imread(str(ASSETS / "bus.jpg"))
    assert ref is not None, "tests/assets/bus.jpg missing"

    store.add("bus", ref)
    print(f"Stored exclusions: {store.list()}")
    assert store.list() == ["bus"]

    same, name, sim = store.is_excluded(ref)
    print(f"Reference vs itself:   excluded={same}  match={name}  sim={sim:.3f}")
    assert same, "the exact reference image must match itself"

    blank = np.full((100, 100, 3), 255, dtype=np.uint8)
    diff, name2, sim2 = store.is_excluded(blank)
    print(f"Unrelated blank image: excluded={diff}  match={name2}  sim={sim2:.3f}")
    assert not diff, "an unrelated image should not match"

    assert store.remove("bus")
    assert store.list() == []

    empty_check, _, _ = store.is_excluded(ref)
    print(f"Empty store (post-remove): excluded={empty_check}")
    assert not empty_check, "an empty store must never exclude anything"

    TEST_STORE.unlink(missing_ok=True)
    print("PASS")


if __name__ == "__main__":
    main()
