"""Shared fixtures, and the one place the project root is put on sys.path.

`pip install -e .` makes the path insert unnecessary, but the tests are also
run straight out of a clone against an interpreter that has never installed
this package (that is how every command in the README works). Doing it here
keeps the bootstrap in one file instead of repeating it at the top of eight.
"""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

ASSETS = Path(__file__).parent / "assets"
FRAME_SHAPE = (240, 320, 3)  # h, w, c


@pytest.fixture
def assets_dir():
    return ASSETS


@pytest.fixture
def bus_image():
    """The one real image the suite uses. Skips rather than fails if absent."""
    import cv2

    path = ASSETS / "bus.jpg"
    img = cv2.imread(str(path))
    if img is None:
        pytest.skip(f"{path} missing")
    return img


# The synthetic static/pan clips live in tests/make_clips.py rather than in a
# fixture here: nothing in the suite consumes them today, they are regenerated
# by hand when the motion thresholds are retuned, and pytest never collects
# that file anyway (no test_ prefix). Two implementations of the same twenty
# lines would be worse than one.
