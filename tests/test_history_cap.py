"""The motion history must not grow without bound as track ids climb.

Marked `model` because constructing Detector imports ultralytics and torch,
not because it runs inference -- none of the assertions below touch the
network. Cheap, but it needs the model stack installed.

    pytest -m model tests/test_history_cap.py
"""
import pytest


@pytest.fixture
def detector():
    from app.detector import Detector

    return Detector()


@pytest.mark.model
def test_history_is_bounded_with_lru_eviction(detector):
    from app import config

    cap = config.MAX_TRACKS_PER_SOURCE
    for tid in range(cap * 3):
        detector._is_moving("drone-01", tid, 0.5, 0.5)

    feed = detector._history["drone-01"]
    assert len(feed) == cap, "history is not bounded"
    assert min(feed) == cap * 3 - cap, "eviction is not least-recently-seen"


@pytest.mark.model
def test_feeds_are_tracked_independently_and_reset(detector):
    from app import config

    cap = config.MAX_TRACKS_PER_SOURCE
    for tid in range(cap * 3):
        detector._is_moving("drone-01", tid, 0.5, 0.5)

    detector._is_moving("helmet-A", 1, 0.5, 0.5)
    assert len(detector._history["helmet-A"]) == 1

    detector.reset_source("drone-01")
    assert "drone-01" not in detector._history
