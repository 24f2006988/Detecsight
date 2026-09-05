"""Verify the reference-image exclusion store: an uploaded reference image
should match itself, but not an unrelated image. Uses a throwaway store path
so this never touches the real weights/exclusions.json.

Marked `model` because ExclusionStore embeds images with MobileNetV3, which
pulls in torch and torchvision.

    pytest -m model tests/test_exclusion.py
"""
import numpy as np
import pytest


@pytest.fixture
def store(tmp_path):
    from app.exclusion import ExclusionStore

    return ExclusionStore(path=tmp_path / "exclusions_test.json")


@pytest.mark.model
def test_reference_matches_itself_but_not_an_unrelated_image(store, bus_image):
    store.add("bus", bus_image)
    assert store.list() == ["bus"]

    same, _name, _sim = store.is_excluded(bus_image)
    assert same, "the exact reference image must match itself"

    blank = np.full((100, 100, 3), 255, dtype=np.uint8)
    diff, _name2, _sim2 = store.is_excluded(blank)
    assert not diff, "an unrelated image should not match"


@pytest.mark.model
def test_empty_store_never_excludes(store, bus_image):
    store.add("bus", bus_image)
    assert store.remove("bus")
    assert store.list() == []

    empty_check, _, _ = store.is_excluded(bus_image)
    assert not empty_check, "an empty store must never exclude anything"
