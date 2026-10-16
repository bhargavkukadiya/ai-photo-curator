"""Optional backends are exercised in the CI extras job without model downloads."""

from __future__ import annotations

import pytest
from PIL import Image

from photo_curator.models import DEEPFACE_AVAILABLE, HEIC_SUPPORTED, get_deepface
from photo_curator.scoring import load_image


@pytest.mark.skipif(not DEEPFACE_AVAILABLE, reason="emotion extra is not installed")
def test_real_deepface_backend_initializes():
    assert get_deepface() is not None


@pytest.mark.skipif(not HEIC_SUPPORTED, reason="HEIC extra is not installed")
def test_real_heif_roundtrip(tmp_path):
    path = tmp_path / "photo.heic"
    Image.new("RGB", (30, 10), (100, 150, 200)).save(path, format="HEIF")
    pil, cv = load_image(path)
    assert pil.size == (30, 10)
    assert cv.shape == (10, 30, 3)
