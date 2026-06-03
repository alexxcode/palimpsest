"""Unit tests for preprocessor — Phase 3."""
import numpy as np
import pytest

from app.services.preprocessor import (
    extract_patches,
    normalize_pair,
    reconstruct_from_patches,
)


@pytest.mark.skip(reason="Phase 3 not yet implemented")
def test_normalize_range():
    before = np.random.randint(0, 4096, (256, 256, 4), dtype=np.uint16)
    after = np.random.randint(0, 4096, (256, 256, 4), dtype=np.uint16)
    b, a = normalize_pair(before, after)
    assert b.min() >= 0.0 and b.max() <= 1.0
    assert a.min() >= 0.0 and a.max() <= 1.0


@pytest.mark.skip(reason="Phase 3 not yet implemented")
def test_extract_patches_count():
    image = np.zeros((1024, 1024, 4), dtype=np.float32)
    patches = extract_patches(image, patch_size=256, overlap=0.5)
    # 1024 with 256 patches and 0.5 overlap → 7 steps per axis → 49 patches
    assert len(patches) == 49
