"""Unit tests for ChangeDetector — Phase 4."""
import numpy as np
import pytest

from app.services.detector import ChangeDetector


@pytest.mark.skip(reason="Phase 4 not yet implemented")
def test_predict_output_shape():
    detector = ChangeDetector()
    before = np.random.rand(256, 256, 4).astype(np.float32)
    after = np.random.rand(256, 256, 4).astype(np.float32)
    mask = detector.predict(before, after)
    assert mask.shape == (256, 256)
    assert set(np.unique(mask)).issubset({0, 1})
