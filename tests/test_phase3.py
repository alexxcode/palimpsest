"""Phase 3 acceptance tests — run with: docker compose run --rm api python tests/test_phase3.py"""
import numpy as np
from app.services.preprocessor import (
    apply_cloud_mask,
    extract_patches,
    normalize_pair,
    reconstruct_from_patches,
)

def test_normalize_pair():
    before = np.random.randint(0, 4096, (1024, 1024, 4), dtype=np.uint16)
    after  = np.random.randint(0, 4096, (1024, 1024, 4), dtype=np.uint16)

    b, a = normalize_pair(before, after)

    assert b.dtype == np.float32, "Output must be float32"
    assert a.dtype == np.float32, "Output must be float32"
    assert b.min() >= 0.0 and b.max() <= 1.0, f"before out of [0,1]: [{b.min():.4f}, {b.max():.4f}]"
    assert a.min() >= 0.0 and a.max() <= 1.0, f"after out of [0,1]: [{a.min():.4f}, {a.max():.4f}]"

    print(f"  normalize_pair  OK  before=[{b.min():.3f}, {b.max():.3f}]  after=[{a.min():.3f}, {a.max():.3f}]")
    return b, a


def test_extract_and_reconstruct(b):
    patches = extract_patches(b, patch_size=256, overlap=0.5)

    expected = 49   # 7x7 for 1024x1024 with step=128
    assert len(patches) == expected, f"Expected {expected} patches, got {len(patches)}"
    assert patches[0].shape == (256, 256, 4), f"Wrong patch shape: {patches[0].shape}"

    reconstructed = reconstruct_from_patches(patches, b.shape, overlap=0.5)

    assert reconstructed.shape == b.shape, f"Shape mismatch: {reconstructed.shape} vs {b.shape}"

    max_err = float(np.abs(b - reconstructed).max())
    assert max_err < 0.01, f"Reconstruction error too high: {max_err:.6f}"

    print(f"  extract_patches          OK  patches={len(patches)}")
    print(f"  reconstruct_from_patches OK  max_error={max_err:.6f}")


def test_apply_cloud_mask(b):
    # SCL band: mostly clear (4=vegetation), some clouds (9), some shadows (3)
    scl = np.full((1024, 1024), 4, dtype=np.uint8)
    scl[100:200, 100:200] = 9   # cloud high probability
    scl[300:350, 300:350] = 3   # cloud shadow

    masked = apply_cloud_mask(b, scl)

    assert masked.shape == b.shape, "Shape must be preserved"
    assert masked[150, 150].sum() == 0.0, "Cloud pixels must be zeroed"
    assert masked[320, 320].sum() == 0.0, "Shadow pixels must be zeroed"
    assert masked[500, 500].sum() > 0.0, "Clear pixels must be preserved"

    n_masked = int((masked == 0).all(axis=-1).sum())
    print(f"  apply_cloud_mask         OK  masked_pixels={n_masked}")


if __name__ == "__main__":
    print("=== Phase 3 — Preprocessing pipeline ===")
    np.random.seed(42)

    b, a = test_normalize_pair()
    test_extract_and_reconstruct(b)
    test_apply_cloud_mask(b)

    print("=== ALL TESTS PASSED ===")
