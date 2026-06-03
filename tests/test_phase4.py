"""
Phase 4 acceptance tests.
Run: docker compose run --rm api python tests/test_phase4.py

Step 1 — model loads (downloads MiT-B2 ImageNet weights on first run ~100MB)
Step 2 — predict() on a 256x256 patch → correct shape and binary values
Step 3 — predict_large() on a real 1024x1024 LEVIR-CD pair → correct shape
Step 4 — inference time per pair (ms)
"""

import time
from pathlib import Path

import numpy as np
from PIL import Image

from app.services.detector import ChangeDetector
from app.services.preprocessor import normalize_pair


def load_levir_pair(index: int = 1) -> tuple[np.ndarray, np.ndarray]:
    """Load a LEVIR-CD test pair as float32 (H, W, 3)."""
    base = Path("data/datasets/test")
    before = np.array(Image.open(base / "A" / f"test_{index}.png").convert("RGB"), dtype=np.float32)
    after  = np.array(Image.open(base / "B" / f"test_{index}.png").convert("RGB"), dtype=np.float32)
    return before, after


def make_4ch(img: np.ndarray) -> np.ndarray:
    """Add a synthetic NIR channel (mean of RGB) to get 4-channel input."""
    nir = img.mean(axis=-1, keepdims=True)
    return np.concatenate([img, nir], axis=-1)


def test_model_loads() -> ChangeDetector:
    print("  [1/4] Loading ChangeDetector (downloads MiT-B2 weights if not cached)...")
    t0 = time.time()
    detector = ChangeDetector()
    print(f"        Model loaded in {time.time() - t0:.1f}s")
    print(f"        Device: {detector.device}")
    total_params = sum(p.numel() for p in detector.model.parameters()) / 1e6
    print(f"        Parameters: {total_params:.1f}M")
    return detector


def test_predict_patch(detector: ChangeDetector) -> None:
    print("  [2/4] predict() on 256x256 synthetic patch...")
    rng = np.random.default_rng(0)
    before = rng.random((256, 256, 4), dtype=np.float32)
    after  = rng.random((256, 256, 4), dtype=np.float32)

    mask = detector.predict(before, after)

    assert mask.shape == (256, 256), f"Wrong shape: {mask.shape}"
    assert mask.dtype == np.uint8,   f"Wrong dtype: {mask.dtype}"
    assert set(np.unique(mask)).issubset({0, 1}), f"Non-binary values: {np.unique(mask)}"

    changed_pct = mask.mean() * 100
    print(f"        Output shape: {mask.shape}  dtype: {mask.dtype}  changed: {changed_pct:.1f}%")


def test_predict_large(detector: ChangeDetector) -> None:
    print("  [3/4] predict_large() on real LEVIR-CD 1024x1024 pair...")

    # Load real test pair
    before_rgb, after_rgb = load_levir_pair(index=1)
    before_4ch = make_4ch(before_rgb) / 255.0   # rough normalization for test
    after_4ch  = make_4ch(after_rgb)  / 255.0

    t0   = time.time()
    mask = detector.predict_large(before_4ch, after_4ch)
    elapsed_ms = (time.time() - t0) * 1000

    assert mask.shape == (1024, 1024), f"Wrong shape: {mask.shape}"
    changed_pct = mask.mean() * 100
    print(f"        Output shape: {mask.shape}  changed: {changed_pct:.1f}%")
    print(f"        Inference time (predict_large): {elapsed_ms:.0f} ms")


def test_normalize_then_predict(detector: ChangeDetector) -> None:
    print("  [4/4] Full pipeline: load → normalize_pair → predict_large...")

    before_rgb, after_rgb = load_levir_pair(index=2)
    before_4ch = make_4ch(before_rgb)
    after_4ch  = make_4ch(after_rgb)

    b_norm, a_norm = normalize_pair(before_4ch, after_4ch)

    t0   = time.time()
    mask = detector.predict_large(b_norm, a_norm)
    elapsed_ms = (time.time() - t0) * 1000

    print(f"        Normalized input range: [{b_norm.min():.3f}, {b_norm.max():.3f}]")
    print(f"        Change mask shape: {mask.shape}  changed: {mask.mean()*100:.1f}%")
    print(f"        Inference time: {elapsed_ms:.0f} ms")


if __name__ == "__main__":
    print("=== Phase 4 — ChangeFormer inference ===")

    detector = test_model_loads()
    test_predict_patch(detector)
    test_predict_large(detector)
    test_normalize_then_predict(detector)

    print("=== ALL TESTS PASSED ===")
