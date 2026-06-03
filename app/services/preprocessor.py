"""
Preprocessing pipeline for satellite image pairs.

All functions operate on numpy arrays.
Images are expected as (H, W, C) — channels last.
Masks / SCL bands are (H, W).
"""

import numpy as np

# Sentinel-2 SCL classes that represent unusable pixels
_CLOUD_MASK_CLASSES = frozenset([
    3,   # Cloud shadows
    8,   # Cloud medium probability
    9,   # Cloud high probability
    10,  # Thin cirrus
    11,  # Snow / Ice
])


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------

def normalize_pair(
    before: np.ndarray,
    after: np.ndarray,
    percentile_low: float = 2.0,
    percentile_high: float = 98.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Percentile-clipping normalization. Both images normalized jointly to [0, 1].

    'Jointly' means the clipping range is computed from the combined pixel
    distribution of both images, so the relative radiometric difference
    between them is preserved — critical for change detection.

    Args:
        before: (H, W, C) uint16 or float array — pre-event image.
        after:  (H, W, C) uint16 or float array — post-event image.
        percentile_low:  lower clip percentile (default 2).
        percentile_high: upper clip percentile (default 98).

    Returns:
        Tuple (before_norm, after_norm), both float32 in [0, 1].
    """
    before = before.astype(np.float32)
    after  = after.astype(np.float32)

    n_channels = before.shape[-1] if before.ndim == 3 else 1

    out_before = np.empty_like(before, dtype=np.float32)
    out_after  = np.empty_like(after,  dtype=np.float32)

    for c in range(n_channels):
        b_ch = before[..., c] if before.ndim == 3 else before
        a_ch = after[..., c]  if after.ndim  == 3 else after

        # Joint distribution: pool pixels from both images
        joint = np.concatenate([b_ch.ravel(), a_ch.ravel()])
        p_low  = np.percentile(joint, percentile_low)
        p_high = np.percentile(joint, percentile_high)

        denom = p_high - p_low
        if denom < 1e-8:
            # Flat channel — set to zero to avoid NaN
            if before.ndim == 3:
                out_before[..., c] = 0.0
                out_after[..., c]  = 0.0
            else:
                out_before[:] = 0.0
                out_after[:]  = 0.0
        else:
            if before.ndim == 3:
                out_before[..., c] = np.clip((b_ch - p_low) / denom, 0.0, 1.0)
                out_after[..., c]  = np.clip((a_ch - p_low) / denom, 0.0, 1.0)
            else:
                out_before[:] = np.clip((b_ch - p_low) / denom, 0.0, 1.0)
                out_after[:]  = np.clip((a_ch - p_low) / denom, 0.0, 1.0)

    return out_before, out_after


# ---------------------------------------------------------------------------
# Cloud masking
# ---------------------------------------------------------------------------

def apply_cloud_mask(image: np.ndarray, scl: np.ndarray) -> np.ndarray:
    """
    Zero out pixels classified as cloud, shadow, cirrus, or snow in the SCL band.

    Args:
        image: (H, W, C) float array — normalized image.
        scl:   (H, W) uint8 array — Sentinel-2 Scene Classification Layer.

    Returns:
        Copy of image with masked pixels set to 0.
    """
    bad_pixels = np.isin(scl, list(_CLOUD_MASK_CLASSES))   # (H, W) bool
    result = image.copy()
    result[bad_pixels] = 0.0
    return result


# ---------------------------------------------------------------------------
# Patch extraction / reconstruction
# ---------------------------------------------------------------------------

def _compute_padding(size: int, patch_size: int, step: int) -> int:
    """Return how many pixels to pad so the image divides evenly into patches.

    If ``size < patch_size`` we first promote the image to at least
    ``patch_size`` before the normal alignment calculation, ensuring exactly
    one patch is produced for images smaller than the patch size.
    """
    if size < patch_size:
        return patch_size - size
    remainder = (size - patch_size) % step
    return (step - remainder) % step


def _patch_positions(h: int, w: int, patch_size: int, step: int) -> list[tuple[int, int]]:
    """Return (row, col) top-left corners for every patch in row-major order."""
    positions = []
    for i in range(0, h - patch_size + 1, step):
        for j in range(0, w - patch_size + 1, step):
            positions.append((i, j))
    return positions


def extract_patches(
    image: np.ndarray,
    patch_size: int,
    overlap: float,
) -> list[np.ndarray]:
    """
    Extract overlapping patches from an image for large-image inference.

    The image is zero-padded (reflect mode) if its dimensions do not divide
    evenly with the given patch_size and overlap. Patches are yielded in
    row-major order — the same order reconstruct_from_patches expects.

    Args:
        image:      (H, W, C) float array.
        patch_size: side length of each square patch in pixels.
        overlap:    fraction of overlap between adjacent patches [0, 1).

    Returns:
        List of (patch_size, patch_size, C) float32 patches.
    """
    step = max(1, int(patch_size * (1.0 - overlap)))
    h, w = image.shape[:2]

    pad_h = _compute_padding(h, patch_size, step)
    pad_w = _compute_padding(w, patch_size, step)

    if pad_h > 0 or pad_w > 0:
        pad_cfg = ((0, pad_h), (0, pad_w)) if image.ndim == 2 else ((0, pad_h), (0, pad_w), (0, 0))
        image = np.pad(image, pad_cfg, mode="reflect")

    positions = _patch_positions(*image.shape[:2], patch_size, step)
    return [image[r:r + patch_size, c:c + patch_size].astype(np.float32)
            for r, c in positions]


def reconstruct_from_patches(
    patches: list[np.ndarray],
    shape: tuple,
    overlap: float,
) -> np.ndarray:
    """
    Reconstruct a full image from overlapping patches using weighted averaging.

    A 2D Hanning window is applied to each patch before accumulation so that
    patch borders are down-weighted and seams are invisible.

    Args:
        patches: list of (patch_size, patch_size[, C]) float32 arrays,
                 in the same row-major order produced by extract_patches.
        shape:   (H, W[, C]) of the original (unpadded) image.
        overlap: same value that was used in extract_patches.

    Returns:
        Reconstructed float32 array of the original shape.
    """
    patch_size = patches[0].shape[0]
    step = max(1, int(patch_size * (1.0 - overlap)))
    h, w = shape[:2]

    pad_h = _compute_padding(h, patch_size, step)
    pad_w = _compute_padding(w, patch_size, step)
    h_pad, w_pad = h + pad_h, w + pad_w

    is_3d = patches[0].ndim == 3
    n_ch  = patches[0].shape[2] if is_3d else 1

    accum   = np.zeros((h_pad, w_pad, n_ch), dtype=np.float64)
    weights = np.zeros((h_pad, w_pad),        dtype=np.float64)

    # 2D Hanning window — tapers at edges but never reaches zero.
    # np.hanning(n) is 0 at index 0 and n-1; using n+2 and slicing [1:-1]
    # shifts the window so all values are strictly > 0, avoiding zero-weight
    # pixels at image borders that would reconstruct as 0.
    win_1d = np.hanning(patch_size + 2)[1:-1].astype(np.float64)
    win_2d = np.outer(win_1d, win_1d)          # (patch_size, patch_size)

    positions = _patch_positions(h_pad, w_pad, patch_size, step)

    for (r, c), patch in zip(positions, patches):
        p = patch[..., np.newaxis] if not is_3d else patch   # ensure 3D
        accum[r:r + patch_size, c:c + patch_size]   += p.astype(np.float64) * win_2d[..., np.newaxis]
        weights[r:r + patch_size, c:c + patch_size] += win_2d

    # Avoid division by zero at corners (shouldn't happen with valid overlap)
    weights = np.maximum(weights, 1e-8)
    result  = accum / weights[..., np.newaxis]          # (H_pad, W_pad, C)

    # Crop padding and restore original dimensionality
    result = result[:h, :w, :]
    if not is_3d:
        result = result[..., 0]

    return result.astype(np.float32)
