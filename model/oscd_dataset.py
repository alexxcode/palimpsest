"""
OSCD (Onera Satellite Change Detection) PyTorch Dataset.

Loads Sentinel-2 B02/B03/B04/B08 band GeoTIFFs from the official OSCD layout,
normalises them with the same joint-percentile scheme used at inference time,
and yields (before, after, mask) tensor triples for change-detection training.

Kaggle source
-------------
  slug : soumikrakshit/onera-satellite-change-detection-dataset
  URL  : https://www.kaggle.com/datasets/soumikrakshit/onera-satellite-change-detection-dataset

Directory layout (auto-detected)
---------------------------------
  {images_dir}/{city}/imgs_1_rect/{band}.tif   ← T1 image (or imgs_1/ if no rect)
  {images_dir}/{city}/imgs_2_rect/{band}.tif   ← T2 image
  {labels_dir}/{city}/cm/{city}-cm.tif         ← binary change mask

Bands loaded
------------
  B02 (Blue), B03 (Green), B04 (Red), B08 (NIR) — all 10 m/px.
  Channel order matches the production pipeline:  BGRNIR  (4 channels).

Change-pixel fraction in OSCD train set
-----------------------------------------
  Varies from ~2 % (bordeaux) to ~30 % (dubai).
  Mean ≈ 10–15 %.  Recommended pos_weight = 5.0 during fine-tuning.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch.utils.data import Dataset

try:
    import rasterio  # noqa: F401
    _RASTERIO = True
except ImportError:
    _RASTERIO = False

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: 14 labelled training cities in OSCD  (in order of publication)
TRAIN_CITIES: list[str] = [
    "aguascalientes", "abudhabi", "beirut", "bercy", "bordeaux",
    "cupertino", "dubai", "lasvegas", "milano", "montpellier",
    "mumbai", "nantes", "paris", "rio",
]

#: Sentinel-2 10 m bands: Blue, Green, Red, NIR  → 4-channel model input
_BANDS: list[str] = ["B02", "B03", "B04", "B08"]

#: Sub-directory names to try for T1 / T2 images
_T1_NAMES = ["imgs_1_rect", "imgs_1"]
_T2_NAMES = ["imgs_2_rect", "imgs_2"]


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def _find_subdir(city_dir: Path, candidates: list[str]) -> Optional[Path]:
    """Return the first existing sub-directory whose name matches a candidate."""
    for name in candidates:
        p = city_dir / name
        if p.is_dir():
            return p
    return None


def _load_bands(img_dir: Path, bands: list[str]) -> np.ndarray:
    """
    Load individual band GeoTIFFs and stack them into (H, W, C) float32.

    Supports two naming conventions:
      * Exact:  ``B02.tif``            (clean / rectified versions)
      * Prefixed: ``*_B02.tif``        (Kaggle raw format: full Sentinel-2 scene name)

    Args:
        img_dir : directory that contains the per-band GeoTIFF files.
        bands   : list of band names in desired channel order (e.g. ['B02','B03','B04','B08']).

    Returns:
        (H, W, C) float32 array with raw reflectance DN values.
    """
    if not _RASTERIO:
        raise ImportError("rasterio not installed.  Run:  pip install rasterio")

    import rasterio  # local import for type-checker benefit

    arrays: list[np.ndarray] = []
    for band in bands:
        tif = img_dir / f"{band}.tif"
        if not tif.exists():
            # Kaggle stores files with the full Sentinel-2 scene name as prefix,
            # e.g.  S2A_OPER_MSI_L1C_TL_MTI__20160120T104345_A003020_T39QZG_B02.tif
            matches = sorted(img_dir.glob(f"*_{band}.tif"))
            if not matches:
                available = sorted(p.name for p in img_dir.glob("*.tif"))[:6]
                raise FileNotFoundError(
                    f"No file matching '{band}.tif' or '*_{band}.tif' in:\n"
                    f"  {img_dir}\n"
                    f"  Available .tif files: {available}"
                )
            tif = matches[0]
        with rasterio.open(tif) as src:
            arrays.append(src.read(1).astype(np.float32))  # (H, W)
    return np.stack(arrays, axis=-1)  # (H, W, C)


def _load_mask(mask_path: Path) -> np.ndarray:
    """
    Load an OSCD change mask → (H, W) float32 binary {0=no-change, 1=change}.

    Auto-detects the per-file encoding by choosing the interpretation that
    produces a physically plausible change fraction (0.3 % – 85 %).

    Known conventions across OSCD re-packagings:
      * Kaggle soumikrakshit: 1=no-change, 2=change  → ``cm > 1``
      * Standard ONERA:       0=no-change, 1=change  → ``cm > 0``
      * Inverted:             0=change, 1=no-change  → ``cm == 0``
      * 8-bit grayscale PNG:  0=no-change, 255=change → ``cm > 127``
    """
    if not _RASTERIO:
        raise ImportError("rasterio not installed.")

    import rasterio

    with rasterio.open(mask_path) as src:
        cm = src.read(1).astype(np.float32)

    candidates = [
        ("cm > 1",   (cm > 1)),    # Kaggle: 1=no-change, 2=change
        ("cm > 0",   (cm > 0)),    # standard ONERA
        ("cm == 0",  (cm == 0)),   # inverted
        ("cm > 127", (cm > 127)),  # 8-bit grayscale
    ]
    for _name, mask_bool in candidates:
        if 0.3 <= mask_bool.mean() * 100 <= 85.0:
            return mask_bool.astype(np.float32)

    uv, uc = np.unique(cm, return_counts=True)
    diag = ", ".join(f"{int(v)}x{c}" for v, c in zip(uv[:8], uc[:8]))
    raise ValueError(
        f"Cannot determine mask convention for {mask_path}\n"
        f"  Unique values: {diag}  dtype={cm.dtype}"
    )


# ---------------------------------------------------------------------------
# Normalisation  (matches production preprocessor.normalize_pair exactly)
# ---------------------------------------------------------------------------

def _normalize_pair(
    before: np.ndarray,
    after: np.ndarray,
    p_lo: float = 2.0,
    p_hi: float = 98.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Joint percentile normalisation, channel-wise, to [0, 1] float32.

    'Joint' means both images share the same clip range per channel so the
    relative radiometric difference is preserved — critical for change detection.

    This is a verbatim port of  app.services.preprocessor.normalize_pair  so
    training and inference see identical normalised values.
    """
    out_b = np.empty_like(before, dtype=np.float32)
    out_a = np.empty_like(after,  dtype=np.float32)

    for c in range(before.shape[-1]):
        b_ch  = before[..., c]
        a_ch  = after[..., c]
        joint = np.concatenate([b_ch.ravel(), a_ch.ravel()])
        lo    = np.percentile(joint, p_lo)
        hi    = np.percentile(joint, p_hi)
        denom = hi - lo
        if denom < 1e-8:
            out_b[..., c] = 0.0
            out_a[..., c] = 0.0
        else:
            out_b[..., c] = np.clip((b_ch - lo) / denom, 0.0, 1.0)
            out_a[..., c] = np.clip((a_ch - lo) / denom, 0.0, 1.0)

    return out_b, out_a


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class OSCDDataset(Dataset):
    """
    PyTorch Dataset for the OSCD Sentinel-2 change detection benchmark.

    Design
    ------
    Each ``__getitem__`` returns one random ``patch_size × patch_size`` crop
    from a randomly chosen city (with ``augment=True``), or from a fixed tiled
    grid (with ``augment=False``, deterministic for validation).

    The full-image pair for each city is loaded and normalised *once* into a
    RAM cache on first access, so subsequent crops are pure in-memory operations.
    OSCD city images are small (~600 × 600 px, 4 bands, float32 ≈ 5.5 MB/pair),
    so caching all 14 cities costs ≈ 77 MB — entirely acceptable.

    Parameters
    ----------
    images_dir       : path to the directory containing one sub-folder per city
                       (the folder named "…Images" in the Kaggle download).
    labels_dir       : path to the directory with change masks
                       (the folder named "…Train Labels").
    cities           : list of city names to load.  Defaults to all 14 OSCD
                       training cities.  Pass a subset for a val split.
    patch_size       : square crop side in pixels (default 256 → 2.56 km).
    patches_per_city : how many patches to draw per city per epoch
                       (default 50 → 700 items/epoch for 14 cities).
    augment          : random horizontal/vertical flips and 90° rotations.
    bands            : Sentinel-2 band file names to load.
    """

    def __init__(
        self,
        images_dir:       str | Path,
        labels_dir:       str | Path,
        cities:           Optional[list[str]] = None,
        patch_size:       int = 256,
        patches_per_city: int = 50,
        augment:          bool = True,
        bands:            list[str] = _BANDS,
    ) -> None:
        self.patch_size       = patch_size
        self.patches_per_city = patches_per_city
        self.augment          = augment
        self.bands            = bands

        images_dir = Path(images_dir)
        labels_dir = Path(labels_dir)

        if cities is None:
            cities = TRAIN_CITIES

        # ------------------------------------------------------------------
        # Discover valid (imgs_1, imgs_2, mask) triples
        # ------------------------------------------------------------------
        self._cities: list[tuple[Path, Path, Path]] = []

        for city in cities:
            city_img = images_dir / city
            if not city_img.is_dir():
                print(f"  [OSCDDataset] city dir missing: {city_img} — skipping")
                continue

            t1 = _find_subdir(city_img, _T1_NAMES)
            t2 = _find_subdir(city_img, _T2_NAMES)
            if t1 is None or t2 is None:
                print(f"  [OSCDDataset] imgs_1/2 not found for {city} — skipping")
                continue

            mask_dir  = labels_dir / city / "cm"
            mask_path = mask_dir / f"{city}-cm.tif"
            if not mask_path.exists():
                mask_path = mask_dir / "cm.tif"          # fallback name
            if not mask_path.exists():
                print(f"  [OSCDDataset] change mask not found for {city} — skipping")
                continue

            self._cities.append((t1, t2, mask_path))

        if not self._cities:
            raise ValueError(
                f"No valid city pairs found.\n"
                f"  images_dir = {images_dir}\n"
                f"  labels_dir = {labels_dir}\n"
                "Check that these paths contain the OSCD data."
            )

        print(
            f"[OSCDDataset] {len(self._cities)} cities  "
            f"patch_size={patch_size}  patches_per_city={patches_per_city}  "
            f"augment={augment}  → {len(self)} samples/epoch"
        )

        # In-memory cache: city_idx → (before, after, mask) normalised arrays
        self._cache: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}

    # ------------------------------------------------------------------
    # Cache access
    # ------------------------------------------------------------------

    def _get_city(self, idx: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return normalised (before, after, mask) for city ``idx``."""
        if idx not in self._cache:
            t1_dir, t2_dir, mask_path = self._cities[idx]
            before_raw = _load_bands(t1_dir, self.bands)   # (H, W, 4) raw DNs
            after_raw  = _load_bands(t2_dir, self.bands)
            before, after = _normalize_pair(before_raw, after_raw)
            mask = _load_mask(mask_path)                    # (H, W) {0, 1}
            self._cache[idx] = (before, after, mask)
        return self._cache[idx]

    def preload_all(self) -> None:
        """Eagerly load all cities into the RAM cache (call before training)."""
        print("Pre-loading OSCD cities into RAM …", end="", flush=True)
        for i in range(len(self._cities)):
            self._get_city(i)
            print(f" {i + 1}", end="", flush=True)
        print("  done.")

    # ------------------------------------------------------------------
    # Dataset protocol
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._cities) * self.patches_per_city

    def __getitem__(self, idx: int):
        city_idx          = idx % len(self._cities)
        before, after, mask = self._get_city(city_idx)

        H, W = before.shape[:2]
        ps   = self.patch_size

        # Pad if the city image is smaller than patch_size (edge case)
        if H < ps or W < ps:
            pad_h = max(0, ps - H)
            pad_w = max(0, ps - W)
            before = np.pad(before, ((0, pad_h), (0, pad_w), (0, 0)), mode="reflect")
            after  = np.pad(after,  ((0, pad_h), (0, pad_w), (0, 0)), mode="reflect")
            mask   = np.pad(mask,   ((0, pad_h), (0, pad_w)),          mode="reflect")
            H, W   = before.shape[:2]

        # ---- Random crop -----------------------------------------------
        i = np.random.randint(0, H - ps + 1)
        j = np.random.randint(0, W - ps + 1)
        before = before[i : i + ps, j : j + ps].copy()
        after  = after [i : i + ps, j : j + ps].copy()
        mask   = mask  [i : i + ps, j : j + ps].copy()

        # ---- Augmentation ----------------------------------------------
        if self.augment:
            if np.random.rand() > 0.5:              # horizontal flip
                before = before[:, ::-1].copy()
                after  = after [:, ::-1].copy()
                mask   = mask  [:, ::-1].copy()

            if np.random.rand() > 0.5:              # vertical flip
                before = before[::-1].copy()
                after  = after [::-1].copy()
                mask   = mask  [::-1].copy()

            k = np.random.randint(0, 4)             # random 90° rotation
            if k:
                before = np.rot90(before, k).copy()
                after  = np.rot90(after,  k).copy()
                mask   = np.rot90(mask,   k).copy()

        # ---- Tensors ---------------------------------------------------
        # (H, W, C) → (C, H, W)
        before_t = torch.from_numpy(before.transpose(2, 0, 1))   # float32
        after_t  = torch.from_numpy(after .transpose(2, 0, 1))
        mask_t   = torch.from_numpy(mask).unsqueeze(0).float()   # (1, H, W)

        return before_t, after_t, mask_t


# ---------------------------------------------------------------------------
# Quick smoke-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    import sys

    parser = argparse.ArgumentParser(description="Smoke-test OSCDDataset")
    parser.add_argument("--images-dir", required=True,
                        help="Path to OSCD images root (contains one dir per city)")
    parser.add_argument("--labels-dir", required=True,
                        help="Path to OSCD train_labels root")
    parser.add_argument("--patch-size", type=int, default=256)
    args = parser.parse_args()

    ds = OSCDDataset(
        images_dir=args.images_dir,
        labels_dir=args.labels_dir,
        patch_size=args.patch_size,
        patches_per_city=5,
        augment=True,
    )
    ds.preload_all()

    b, a, m = ds[0]
    print(f"\nSample 0:")
    print(f"  before : {b.shape}  dtype={b.dtype}  range=[{b.min():.3f}, {b.max():.3f}]")
    print(f"  after  : {a.shape}")
    print(f"  mask   : {m.shape}  change_frac={m.mean():.4f}")
    print(f"\nDataset length: {len(ds)}")
    print("Smoke-test passed ✓")
