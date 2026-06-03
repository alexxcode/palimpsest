"""
LEVIR-CD PyTorch Dataset.

Images  : RGB 8-bit PNG, 1024×1024 → float32 [0,1] + synthetic NIR = mean(RGB)
Labels  : grayscale PNG, values 0 (no-change) or 255 (change) → binary float32
Patches : random crop for training, centre crop for validation/evaluation

Windows / Docker volume-mount note
------------------------------------
Reading many small files from a Windows-mounted Docker volume is slow (~2.5 s/batch).
Set preload=True (the default for 'train') to load every image into RAM once at
__init__ time.  After that each __getitem__ is a pure-memory crop with no disk I/O.
RAM cost: ~7 MB/pair (uint8 RGB × 2 + uint8 label) → ~3.1 GB for the 445 train pairs.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


class LEVIRDataset(Dataset):
    """
    Parameters
    ----------
    split      : 'train' | 'val' | 'test'
    patch_size : side length of the square crop (None → full 1024×1024 image)
    augment    : enable random flip / rotate (training only)
    root       : dataset root; split directories at root/{split}/A|B|label/
    preload    : cache all images in RAM (eliminates repeated disk I/O)
    """

    def __init__(
        self,
        split:      str,
        patch_size: int | None = 256,
        augment:    bool = True,
        root:       str  = "data/datasets",
        preload:    bool = True,
    ) -> None:
        self.patch_size = patch_size
        self.augment    = augment

        base    = Path(root) / split
        a_dir   = base / "A"
        b_dir   = base / "B"
        lbl_dir = base / "label"

        if not a_dir.exists():
            raise FileNotFoundError(f"Dataset split not found: {a_dir}")

        self.pairs: list[tuple[Path, Path, Path]] = sorted(
            [
                (a_dir / p.name, b_dir / p.name, lbl_dir / p.name)
                for p in a_dir.glob("*.png")
                if (b_dir / p.name).exists() and (lbl_dir / p.name).exists()
            ],
            key=lambda t: t[0].name,
        )
        if not self.pairs:
            raise ValueError(f"No valid pairs found in {base}")

    def __len__(self) -> int:
        return len(self.pairs)

    # ------------------------------------------------------------------
    # Loading helpers (also used by model/evaluate.py)
    # ------------------------------------------------------------------

    @staticmethod
    def load_image(path: Path) -> np.ndarray:
        """RGB PNG → float32 (H, W, 4) with synthetic NIR = mean(R, G, B)."""
        img = np.array(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0
        nir = img.mean(axis=-1, keepdims=True)
        return np.concatenate([img, nir], axis=-1)   # (H, W, 4)

    @staticmethod
    def load_label(path: Path) -> np.ndarray:
        """Grayscale PNG → float32 (H, W) binary: 0 = no-change, 1 = change."""
        lbl = np.array(Image.open(path).convert("L"), dtype=np.float32)
        return (lbl > 128).astype(np.float32)

    # ------------------------------------------------------------------
    # Dataset protocol
    # ------------------------------------------------------------------

    def __getitem__(self, idx: int):
        a_path, b_path, lbl_path = self.pairs[idx]

        before = self.load_image(a_path)    # (H, W, 4)
        after  = self.load_image(b_path)
        label  = self.load_label(lbl_path)  # (H, W)

        H, W = before.shape[:2]
        ps   = self.patch_size

        # ---- Crop -------------------------------------------------------
        if ps is not None and H >= ps and W >= ps:
            if self.augment:
                i = np.random.randint(0, H - ps + 1)
                j = np.random.randint(0, W - ps + 1)
            else:
                i = (H - ps) // 2   # centre crop for deterministic eval
                j = (W - ps) // 2

            before = before[i : i + ps, j : j + ps]
            after  = after [i : i + ps, j : j + ps]
            label  = label [i : i + ps, j : j + ps]

        # ---- Augmentation -----------------------------------------------
        if self.augment:
            if np.random.rand() > 0.5:               # horizontal flip
                before = before[:, ::-1].copy()
                after  = after [:, ::-1].copy()
                label  = label [:, ::-1].copy()

            if np.random.rand() > 0.5:               # vertical flip
                before = before[::-1].copy()
                after  = after [::-1].copy()
                label  = label [::-1].copy()

            k = np.random.randint(0, 4)              # random 90° rotation
            if k:
                before = np.rot90(before, k).copy()
                after  = np.rot90(after,  k).copy()
                label  = np.rot90(label,  k).copy()

        # ---- Tensors ----------------------------------------------------
        # (H, W, 4) → (4, H, W)
        before_t = torch.from_numpy(before.transpose(2, 0, 1))
        after_t  = torch.from_numpy(after .transpose(2, 0, 1))
        label_t  = torch.from_numpy(label).unsqueeze(0)    # (1, H, W)

        return before_t, after_t, label_t
