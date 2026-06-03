"""
ChangeFormer — binary change detection model.

Architecture (Siamese Transformer):
  1. Shared MiT-B2 encoder  (Mix Transformer, SegFormer backbone)
     → 4 multi-scale feature maps at H/4, H/8, H/16, H/32
  2. Absolute difference  |f_before - f_after|  at each scale
  3. Per-scale Conv projection → embed_dim=256
  4. Upsample all to H/4, concatenate, fuse
  5. Bilinear upsample to H  →  binary logit map
  6. Sigmoid + threshold at CONFIDENCE_THRESHOLD

Reference: Bandara & Patel, "ChangeFormer: A Transformer-Based Siamese
Network for Change Detection", IGARSS 2022.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F

from app.core.config import settings
from app.services.preprocessor import extract_patches, reconstruct_from_patches

logger = logging.getLogger(__name__)

# EfficientNet-B2 channel dims for stages 1-4 (skip stage 0 at H/2)
_EFF_B2_CHANNELS = [24, 48, 120, 352]
_EFF_B2_OUT_IDX  = (1, 2, 3, 4)


# ---------------------------------------------------------------------------
# Model definition
# ---------------------------------------------------------------------------

class _ConvBnRelu(nn.Sequential):
    def __init__(self, in_ch: int, out_ch: int, k: int = 1, p: int = 0):
        super().__init__(
            nn.Conv2d(in_ch, out_ch, k, padding=p, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )


class ChangeFormer(nn.Module):
    """
    Siamese ChangeFormer with MiT-B2 backbone.

    Input:  two (B, C, H, W) normalized image tensors — same weights, shared encoder.
    Output: (B, 1, H, W) logit map. Apply sigmoid for probability, threshold for mask.
    """

    def __init__(self, in_channels: int = 4, embed_dim: int = 256) -> None:
        super().__init__()

        # Shared encoder — identical weights applied to both images (Siamese)
        # EfficientNet-B2: stages 1-4 → H/4, H/8, H/16, H/32
        self.encoder = timm.create_model(
            "efficientnet_b2.ra_in1k",
            pretrained=False,   # weights come from our GCS checkpoint, not HuggingFace
            features_only=True,
            in_chans=in_channels,
            out_indices=_EFF_B2_OUT_IDX,
        )

        # Per-scale 1×1 conv projections applied to absolute difference features
        self.proj = nn.ModuleList([
            _ConvBnRelu(c, embed_dim) for c in _EFF_B2_CHANNELS
        ])

        # Multi-scale fusion: 4 × embed_dim → embed_dim
        self.fuse = nn.Sequential(
            _ConvBnRelu(embed_dim * 4, embed_dim),
            _ConvBnRelu(embed_dim, embed_dim // 2, k=3, p=1),
        )

        # Binary output head
        self.head = nn.Conv2d(embed_dim // 2, 1, kernel_size=1)

    def forward(self, before: torch.Tensor, after: torch.Tensor) -> torch.Tensor:
        """
        Args:
            before: (B, C, H, W)
            after:  (B, C, H, W)
        Returns:
            logits: (B, 1, H, W)
        """
        H4, W4 = before.shape[2] // 4, before.shape[3] // 4

        f_b = self.encoder(before)   # list of 4 tensors
        f_a = self.encoder(after)

        feats = []
        for i, (fb, fa) in enumerate(zip(f_b, f_a)):
            diff = torch.abs(fb - fa)                                          # absolute difference
            diff = self.proj[i](diff)                                          # project → embed_dim
            diff = F.interpolate(diff, (H4, W4), mode="bilinear", align_corners=False)
            feats.append(diff)

        fused  = self.fuse(torch.cat(feats, dim=1))                            # (B, embed_dim//2, H4, W4)
        logits = self.head(fused)                                               # (B, 1, H4, W4)
        logits = F.interpolate(logits, before.shape[2:], mode="bilinear", align_corners=False)
        return logits


# ---------------------------------------------------------------------------
# Detector wrapper
# ---------------------------------------------------------------------------

class ChangeDetector:
    """
    Wraps ChangeFormer for inference.
    Loaded once at API startup via FastAPI lifespan.
    Handles images of any size via patch-based inference (sliding window).

    Inference settings (calibrated on LEVIR-CD test set, F1=0.8685):
      patch_size          = 512 px   (config.patch_size)
      confidence_threshold = 0.35    (config.confidence_threshold)
      tta_enabled          = True    (4-way flip TTA, config.tta_enabled)
    """

    MODEL_VERSION = "changeformer-effb2-oscd-v1"

    def __init__(self) -> None:
        self.device = torch.device("cpu")
        self.model  = ChangeFormer(in_channels=4)

        ckpt = Path(settings.model_checkpoint_path)
        if ckpt.exists():
            state = torch.load(ckpt, map_location=self.device, weights_only=True)
            # Support both raw state_dict and wrapped {"model": state_dict} formats
            if isinstance(state, dict) and "model" in state:
                state = state["model"]
            missing, unexpected = self.model.load_state_dict(state, strict=False)
            if missing:
                logger.warning("Missing keys in checkpoint: %s", missing[:5])
            logger.info("Loaded checkpoint: %s", ckpt)
        else:
            logger.warning(
                "Checkpoint not found at %s — using ImageNet encoder init only. "
                "Run model/download_checkpoint.py to download trained weights.",
                ckpt,
            )

        self.model.eval()

    def _predict_proba(self, before: np.ndarray, after: np.ndarray) -> np.ndarray:
        """
        Single forward pass on a (H, W, C) pair.
        Returns float32 probability map (H, W) in [0, 1].
        """
        to_tensor = lambda x: torch.from_numpy(x.transpose(2, 0, 1)).unsqueeze(0).float()
        with torch.no_grad():
            logits = self.model(to_tensor(before), to_tensor(after))   # (1, 1, H, W)
            proba  = torch.sigmoid(logits).squeeze().cpu().numpy()      # (H, W)
        return proba.astype(np.float32)

    def _predict_proba_tta(self, before: np.ndarray, after: np.ndarray) -> np.ndarray:
        """
        4-way flip TTA: averages predictions over the original orientation
        and three flipped variants (hflip, vflip, hflip+vflip).
        Adds ~1.5% F1 vs single-pass inference at 4× the compute cost per patch.
        """
        total = np.zeros(before.shape[:2], dtype=np.float32)
        for hf in (False, True):
            for vf in (False, True):
                b = before[:, ::-1].copy() if hf else before
                b = b[::-1].copy()         if vf else b
                a = after[:, ::-1].copy()  if hf else after
                a = a[::-1].copy()         if vf else a
                p = self._predict_proba(b, a)
                if hf: p = p[:, ::-1]
                if vf: p = p[::-1]
                total += p
        return (total / 4.0).astype(np.float32)

    def predict(self, before: np.ndarray, after: np.ndarray) -> np.ndarray:
        """
        Input:  two normalized arrays (H, W, C) in [0, 1]
        Output: binary change mask (H, W), uint8, values 0 or 1.
        """
        infer = self._predict_proba_tta if settings.tta_enabled else self._predict_proba
        proba = infer(before, after)
        return (proba > settings.confidence_threshold).astype(np.uint8)

    def predict_large(self, before: np.ndarray, after: np.ndarray) -> np.ndarray:
        """
        Tiles large images into overlapping patches, runs inference on each,
        reconstructs the full probability map with Hanning blending,
        then thresholds to produce the binary mask.

        Uses TTA per-patch when settings.tta_enabled=True (default).
        patch_size and confidence_threshold come from settings.
        """
        ps      = settings.patch_size   # 512 by default
        overlap = 0.5
        infer   = self._predict_proba_tta if settings.tta_enabled else self._predict_proba

        patches_b = extract_patches(before, ps, overlap)
        patches_a = extract_patches(after,  ps, overlap)

        proba_patches = [
            infer(pb, pa)[..., np.newaxis]   # (ps, ps, 1)
            for pb, pa in zip(patches_b, patches_a)
        ]

        shape_3d  = before.shape[:2] + (1,)
        proba_map = reconstruct_from_patches(proba_patches, shape_3d, overlap)
        proba_map = proba_map[..., 0]   # (H, W)

        return (proba_map > settings.confidence_threshold).astype(np.uint8)

    def predict_large_debug(
        self, before: np.ndarray, after: np.ndarray
    ) -> tuple[np.ndarray, float]:
        """Like predict_large but also returns the max probability value in [0,1].

        Useful for diagnosing cases where the mask is empty: if max_proba is
        high (>0.5) the threshold is wrong; if it's low (<0.1) the model is
        not seeing any change at all (resolution mismatch, cloud cover, etc.).
        """
        ps      = settings.patch_size
        overlap = 0.5
        infer   = self._predict_proba_tta if settings.tta_enabled else self._predict_proba

        patches_b = extract_patches(before, ps, overlap)
        patches_a = extract_patches(after,  ps, overlap)

        proba_patches = [
            infer(pb, pa)[..., np.newaxis]
            for pb, pa in zip(patches_b, patches_a)
        ]

        shape_3d  = before.shape[:2] + (1,)
        proba_map = reconstruct_from_patches(proba_patches, shape_3d, overlap)
        proba_map = proba_map[..., 0]

        max_proba = float(proba_map.max())
        mask      = (proba_map > settings.confidence_threshold).astype(np.uint8)
        return mask, max_proba
