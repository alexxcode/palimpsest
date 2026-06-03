"""
Fine-tune ChangeFormer on LEVIR-CD.

Usage:
  python model/train.py                        # 50 epochs
  python model/train.py --epochs 100 --resume  # continue training
  python model/train.py --data-root /data/levir --gcs-upload  # VM workflow

GPU is used automatically when available (CUDA).  On a T4 GPU: ~15-25 s/epoch.
On CPU: ~260 s/epoch.

Class imbalance
---------------
LEVIR-CD has ~2.5% change pixels.  Addressed with:
  • weighted BCE  (pos_weight=10 → change pixels penalised 10×)
  • Dice loss     (invariant to class frequency)
  Combined: 0.5 × wBCE + 0.5 × Dice
"""
from __future__ import annotations

import argparse
import math
import os
import sys
import time
from pathlib import Path

sys.stdout.reconfigure(line_buffering=True)

_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from app.services.detector import ChangeFormer
from model.dataset import LEVIRDataset


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------

def dice_loss(logits: torch.Tensor, targets: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    proba = torch.sigmoid(logits)
    p = proba.reshape(-1)
    t = targets.reshape(-1)
    inter = (p * t).sum()
    return 1.0 - (2.0 * inter + eps) / (p.sum() + t.sum() + eps)


def combined_loss(
    logits:     torch.Tensor,
    targets:    torch.Tensor,
    pos_weight: float = 10.0,
) -> torch.Tensor:
    """0.5 × weighted-BCE  +  0.5 × Dice."""
    pw  = torch.tensor([pos_weight], device=logits.device)
    bce = F.binary_cross_entropy_with_logits(logits, targets, pos_weight=pw)
    dce = dice_loss(logits, targets)
    return 0.5 * bce + 0.5 * dce


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

@torch.no_grad()
def validate(
    model:      ChangeFormer,
    val_loader: DataLoader,
    device:     torch.device,
    pos_weight: float = 10.0,
    threshold:  float = 0.5,
) -> dict:
    model.eval()
    total_loss = 0.0
    tp = fp = fn = 0

    for before, after, labels in val_loader:
        before, after, labels = before.to(device), after.to(device), labels.to(device)
        logits = model(before, after)
        total_loss += combined_loss(logits, labels, pos_weight).item()

        preds = (torch.sigmoid(logits) > threshold).long()
        tgts  = labels.long()

        tp += int((preds &  tgts).sum())
        fp += int((preds & ~tgts).sum())
        fn += int((~preds & tgts).sum())

    model.train()

    eps       = 1e-7
    precision = tp / (tp + fp + eps)
    recall    = tp / (tp + fn + eps)
    f1        = 2 * precision * recall / (precision + recall + eps)
    iou       = tp / (tp + fp + fn + eps)

    return {
        "loss":      total_loss / max(len(val_loader), 1),
        "f1":        f1,
        "iou":       iou,
        "precision": precision,
        "recall":    recall,
    }


def evaluate_model(model: ChangeFormer, dataset: LEVIRDataset,
                   device: torch.device | None = None) -> dict:
    """Fast centre-crop evaluation — imported by evaluate.py."""
    if device is None:
        device = next(model.parameters()).device
    loader = DataLoader(dataset, batch_size=16, shuffle=False, num_workers=0)
    return validate(model, loader, device)


# ---------------------------------------------------------------------------
# GCS upload helper (optional, for VM workflow)
# ---------------------------------------------------------------------------

def _upload_checkpoint_to_gcs(local_path: Path, gcs_prefix: str) -> None:
    try:
        from google.cloud import storage as gcs
        from app.core.config import settings
        client = gcs.Client(project=settings.gcp_project_id)
        bucket = client.bucket(settings.gcs_bucket_name)
        blob_name = f"{gcs_prefix}{local_path.name}"
        bucket.blob(blob_name).upload_from_filename(str(local_path))
        print(f"  Checkpoint uploaded → gs://{settings.gcs_bucket_name}/{blob_name}")
    except Exception as exc:
        print(f"  GCS upload skipped: {exc}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Train ChangeFormer on LEVIR-CD")
    parser.add_argument("--epochs",       type=int,   default=50)
    parser.add_argument("--batch-size",   type=int,   default=16,   help="Per-GPU batch size")
    parser.add_argument("--lr",           type=float, default=6e-5, help="Peak AdamW LR")
    parser.add_argument("--patch-size",   type=int,   default=256)
    parser.add_argument("--pos-weight",   type=float, default=10.0, help="BCE positive-class weight")
    parser.add_argument("--val-interval", type=int,   default=5,    help="Validate every N epochs")
    parser.add_argument("--warmup",       type=int,   default=5,    help="LR warmup epochs")
    parser.add_argument("--workers",      type=int,   default=4,    help="DataLoader workers")
    parser.add_argument("--data-root",    default="data/datasets",  help="Path to LEVIR-CD splits")
    parser.add_argument("--checkpoint",   default="checkpoints/changeformer_levir.pth")
    parser.add_argument("--resume",       action="store_true")
    parser.add_argument("--gcs-upload",   action="store_true",
                        help="Upload best checkpoint to GCS after each save")
    args = parser.parse_args()

    ckpt_path = Path(args.checkpoint)
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Device
    # ------------------------------------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}", flush=True)
    if device.type == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------
    train_ds = LEVIRDataset("train", patch_size=args.patch_size, augment=True,
                            root=args.data_root, preload=False)
    val_ds   = LEVIRDataset("val",   patch_size=args.patch_size, augment=False,
                            root=args.data_root, preload=False)

    pin = device.type == "cuda"
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=pin, drop_last=True,
        persistent_workers=(args.workers > 0),
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=pin,
        persistent_workers=(args.workers > 0),
    )

    n_batches = len(train_loader)
    print(f"Train: {len(train_ds)} pairs  Val: {len(val_ds)} pairs")
    print(f"Batch: {args.batch_size}  Batches/epoch: {n_batches}  Workers: {args.workers}")

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------
    model       = ChangeFormer(in_channels=4).to(device)
    start_epoch = 1
    best_f1     = 0.0

    if args.resume and ckpt_path.exists():
        state = torch.load(ckpt_path, map_location=device, weights_only=True)
        if isinstance(state, dict) and "model" in state:
            model.load_state_dict(state["model"], strict=False)
            start_epoch = state.get("epoch", 0) + 1
            best_f1     = state.get("best_f1", 0.0)
            print(f"Resumed from epoch {start_epoch - 1}  best F1={best_f1:.4f}")
        else:
            model.load_state_dict(state, strict=False)

    model.train()
    total_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Parameters: {total_params:.1f}M")

    # ------------------------------------------------------------------
    # Optimiser + LR schedule (linear warmup → cosine decay to 1% of peak)
    # ------------------------------------------------------------------
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    def _lr_lambda(epoch: int) -> float:
        if epoch < args.warmup:
            return (epoch + 1) / max(args.warmup, 1)
        t = (epoch - args.warmup) / max(args.epochs - args.warmup, 1)
        return 0.01 + 0.99 * 0.5 * (1 + math.cos(math.pi * t))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, _lr_lambda)

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    print(f"\n{'Epoch':>6}  {'Loss':>8}  {'F1':>7}  {'IoU':>7}  "
          f"{'Prec':>7}  {'Recall':>7}  {'LR':>9}  {'s/ep':>6}")
    print("-" * 72)

    for epoch in range(start_epoch, args.epochs + 1):
        t0         = time.time()
        epoch_loss = 0.0

        for before, after, labels in train_loader:
            before = before.to(device, non_blocking=True)
            after  = after.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad()
            logits = model(before, after)
            loss   = combined_loss(logits, labels, pos_weight=args.pos_weight)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_loss += loss.item()

        scheduler.step()

        avg_loss = epoch_loss / n_batches
        elapsed  = time.time() - t0
        lr_now   = optimizer.param_groups[0]["lr"]

        if epoch % args.val_interval == 0 or epoch == args.epochs:
            m    = validate(model, val_loader, device, pos_weight=args.pos_weight)
            flag = ""

            if m["f1"] > best_f1:
                best_f1 = m["f1"]
                torch.save(
                    {"model": model.state_dict(), "epoch": epoch,
                     "best_f1": best_f1, "args": vars(args)},
                    ckpt_path,
                )
                flag = "  ← saved"
                if args.gcs_upload:
                    _upload_checkpoint_to_gcs(ckpt_path, "checkpoints/")

            print(
                f"{epoch:>6d}  {avg_loss:>8.4f}  {m['f1']:>7.4f}  "
                f"{m['iou']:>7.4f}  {m['precision']:>7.4f}  {m['recall']:>7.4f}  "
                f"{lr_now:>9.2e}  {elapsed:>5.0f}s{flag}"
            )
        else:
            print(f"{epoch:>6d}  {avg_loss:>8.4f}"
                  f"{'':>50}  {lr_now:>9.2e}  {elapsed:>5.0f}s")

    print("-" * 72)
    print(f"\nDone.  Best val F1 = {best_f1:.4f}   checkpoint → {ckpt_path}")
    if best_f1 >= 0.85:
        print("  ✓  F1 ≥ 0.85 target achieved!")
    else:
        print("  Target F1 ≥ 0.85 not reached — run with --resume to continue.")


if __name__ == "__main__":
    main()
