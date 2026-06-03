"""
Evaluate ChangeFormer on the LEVIR-CD val or test set.

Usage (inside the Docker container):
  python model/evaluate.py                        # full test set, predict_large
  python model/evaluate.py --split val
  python model/evaluate.py --split test --fast    # centre-crop patches (quick)
  python model/evaluate.py --max-images 20        # quick sanity check

Full mode  — runs predict_large() on each 1024×1024 image (accurate boundary
             metrics, slow: ~6 s/image → ~13 min for 128 test images).
Fast mode  — runs a single 256×256 centre-crop batch (< 1 s total, used during
             training via the evaluate_model() helper imported by train.py).
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

import numpy as np
import torch
from torch.utils.data import DataLoader

from app.core.config import settings
from app.services.detector import ChangeDetector, ChangeFormer
from model.dataset import LEVIRDataset


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------

def _pixel_metrics(pred: np.ndarray, label: np.ndarray) -> dict:
    """
    Binary pixel-level metrics for one image.
    pred, label : (H, W) bool or uint8
    """
    p = pred.astype(bool)
    g = label.astype(bool)

    tp = int((p  &  g).sum())
    fp = int((p  & ~g).sum())
    fn = int((~p &  g).sum())
    tn = int((~p & ~g).sum())

    eps       = 1e-7
    precision = tp / (tp + fp + eps)
    recall    = tp / (tp + fn + eps)
    f1        = 2 * precision * recall / (precision + recall + eps)
    iou       = tp / (tp + fp + fn + eps)
    oa        = (tp + tn) / (tp + fp + fn + tn + eps)

    return dict(tp=tp, fp=fp, fn=fn, tn=tn,
                precision=precision, recall=recall, f1=f1, iou=iou, oa=oa)


def _aggregate(counts: dict) -> dict:
    """Compute global metrics from accumulated TP/FP/FN/TN."""
    tp, fp, fn, tn = counts["tp"], counts["fp"], counts["fn"], counts["tn"]
    eps       = 1e-7
    precision = tp / (tp + fp + eps)
    recall    = tp / (tp + fn + eps)
    f1        = 2 * precision * recall / (precision + recall + eps)
    iou       = tp / (tp + fp + fn + eps)
    oa        = (tp + tn) / (tp + fp + fn + tn + eps)
    return dict(precision=precision, recall=recall, f1=f1, iou=iou, oa=oa)


# ---------------------------------------------------------------------------
# evaluate_model — fast helper imported by train.py
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_model(model: ChangeFormer, dataset: LEVIRDataset) -> dict:
    """
    Centre-crop patch evaluation — fast, for use inside the training loop.
    Returns dict with f1, iou, precision, recall.
    """
    loader = DataLoader(dataset, batch_size=8, shuffle=False, num_workers=0)
    model.eval()

    tp = fp = fn = 0
    for before, after, labels in loader:
        logits = model(before, after)
        preds  = (torch.sigmoid(logits) > settings.confidence_threshold).long()
        tgts   = labels.long()

        tp += int((preds &  tgts).sum())
        fp += int((preds & ~tgts).sum())
        fn += int((~preds & tgts).sum())

    model.train()

    eps       = 1e-7
    precision = tp / (tp + fp + eps)
    recall    = tp / (tp + fn + eps)
    f1        = 2 * precision * recall / (precision + recall + eps)
    iou       = tp / (tp + fp + fn + eps)
    return dict(f1=f1, iou=iou, precision=precision, recall=recall)


# ---------------------------------------------------------------------------
# Main CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate ChangeFormer on LEVIR-CD")
    parser.add_argument("--split",           default="test", choices=["train", "val", "test"])
    parser.add_argument("--checkpoint",      default="checkpoints/changeformer_levir.pth")
    parser.add_argument("--data-root",       default="data/datasets", help="Path to LEVIR-CD splits")
    parser.add_argument("--threshold",       type=float, default=0.5, help="Sigmoid threshold")
    parser.add_argument("--threshold-sweep", action="store_true",
                        help="Sweep thresholds 0.3–0.8 and print F1/P/R table (fast mode only)")
    parser.add_argument("--tta",             action="store_true",
                        help="Test-time augmentation: average over 4 flips before threshold")
    parser.add_argument("--patch-size",      type=int, default=256,
                        help="Crop size for fast/sweep mode (default 256)")
    parser.add_argument("--max-images",      type=int, default=None, help="Limit for quick checks")
    parser.add_argument("--fast",            action="store_true",
                        help="Centre-crop patches instead of full predict_large")
    args = parser.parse_args()

    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {ckpt_path}\n"
            f"Train the model first:  python model/train.py"
        )

    # ------------------------------------------------------------------
    # Fast mode — batch inference on centre-crop patches
    # ------------------------------------------------------------------
    if args.fast or args.threshold_sweep:
        print(f"Loading model from {ckpt_path} ...")
        state = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        model = ChangeFormer(in_channels=4)
        sd    = state["model"] if isinstance(state, dict) and "model" in state else state
        model.load_state_dict(sd, strict=False)

        ds = LEVIRDataset(args.split, patch_size=args.patch_size, augment=False,
                          root=args.data_root)
        if args.max_images:
            ds.pairs = ds.pairs[: args.max_images]

        def _collect_proba(loader, model, tta: bool) -> tuple[torch.Tensor, torch.Tensor]:
            """Returns (proba [N,1,H,W], labels [N,1,H,W]) averaged over TTA if requested."""
            all_proba, all_labels = [], []
            model.eval()
            with torch.no_grad():
                for before, after, labels in loader:
                    if tta:
                        # 4-way flip TTA: sum probabilities then normalise
                        prob = torch.zeros_like(torch.sigmoid(model(before, after)))
                        for hf in (False, True):
                            for vf in (False, True):
                                b = before.flip(-1) if hf else before
                                b = b.flip(-2)      if vf else b
                                a = after.flip(-1)  if hf else after
                                a = a.flip(-2)      if vf else a
                                p = torch.sigmoid(model(b, a))
                                if hf: p = p.flip(-1)
                                if vf: p = p.flip(-2)
                                prob = prob + p
                        prob = prob / 4.0
                    else:
                        prob = torch.sigmoid(model(before, after))
                    all_proba.append(prob.cpu())
                    all_labels.append(labels.cpu())
            return torch.cat(all_proba), torch.cat(all_labels)

        if args.threshold_sweep:
            tta_tag = " (+TTA)" if args.tta else ""
            print(f"Collecting logits for {len(ds)} {args.split} patches{tta_tag} …")
            loader = DataLoader(ds, batch_size=8, shuffle=False, num_workers=0)
            proba, labels = _collect_proba(loader, model, args.tta)

            print(f"\n{'Thresh':>7}  {'F1':>7}  {'IoU':>7}  {'Prec':>7}  {'Recall':>7}")
            print("-" * 45)
            best_f1, best_thr = 0.0, 0.5
            for thr in [t / 100 for t in range(30, 85, 5)]:
                preds = (proba > thr).long()
                tgts  = labels.long()
                eps   = 1e-7
                tp = int((preds &  tgts).sum())
                fp = int((preds & ~tgts).sum())
                fn = int((~preds & tgts).sum())
                p  = tp / (tp + fp + eps)
                r  = tp / (tp + fn + eps)
                f1 = 2 * p * r / (p + r + eps)
                iou = tp / (tp + fp + fn + eps)
                marker = "  ←" if f1 > best_f1 else ""
                print(f"  {thr:.2f}   {f1:.4f}  {iou:.4f}  {p:.4f}  {r:.4f}{marker}")
                if f1 > best_f1:
                    best_f1, best_thr = f1, thr
            print("-" * 45)
            print(f"  Best F1={best_f1:.4f} at threshold={best_thr:.2f}")
            return

        tta_tag = " (+TTA)" if args.tta else ""
        print(f"Fast eval on {len(ds)} {args.split} {args.patch_size}px patches{tta_tag} …")
        loader = DataLoader(ds, batch_size=8, shuffle=False, num_workers=0)
        proba, labels = _collect_proba(loader, model, args.tta)
        preds = (proba > args.threshold).long()
        tgts  = labels.long()
        eps   = 1e-7
        tp = int((preds &  tgts).sum())
        fp = int((preds & ~tgts).sum())
        fn = int((~preds & tgts).sum())
        p  = tp / (tp + fp + eps)
        r  = tp / (tp + fn + eps)
        f1 = 2 * p * r / (p + r + eps)
        iou = tp / (tp + fp + fn + eps)
        print(f"\nF1={f1:.4f}  IoU={iou:.4f}  P={p:.4f}  R={r:.4f}")
        if args.split == "test":
            status = "✓  PASSES" if f1 >= 0.85 else "✗  BELOW TARGET"
            print(f"Phase target F1 ≥ 0.85: {status} ({f1:.4f})")
        return

    # ------------------------------------------------------------------
    # Full mode — predict_large on every full image
    # ------------------------------------------------------------------
    print(f"Loading ChangeDetector from {ckpt_path} ...")
    detector = ChangeDetector()

    ds    = LEVIRDataset(args.split, patch_size=None, augment=False, root=args.data_root)
    pairs = ds.pairs
    if args.max_images:
        pairs = pairs[: args.max_images]

    print(f"Full-image evaluation  split={args.split}  images={len(pairs)}\n")

    counts   = dict(tp=0, fp=0, fn=0, tn=0)
    t0_total = time.time()

    for i, (a_path, b_path, lbl_path) in enumerate(pairs, 1):
        before = LEVIRDataset.load_image(a_path)   # (1024, 1024, 4)
        after  = LEVIRDataset.load_image(b_path)
        label  = (LEVIRDataset.load_label(lbl_path) > 0.5).astype(np.uint8)

        t0   = time.time()
        pred = detector.predict_large(before, after)
        ms   = (time.time() - t0) * 1000

        m = _pixel_metrics(pred > args.threshold, label)
        for k in ("tp", "fp", "fn", "tn"):
            counts[k] += m[k]

        print(
            f"  [{i:3d}/{len(pairs)}] {a_path.stem[:22]:22s}  "
            f"F1={m['f1']:.4f}  IoU={m['iou']:.4f}  "
            f"P={m['precision']:.4f}  R={m['recall']:.4f}  {ms:.0f}ms"
        )

    elapsed = time.time() - t0_total
    g       = _aggregate(counts)

    print(f"\n{'═' * 62}")
    print(f"  Split : {args.split}   Images : {len(pairs)}   Time : {elapsed:.0f}s")
    print(f"{'─' * 62}")
    print(f"  F1        : {g['f1']:.4f}")
    print(f"  IoU       : {g['iou']:.4f}")
    print(f"  Precision : {g['precision']:.4f}")
    print(f"  Recall    : {g['recall']:.4f}")
    print(f"  OA        : {g['oa']:.4f}")
    print(f"{'═' * 62}")

    if args.split == "test":
        target = 0.85
        ok     = g["f1"] >= target
        status = "✓  PASSES" if ok else "✗  BELOW TARGET"
        print(f"  Phase target F1 ≥ {target}: {status} ({g['f1']:.4f})")


if __name__ == "__main__":
    main()
