"""
ForestSense — Evaluation Metrics
==================================
Paper-ready metrics for 4-class change detection segmentation.

Computed metrics:
  - Overall Accuracy (OA)
  - Per-class Precision, Recall, F1, IoU
  - Mean Precision, Recall, F1 (mF1), Mean IoU (mIoU)
  - Cohen's Kappa Coefficient
  - Confusion Matrix
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Tuple


CLASS_NAMES = [
    "Unchanged Non-Forest",
    "Stable Forest",
    "Forest Loss",
    "Forest Gain",
]


class ChangeDetectionMetrics:
    """
    Accumulates predictions over batches, computes all metrics at epoch end.

    Usage:
        metrics = ChangeDetectionMetrics(num_classes=4)
        for batch in loader:
            preds, labels = model(batch)
            metrics.update(preds, labels)
        results = metrics.compute()
        metrics.reset()
    """

    def __init__(self, num_classes: int = 4, ignore_index: int = -1):
        self.num_classes  = num_classes
        self.ignore_index = ignore_index
        self.reset()

    def reset(self):
        """Reset accumulated confusion matrix."""
        self.conf_matrix = np.zeros(
            (self.num_classes, self.num_classes), dtype=np.int64
        )

    def update(self, logits: torch.Tensor, targets: torch.Tensor):
        """
        Update confusion matrix with one batch.

        Args:
            logits : (B, C, H, W) — raw model output or probabilities
            targets: (B, H, W)   — ground truth class indices (long)
        """
        # Argmax for predictions
        preds = logits.argmax(dim=1)   # (B, H, W)

        # Flatten
        preds_np   = preds.cpu().numpy().ravel()
        targets_np = targets.cpu().numpy().ravel()

        # Filter ignore_index
        if self.ignore_index >= 0:
            valid  = targets_np != self.ignore_index
            preds_np   = preds_np[valid]
            targets_np = targets_np[valid]

        # Accumulate confusion matrix
        for t, p in zip(targets_np, preds_np):
            if 0 <= t < self.num_classes and 0 <= p < self.num_classes:
                self.conf_matrix[t, p] += 1

    def compute(self) -> Dict:
        """
        Compute all metrics from the accumulated confusion matrix.

        Returns:
            Dict with keys: OA, precision, recall, f1, iou, mPrecision,
                            mRecall, mF1, mIoU, kappa, conf_matrix
        """
        cm = self.conf_matrix.astype(np.float64)
        total = cm.sum()

        if total == 0:
            raise ValueError("No samples accumulated. Call update() first.")

        # Overall Accuracy
        oa = np.diag(cm).sum() / total

        # Per-class metrics
        precision = np.zeros(self.num_classes)
        recall    = np.zeros(self.num_classes)
        f1        = np.zeros(self.num_classes)
        iou       = np.zeros(self.num_classes)

        for c in range(self.num_classes):
            tp = cm[c, c]
            fp = cm[:, c].sum() - tp   # predicted as c, actually not c
            fn = cm[c, :].sum() - tp   # actually c, predicted as something else
            tn = total - tp - fp - fn

            precision[c] = tp / (tp + fp + 1e-8)
            recall[c]    = tp / (tp + fn + 1e-8)
            f1[c]        = 2 * precision[c] * recall[c] / (precision[c] + recall[c] + 1e-8)
            iou[c]       = tp / (tp + fp + fn + 1e-8)

        # Cohen's Kappa
        po = oa
        pe_num = (cm.sum(axis=0) * cm.sum(axis=1)).sum()
        pe = pe_num / (total ** 2 + 1e-8)
        kappa = (po - pe) / (1.0 - pe + 1e-8)

        # ── Class support weights (pixels per class / total) ──────────────────
        class_support = cm.sum(axis=1)
        class_weights = class_support / (class_support.sum() + 1e-8)
        weighted_f1   = float(np.dot(f1,  class_weights))
        weighted_iou  = float(np.dot(iou, class_weights))

        results = {
            # Overall
            "OA"               : float(round(oa * 100, 3)),
            "kappa"            : float(round(kappa, 4)),
            # Macro-average (all 4 classes equally)
            "mPrecision"       : float(round(precision.mean() * 100, 3)),
            "mRecall"          : float(round(recall.mean() * 100, 3)),
            "mF1"              : float(round(f1.mean() * 100, 3)),
            "mIoU"             : float(round(iou.mean() * 100, 3)),
            # Weighted-average (by class pixel frequency — real-world performance)
            "weighted_F1"      : float(round(weighted_f1 * 100, 3)),
            "weighted_IoU"     : float(round(weighted_iou * 100, 3)),
            # Change classes only (Loss=2, Gain=3) — most important for paper
            "mF1_change_only"  : float(round(f1[2:].mean() * 100, 3)),
            "mIoU_change_only" : float(round(iou[2:].mean() * 100, 3)),
            "change_F1"        : float(round(f1[2:].mean() * 100, 3)),   # alias
            "change_IoU"       : float(round(iou[2:].mean() * 100, 3)),  # alias
            # Per-class (%)
            "precision"        : {CLASS_NAMES[i]: round(precision[i] * 100, 3) for i in range(self.num_classes)},
            "recall"           : {CLASS_NAMES[i]: round(recall[i] * 100, 3) for i in range(self.num_classes)},
            "f1"               : {CLASS_NAMES[i]: round(f1[i] * 100, 3) for i in range(self.num_classes)},
            "iou"              : {CLASS_NAMES[i]: round(iou[i] * 100, 3) for i in range(self.num_classes)},
            # Raw confusion matrix
            "conf_matrix"      : self.conf_matrix.copy(),
        }
        return results


    def print_report(self, results: Optional[Dict] = None, epoch: int = -1):
        """Print a formatted metrics table (for paper / logging)."""
        if results is None:
            results = self.compute()

        header = f" Epoch {epoch} " if epoch >= 0 else ""
        print(f"\n{'='*60}")
        print(f"  📊 METRICS REPORT {header}")
        print(f"{'='*60}")
        print(f"  Overall Accuracy  : {results['OA']:.2f}%")
        print(f"  Mean F1-Score     : {results['mF1']:.2f}%")
        print(f"  Mean IoU          : {results['mIoU']:.2f}%")
        print(f"  Cohen's Kappa     : {results['kappa']:.4f}")
        print(f"  Change F1 (Cls2+3): {results['change_F1']:.2f}%")
        print(f"  Change IoU        : {results['change_IoU']:.2f}%")
        print(f"\n  {'Class':<25} {'Prec':>7} {'Rec':>7} {'F1':>7} {'IoU':>7}")
        print(f"  {'-'*53}")
        for cls in CLASS_NAMES:
            p = results['precision'][cls]
            r = results['recall'][cls]
            f = results['f1'][cls]
            u = results['iou'][cls]
            print(f"  {cls:<25} {p:>6.2f}% {r:>6.2f}% {f:>6.2f}% {u:>6.2f}%")
        print(f"{'='*60}\n")


def compute_batch_accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    """
    Fast per-batch pixel accuracy (for progress bar display during training).
    Does NOT accumulate — just a quick estimate.
    """
    preds = logits.argmax(dim=1)
    correct = (preds == targets).float().sum().item()
    total   = targets.numel()
    return correct / total * 100.0
