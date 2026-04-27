"""
ForestSense — Loss Functions
==============================
Combined Dice + Weighted Cross-Entropy loss for 4-class change detection.

Why this combination?
  - CrossEntropyLoss: Handles class imbalance via weights; good for per-pixel accuracy.
  - DiceLoss: Penalises poor overlap on rare classes (Forest Loss/Gain); better for F1.
  - Together they balance pixel-level and region-level accuracy.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class DiceLoss(nn.Module):
    """
    Multi-class Weighted Soft Dice Loss.
    Weights each class by inverse pixel frequency so rare classes
    (Forest Loss, Forest Gain) contribute more to the loss.

    Args:
        smooth     : Laplace smoothing to prevent divide-by-zero.
        ignore_index: Class index to ignore (set to -1 to disable).
    """

    def __init__(self, smooth: float = 1e-6, ignore_index: int = -1):
        super().__init__()
        self.smooth       = smooth
        self.ignore_index = ignore_index

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits : (B, C, H, W) — raw model output
            targets: (B, H, W)   — ground-truth class indices (long)
        Returns:
            Scalar weighted dice loss.
        """
        num_classes = logits.shape[1]
        probs = F.softmax(logits, dim=1)                          # (B, C, H, W)

        # One-hot encode targets → (B, C, H, W)
        targets_oh = F.one_hot(targets.clamp(0), num_classes)     # (B, H, W, C)
        targets_oh = targets_oh.permute(0, 3, 1, 2).float()       # (B, C, H, W)

        # Handle ignore_index: zero out those pixels in both pred and target
        if self.ignore_index >= 0:
            mask = (targets != self.ignore_index).unsqueeze(1).float()
            probs      = probs * mask
            targets_oh = targets_oh * mask

        # Dice per class: 2 * |P ∩ T| / (|P| + |T|)
        intersection = (probs * targets_oh).sum(dim=(0, 2, 3))    # (C,)
        cardinality  = probs.sum(dim=(0, 2, 3)) + targets_oh.sum(dim=(0, 2, 3))  # (C,)
        dice_per_cls = (2.0 * intersection + self.smooth) / (cardinality + self.smooth)

        # ── Weighted average: rare classes get higher weight ──────────────────
        # Class frequency = how many pixels of each class exist in this batch
        class_freq = targets_oh.sum(dim=(0, 2, 3)) + self.smooth  # (C,)
        # Inverse-frequency weights — normalised to sum to num_classes
        weights = 1.0 / class_freq
        weights = weights / weights.sum() * num_classes

        return 1.0 - (dice_per_cls * weights).sum() / weights.sum()


class FocalLoss(nn.Module):
    """
    Focal Loss for handling class imbalance.
    Focuses on hard-to-classify pixels.

    Args:
        alpha : Class weights (same format as CrossEntropyLoss weight).
        gamma : Focusing parameter. Higher = more focus on hard examples.
    """

    def __init__(
        self,
        alpha : Optional[torch.Tensor] = None,
        gamma : float = 2.0,
        reduction: str = 'mean',
    ):
        super().__init__()
        self.alpha     = alpha
        self.gamma     = gamma
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = F.cross_entropy(logits, targets, weight=self.alpha, reduction='none')
        pt = torch.exp(-ce_loss)                      # Probability of correct class
        focal = ((1 - pt) ** self.gamma) * ce_loss   # Downweight easy examples

        if self.reduction == 'mean':
            return focal.mean()
        elif self.reduction == 'sum':
            return focal.sum()
        return focal


class CombinedLoss(nn.Module):
    """
    Combined Dice + CrossEntropy (or Focal) loss.

    Loss = ce_weight × CE(logits, targets) + dice_weight × WeightedDice(logits, targets)

    Default weights: dice=0.7, ce=0.3
    Why? Forest change detection is a segmentation task where regional overlap
    (Dice) matters more than pixel-level accuracy (CE). Higher Dice weight
    also naturally upweights rare Forest Loss/Gain classes.

    Args:
        class_weights: (C,) tensor for cross-entropy class imbalance.
        dice_weight  : Weight for dice component (default 0.7).
        ce_weight    : Weight for cross-entropy component (default 0.3).
        use_focal    : If True, use Focal Loss instead of CE.
        focal_gamma  : Focal loss gamma parameter.
    """

    def __init__(
        self,
        class_weights : Optional[torch.Tensor] = None,
        dice_weight   : float = 0.7,   # Increased from 0.5 — Dice is more important for segmentation
        ce_weight     : float = 0.3,   # Reduced from 0.5
        use_focal     : bool  = False,
        focal_gamma   : float = 2.0,
    ):
        super().__init__()
        self.dice_weight = dice_weight
        self.ce_weight   = ce_weight
        self.dice_loss   = DiceLoss()

        if use_focal:
            self.ce_loss = FocalLoss(alpha=class_weights, gamma=focal_gamma)
        else:
            self.ce_loss = nn.CrossEntropyLoss(weight=class_weights)

    def forward(
        self,
        logits : torch.Tensor,
        targets: torch.Tensor,
    ) -> tuple:
        """
        Returns:
            total_loss : Scalar combined loss.
            ce_val     : Cross-entropy component (for logging).
            dice_val   : Dice component (for logging).
        """
        ce_val   = self.ce_loss(logits, targets)
        dice_val = self.dice_loss(logits, targets)
        total    = self.ce_weight * ce_val + self.dice_weight * dice_val
        return total, ce_val.detach(), dice_val.detach()


# ── Factory ────────────────────────────────────────────────────────────────────

def build_loss(
    class_weights : Optional[torch.Tensor] = None,
    use_focal     : bool = False,
) -> CombinedLoss:
    """
    Build the loss function.

    Args:
        class_weights: Tensor of shape (4,) from dataset stats.
                       Pass None to use uniform weights.
        use_focal    : Use Focal Loss instead of standard CE.
    """
    loss_fn = CombinedLoss(
        class_weights = class_weights,
        dice_weight   = 0.5,
        ce_weight     = 0.5,
        use_focal     = use_focal,
    )
    print(f"✅ Loss: 0.5 × {'Focal' if use_focal else 'CE'} + 0.5 × Dice")
    if class_weights is not None:
        print(f"   Class weights: {class_weights.tolist()}")
    return loss_fn
