"""
ForestSense — Visualization Utilities
======================================
Publication-ready plotting helpers used throughout the project.
Enforces a consistent dark-theme aesthetic for all figures.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.colors import ListedColormap, Normalize
from mpl_toolkits.axes_grid1 import make_axes_locatable
from typing import Optional, List
import os


# ── Style Constants ───────────────────────────────────────────
DARK_BG      = "#0f0f1a"
PANEL_BG     = "#1a1a2e"
TEXT_COLOR   = "#e8e8f0"
ACCENT_COLOR = "#7c3aed"

LABEL_CMAP   = ListedColormap(["#2c2c2c", "#2ecc71", "#e74c3c", "#3498db"])
LABEL_NAMES  = ["Unchanged Non-Forest", "Stable Forest", "Forest Loss", "Forest Gain"]
LABEL_COLORS = ["#2c2c2c", "#2ecc71", "#e74c3c", "#3498db"]

plt.rcParams.update({
    "font.family"        : "DejaVu Sans",
    "font.size"          : 10,
    "axes.facecolor"     : PANEL_BG,
    "axes.edgecolor"     : "#333355",
    "axes.labelcolor"    : TEXT_COLOR,
    "xtick.color"        : TEXT_COLOR,
    "ytick.color"        : TEXT_COLOR,
    "text.color"         : TEXT_COLOR,
    "figure.facecolor"   : DARK_BG,
    "grid.color"         : "#2a2a4a",
    "grid.alpha"         : 0.5,
    "legend.facecolor"   : PANEL_BG,
    "legend.edgecolor"   : "#333355",
    "legend.labelcolor"  : TEXT_COLOR,
})


def save_figure(fig, path: str, dpi: int = 200):
    """Save figure with consistent facecolor and tight layout."""
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor=DARK_BG)
    plt.close(fig)
    print(f"  💾 Saved → {path}")


def label_legend_handles() -> List[mpatches.Patch]:
    """Return matplotlib patch handles for the 4 change classes."""
    return [
        mpatches.Patch(facecolor=LABEL_COLORS[i], label=LABEL_NAMES[i])
        for i in range(4)
    ]


def plot_change_panel(
    t1_rgb    : np.ndarray,
    t2_rgb    : np.ndarray,
    t1_sar    : np.ndarray,
    t2_sar    : np.ndarray,
    pred_mask : np.ndarray,
    gt_mask   : Optional[np.ndarray] = None,
    ndvi_t1   : Optional[np.ndarray] = None,
    ndvi_t2   : Optional[np.ndarray] = None,
    title     : str = "Change Detection Result",
    save_path : Optional[str] = None,
) -> plt.Figure:
    """
    Publication-ready panel showing:
    Row 1: T1 RGB | T2 RGB | Prediction
    Row 2: T1 SAR | T2 SAR | Ground Truth (or NDVI comparison)
    """
    n_cols = 3
    n_rows = 2 if gt_mask is not None or ndvi_t1 is not None else 1

    fig = plt.figure(figsize=(n_cols * 4, n_rows * 4 + 0.5), facecolor=DARK_BG)
    fig.suptitle(title, color=TEXT_COLOR, fontsize=13, fontweight="bold", y=1.01)
    gs = gridspec.GridSpec(n_rows, n_cols, figure=fig, hspace=0.25, wspace=0.15)

    # Row 1
    ax_t1  = fig.add_subplot(gs[0, 0])
    ax_t2  = fig.add_subplot(gs[0, 1])
    ax_pred = fig.add_subplot(gs[0, 2])

    ax_t1.imshow(np.clip(t1_rgb, 0, 1))
    ax_t1.set_title("T1 (2021) True Color", color=TEXT_COLOR, fontsize=9)
    ax_t1.axis("off")

    ax_t2.imshow(np.clip(t2_rgb, 0, 1))
    ax_t2.set_title("T2 (2023) True Color", color=TEXT_COLOR, fontsize=9)
    ax_t2.axis("off")

    ax_pred.imshow(pred_mask, cmap=LABEL_CMAP, vmin=0, vmax=3, interpolation="nearest")
    ax_pred.set_title("Predicted Change Map", color=TEXT_COLOR, fontsize=9)
    ax_pred.axis("off")

    # Row 2
    if n_rows == 2:
        ax_sar1 = fig.add_subplot(gs[1, 0])
        ax_sar2 = fig.add_subplot(gs[1, 1])
        ax_gt   = fig.add_subplot(gs[1, 2])

        ax_sar1.imshow(t1_sar, cmap="gray")
        ax_sar1.set_title("T1 SAR (VV)", color=TEXT_COLOR, fontsize=9)
        ax_sar1.axis("off")

        ax_sar2.imshow(t2_sar, cmap="gray")
        ax_sar2.set_title("T2 SAR (VV)", color=TEXT_COLOR, fontsize=9)
        ax_sar2.axis("off")

        if gt_mask is not None:
            ax_gt.imshow(gt_mask, cmap=LABEL_CMAP, vmin=0, vmax=3, interpolation="nearest")
            ax_gt.set_title("Ground Truth", color=TEXT_COLOR, fontsize=9)
        elif ndvi_t1 is not None and ndvi_t2 is not None:
            ndvi_diff = (ndvi_t2 * 2 - 1) - (ndvi_t1 * 2 - 1)    # rescale to [-1,1]
            im = ax_gt.imshow(ndvi_diff, cmap="RdYlGn", vmin=-0.5, vmax=0.5)
            ax_gt.set_title("ΔNDVI (T2 - T1)", color=TEXT_COLOR, fontsize=9)
            plt.colorbar(im, ax=ax_gt, fraction=0.046, pad=0.04)
        ax_gt.axis("off")

    # Legend
    fig.legend(
        handles   = label_legend_handles(),
        loc       = "lower center",
        ncol      = 4,
        fontsize  = 8,
        framealpha= 0.15,
        bbox_to_anchor = (0.5, -0.04),
    )

    if save_path:
        save_figure(fig, save_path)
    return fig


def plot_confusion_matrix(
    cm        : np.ndarray,
    class_names: List[str] = LABEL_NAMES,
    save_path : Optional[str] = None,
    title     : str = "Confusion Matrix",
) -> plt.Figure:
    """Plot a colour-coded confusion matrix suitable for a paper figure."""
    import seaborn as sns

    fig, ax = plt.subplots(figsize=(6, 5), facecolor=DARK_BG)
    cm_norm = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-8)

    sns.heatmap(
        cm_norm,
        annot        = cm,           # raw counts as annotation
        fmt          = "d",
        cmap         = "viridis",
        ax           = ax,
        linewidths   = 0.5,
        linecolor    = DARK_BG,
        xticklabels  = [n.split()[0] for n in class_names],
        yticklabels  = [n.split()[0] for n in class_names],
        vmin         = 0,
        vmax         = 1,
        cbar_kws     = {"shrink": 0.8},
    )
    ax.set_xlabel("Predicted", color=TEXT_COLOR)
    ax.set_ylabel("Ground Truth", color=TEXT_COLOR)
    ax.set_title(title, color=TEXT_COLOR, fontweight="bold")
    ax.tick_params(colors=TEXT_COLOR)

    if save_path:
        save_figure(fig, save_path)
    return fig


def plot_training_curves(
    train_losses : List[float],
    val_losses   : List[float],
    val_f1s      : List[float],
    val_ious     : List[float],
    save_path    : Optional[str] = None,
) -> plt.Figure:
    """Plot training/validation loss and metric curves."""
    epochs = list(range(1, len(train_losses) + 1))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4), facecolor=DARK_BG)
    fig.suptitle("Training Progress", color=TEXT_COLOR, fontsize=13, fontweight="bold")

    # Loss curves
    ax1.plot(epochs, train_losses, color="#7c3aed", label="Train Loss", linewidth=2)
    ax1.plot(epochs, val_losses,   color="#e74c3c", label="Val Loss",   linewidth=2, linestyle="--")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss (Dice + BCE)")
    ax1.set_title("Loss Curves")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Metric curves
    ax2.plot(epochs, val_f1s,  color="#2ecc71", label="Val F1-Score", linewidth=2)
    ax2.plot(epochs, val_ious, color="#3498db", label="Val IoU",      linewidth=2, linestyle="--")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Score")
    ax2.set_title("Validation Metrics (Change Class)")
    ax2.set_ylim(0, 1)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        save_figure(fig, save_path)
    return fig
