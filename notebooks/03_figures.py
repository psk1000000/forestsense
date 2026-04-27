"""
ForestSense — Phase 3: Publication Figures & Regional Analytics
================================================================
Run this as a Colab notebook (paste each cell) after Phase 2 training.

CELL STRUCTURE:
  Cell 1 — Setup & model reload
  Cell 2 — Confusion matrix figure
  Cell 3 — Training curves figure
  Cell 4 — Prediction panels (6 samples)
  Cell 5 — Per-region analytics bar chart
  Cell 6 — Summary metrics table (copy-paste into paper)
  Cell 7 — Save everything to Drive
"""

# ── CELL 1: Setup & Reload Model ───────────────────────────────────────────────
import os, sys, json
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

# Paths
PROJECT_ROOT = "/content/drive/MyDrive/forestsense"
sys.path.insert(0, PROJECT_ROOT)
os.chdir(PROJECT_ROOT)

OUT_VIZ  = "/content/outputs/visualizations"
os.makedirs(OUT_VIZ, exist_ok=True)
os.makedirs(f"{OUT_VIZ}/panels", exist_ok=True)

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"📱 Device: {device}")

# Load model
from model.architecture import build_model
model = build_model(backbone="resnet18", pretrained=False).to(device)
ckpt  = torch.load(
    "/content/drive/MyDrive/forestsense/best_model.pth",
    map_location=device, weights_only=False
)
model.load_state_dict(ckpt["model_state_dict"])
model.eval()
print(f"✅ Model loaded — Epoch {ckpt['epoch']}, Best mF1={ckpt['best_mf1']:.2f}%")

# Test results (already computed — paste from Phase 2 output or load from json)
# If you saved test_metrics.json, load it:
METRICS_PATH = "/content/drive/MyDrive/forestsense/outputs/test_metrics.json"
if os.path.exists(METRICS_PATH):
    with open(METRICS_PATH) as f:
        test_results = json.load(f)
    test_results["conf_matrix"] = np.array(test_results["conf_matrix"])
    print("✅ Test metrics loaded from Drive.")
else:
    # Re-run evaluation if json not found
    from data.dataset import get_dataloaders
    from model.metrics import ChangeDetectionMetrics
    from model.losses import build_loss
    from torch.amp import autocast

    _, _, test_loader = get_dataloaders(
        manifest_path="/content/data/processed/manifest.csv",
        batch_size=8, num_workers=2
    )
    loss_fn = build_loss(use_focal=True)
    metrics = ChangeDetectionMetrics(num_classes=4)

    with torch.no_grad():
        for batch in test_loader:
            t1  = batch["t1"].to(device)
            t2  = batch["t2"].to(device)
            lbl = batch["label"].to(device)
            with autocast("cuda"):
                logits = model(t1, t2)
            metrics.update(logits, lbl)

    test_results = metrics.compute()
    print("✅ Test evaluation complete.")


# ── CELL 2: Confusion Matrix ────────────────────────────────────────────────────
from utils.visualize import plot_confusion_matrix, save_figure
import seaborn as sns

CLASS_NAMES = ["Unchanged\nNon-Forest", "Stable\nForest", "Forest\nLoss", "Forest\nGain"]
DARK_BG = "#0f0f1a"
TEXT_COLOR = "#e8e8f0"

cm      = test_results["conf_matrix"].astype(float)
cm_norm = cm / (cm.sum(axis=1, keepdims=True) + 1e-8)

fig, ax = plt.subplots(figsize=(7, 6))
fig.patch.set_facecolor(DARK_BG)
ax.set_facecolor("#1a1a2e")

hm = sns.heatmap(
    cm_norm * 100,
    annot=True,
    fmt=".1f",
    cmap="viridis",
    ax=ax,
    linewidths=0.8,
    linecolor=DARK_BG,
    xticklabels=CLASS_NAMES,
    yticklabels=CLASS_NAMES,
    vmin=0, vmax=100,
    cbar_kws={"shrink": 0.82, "label": "Recall (%)"},
    annot_kws={"size": 11, "weight": "bold"},
)

ax.set_xlabel("Predicted Class", color=TEXT_COLOR, fontsize=12, labelpad=10)
ax.set_ylabel("True Class", color=TEXT_COLOR, fontsize=12, labelpad=10)
ax.set_title("ForestSense — Confusion Matrix (Test Set)", color=TEXT_COLOR,
             fontsize=13, fontweight="bold", pad=15)

ax.tick_params(colors=TEXT_COLOR, labelsize=9)
hm.collections[0].colorbar.ax.tick_params(colors=TEXT_COLOR)
hm.collections[0].colorbar.ax.yaxis.label.set_color(TEXT_COLOR)

plt.tight_layout()
CM_PATH = f"{OUT_VIZ}/confusion_matrix.png"
fig.savefig(CM_PATH, dpi=200, bbox_inches="tight", facecolor=DARK_BG)
plt.show()
print(f"✅ Confusion matrix saved → {CM_PATH}")


# ── CELL 3: Training Curves ─────────────────────────────────────────────────────
# Load training log CSV
LOG_PATH = "/content/drive/MyDrive/forestsense/outputs/training_log.csv"

if os.path.exists(LOG_PATH):
    log_df = pd.read_csv(LOG_PATH)
    train_df = log_df[log_df["phase"] == "train"].reset_index(drop=True)
    val_df   = log_df[log_df["phase"] == "val"].reset_index(drop=True)

    epochs = train_df["epoch"].tolist()

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.patch.set_facecolor(DARK_BG)
    fig.suptitle("ForestSense — Training Progress", color=TEXT_COLOR,
                 fontsize=14, fontweight="bold", y=1.02)

    panel_style = dict(facecolor="#1a1a2e")

    # Plot 1: Loss
    ax = axes[0]
    ax.set_facecolor("#1a1a2e")
    ax.plot(epochs, train_df["loss"], color="#7c3aed", lw=2, label="Train Loss")
    ax.plot(val_df["epoch"],   val_df["loss"],  color="#e74c3c", lw=2, ls="--", label="Val Loss")
    ax.set_xlabel("Epoch", color=TEXT_COLOR)
    ax.set_ylabel("Loss", color=TEXT_COLOR)
    ax.set_title("Loss Curves", color=TEXT_COLOR, fontweight="bold")
    ax.legend(framealpha=0.2)
    ax.grid(alpha=0.25)
    ax.tick_params(colors=TEXT_COLOR)

    # Plot 2: mF1 & mIoU
    ax = axes[1]
    ax.set_facecolor("#1a1a2e")
    ax.plot(val_df["epoch"], val_df["mF1"],  color="#2ecc71", lw=2, label="mF1 (macro)")
    ax.plot(val_df["epoch"], val_df["mIoU"], color="#3498db", lw=2, ls="--", label="mIoU")
    ax.set_xlabel("Epoch", color=TEXT_COLOR)
    ax.set_ylabel("Score (%)", color=TEXT_COLOR)
    ax.set_title("Validation F1 & IoU", color=TEXT_COLOR, fontweight="bold")
    ax.axhline(y=test_results["mF1"], color="#2ecc71", lw=1, ls=":", alpha=0.7,
               label=f"Test mF1={test_results['mF1']:.1f}%")
    ax.legend(framealpha=0.2)
    ax.grid(alpha=0.25)
    ax.tick_params(colors=TEXT_COLOR)

    # Plot 3: Change F1
    ax = axes[2]
    ax.set_facecolor("#1a1a2e")
    ax.plot(val_df["epoch"], val_df["mF1_change_only"], color="#f39c12", lw=2.5,
            label="Change F1 (Loss+Gain)")
    ax.plot(val_df["epoch"], val_df["OA"],              color="#9b59b6", lw=2, ls="--",
            label="Overall Accuracy")
    ax.set_xlabel("Epoch", color=TEXT_COLOR)
    ax.set_ylabel("Score (%)", color=TEXT_COLOR)
    ax.set_title("Change Classes & OA", color=TEXT_COLOR, fontweight="bold")
    ax.axhline(y=test_results["mF1_change_only"], color="#f39c12", lw=1, ls=":",
               alpha=0.7, label=f"Test Change F1={test_results['mF1_change_only']:.1f}%")
    ax.legend(framealpha=0.2)
    ax.grid(alpha=0.25)
    ax.tick_params(colors=TEXT_COLOR)

    plt.tight_layout()
    CURVES_PATH = f"{OUT_VIZ}/training_curves.png"
    fig.savefig(CURVES_PATH, dpi=200, bbox_inches="tight", facecolor=DARK_BG)
    plt.show()
    print(f"✅ Training curves saved → {CURVES_PATH}")
else:
    print("⚠️  training_log.csv not found. Run Cell 7 first to copy outputs from Drive.")


# ── CELL 4: Prediction Panels ───────────────────────────────────────────────────
from data.dataset import ForestSenseDataset, get_val_transforms
from utils.visualize import (LABEL_CMAP, TEXT_COLOR, DARK_BG,
                              label_legend_handles, save_figure)
from torch.amp import autocast
import matplotlib.gridspec as gridspec

MANIFEST = "/content/data/processed/manifest.csv"
if not os.path.exists(MANIFEST):
    # Fallback to Drive
    MANIFEST = "/content/drive/MyDrive/forestsense/data/processed/manifest.csv"

test_ds = ForestSenseDataset(
    manifest_path=MANIFEST,
    split="test",
    transforms=get_val_transforms(),
)
print(f"✅ Test dataset: {len(test_ds)} patches")

def to_rgb(stack_np):
    """Bands: 3=Blue, 4=Green, 5=Red (0-indexed). Returns [H,W,3] in [0,1]."""
    r = stack_np[5]; g = stack_np[4]; b = stack_np[3]
    rgb = np.stack([r, g, b], axis=-1)
    p2, p98 = np.percentile(rgb, 2), np.percentile(rgb, 98)
    return np.clip((rgb - p2) / (p98 - p2 + 1e-8), 0, 1)

def to_sar(stack_np):
    s = stack_np[0]
    p2, p98 = np.percentile(s, 2), np.percentile(s, 98)
    return np.clip((s - p2) / (p98 - p2 + 1e-8), 0, 1)

def to_ndvi(stack_np):
    n = stack_np[9]   # NDVI band (precomputed)
    return np.clip(n, 0, 1)

# Pick 6 patches with most change
df_test = pd.read_csv(MANIFEST)
df_test = df_test[df_test["split"] == "test"].sort_values("change_pct", ascending=False)
sample_indices = df_test.index[:6].tolist()

# Re-map indices to dataset positions
test_indices = df_test.reset_index(drop=True).head(6).index.tolist()

for panel_i, ds_idx in enumerate(test_indices[:6]):
    try:
        sample = test_ds[ds_idx]
    except Exception:
        sample = test_ds[panel_i]

    t1_np  = sample["t1"].numpy()
    t2_np  = sample["t2"].numpy()
    lbl    = sample["label"].numpy()
    region = sample["meta"]["region"]
    chg_pct= sample["meta"]["change_pct"]

    t1_ten = sample["t1"].unsqueeze(0).to(device)
    t2_ten = sample["t2"].unsqueeze(0).to(device)

    with torch.no_grad():
        with autocast("cuda"):
            logit = model(t1_ten, t2_ten)
    pred = logit.argmax(dim=1).squeeze().cpu().numpy()

    # Build figure — 2 rows × 4 cols
    fig = plt.figure(figsize=(16, 8), facecolor=DARK_BG)
    fig.suptitle(
        f"ForestSense — Region: {region.replace('_', ' ').title()}  |  "
        f"Change pixels: {chg_pct:.1f}%",
        color=TEXT_COLOR, fontsize=13, fontweight="bold"
    )
    gs = gridspec.GridSpec(2, 4, figure=fig, hspace=0.3, wspace=0.08)

    panels = [
        (0, 0, to_rgb(t1_np),  "T1 (2021) — True Colour",    None),
        (0, 1, to_rgb(t2_np),  "T2 (2023) — True Colour",    None),
        (0, 2, pred,           "Predicted Change Map",         LABEL_CMAP),
        (0, 3, lbl,            "Ground Truth",                 LABEL_CMAP),
        (1, 0, to_sar(t1_np),  "T1 SAR (VV dB)",             "gray"),
        (1, 1, to_sar(t2_np),  "T2 SAR (VV dB)",             "gray"),
        (1, 2, to_ndvi(t1_np), "T1 NDVI",                    "RdYlGn"),
        (1, 3, to_ndvi(t2_np), "T2 NDVI",                    "RdYlGn"),
    ]

    for row, col, img, title, cmap in panels:
        ax = fig.add_subplot(gs[row, col])
        ax.set_facecolor("#1a1a2e")
        kw = {"cmap": cmap, "vmin": 0, "vmax": 3} if cmap == LABEL_CMAP else \
             {"cmap": cmap} if cmap else {}
        ax.imshow(img, interpolation="nearest", **kw)
        ax.set_title(title, color=TEXT_COLOR, fontsize=8, pad=4)
        ax.axis("off")

    # Legend
    fig.legend(
        handles=label_legend_handles(),
        loc="lower center", ncol=4, fontsize=9,
        framealpha=0.15, bbox_to_anchor=(0.5, -0.03)
    )

    panel_path = f"{OUT_VIZ}/panels/panel_{panel_i+1}_{region}.png"
    fig.savefig(panel_path, dpi=180, bbox_inches="tight", facecolor=DARK_BG)
    plt.show()
    plt.close(fig)
    print(f"  ✅ Panel {panel_i+1} saved → {panel_path}")

print(f"\n✅ All {len(test_indices[:6])} panels generated!")


# ── CELL 5: Per-Region Analytics ───────────────────────────────────────────────
"""
Breaks down per-class performance by region (Western Ghats, Sundarbans, Saranda).
"""
from model.metrics import ChangeDetectionMetrics, CLASS_NAMES

MANIFEST = "/content/data/processed/manifest.csv"
if not os.path.exists(MANIFEST):
    MANIFEST = "/content/drive/MyDrive/forestsense/data/processed/manifest.csv"

from data.dataset import ForestSenseDataset, get_val_transforms
from torch.utils.data import DataLoader
from torch.amp import autocast

region_metrics = {}
all_regions    = ["western_ghats", "sundarbans", "saranda"]

for region in all_regions:
    # Create a region-specific test dataset using filtered manifest
    df_all    = pd.read_csv(MANIFEST)
    df_region = df_all[(df_all["split"] == "test") & (df_all["region"] == region)]

    if len(df_region) == 0:
        print(f"⚠️  No test patches for region: {region}")
        continue

    # Write temp manifest
    tmp_manifest = f"/content/tmp_{region}.csv"
    df_region.to_csv(tmp_manifest, index=False)

    ds = ForestSenseDataset(
        manifest_path=tmp_manifest,
        split="test",
        transforms=get_val_transforms(),
    )
    loader = DataLoader(ds, batch_size=8, num_workers=2, pin_memory=True)

    met = ChangeDetectionMetrics(num_classes=4)
    with torch.no_grad():
        for batch in loader:
            t1  = batch["t1"].to(device)
            t2  = batch["t2"].to(device)
            lbl = batch["label"].to(device)
            with autocast("cuda"):
                logits = model(t1, t2)
            met.update(logits, lbl)

    region_metrics[region] = met.compute()
    print(f"  ✅ {region}: OA={region_metrics[region]['OA']:.1f}%  "
          f"mF1={region_metrics[region]['mF1']:.1f}%  "
          f"ChangeF1={region_metrics[region]['mF1_change_only']:.1f}%")

# ── Plot regional comparison ──────────────────────────────────────────────────
regions_clean = [r.replace("_", " ").title() for r in region_metrics.keys()]
metrics_keys  = ["OA", "mF1", "mF1_change_only", "kappa"]
metrics_labels= ["Overall Accuracy (%)", "Mean F1 (%)", "Change F1 (%)", "Kappa × 100"]

fig, axes = plt.subplots(1, 4, figsize=(16, 5))
fig.patch.set_facecolor(DARK_BG)
fig.suptitle("ForestSense — Per-Region Performance (Test Set)",
             color=TEXT_COLOR, fontsize=14, fontweight="bold", y=1.02)

COLORS = ["#7c3aed", "#2ecc71", "#e74c3c"]

for ax_i, (key, label) in enumerate(zip(metrics_keys, metrics_labels)):
    ax = axes[ax_i]
    ax.set_facecolor("#1a1a2e")

    vals = []
    for r in region_metrics.keys():
        v = region_metrics[r][key]
        if key == "kappa":
            v = v * 100   # scale kappa to 0-100 for the bar chart
        vals.append(v)

    bars = ax.bar(regions_clean, vals, color=COLORS[:len(vals)],
                  edgecolor="#1a1a2e", width=0.5)

    # Add value labels on bars
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + 0.5,
                f"{v:.1f}", ha="center", va="bottom",
                color=TEXT_COLOR, fontsize=9, fontweight="bold")

    ax.set_title(label, color=TEXT_COLOR, fontweight="bold", fontsize=10)
    ax.set_ylim(0, 105)
    ax.tick_params(colors=TEXT_COLOR, labelsize=8)
    ax.set_ylabel("Score", color=TEXT_COLOR, fontsize=9)
    for spine in ax.spines.values():
        spine.set_edgecolor("#333355")

plt.tight_layout()
REGION_PATH = f"{OUT_VIZ}/regional_analytics.png"
fig.savefig(REGION_PATH, dpi=200, bbox_inches="tight", facecolor=DARK_BG)
plt.show()
print(f"✅ Regional analytics chart saved → {REGION_PATH}")


# ── CELL 6: Paper Results Table ─────────────────────────────────────────────────
"""
Prints a LaTeX-ready results table and a plain-text summary.
Copy the LaTeX directly into your paper.
"""
print("\n" + "="*65)
print("  📝 PAPER RESULTS TABLE")
print("="*65)

CLASS_LABELS = ["Unchanged Non-Forest", "Stable Forest", "Forest Loss", "Forest Gain"]

# Overall
print(f"\n  Overall Accuracy  : {test_results['OA']:.2f}%")
print(f"  Mean F1 (macro)   : {test_results['mF1']:.2f}%")
print(f"  Mean IoU          : {test_results['mIoU']:.2f}%")
print(f"  Weighted F1       : {test_results.get('weighted_F1', 'N/A')}")
print(f"  Change F1 (cls2+3): {test_results['mF1_change_only']:.2f}%")
print(f"  Change IoU        : {test_results['mIoU_change_only']:.2f}%")
print(f"  Cohen's Kappa     : {test_results['kappa']:.4f}")

# Per-class
print(f"\n  {'Class':<25} {'Prec':>7} {'Rec':>7} {'F1':>7} {'IoU':>7}")
print(f"  {'-'*55}")
for cls in CLASS_LABELS:
    p = test_results['precision'][cls]
    r = test_results['recall'][cls]
    f = test_results['f1'][cls]
    u = test_results['iou'][cls]
    print(f"  {cls:<25} {p:>6.2f}% {r:>6.2f}% {f:>6.2f}% {u:>6.2f}%")
print("="*65)

# LaTeX table
print("\n\n  📄 LATEX TABLE (copy into paper):\n")
print(r"  \begin{table}[h]")
print(r"  \centering")
print(r"  \caption{ForestSense Test Set Results}")
print(r"  \begin{tabular}{lcccc}")
print(r"  \hline")
print(r"  \textbf{Class} & \textbf{Prec.} & \textbf{Recall} & \textbf{F1} & \textbf{IoU} \\")
print(r"  \hline")
for cls in CLASS_LABELS:
    p = test_results['precision'][cls]
    r = test_results['recall'][cls]
    f = test_results['f1'][cls]
    u = test_results['iou'][cls]
    short = cls.replace("Unchanged Non-Forest", "Non-Forest").replace("Stable Forest", "Stable Forest")
    print(f"  {short} & {p:.2f}\\% & {r:.2f}\\% & {f:.2f}\\% & {u:.2f}\\% \\\\")
print(r"  \hline")
print(f"  \\textbf{{Overall Accuracy}} & \\multicolumn{{4}}{{c}}{{{test_results['OA']:.2f}\\%}} \\\\")
print(f"  \\textbf{{Mean F1}} & \\multicolumn{{4}}{{c}}{{{test_results['mF1']:.2f}\\%}} \\\\")
print(f"  \\textbf{{Cohen's Kappa}} & \\multicolumn{{4}}{{c}}{{{test_results['kappa']:.4f}}} \\\\")
print(r"  \hline")
print(r"  \end{tabular}")
print(r"  \end{table}")


# ── CELL 7: Save All to Drive ───────────────────────────────────────────────────
import shutil, json

# Save regional metrics
with open("/content/outputs/regional_metrics.json", "w") as f:
    safe = {}
    for r, m in region_metrics.items():
        safe[r] = {k: (v.tolist() if hasattr(v, "tolist") else v)
                   for k, v in m.items() if k != "conf_matrix"}
    json.dump(safe, f, indent=2)
print("✅ Regional metrics saved.")

# Copy all outputs to Drive
shutil.copytree(
    "/content/outputs",
    "/content/drive/MyDrive/forestsense/outputs",
    dirs_exist_ok=True
)
print("""
✅ All outputs saved to Drive!

📁 forestsense/outputs/
├── visualizations/
│   ├── confusion_matrix.png      ← Paper Figure 1
│   ├── training_curves.png       ← Paper Figure 2
│   ├── regional_analytics.png    ← Paper Figure 3
│   └── panels/
│       └── panel_*.png           ← Paper Figure 4 (pick best 2)
├── test_metrics.json             ← Table 1 numbers
├── regional_metrics.json         ← Table 2 numbers
└── training_log.csv              ← Supplementary material
""")
