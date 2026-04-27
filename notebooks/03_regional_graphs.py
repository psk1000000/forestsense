"""
ForestSense — Phase 3: Regional Graph Analysis (Colab Notebook)
================================================================
Run AFTER Phase 2 training is complete and best_model.pth is saved to Drive.

CELL STRUCTURE:
  Cell 1 — Setup, install, reload model
  Cell 2 — Run inference on all test patches
  Cell 3 — SLIC superpixels + RAG construction
  Cell 4 — Spatial metrics per region
  Cell 5 — Fragmentation & patch-size charts (paper figures)
  Cell 6 — RAG overlay figures (paper figures)
  Cell 7 — Summary table + LaTeX output
  Cell 8 — Save all to Drive
"""

# ── CELL 1: Setup & Reload Model ───────────────────────────────────────────────
!pip install -q scikit-image scipy

import os, sys, json, warnings
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
warnings.filterwarnings("ignore")

PROJECT_ROOT = "/content/drive/MyDrive/forestsense"
sys.path.insert(0, PROJECT_ROOT)
os.chdir(PROJECT_ROOT)

OUT_DIR = "/content/outputs/phase3_graphs"
os.makedirs(OUT_DIR, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"📱 Device: {device}")

# ── Load model ────────────────────────────────────────────────────────────────
from model.architecture import build_model
model = build_model(backbone="resnet18", pretrained=False).to(device)
ckpt  = torch.load(
    "/content/drive/MyDrive/forestsense/best_model.pth",
    map_location=device, weights_only=False
)
model.load_state_dict(ckpt["model_state_dict"])
model.eval()
print(f"✅ Model loaded — Epoch {ckpt['epoch']}, mF1={ckpt['best_mf1']:.2f}%")

# ── Constants ─────────────────────────────────────────────────────────────────
DARK_BG    = "#0f0f1a"
TEXT_COLOR = "#e8e8f0"
from utils.visualize import LABEL_CMAP, LABEL_COLORS, LABEL_NAMES, label_legend_handles

REGION_COLORS = {
    "western_ghats": "#7c3aed",
    "sundarbans"   : "#2ecc71",
    "saranda"      : "#e74c3c",
}
REGIONS = list(REGION_COLORS.keys())


# ── CELL 2: Run Inference on All Test Patches ───────────────────────────────────
from data.dataset import ForestSenseDataset, get_val_transforms
from torch.utils.data import DataLoader
from torch.amp import autocast

MANIFEST = "/content/data/processed/manifest.csv"
if not os.path.exists(MANIFEST):
    MANIFEST = "/content/drive/MyDrive/forestsense/data/processed/manifest.csv"

test_ds = ForestSenseDataset(
    manifest_path=MANIFEST, split="test", transforms=get_val_transforms()
)
test_loader = DataLoader(test_ds, batch_size=1, num_workers=2, shuffle=False)
print(f"✅ Test set: {len(test_ds)} patches")

# Collect predictions, GT labels, metadata, and raw T1/T2 for each test patch
all_records = []

with torch.no_grad():
    for i, batch in enumerate(test_loader):
        t1     = batch["t1"].to(device)
        t2     = batch["t2"].to(device)
        lbl    = batch["label"].squeeze(0).numpy()   # (H, W)
        region = batch["meta"]["region"][0]
        chg_pct= float(batch["meta"]["change_pct"][0])

        with autocast("cuda"):
            logit = model(t1, t2)
        pred = logit.argmax(dim=1).squeeze(0).cpu().numpy()   # (H, W)

        # Store RGB bands for SLIC (bands 5=R, 4=G, 3=B → indices 5,4,3)
        t1_np = batch["t1"].squeeze(0).numpy()   # (12, H, W)
        t2_np = batch["t2"].squeeze(0).numpy()

        def make_rgb(stack):
            r, g, b = stack[5], stack[4], stack[3]
            rgb = np.stack([r, g, b], axis=-1)
            p2, p98 = np.percentile(rgb, 2), np.percentile(rgb, 98)
            return np.clip((rgb - p2) / (p98 - p2 + 1e-8), 0, 1)

        all_records.append({
            "region"  : region,
            "chg_pct" : chg_pct,
            "pred"    : pred,
            "label"   : lbl,
            "t1_rgb"  : make_rgb(t1_np),
            "t2_rgb"  : make_rgb(t2_np),
            "t1_sar"  : t1_np[0],
            "t2_sar"  : t2_np[0],
        })

        if (i + 1) % 50 == 0:
            print(f"  Processed {i+1}/{len(test_ds)} patches")

print(f"\n✅ Inference done on {len(all_records)} test patches")
for r in REGIONS:
    n = sum(1 for rec in all_records if rec["region"] == r)
    print(f"   {r}: {n} patches")


# ── CELL 3: SLIC + RAG Construction ────────────────────────────────────────────
from model.graph_analysis import (
    compute_slic, RegionAdjacencyGraph,
    compute_patch_metrics, compare_metrics
)

print("⏳ Building SLIC superpixels and RAGs for each test patch...")

# Compute spatial metrics for every patch, split by region
region_patch_metrics = {r: [] for r in REGIONS}

for i, rec in enumerate(all_records):
    pred   = rec["pred"]
    region = rec["region"]

    # SLIC on the predicted map using RGB image for coherence
    segments = compute_slic(pred, rgb_image=rec["t1_rgb"], n_segments=150)

    # Compute patch-level metrics on the prediction map
    # We treat: Forest Loss pixels → they WERE forest at T1
    # So T1 "virtual" forest = stable_forest + forest_loss
    # T2 "virtual" forest = stable_forest + forest_gain
    t1_virtual = np.where((pred == 1) | (pred == 2), 1, pred)  # treat loss as T1 forest
    t2_virtual = np.where((pred == 1) | (pred == 3), 1, pred)  # treat gain as T2 forest
    # Simple approach: just use the pred map directly
    m = compute_patch_metrics(pred)
    m["region"] = region
    m["chg_pct"] = rec["chg_pct"]
    region_patch_metrics[region].append(m)

    if (i + 1) % 50 == 0:
        print(f"  {i+1}/{len(all_records)} patches processed")

print("✅ Spatial metrics computed for all patches!")

# Aggregate per region
region_summary = {}
for region in REGIONS:
    if not region_patch_metrics[region]:
        continue
    df = pd.DataFrame(region_patch_metrics[region])
    region_summary[region] = {
        "n_patches_mean"      : df["n_forest_patches"].mean(),
        "n_patches_std"       : df["n_forest_patches"].std(),
        "total_forest_ha"     : df["total_forest_ha"].sum(),
        "mean_patch_size_ha"  : df["mean_patch_size_ha"].mean(),
        "largest_patch_idx"   : df["largest_patch_idx"].mean(),
        "fragmentation_idx"   : df["fragmentation_idx"].mean(),
        "edge_density"        : df["edge_density"].mean(),
        "forest_loss_ha"      : df["forest_loss_ha"].sum(),
        "forest_gain_ha"      : df["forest_gain_ha"].sum(),
        "net_change_ha"       : df["net_change_ha"].sum(),
        "n_test_patches"      : len(df),
    }
    print(f"\n📍 {region.replace('_',' ').title()}:")
    print(f"   Test patches        : {region_summary[region]['n_test_patches']}")
    print(f"   Total forest area   : {region_summary[region]['total_forest_ha']:.1f} ha")
    print(f"   Forest loss         : {region_summary[region]['forest_loss_ha']:.1f} ha")
    print(f"   Forest gain         : {region_summary[region]['forest_gain_ha']:.1f} ha")
    print(f"   Net change          : {region_summary[region]['net_change_ha']:.1f} ha")
    print(f"   Mean patch size     : {region_summary[region]['mean_patch_size_ha']:.3f} ha")
    print(f"   Fragmentation index : {region_summary[region]['fragmentation_idx']:.3f}")
    print(f"   Largest Patch Index : {region_summary[region]['largest_patch_idx']:.3f}")


# ── CELL 4: Fragmentation & Spatial Metrics Figure ─────────────────────────────
"""
4 subplots, publication quality:
  1. Forest area by region (stacked: stable, loss, gain)
  2. Mean patch size comparison
  3. Fragmentation index
  4. Largest Patch Index (connectivity proxy)
"""
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.patch.set_facecolor(DARK_BG)
fig.suptitle(
    "ForestSense — Regional Spatial Graph Metrics (2021→2023)",
    color=TEXT_COLOR, fontsize=15, fontweight="bold", y=1.02
)

region_labels = [r.replace("_", " ").title() for r in REGIONS if r in region_summary]
region_keys   = [r for r in REGIONS if r in region_summary]
colors        = [REGION_COLORS[r] for r in region_keys]

def style_ax(ax, title, ylabel):
    ax.set_facecolor("#1a1a2e")
    ax.set_title(title, color=TEXT_COLOR, fontweight="bold", fontsize=11)
    ax.set_ylabel(ylabel, color=TEXT_COLOR)
    ax.tick_params(colors=TEXT_COLOR, labelsize=9)
    for spine in ax.spines.values():
        spine.set_edgecolor("#333355")
    ax.set_xticks(range(len(region_labels)))
    ax.set_xticklabels(region_labels, color=TEXT_COLOR)

# Plot 1: Stacked forest area (stable / loss / gain)
ax = axes[0, 0]
style_ax(ax, "Forest Area by Change Type (ha)", "Area (ha)")
x    = np.arange(len(region_keys))
w    = 0.25
stable_ha = [region_summary[r]["total_forest_ha"] -
             region_summary[r]["forest_loss_ha"] for r in region_keys]
loss_ha   = [region_summary[r]["forest_loss_ha"]  for r in region_keys]
gain_ha   = [region_summary[r]["forest_gain_ha"]  for r in region_keys]

b1 = ax.bar(x - w, stable_ha, w, color="#2ecc71", label="Stable Forest",     edgecolor=DARK_BG)
b2 = ax.bar(x,     loss_ha,   w, color="#e74c3c", label="Forest Loss",        edgecolor=DARK_BG)
b3 = ax.bar(x + w, gain_ha,   w, color="#3498db", label="Forest Gain",        edgecolor=DARK_BG)
ax.legend(framealpha=0.2, fontsize=8)
for bars in [b1, b2, b3]:
    for bar in bars:
        h = bar.get_height()
        if h > 0:
            ax.text(bar.get_x() + bar.get_width()/2, h + 0.2,
                    f"{h:.1f}", ha="center", va="bottom",
                    color=TEXT_COLOR, fontsize=7)

# Plot 2: Mean patch size
ax = axes[0, 1]
style_ax(ax, "Mean Forest Patch Size (ha)", "Patch Size (ha)")
vals  = [region_summary[r]["mean_patch_size_ha"] for r in region_keys]
bars  = ax.bar(range(len(region_keys)), vals, color=colors, edgecolor=DARK_BG, width=0.5)
for bar, v in zip(bars, vals):
    ax.text(bar.get_x() + bar.get_width()/2, v + 0.001,
            f"{v:.4f}", ha="center", va="bottom", color=TEXT_COLOR, fontsize=9)

# Plot 3: Fragmentation Index
ax = axes[1, 0]
style_ax(ax, "Fragmentation Index\n(patches / ha — higher = more fragmented)",
         "Fragmentation Index")
vals = [region_summary[r]["fragmentation_idx"] for r in region_keys]
bars = ax.bar(range(len(region_keys)), vals, color=colors, edgecolor=DARK_BG, width=0.5)
for bar, v in zip(bars, vals):
    ax.text(bar.get_x() + bar.get_width()/2, v + 0.001,
            f"{v:.3f}", ha="center", va="bottom", color=TEXT_COLOR, fontsize=9)
ax.axhline(y=np.mean(vals), color="#f39c12", lw=1.5, ls="--", alpha=0.7,
           label=f"Mean = {np.mean(vals):.3f}")
ax.legend(framealpha=0.2)

# Plot 4: Largest Patch Index
ax = axes[1, 1]
style_ax(ax, "Largest Patch Index\n(1.0 = single contiguous forest)",
         "LPI (0 – 1)")
vals = [region_summary[r]["largest_patch_idx"] for r in region_keys]
bars = ax.bar(range(len(region_keys)), vals, color=colors, edgecolor=DARK_BG, width=0.5)
ax.set_ylim(0, 1.05)
for bar, v in zip(bars, vals):
    ax.text(bar.get_x() + bar.get_width()/2, v + 0.01,
            f"{v:.3f}", ha="center", va="bottom", color=TEXT_COLOR, fontsize=9)
ax.axhline(y=0.5, color="#9b59b6", lw=1, ls=":", alpha=0.6, label="0.5 threshold")
ax.legend(framealpha=0.2)

plt.tight_layout()
SPATIAL_FIG = f"{OUT_DIR}/spatial_metrics.png"
fig.savefig(SPATIAL_FIG, dpi=200, bbox_inches="tight", facecolor=DARK_BG)
plt.show()
print(f"✅ Spatial metrics figure saved → {SPATIAL_FIG}")


# ── CELL 5: RAG Overlay Figures ─────────────────────────────────────────────────
"""
For each region: pick the patch with highest change_pct.
Show: T1 RGB | T1 RAG overlay | T2 RGB | T2 Prediction | RAG graph
"""
from model.graph_analysis import compute_slic, RegionAdjacencyGraph, draw_rag_on_map

for region in region_keys:
    # Pick patch with highest change_pct
    region_recs = [r for r in all_records if r["region"] == region]
    if not region_recs:
        continue
    rec = max(region_recs, key=lambda x: x["chg_pct"])

    pred   = rec["pred"]
    t1_rgb = rec["t1_rgb"]
    t2_rgb = rec["t2_rgb"]
    t1_sar = rec["t1_sar"]

    # SLIC
    segments = compute_slic(pred, rgb_image=t1_rgb, n_segments=150)
    rag      = RegionAdjacencyGraph(segments, pred)

    fig = plt.figure(figsize=(18, 6), facecolor=DARK_BG)
    fig.suptitle(
        f"Region Adjacency Graph — {region.replace('_',' ').title()}  "
        f"(Change: {rec['chg_pct']:.1f}%)",
        color=TEXT_COLOR, fontsize=13, fontweight="bold"
    )
    gs = gridspec.GridSpec(1, 5, figure=fig, wspace=0.08)

    # Panel 1: T1 RGB
    ax = fig.add_subplot(gs[0, 0])
    ax.imshow(t1_rgb, interpolation="nearest")
    ax.set_title("T1 True Colour", color=TEXT_COLOR, fontsize=9)
    ax.axis("off")

    # Panel 2: T2 RGB
    ax = fig.add_subplot(gs[0, 1])
    ax.imshow(rec["t2_rgb"], interpolation="nearest")
    ax.set_title("T2 True Colour", color=TEXT_COLOR, fontsize=9)
    ax.axis("off")

    # Panel 3: Prediction map
    ax = fig.add_subplot(gs[0, 2])
    ax.imshow(pred, cmap=LABEL_CMAP, vmin=0, vmax=3, interpolation="nearest")
    ax.set_title("Predicted Change Map", color=TEXT_COLOR, fontsize=9)
    ax.axis("off")

    # Panel 4: SLIC superpixels on RGB
    from skimage.segmentation import mark_boundaries
    ax = fig.add_subplot(gs[0, 3])
    slic_vis = mark_boundaries(t1_rgb, segments, color=(1, 0.9, 0.1), mode="outer")
    ax.imshow(slic_vis, interpolation="nearest")
    ax.set_title(f"SLIC ({len(np.unique(segments))} superpixels)", color=TEXT_COLOR, fontsize=9)
    ax.axis("off")

    # Panel 5: RAG overlay on prediction
    ax = fig.add_subplot(gs[0, 4])
    draw_rag_on_map(pred, segments, rag, ax, LABEL_CMAP,
                    title=f"RAG (nodes={len(rag.nodes)}, edges={len(rag.edges)})",
                    text_color=TEXT_COLOR)

    # Legend
    fig.legend(
        handles=label_legend_handles(),
        loc="lower center", ncol=4, fontsize=9,
        framealpha=0.15, bbox_to_anchor=(0.5, -0.05)
    )

    plt.tight_layout()
    RAG_PATH = f"{OUT_DIR}/rag_{region}.png"
    fig.savefig(RAG_PATH, dpi=180, bbox_inches="tight", facecolor=DARK_BG)
    plt.show()
    plt.close(fig)
    print(f"  ✅ RAG figure saved → {RAG_PATH}")


# ── CELL 6: Net Change Timeline Figure ─────────────────────────────────────────
"""
Horizontal bar chart: Net forest change per region.
Green = net gain, Red = net loss.
"""
fig, ax = plt.subplots(figsize=(10, 5))
fig.patch.set_facecolor(DARK_BG)
ax.set_facecolor("#1a1a2e")

net_vals = [region_summary[r]["net_change_ha"] for r in region_keys]
bar_colors = ["#e74c3c" if v < 0 else "#2ecc71" for v in net_vals]

bars = ax.barh(region_labels, net_vals, color=bar_colors, edgecolor=DARK_BG, height=0.4)

for bar, v in zip(bars, net_vals):
    x_pos = v + (0.05 * max(abs(min(net_vals)), abs(max(net_vals))))
    ax.text(x_pos, bar.get_y() + bar.get_height()/2,
            f"{v:+.2f} ha", va="center",
            color=TEXT_COLOR, fontsize=10, fontweight="bold")

ax.axvline(x=0, color=TEXT_COLOR, lw=1, alpha=0.4)
ax.set_xlabel("Net Forest Change (ha)  —  Negative = Loss", color=TEXT_COLOR)
ax.set_title("ForestSense — Net Forest Change per Region (2021 → 2023)",
             color=TEXT_COLOR, fontweight="bold", fontsize=13)
ax.tick_params(colors=TEXT_COLOR)
for spine in ax.spines.values():
    spine.set_edgecolor("#333355")

plt.tight_layout()
NET_PATH = f"{OUT_DIR}/net_forest_change.png"
fig.savefig(NET_PATH, dpi=200, bbox_inches="tight", facecolor=DARK_BG)
plt.show()
print(f"✅ Net change figure saved → {NET_PATH}")


# ── CELL 7: Summary Table + LaTeX ───────────────────────────────────────────────
print("\n" + "="*75)
print("  📝 PHASE 3 — REGIONAL SPATIAL METRICS TABLE")
print("="*75)
print(f"\n  {'Region':<20} {'Forest(ha)':>10} {'Loss(ha)':>9} {'Gain(ha)':>9} "
      f"{'NetChg(ha)':>11} {'Frag':>7} {'LPI':>7}")
print(f"  {'-'*73}")
for r in region_keys:
    s = region_summary[r]
    print(f"  {r.replace('_',' ').title():<20} "
          f"{s['total_forest_ha']:>10.1f} "
          f"{s['forest_loss_ha']:>9.1f} "
          f"{s['forest_gain_ha']:>9.1f} "
          f"{s['net_change_ha']:>+11.1f} "
          f"{s['fragmentation_idx']:>7.3f} "
          f"{s['largest_patch_idx']:>7.3f}")
print("="*75)

print("\n\n  📄 LATEX TABLE:\n")
print(r"  \begin{table}[h]")
print(r"  \centering")
print(r"  \caption{Regional Spatial Graph Metrics (ForestSense, 2021--2023)}")
print(r"  \begin{tabular}{lccccc}")
print(r"  \hline")
print(r"  \textbf{Region} & \textbf{Forest (ha)} & \textbf{Loss (ha)} & "
      r"\textbf{Gain (ha)} & \textbf{Frag. Idx} & \textbf{LPI} \\")
print(r"  \hline")
for r in region_keys:
    s = region_summary[r]
    rname = r.replace("_", " ").title()
    print(f"  {rname} & {s['total_forest_ha']:.1f} & {s['forest_loss_ha']:.1f} "
          f"& {s['forest_gain_ha']:.1f} & {s['fragmentation_idx']:.3f} "
          f"& {s['largest_patch_idx']:.3f} \\\\")
print(r"  \hline")
print(r"  \end{tabular}")
print(r"  \end{table}")


# ── CELL 8: Save to Drive ────────────────────────────────────────────────────────
import shutil

# Save metrics JSON
with open(f"{OUT_DIR}/regional_graph_metrics.json", "w") as f:
    json.dump(region_summary, f, indent=2)

shutil.copytree(OUT_DIR,
                f"/content/drive/MyDrive/forestsense/outputs/phase3_graphs",
                dirs_exist_ok=True)
print(f"""
✅ Phase 3 outputs saved to Drive!

📁 outputs/phase3_graphs/
├── spatial_metrics.png        ← Paper Figure: 4-panel spatial metrics
├── rag_western_ghats.png      ← Paper Figure: RAG overlay
├── rag_sundarbans.png         ← Paper Figure: RAG overlay
├── rag_saranda.png            ← Paper Figure: RAG overlay
├── net_forest_change.png      ← Paper Figure: Net change bar chart
└── regional_graph_metrics.json← Table 2 numbers
""")
