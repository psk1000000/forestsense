"""
ForestSense — Patch Sample Generator
======================================
Generates visual sample grids from the processed dataset for verification.
Produces publication-ready figures showing:
  - T1 RGB (Optical True Color)
  - T2 RGB (Optical True Color)
  - T1 SAR (VV in greyscale)
  - T2 SAR (VV in greyscale)
  - NDVI T1 vs T2
  - Label Map (colour-coded change classes)

Usage:
    python data/generate_samples.py \
        --manifest data/processed/manifest.csv \
        --output   outputs/visualizations/samples \
        --n        8

Outputs:
    outputs/visualizations/samples/
    ├── western_ghats_sample_grid.png
    ├── sundarbans_sample_grid.png
    └── saranda_sample_grid.png
"""

import argparse
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap
from pathlib import Path


# ── Colour map for label classes ──────────────────────────────
LABEL_CMAP   = ListedColormap(["#2c2c2c", "#2ecc71", "#e74c3c", "#3498db"])
LABEL_NAMES  = ["Unchanged\nNon-Forest", "Stable\nForest", "Forest\nLoss", "Forest\nGain"]
LABEL_COLORS = ["#2c2c2c", "#2ecc71", "#e74c3c", "#3498db"]

# ── Band indices in the 12-band stack ─────────────────────────
IDX = {
    "SAR_VV"  : 0,
    "SAR_VH"  : 1,
    "SAR_ratio": 2,
    "B2_Blue" : 3,
    "B3_Green": 4,
    "B4_Red"  : 5,
    "B8_NIR"  : 6,
    "NDVI"    : 9,
    "EVI"     : 10,
    "NDWI"    : 11,
}


def to_rgb(stack: np.ndarray) -> np.ndarray:
    """Extract and scale RGB for display from a (12, H, W) stack."""
    r = stack[IDX["B4_Red"]]
    g = stack[IDX["B3_Green"]]
    b = stack[IDX["B2_Blue"]]
    rgb = np.stack([r, g, b], axis=-1)
    # Stretch to [0, 1]
    rgb = (rgb - rgb.min()) / (rgb.max() - rgb.min() + 1e-8)
    return np.clip(rgb, 0, 1)


def to_sar_vv(stack: np.ndarray) -> np.ndarray:
    """Extract SAR VV band (already normalized 0–1)."""
    return stack[IDX["SAR_VV"]]


def to_ndvi(stack: np.ndarray) -> np.ndarray:
    """Extract NDVI band; values were scaled to [0,1], rescale to [-1,1] for display."""
    ndvi_01 = stack[IDX["NDVI"]]
    return ndvi_01 * 2.0 - 1.0     # back to [-1, 1] for proper colormap


def to_sar_false_color(stack: np.ndarray) -> np.ndarray:
    """SAR false-color composite: VV/VH/ratio as R/G/B."""
    vv     = stack[IDX["SAR_VV"]]
    vh     = stack[IDX["SAR_VH"]]
    ratio  = stack[IDX["SAR_ratio"]]
    fcc    = np.stack([vv, vh, ratio], axis=-1)
    fcc    = (fcc - fcc.min()) / (fcc.max() - fcc.min() + 1e-8)
    return np.clip(fcc, 0, 1)


def make_legend():
    """Return matplotlib legend handles for label classes."""
    handles = [
        mpatches.Patch(color=LABEL_COLORS[i], label=LABEL_NAMES[i])
        for i in range(4)
    ]
    return handles


def plot_sample_grid(
    records     : pd.DataFrame,
    n_samples   : int,
    region_id   : str,
    out_path    : str,
):
    """
    Plot a 7-column grid for n_samples patches from a region.
    Columns: T1 RGB | T2 RGB | T1 SAR-FCC | T2 SAR-FCC | NDVI T1 | NDVI T2 | Label
    """
    records = records.sample(min(n_samples, len(records)), random_state=42).reset_index(drop=True)
    n = len(records)
    fig, axes = plt.subplots(n, 7, figsize=(21, 3 * n), facecolor="#1a1a2e")
    fig.suptitle(
        f"ForestSense — Sample Patches: {region_id.replace('_', ' ').title()}",
        fontsize=14, color="white", fontweight="bold", y=1.01,
    )

    col_titles = [
        "T1 (2021)\nTrue Color",
        "T2 (2023)\nTrue Color",
        "T1 SAR\nFalse Color",
        "T2 SAR\nFalse Color",
        "T1 NDVI",
        "T2 NDVI",
        "Change\nLabel",
    ]

    for col_idx, title in enumerate(col_titles):
        (axes[0] if n == 1 else axes[0])[col_idx].set_title(
            title, color="white", fontsize=8, pad=4
        )

    for row_idx, row in records.iterrows():
        t1    = np.load(row["t1_path"],    allow_pickle=False)
        t2    = np.load(row["t2_path"],    allow_pickle=False)
        label = np.load(row["label_path"], allow_pickle=False)

        row_axes = axes[row_idx] if n > 1 else axes

        # T1 RGB
        row_axes[0].imshow(to_rgb(t1))
        row_axes[0].axis("off")

        # T2 RGB
        row_axes[1].imshow(to_rgb(t2))
        row_axes[1].axis("off")

        # T1 SAR false color
        row_axes[2].imshow(to_sar_false_color(t1))
        row_axes[2].axis("off")

        # T2 SAR false color
        row_axes[3].imshow(to_sar_false_color(t2))
        row_axes[3].axis("off")

        # T1 NDVI
        im1 = row_axes[4].imshow(to_ndvi(t1), cmap="RdYlGn", vmin=-1, vmax=1)
        row_axes[4].axis("off")

        # T2 NDVI
        im2 = row_axes[5].imshow(to_ndvi(t2), cmap="RdYlGn", vmin=-1, vmax=1)
        row_axes[5].axis("off")

        # Label map
        row_axes[6].imshow(label, cmap=LABEL_CMAP, vmin=0, vmax=3, interpolation="nearest")
        row_axes[6].axis("off")

        # Row info label
        change_pct = float(row["change_pct"])
        row_axes[0].set_ylabel(
            f"#{row_idx+1}\nΔ={change_pct:.1f}%",
            color="white", fontsize=7, rotation=0, labelpad=35
        )

    # Add NDVI colorbar
    cbar_ax = fig.add_axes([0.63, -0.02, 0.08, 0.015])
    plt.colorbar(im2, cax=cbar_ax, orientation="horizontal", label="NDVI")
    cbar_ax.xaxis.label.set_color("white")
    cbar_ax.tick_params(colors="white")

    # Add legend for label map
    legend_ax = fig.add_axes([0.75, -0.04, 0.22, 0.04])
    legend_ax.axis("off")
    legend_ax.legend(
        handles    = make_legend(),
        loc        = "center",
        ncol       = 2,
        fontsize   = 7,
        labelcolor = "white",
        framealpha = 0,
    )

    plt.tight_layout(pad=0.5)
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="#1a1a2e")
    plt.close()
    print(f"  ✅ Saved sample grid → {out_path}")


def run(manifest_path: str, output_dir: str, n_samples: int):
    """Generate sample grids for all regions."""
    os.makedirs(output_dir, exist_ok=True)
    df = pd.read_csv(manifest_path)

    regions = df["region"].unique()
    for region_id in regions:
        region_df = df[df["region"] == region_id]
        # Prefer patches with actual change (change_pct > 5%)
        changed = region_df[region_df["change_pct"] > 5.0]
        if len(changed) >= n_samples // 2:
            region_df = pd.concat([
                changed.sample(min(n_samples // 2, len(changed)), random_state=42),
                region_df[region_df["change_pct"] <= 5.0].sample(
                    min(n_samples // 2, len(region_df[region_df["change_pct"] <= 5.0])), random_state=42
                )
            ])

        out_path = os.path.join(output_dir, f"{region_id}_sample_grid.png")
        print(f"\n🖼️  Generating sample grid: {region_id} ({len(region_df)} available)")
        plot_sample_grid(region_df, n_samples, region_id, out_path)

    print("\n✅ Sample generation complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate ForestSense sample grids")
    parser.add_argument("--manifest", default="data/processed/manifest.csv")
    parser.add_argument("--output",   default="outputs/visualizations/samples")
    parser.add_argument("--n",        type=int, default=8, help="Samples per region")
    args = parser.parse_args()

    run(args.manifest, args.output, args.n)
