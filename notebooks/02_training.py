"""
ForestSense — Colab Training Notebook (Phase 2)
================================================
Run this cell by cell in Google Colab after Phase 1 preprocessing is complete.
The processed patches must be in your Google Drive at:
    /content/drive/MyDrive/forestsense/data/processed/

CELL STRUCTURE:
  Cell 1  — Install dependencies
  Cell 2  — Mount Drive, set paths
  Cell 3  — Fix manifest paths (IMPORTANT)
  Cell 4  — Verify dataset
  Cell 5  — Build model
  Cell 6  — Run training
  Cell 7  — Evaluate on test set + save figures
  Cell 8  — Copy outputs to Drive
"""

# ── CELL 1: Install Dependencies ───────────────────────────────────────────────
import subprocess
subprocess.run([
    "pip", "install", "-q",
    "torch==2.2.2", "torchvision==0.17.2",
    "timm==1.0.3",
    "albumentations==1.4.7",
    "numpy", "pandas", "scikit-learn",
    "matplotlib", "seaborn",
    "tqdm", "einops",
], check=True)
print("✅ Dependencies installed.")


# ── CELL 2: Mount Drive & Set Up Paths ─────────────────────────────────────────
from google.colab import drive
drive.mount("/content/drive")

import os, sys

# ── WHERE YOUR FORESTSENSE PROJECT LIVES ──────────────
PROJECT_ROOT = "/content/drive/MyDrive/forestsense"
# If you placed it elsewhere, change the path above ↑

sys.path.insert(0, PROJECT_ROOT)
os.chdir(PROJECT_ROOT)     # Critical: makes relative paths in manifest work

print(f"✅ Working directory set to: {os.getcwd()}")
print(f"✅ Python path includes: {PROJECT_ROOT}")


# ── CELL 3: Fix Manifest Paths (IMPORTANT!) ────────────────────────────────────
"""
The manifest.csv was generated with relative paths during Phase 1.
We need to make sure those relative paths resolve correctly from PROJECT_ROOT.
This cell does a quick sanity check.
"""
import pandas as pd
from pathlib import Path

manifest_path = "data/processed/manifest.csv"
df = pd.read_csv(manifest_path)

print(f"📋 Manifest loaded: {len(df)} patches")
print(f"   Columns: {list(df.columns)}")

# Check a few paths
sample = df.iloc[0]
for col in ["t1_path", "t2_path", "label_path"]:
    full_path = Path(PROJECT_ROOT) / sample[col]
    exists    = full_path.exists()
    status    = "✅" if exists else "❌"
    print(f"  {status} {col}: {str(full_path)[:80]}")

if not Path(PROJECT_ROOT, sample["t1_path"]).exists():
    print("\n⚠️  Paths not resolving. Trying to remap...")
    # Attempt to remap if paths are absolute from old Colab session
    def remap_path(p):
        p = str(p)
        # Replace old absolute Colab path with the project root relative path
        for old_prefix in ["/content/data/", "/content/forestsense/data/"]:
            if p.startswith(old_prefix):
                return p.replace(old_prefix, "data/")
        return p

    for col in ["t1_path", "t2_path", "label_path"]:
        df[col] = df[col].apply(remap_path)

    # Save fixed manifest
    df.to_csv(manifest_path, index=False)
    print("✅ Manifest paths fixed and saved.")

print(f"\n📊 Split distribution:")
print(df["split"].value_counts().to_string())
print(f"\n📊 Region distribution:")
print(df["region"].value_counts().to_string())


# ── CELL 4: Verify Dataset ─────────────────────────────────────────────────────
from data.dataset import get_dataloaders
import numpy as np

train_loader, val_loader, test_loader = get_dataloaders(
    manifest_path = manifest_path,
    batch_size    = 8,
    num_workers   = 2,
)

# Load one batch to verify shapes
batch = next(iter(train_loader))
print(f"\n✅ Batch verification:")
print(f"   t1 shape   : {tuple(batch['t1'].shape)}")
print(f"   t2 shape   : {tuple(batch['t2'].shape)}")
print(f"   label shape: {tuple(batch['label'].shape)}")
print(f"   t1 range   : [{batch['t1'].min():.3f}, {batch['t1'].max():.3f}]")
print(f"   label vals : {batch['label'].unique().tolist()}")
print(f"   regions    : {set(batch['meta']['region'])}")


# ── CELL 5: Build Model ─────────────────────────────────────────────────────────
import torch
from model.architecture import build_model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"📱 Device: {device}")
if device.type == "cuda":
    print(f"   GPU: {torch.cuda.get_device_name(0)}")
    print(f"   VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

# Build model — ResNet-18 is optimal for ~2k patches (less overfitting, faster training)
# Upgrade to backbone='resnet34' only if you collect more data (>10k patches)
model = build_model(
    in_channels = 12,
    num_classes = 4,
    backbone    = "resnet18",
    pretrained  = True,
).to(device)


# ── CELL 6: Run Training ────────────────────────────────────────────────────────
"""
Training Configuration:
  - epochs=50     : Max 50 epochs (early stopping may end it sooner)
  - batch_size=8  : Safe for T4 GPU (16GB)
  - lr=1e-4       : Good starting point for fine-tuning
  - patience=10   : Stop if no improvement for 10 epochs

Expected training time:
  T4 GPU  : ~3-5 min/epoch  → 50 epochs ≈ 3-4 hours
  V100 GPU: ~1-2 min/epoch  → 50 epochs ≈ 1.5-2 hours
  CPU     : ~30 min/epoch   → NOT RECOMMENDED

Tip: Enable GPU runtime: Runtime → Change runtime type → T4 GPU
"""
from train import train

# ── Reproducibility seed (REQUIRED for paper) ──────────────────────────────────
import random

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Makes CUDA ops deterministic (slightly slower, but reproducible)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False
    print(f"🌱 Random seed set to {seed} — results are reproducible.")

set_seed(42)

config = dict(
    manifest    = manifest_path,
    stats_dir   = "data/processed",
    out_dir     = "outputs",
    epochs      = 50,
    batch_size  = 8,       # Optimal for T4 GPU — do NOT reduce to 4 (noisier gradients)
    lr          = 3e-4,    # Slightly higher than default for small dataset with pretrained backbone
    weight_decay= 1e-4,
    backbone    = "resnet18",
    pretrained  = True,
    use_focal   = True,    # Forest Loss ≈ 1.7% — severe imbalance requires Focal Loss
    patience    = 10,
    num_workers = 2,
    amp         = True,
    seed        = 42,      # For reproducibility (paper requirement)
)

trained_model, test_results = train(config)


# ── CELL 7: Generate Visual Outputs ────────────────────────────────────────────
"""
Generates:
  1. Sample change detection panels (T1 | T2 | Prediction | GT)
  2. Confusion matrix figure
  3. Per-region analytics charts
All saved to outputs/visualizations/
"""
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from model.architecture import build_model
from model.metrics import ChangeDetectionMetrics
from data.dataset import ForestSenseDataset, get_val_transforms
from utils.visualize import (
    plot_change_panel, plot_confusion_matrix, save_figure,
    LABEL_CMAP
)

# Load best model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = build_model(backbone="resnet18", pretrained=False).to(device)
ckpt  = torch.load("outputs/checkpoints/best_model.pth", map_location=device)
model.load_state_dict(ckpt["model_state_dict"])
model.eval()

# ── Confusion Matrix ──────────────────────────────────────────
from sklearn.metrics import confusion_matrix as sk_cm
import numpy as np

conf_mat = np.array(test_results["conf_matrix"])
fig = plot_confusion_matrix(
    conf_mat,
    title     = "ForestSense — Test Set Confusion Matrix",
    save_path = "outputs/visualizations/confusion_matrix.png",
)
print("✅ Confusion matrix saved")

# ── Sample Prediction Panels ──────────────────────────────────
test_ds = ForestSenseDataset(
    manifest_path = manifest_path,
    split         = "test",
    transforms    = get_val_transforms(),
)

os.makedirs("outputs/visualizations/panels", exist_ok=True)
n_samples = min(6, len(test_ds))

for i in range(n_samples):
    sample = test_ds[i]
    t1  = sample["t1"].unsqueeze(0).to(device)
    t2  = sample["t2"].unsqueeze(0).to(device)
    lbl = sample["label"].numpy()

    with torch.no_grad():
        logit = model(t1, t2)
    pred_mask = logit.argmax(dim=1).squeeze().cpu().numpy()

    def _rgb(stack_np):
        # Bands 5=Red, 4=Green, 3=Blue (indices 5,4,3)
        r, g, b = stack_np[5], stack_np[4], stack_np[3]
        rgb = np.stack([r, g, b], axis=-1)
        return np.clip((rgb - rgb.min()) / (rgb.max() - rgb.min() + 1e-8), 0, 1)

    def _sar(stack_np):
        return stack_np[0]  # SAR VV band

    def _ndvi(stack_np):
        return stack_np[9]  # NDVI band

    t1_np = sample["t1"].numpy()
    t2_np = sample["t2"].numpy()

    region  = sample["meta"]["region"]
    chg_pct = sample["meta"]["change_pct"]

    fig = plot_change_panel(
        t1_rgb    = _rgb(t1_np),
        t2_rgb    = _rgb(t2_np),
        t1_sar    = _sar(t1_np),
        t2_sar    = _sar(t2_np),
        pred_mask = pred_mask,
        gt_mask   = lbl,
        ndvi_t1   = _ndvi(t1_np),
        ndvi_t2   = _ndvi(t2_np),
        title     = f"Region: {region} | Change: {chg_pct:.1f}%",
        save_path = f"outputs/visualizations/panels/sample_{i+1}_{region}.png",
    )

print(f"✅ {n_samples} prediction panels saved")


# ── CELL 8: Copy All Outputs to Google Drive ───────────────────────────────────
import shutil

drive_output = "/content/drive/MyDrive/forestsense/outputs"
shutil.copytree("outputs", drive_output, dirs_exist_ok=True)
print(f"✅ All outputs saved to Drive: {drive_output}")
print("""
📁 Your Drive now contains:
   outputs/
   ├── checkpoints/
   │   ├── best_model.pth         ← Best model weights (for paper)
   │   └── latest_model.pth
   ├── visualizations/
   │   ├── confusion_matrix.png   ← Paper figure
   │   ├── training_curves.png    ← Paper figure
   │   └── panels/
   │       └── sample_*.png       ← Paper figures
   ├── training_log.csv           ← All epoch metrics
   └── test_metrics.json          ← Final accuracy table for paper
""")
