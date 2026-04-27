"""
ForestSense — GEE Data Processor
==================================
After GEE export tasks complete and GeoTIFF files land in Google Drive:
  1. Mount Drive (Colab) or set RAW_DATA_DIR to local path
  2. Run this script to:
       - Tile large GeoTIFFs into 256×256 patches
       - Normalize bands (SAR in dB, Optical 0–1)
       - Pair T1/T2/Label patches
       - Split into train/val/test sets
       - Save as .npy patch files (fast loading during training)
       - Write a CSV manifest of all patches

Output structure:
    data/processed/
    ├── patches/
    │   ├── western_ghats/
    │   │   ├── train/  val/  test/
    │   │   │   ├── <id>_t1.npy       # (12, 256, 256) float32
    │   │   │   ├── <id>_t2.npy       # (12, 256, 256) float32
    │   │   │   └── <id>_label.npy    # (256, 256) uint8
    │   ├── sundarbans/
    │   └── saranda/
    └── manifest.csv
"""

import os
import json
import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import Window
from pathlib import Path
from tqdm import tqdm
import math
import hashlib
import warnings
warnings.filterwarnings("ignore")

# ============================================================
#   CONFIGURATION
# ============================================================

# ── Paths ─────────────────────────────────────────────────────
# Change RAW_DATA_DIR to the path where your GEE exports are stored.
# In Google Colab with Drive mounted: "/content/drive/MyDrive/ForestSense_Data"
RAW_DATA_DIR  = os.environ.get("FORESTSENSE_RAW_DATA", "/content/drive/MyDrive/ForestSense_Data")
PROCESSED_DIR = "data/processed"

# ── Patch Configuration ───────────────────────────────────────
PATCH_SIZE   = 256    # Pixels per patch (256×256)
STRIDE       = 128    # Overlap stride (50% overlap for augmentation richness)
MIN_VALID    = 0.70   # Minimum fraction of valid pixels to keep a patch

# ── Band Information ──────────────────────────────────────────
SAR_BANDS     = ["SAR_VV", "SAR_VH", "SAR_ratio"]        # indices 0,1,2
OPTICAL_BANDS = ["OPT_B2", "OPT_B3", "OPT_B4", "OPT_B8",
                 "OPT_B11", "OPT_B12", "NDVI", "EVI", "NDWI"]  # indices 3–11
ALL_BANDS     = SAR_BANDS + OPTICAL_BANDS                 # 12 total

N_BANDS   = len(ALL_BANDS)    # 12
N_CLASSES = 4                 # 0=Unchanged NF, 1=Stable Forest, 2=Loss, 3=Gain

# ── Dataset Split ─────────────────────────────────────────────
SPLIT_RATIOS = {"train": 0.70, "val": 0.15, "test": 0.15}

# ── Normalization Stats (global percentile clipping) ──────────
# SAR bands in dB: typically -25 to 5 dB
SAR_CLIP = {"min": -25.0, "max": 5.0}
# Optical indices already 0–1 (NDVI/EVI/NDWI)
# Optical reflectance bands 0–1 after /10000 in GEE
OPT_CLIP = {"min": 0.0, "max": 1.0}


# ============================================================
#   NORMALIZATION
# ============================================================

def normalize_sar_band(data, clip_min=SAR_CLIP["min"], clip_max=SAR_CLIP["max"]):
    """Clip and min-max normalize SAR band to [0, 1]."""
    data = np.clip(data, clip_min, clip_max)
    return (data - clip_min) / (clip_max - clip_min)


def normalize_optical_band(data, p2=None, p98=None):
    """
    Normalize optical reflectance band using 2nd–98th percentile clipping
    on valid pixels to handle outliers robustly.
    """
    valid = data[data > 0]
    if len(valid) == 0:
        return np.zeros_like(data, dtype=np.float32)
    if p2 is None:
        p2  = np.percentile(valid, 2)
    if p98 is None:
        p98 = np.percentile(valid, 98)
    data = np.clip(data, p2, p98)
    denom = p98 - p2
    if denom == 0:
        return np.zeros_like(data, dtype=np.float32)
    return ((data - p2) / denom).astype(np.float32)


def normalize_stack(stack_data, band_names):
    """
    Normalize a (C, H, W) stack where:
      - SAR bands (VV, VH, ratio) → dB clipping
      - Optical reflectance bands (B2–B12) → percentile clipping
      - Index bands (NDVI, EVI, NDWI) → already -1 to 1, clip to [-1, 1] then scale
    
    Returns:
        np.ndarray of shape (C, H, W), float32, all values in [0, 1]
    """
    normalized = np.zeros_like(stack_data, dtype=np.float32)
    for i, band_name in enumerate(band_names):
        band_data = stack_data[i].astype(np.float32)
        if band_name in ["SAR_VV", "SAR_VH", "SAR_ratio"]:
            normalized[i] = normalize_sar_band(band_data)
        elif band_name in ["NDVI", "EVI", "NDWI"]:
            # Vegetation indices: clip to [-1, 1] then scale to [0, 1]
            band_data = np.clip(band_data, -1.0, 1.0)
            normalized[i] = (band_data + 1.0) / 2.0
        else:
            # Optical reflectance bands
            normalized[i] = normalize_optical_band(band_data)
    return normalized


# ============================================================
#   PATCH EXTRACTION
# ============================================================

def read_geotiff(filepath):
    """
    Read a GeoTIFF into a numpy array (C, H, W).
    Returns (data, profile) where profile contains CRS/transform metadata.
    """
    with rasterio.open(filepath) as src:
        data    = src.read().astype(np.float32)
        profile = src.profile
    return data, profile


def compute_patch_validity(patch):
    """
    Returns the fraction of valid (non-NaN, non-zero) pixels in a patch.
    Used to filter out mostly-nodata patches near image edges.
    """
    total = patch.shape[-1] * patch.shape[-2]
    if patch.ndim == 3:
        valid = np.sum(~np.isnan(patch[0]) & (patch[0] != 0))
    else:
        valid = np.sum(~np.isnan(patch) & (patch != 0))
    return valid / total


def extract_patches(t1_data, t2_data, label_data, patch_size=PATCH_SIZE, stride=STRIDE):
    """
    Extract aligned (T1, T2, Label) patches from full-scene arrays.
    
    Args:
        t1_data    : np.ndarray (C, H, W) — T1 multimodal stack (normalized)
        t2_data    : np.ndarray (C, H, W) — T2 multimodal stack (normalized)
        label_data : np.ndarray (H, W)   — pseudo-label map (uint8)
        patch_size : int — square patch size in pixels
        stride     : int — sliding window stride
    
    Returns:
        List of (t1_patch, t2_patch, label_patch) tuples
    """
    _, H, W = t1_data.shape
    patches = []

    rows = range(0, H - patch_size + 1, stride)
    cols = range(0, W - patch_size + 1, stride)

    for r in rows:
        for c in cols:
            t1_patch  = t1_data[:, r:r+patch_size, c:c+patch_size]
            t2_patch  = t2_data[:, r:r+patch_size, c:c+patch_size]
            lbl_patch = label_data[r:r+patch_size, c:c+patch_size]

            # Skip low-validity patches
            if compute_patch_validity(t1_patch) < MIN_VALID:
                continue
            if compute_patch_validity(t2_patch) < MIN_VALID:
                continue

            patches.append((
                t1_patch.astype(np.float32),
                t2_patch.astype(np.float32),
                lbl_patch.astype(np.uint8),
            ))

    return patches


# ============================================================
#   CLASS BALANCE ANALYSIS
# ============================================================

def compute_class_distribution(patches):
    """
    Compute per-class pixel count across all patches.
    Used to set class weights for weighted loss during training.
    
    Returns:
        dict: {'class_name': count, ...}
    """
    class_names = {0: "Unchanged_NF", 1: "Stable_Forest", 2: "Forest_Loss", 3: "Forest_Gain"}
    counts = {0: 0, 1: 0, 2: 0, 3: 0}
    for _, _, lbl in patches:
        for cls in counts:
            counts[cls] += int(np.sum(lbl == cls))
    
    total = sum(counts.values())
    distribution = {}
    for cls, count in counts.items():
        pct = 100.0 * count / total if total > 0 else 0
        distribution[class_names[cls]] = {"count": count, "percentage": round(pct, 2)}
    
    return distribution, counts


def compute_class_weights(counts):
    """
    Compute inverse-frequency class weights for weighted loss.
    
    Returns:
        np.ndarray of shape (N_CLASSES,)
    """
    total  = sum(counts.values())
    n_cls  = len(counts)
    weights = np.array([
        total / (n_cls * counts[c]) if counts[c] > 0 else 1.0
        for c in sorted(counts.keys())
    ], dtype=np.float32)
    # Normalize so mean weight ≈ 1
    weights = weights / weights.mean()
    return weights


# ============================================================
#   DATASET SPLIT & SAVE
# ============================================================

def deterministic_split(patches, split_ratios=SPLIT_RATIOS, seed=42):
    """
    Deterministically split patch list into train/val/test.
    Uses a seeded shuffle to ensure reproducibility.
    
    Returns:
        dict: {'train': [...], 'val': [...], 'test': [...]}
    """
    rng = np.random.default_rng(seed)
    indices = np.arange(len(patches))
    rng.shuffle(indices)

    n = len(patches)
    n_train = math.floor(n * split_ratios["train"])
    n_val   = math.floor(n * split_ratios["val"])

    splits = {
        "train": [patches[i] for i in indices[:n_train]],
        "val"  : [patches[i] for i in indices[n_train:n_train + n_val]],
        "test" : [patches[i] for i in indices[n_train + n_val:]],
    }
    return splits


def save_patches(splits, region_id, out_dir, manifest_rows):
    """
    Save (T1, T2, Label) patch triplets as .npy files.
    Appends row info to manifest_rows list.
    """
    for split_name, patch_list in splits.items():
        split_dir = Path(out_dir) / region_id / split_name
        split_dir.mkdir(parents=True, exist_ok=True)

        for idx, (t1, t2, lbl) in enumerate(tqdm(patch_list, desc=f"  Saving {region_id}/{split_name}", leave=False)):
            patch_id = f"{region_id}_{split_name}_{idx:05d}"
            np.save(str(split_dir / f"{patch_id}_t1.npy"),    t1)
            np.save(str(split_dir / f"{patch_id}_t2.npy"),    t2)
            np.save(str(split_dir / f"{patch_id}_label.npy"), lbl)

            # Compute class distribution for this patch
            change_pixels = int(np.sum(lbl >= 2))
            total_pixels  = lbl.size
            change_pct    = round(100.0 * change_pixels / total_pixels, 2)

            manifest_rows.append({
                "patch_id"     : patch_id,
                "region"       : region_id,
                "split"        : split_name,
                "t1_path"      : str(split_dir / f"{patch_id}_t1.npy"),
                "t2_path"      : str(split_dir / f"{patch_id}_t2.npy"),
                "label_path"   : str(split_dir / f"{patch_id}_label.npy"),
                "patch_size"   : PATCH_SIZE,
                "n_bands"      : N_BANDS,
                "change_pixels": change_pixels,
                "total_pixels" : total_pixels,
                "change_pct"   : change_pct,
            })


# ============================================================
#   MAIN PROCESSING PIPELINE
# ============================================================

def process_region(region_id, raw_dir, out_dir):
    """
    Full preprocessing pipeline for a single region.
    
    Args:
        region_id: str — e.g., 'western_ghats'
        raw_dir  : str — path to raw GeoTIFF exports
        out_dir  : str — where to save processed patches
    
    Returns:
        (patch_count, distribution, class_weights)
    """
    print(f"\n{'='*60}")
    print(f"🌿 Processing region: {region_id}")
    print(f"{'='*60}")

    prefix = f"forestsense_{region_id}"
    t1_path  = os.path.join(raw_dir, f"{prefix}_T1_2021_stack.tif")
    t2_path  = os.path.join(raw_dir, f"{prefix}_T2_2023_stack.tif")
    lbl_path = os.path.join(raw_dir, f"{prefix}_labels.tif")

    # Validate files
    for p, name in [(t1_path, "T1"), (t2_path, "T2"), (lbl_path, "Labels")]:
        if not os.path.exists(p):
            raise FileNotFoundError(
                f"❌ {name} file not found: {p}\n"
                f"   Make sure GEE export tasks completed and files are in: {raw_dir}"
            )

    # ── Read ─────────────────────────────────────────────────
    print("  📂 Reading GeoTIFFs...")
    t1_raw, t1_profile = read_geotiff(t1_path)
    t2_raw, _          = read_geotiff(t2_path)
    lbl_raw, _         = read_geotiff(lbl_path)

    print(f"  Shape: T1={t1_raw.shape}, T2={t2_raw.shape}, Labels={lbl_raw.shape}")

    # Squeeze label band: (1, H, W) → (H, W)
    if lbl_raw.ndim == 3:
        lbl_raw = lbl_raw[0]

    # ── Normalize ────────────────────────────────────────────
    print("  🔧 Normalizing bands...")
    t1_norm = normalize_stack(t1_raw, ALL_BANDS)
    t2_norm = normalize_stack(t2_raw, ALL_BANDS)

    # ── Replace NaN with 0 ───────────────────────────────────
    t1_norm = np.nan_to_num(t1_norm, nan=0.0)
    t2_norm = np.nan_to_num(t2_norm, nan=0.0)

    # ── Extract Patches ──────────────────────────────────────
    print(f"  ✂️  Extracting {PATCH_SIZE}×{PATCH_SIZE} patches (stride={STRIDE})...")
    patches = extract_patches(t1_norm, t2_norm, lbl_raw.astype(np.uint8))
    print(f"  → {len(patches)} valid patches extracted")

    if len(patches) == 0:
        raise ValueError(f"No valid patches for region '{region_id}'. Check data quality.")

    # ── Class Distribution ───────────────────────────────────
    distribution, counts = compute_class_distribution(patches)
    class_weights        = compute_class_weights(counts)
    print("  📊 Class distribution:")
    for cls, info in distribution.items():
        print(f"     {cls:20s}: {info['percentage']:5.1f}%  ({info['count']:,} px)")

    # ── Split ────────────────────────────────────────────────
    splits = deterministic_split(patches)
    for split, p_list in splits.items():
        print(f"  {split:5s}: {len(p_list)} patches")

    # ── Save ─────────────────────────────────────────────────
    manifest_rows = []
    save_patches(splits, region_id, out_dir, manifest_rows)

    # ── Save Region Stats ────────────────────────────────────
    region_stats = {
        "region_id"     : region_id,
        "total_patches" : len(patches),
        "split_counts"  : {k: len(v) for k, v in splits.items()},
        "class_dist"    : distribution,
        "class_weights" : class_weights.tolist(),
        "patch_size"    : PATCH_SIZE,
        "n_bands"       : N_BANDS,
        "band_names"    : ALL_BANDS,
    }
    stats_path = Path(out_dir) / region_id / "stats.json"
    with open(stats_path, "w") as f:
        json.dump(region_stats, f, indent=2)

    print(f"  ✅ Done. Stats saved → {stats_path}")
    return len(patches), distribution, class_weights, manifest_rows


def run_preprocessing():
    """Run preprocessing pipeline for all three regions."""
    print("🚀 ForestSense — Data Preprocessing")
    print(f"   Raw data dir : {RAW_DATA_DIR}")
    print(f"   Output dir   : {PROCESSED_DIR}")

    patches_dir = os.path.join(PROCESSED_DIR, "patches")
    os.makedirs(patches_dir, exist_ok=True)

    regions = ["western_ghats", "sundarbans", "saranda"]
    all_manifest_rows = []
    all_stats = {}

    for region_id in regions:
        patch_count, distribution, class_weights, manifest_rows = process_region(
            region_id,
            raw_dir = RAW_DATA_DIR,
            out_dir = patches_dir,
        )
        all_manifest_rows.extend(manifest_rows)
        all_stats[region_id] = {
            "total_patches": patch_count,
            "class_dist"   : distribution,
            "class_weights": class_weights.tolist(),
        }

    # ── Save Global Manifest ─────────────────────────────────
    manifest_df   = pd.DataFrame(all_manifest_rows)
    manifest_path = os.path.join(PROCESSED_DIR, "manifest.csv")
    manifest_df.to_csv(manifest_path, index=False)

    # ── Save Global Stats ─────────────────────────────────────
    stats_path = os.path.join(PROCESSED_DIR, "global_stats.json")
    with open(stats_path, "w") as f:
        json.dump(all_stats, f, indent=2)

    # ── Summary Report ────────────────────────────────────────
    print("\n" + "=" * 60)
    print("📋 PREPROCESSING SUMMARY")
    print("=" * 60)
    total = len(manifest_df)
    train = len(manifest_df[manifest_df["split"] == "train"])
    val   = len(manifest_df[manifest_df["split"] == "val"])
    test  = len(manifest_df[manifest_df["split"] == "test"])
    print(f"  Total patches  : {total:,}")
    print(f"  Train          : {train:,} ({100*train/total:.1f}%)")
    print(f"  Val            : {val:,}   ({100*val/total:.1f}%)")
    print(f"  Test           : {test:,}  ({100*test/total:.1f}%)")
    print(f"  Manifest saved : {manifest_path}")
    print(f"  Stats saved    : {stats_path}")
    print("\n✅ Preprocessing complete! Ready for training.")


if __name__ == "__main__":
    run_preprocessing()
