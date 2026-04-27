"""
ForestSense — PyTorch Dataset
==============================
Provides torch.utils.data.Dataset classes for loading the pre-processed
ForestSense patch triplets (T1, T2, Label) during training and evaluation.

Usage:
    from data.dataset import ForestSenseDataset, get_dataloaders
    
    train_loader, val_loader, test_loader = get_dataloaders(
        manifest_path="data/processed/manifest.csv",
        batch_size=8,
        num_workers=4,
    )
"""

import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import albumentations as A
from pathlib import Path
from typing import Optional, List, Tuple, Dict


# ── Augmentation Pipelines ─────────────────────────────────────

def get_train_transforms(patch_size: int = 256) -> A.Compose:
    """
    Training augmentation pipeline.
    Applies identical spatial transforms to T1, T2, and Label simultaneously.
    """
    return A.Compose([
        A.RandomRotate90(p=0.5),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.ShiftScaleRotate(
            shift_limit  = 0.05,
            scale_limit  = 0.1,
            rotate_limit = 15,
            border_mode  = 0,
            p            = 0.3,
        ),
        # Elastic distortion simulates terrain variability
        A.ElasticTransform(alpha=120, sigma=6, p=0.15),
        # Gaussian noise on image data (not label)
        A.GaussNoise(var_limit=(0.001, 0.005), p=0.2),
    ], additional_targets={
        "t2_image": "image",   # Apply same spatial aug to T2
    })


def get_val_transforms() -> A.Compose:
    """Validation/test pipeline — no augmentation, just returns as-is."""
    return A.Compose([], additional_targets={"t2_image": "image"})


# ============================================================
#   DATASET CLASS
# ============================================================

class ForestSenseDataset(Dataset):
    """
    PyTorch Dataset for ForestSense multimodal change detection.
    
    Each sample is a dict:
    {
        't1'     : torch.Tensor (C, H, W) — T1 (2021) SAR+Optical stack, float32
        't2'     : torch.Tensor (C, H, W) — T2 (2023) SAR+Optical stack, float32
        'label'  : torch.Tensor (H, W)   — change label uint8 → long
        'meta'   : dict                   — patch_id, region, split, change_pct
    }
    
    Band layout (12 bands):
        0: SAR_VV
        1: SAR_VH
        2: SAR_ratio
        3: OPT_B2 (Blue)
        4: OPT_B3 (Green)
        5: OPT_B4 (Red)
        6: OPT_B8 (NIR)
        7: OPT_B11 (SWIR1)
        8: OPT_B12 (SWIR2)
        9: NDVI
       10: EVI
       11: NDWI
    
    Label classes:
        0 = Unchanged Non-Forest
        1 = Unchanged Forest (Stable)
        2 = Forest Loss
        3 = Forest Gain
    """

    def __init__(
        self,
        manifest_path : str,
        split         : str,
        regions       : Optional[List[str]] = None,
        transforms    = None,
        use_sar       : bool = True,
        use_optical   : bool = True,
        indices_only  : bool = False,
    ):
        """
        Args:
            manifest_path : Path to manifest.csv
            split         : 'train', 'val', or 'test'
            regions       : Filter by region(s), None = all regions
            transforms    : Albumentations Compose pipeline
            use_sar       : Include SAR bands (bands 0-2)
            use_optical   : Include optical bands (bands 3-8)
            indices_only  : Only use NDVI/EVI/NDWI (bands 9-11); overrides above
        """
        assert split in ("train", "val", "test"), f"split must be train/val/test, got '{split}'"

        self.transforms   = transforms
        self.use_sar      = use_sar
        self.use_optical  = use_optical
        self.indices_only = indices_only

        # ── Determine which bands to load ─────────────────────
        if indices_only:
            self.band_indices = [9, 10, 11]    # NDVI, EVI, NDWI only
        else:
            selected = []
            if use_sar:
                selected += [0, 1, 2]
            if use_optical:
                selected += list(range(3, 12))
            self.band_indices = selected or list(range(12))

        self.n_bands = len(self.band_indices)

        # ── Load & filter manifest ─────────────────────────────
        df = pd.read_csv(manifest_path)
        df = df[df["split"] == split]
        if regions:
            df = df[df["region"].isin(regions)]

        if len(df) == 0:
            raise ValueError(f"No patches found for split='{split}', regions={regions}")

        self.records = df.reset_index(drop=True)
        self.split   = split

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx: int) -> Dict:
        row = self.records.iloc[idx]

        # ── Load .npy patches ──────────────────────────────────
        t1    = np.load(row["t1_path"],    allow_pickle=False)   # (12, H, W)
        t2    = np.load(row["t2_path"],    allow_pickle=False)   # (12, H, W)
        label = np.load(row["label_path"], allow_pickle=False)   # (H, W)

        # ── Band selection ─────────────────────────────────────
        t1 = t1[self.band_indices]
        t2 = t2[self.band_indices]

        # ── Augmentation (albumentations expects HWC) ──────────
        if self.transforms is not None:
            t1_hwc  = t1.transpose(1, 2, 0)   # CHW → HWC
            t2_hwc  = t2.transpose(1, 2, 0)

            augmented = self.transforms(
                image    = t1_hwc,
                t2_image = t2_hwc,
                mask     = label,
            )
            t1    = augmented["image"].transpose(2, 0, 1)       # HWC → CHW
            t2    = augmented["t2_image"].transpose(2, 0, 1)
            label = augmented["mask"]

        # ── To Tensor ─────────────────────────────────────────
        t1_tensor    = torch.from_numpy(t1.copy()).float()
        t2_tensor    = torch.from_numpy(t2.copy()).float()
        label_tensor = torch.from_numpy(label.astype(np.int64)).long()

        return {
            "t1"   : t1_tensor,
            "t2"   : t2_tensor,
            "label": label_tensor,
            "meta" : {
                "patch_id"  : row["patch_id"],
                "region"    : row["region"],
                "split"     : row["split"],
                "change_pct": float(row["change_pct"]),
            },
        }

    def get_class_weights(self, stats_path: str) -> torch.Tensor:
        """Load pre-computed class weights from stats JSON."""
        import json
        with open(stats_path) as f:
            stats = json.load(f)
        # Average weights across regions present in this dataset
        all_weights = []
        for region in self.records["region"].unique():
            if region in stats:
                all_weights.append(stats[region]["class_weights"])
        if all_weights:
            w = np.mean(all_weights, axis=0)
        else:
            w = np.ones(4, dtype=np.float32)
        return torch.tensor(w, dtype=torch.float32)


# ============================================================
#   DATALOADER FACTORY
# ============================================================

def get_dataloaders(
    manifest_path : str,
    batch_size    : int = 8,
    num_workers   : int = 4,
    pin_memory    : bool = True,
    regions       : Optional[List[str]] = None,
    use_sar       : bool = True,
    use_optical   : bool = True,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Build train, val, and test DataLoaders with appropriate transforms.
    
    Returns:
        (train_loader, val_loader, test_loader)
    """
    train_tf = get_train_transforms()
    val_tf   = get_val_transforms()

    shared_kwargs = dict(
        manifest_path = manifest_path,
        regions       = regions,
        use_sar       = use_sar,
        use_optical   = use_optical,
    )

    train_ds = ForestSenseDataset(split="train", transforms=train_tf, **shared_kwargs)
    val_ds   = ForestSenseDataset(split="val",   transforms=val_tf,   **shared_kwargs)
    test_ds  = ForestSenseDataset(split="test",  transforms=val_tf,   **shared_kwargs)

    loader_kwargs = dict(
        batch_size  = batch_size,
        num_workers = num_workers,
        pin_memory  = pin_memory,
    )

    train_loader = DataLoader(train_ds, shuffle=True,  **loader_kwargs)
    val_loader   = DataLoader(val_ds,   shuffle=False, **loader_kwargs)
    test_loader  = DataLoader(test_ds,  shuffle=False, **loader_kwargs)

    print(f"📦 DataLoaders ready:")
    print(f"   Train : {len(train_ds):,} patches ({len(train_loader)} batches)")
    print(f"   Val   : {len(val_ds):,} patches ({len(val_loader)} batches)")
    print(f"   Test  : {len(test_ds):,} patches ({len(test_loader)} batches)")
    print(f"   Input : {train_ds.n_bands} bands × {256}×{256} pixels")

    return train_loader, val_loader, test_loader
