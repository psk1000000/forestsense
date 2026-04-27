"""
ForestSense — Google Colab Data Acquisition Notebook
=======================================================
Copy-paste this into a Google Colab notebook OR run as:
    !python notebooks/01_data_acquisition.py

Cells are marked with # ── CELL N ──── for easy Colab splitting.

Prerequisites:
  - Sign up for Google Earth Engine at: https://earthengine.google.com/
  - Create a GEE Cloud Project at: https://console.cloud.google.com/
  - Set GEE_PROJECT below to your project ID
"""

# ── CELL 1: Install Dependencies ───────────────────────────────────────────────
"""
Run this cell first. Restart runtime after installation.
"""
import subprocess
subprocess.run([
    "pip", "install", "-q",
    "earthengine-api==0.1.390",
    "geemap==0.32.0",
    "rasterio==1.3.10",
    "numpy", "pandas", "tqdm",
])
print("✅ Dependencies installed. Restart the runtime if this is the first run.")


# ── CELL 2: Mount Google Drive ─────────────────────────────────────────────────
"""
Mount Drive to save exported GeoTIFFs there.
"""
try:
    from google.colab import drive
    drive.mount("/content/drive")
    DRIVE_MOUNTED = True
    print("✅ Google Drive mounted at /content/drive")
except ImportError:
    DRIVE_MOUNTED = False
    print("⚠️  Not running in Colab. Drive mount skipped.")


# ── CELL 3: GEE Authentication & Configuration ────────────────────────────────
import ee
import os
import json
import sys

# ╔══════════════════════════════════════════════════════════════════╗
# ║  ⚠️  CONFIGURE THIS SECTION BEFORE RUNNING                      ║
# ║  Replace the placeholder with your actual GEE Cloud Project ID  ║
# ╚══════════════════════════════════════════════════════════════════╝
GEE_PROJECT  = "beaming-prism-494018-k1"    # ← YOUR GEE PROJECT ID HERE
DRIVE_FOLDER = "ForestSense_Data"       # Folder in Google Drive

# Local output (for non-Colab or after downloading from Drive)
LOCAL_OUT = "/content/drive/MyDrive/ForestSense_Data" if DRIVE_MOUNTED else "data/raw"
os.makedirs(LOCAL_OUT, exist_ok=True)

# ── Authenticate ──────────────────────────────────────────────────
try:
    ee.Initialize(project=GEE_PROJECT)
    print(f"✅ GEE initialized with project: {GEE_PROJECT}")
except Exception as e:
    print(f"⚠️  GEE init failed: {e}")
    print("   Running ee.Authenticate() ...")
    ee.Authenticate()
    ee.Initialize(project=GEE_PROJECT)
    print("✅ GEE authenticated & initialized.")


# ── CELL 4: Import Acquisition Module ─────────────────────────────────────────
# Add project root to path
sys.path.insert(0, "/content")

# If running standalone in Colab, paste the functions below directly.
# Otherwise, import from the module:
try:
    from data.data_acquisition import (
        REGIONS, get_study_geometry,
        get_sentinel1_composite, get_sentinel2_composite,
        build_multimodal_stack, generate_pseudo_labels,
        get_ndvi_timeseries, export_to_drive, check_tasks,
        T1_START, T1_END, T2_START, T2_END, EXPORT_SCALE
    )
    print("✅ Acquisition module imported.")
except ImportError:
    print("⚠️  Module not found. Make sure forestsense/ is in /content/")
    print("   Upload the project folder to Colab or clone from GitHub.")


# ── CELL 5: Preview Region Info ───────────────────────────────────────────────
print("\n📍 Study Regions:")
print("-" * 70)
for region_id, info in REGIONS.items():
    g = get_study_geometry(info["bbox"])
    area = g.area().getInfo() / 1e6  # km²
    print(f"  [{region_id}]")
    print(f"    Name       : {info['name']}")
    print(f"    Forest Type: {info['forest_type']}")
    print(f"    Area (est) : ~{area:.0f} km²")
    print(f"    BBox       : {info['bbox']}")
    print()


# ── CELL 6: Test Single Region — Verify GEE Access ────────────────────────────
"""
Test with just Western Ghats first to verify GEE connectivity.
This cell computes stats without exporting.
"""
print("🔎 Testing GEE access for Western Ghats...")
test_geom = get_study_geometry(REGIONS["western_ghats"]["bbox"])

# Quick test: get mean SAR backscatter
s1_test = get_sentinel1_composite(test_geom, "2023-01-01", "2023-03-31")
stats = s1_test.select("VV").reduceRegion(
    reducer  = ee.Reducer.mean(),
    geometry = test_geom,
    scale    = 100,        # coarse scale for speed
    maxPixels= 1e9,
).getInfo()
print(f"  Mean SAR VV (Western Ghats, Q1 2023): {stats.get('VV', 'N/A'):.3f} dB")
print("✅ GEE access confirmed!")


# ── CELL 7: Run Full Acquisition (ALL REGIONS) ────────────────────────────────
"""
This cell submits export tasks to GEE for all 3 regions.
Each export runs in GEE's background — tasks typically complete in 30–90 minutes.

Monitor at: https://code.earthengine.google.com/tasks
Files appear in Google Drive → ForestSense_Data/
"""

metadata = {}
all_tasks = []

for region_id, region_info in REGIONS.items():
    print(f"\n{'='*60}")
    print(f"🌿 Processing: {region_info['name']}")
    geometry = get_study_geometry(region_info["bbox"])

    # Build stacks
    t1_stack = build_multimodal_stack(geometry, T1_START, T1_END, "T1-2021")
    t2_stack = build_multimodal_stack(geometry, T2_START, T2_END, "T2-2023")

    # Generate pseudo-labels
    label_map = generate_pseudo_labels(t1_stack, t2_stack, geometry)

    # NDVI time-series for analytics
    ndvi_series = get_ndvi_timeseries(geometry, year_start=2018, year_end=2023)

    # Submit export tasks
    prefix = f"forestsense_{region_id}"
    region_bbox = geometry.bounds().getInfo()["coordinates"]

    tasks = [
        export_to_drive(t1_stack,  f"{prefix}_T1_2021_stack", DRIVE_FOLDER, region_bbox),
        export_to_drive(t2_stack,  f"{prefix}_T2_2023_stack", DRIVE_FOLDER, region_bbox),
        export_to_drive(label_map, f"{prefix}_labels",         DRIVE_FOLDER, region_bbox),
    ]
    all_tasks.extend(tasks)

    metadata[region_id] = {
        **region_info,
        "t1_period"   : {"start": T1_START, "end": T1_END},
        "t2_period"   : {"start": T2_START, "end": T2_END},
        "ndvi_series" : ndvi_series,
        "export_scale": EXPORT_SCALE,
    }

# Save metadata
meta_path = os.path.join(LOCAL_OUT, "acquisition_metadata.json")
with open(meta_path, "w") as f:
    json.dump(metadata, f, indent=2)
print(f"\n📝 Metadata → {meta_path}")

print("\n⏳ All tasks submitted. Monitor at https://code.earthengine.google.com/tasks")


# ── CELL 8: Check Task Status ──────────────────────────────────────────────────
"""
Run this cell periodically to check export task status.
"""
print("📊 GEE Task Status:")
check_tasks()


# ── CELL 9: Run Preprocessing (after exports complete) ────────────────────────
"""
Run this cell ONLY after all GEE export tasks show 'COMPLETED' in the task monitor.
"""
import sys
sys.path.insert(0, "/content")

import os
os.environ["FORESTSENSE_RAW_DATA"] = LOCAL_OUT

from data.process_gee_data import run_preprocessing
run_preprocessing()

print("\n✅ Phase 1 COMPLETE!")
print("   Next step: Open notebooks/02_training.ipynb")
