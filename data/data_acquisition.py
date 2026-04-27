"""
ForestSense — Data Acquisition Script
======================================
This script fetches Sentinel-1 (SAR) and Sentinel-2 (Optical) data
from Google Earth Engine for three Indian forest regions across two
temporal windows (T1: 2021, T2: 2023).

Usage:
    - Run in Google Colab (recommended for GEE authentication)
    - Or run locally after `earthengine authenticate`

Output:
    - GeoTIFF files exported to Google Drive (or local if USE_DRIVE=False)
    - Metadata JSON per region per timestep
    - Approx. total size: ~10 GB across all 3 regions
"""

# ─── CELL 1: Install dependencies ────────────────────────────────────────────
# Run this cell first in Colab
INSTALL_CELL = """
!pip install earthengine-api geemap rasterio numpy pandas tqdm -q
"""

# ─── CELL 2: Imports & Configuration ─────────────────────────────────────────
import ee
import geemap
import json
import os
import math
from datetime import datetime

# ============================================================
#   PROJECT CONFIGURATION
# ============================================================

PROJECT_NAME = "forestsense"
GEE_PROJECT  = "ee-your-project-id"          # ← Replace with your GEE Cloud Project ID

# Google Drive output folder (Colab)
DRIVE_FOLDER = "ForestSense_Data"

# Toggle: True → export to Google Drive; False → local geemap download
USE_DRIVE = True

# ── Temporal Windows ──────────────────────────────────────────
T1_START = "2021-01-01"
T1_END   = "2021-12-31"

T2_START = "2023-01-01"
T2_END   = "2023-12-31"

# ── Patch Configuration ───────────────────────────────────────
PATCH_SIZE_M = 5120      # Each export tile = 5.12 km × 5.12 km
EXPORT_SCALE = 20        # 20m resolution (compromise between S1@10m, S2@10m)
MAX_PIXELS   = 1e13      # GEE export limit

# ── Study Regions (lon_min, lat_min, lon_max, lat_max) ──────────
REGIONS = {
    "western_ghats": {
        "name"       : "Western Ghats — Wayanad, Kerala",
        "forest_type": "Tropical Evergreen",
        "bbox"       : [75.7, 11.5, 76.3, 12.1],   # ~66 km × 66 km ≈ ~880 km²
        "description": "UNESCO World Heritage biodiversity hotspot; high encroachment pressure"
    },
    "sundarbans": {
        "name"       : "Sundarbans — West Bengal",
        "forest_type": "Mangrove",
        "bbox"       : [88.3, 21.5, 89.1, 22.1],   # ~80 km × 66 km ≈ ~960 km²
        "description": "World's largest mangrove delta; cyclone and sea-level change impact"
    },
    "saranda": {
        "name"       : "Saranda Forest — Jharkhand",
        "forest_type": "Dry Deciduous (Sal)",
        "bbox"       : [85.0, 22.0, 85.7, 22.6],   # ~70 km × 66 km ≈ ~820 km²
        "description": "Asia's largest Sal forest; historical mining deforestation pressure"
    }
}

# ============================================================
#   HELPER FUNCTIONS
# ============================================================

def get_study_geometry(bbox):
    """Return an ee.Geometry.Rectangle from [lon_min, lat_min, lon_max, lat_max]."""
    return ee.Geometry.Rectangle(bbox)


# ── Sentinel-1 SAR Processing ─────────────────────────────────

def get_sentinel1_composite(geometry, start_date, end_date):
    """
    Fetch a Sentinel-1 GRD median composite (VV + VH) for a given geometry and date range.
    Applies speckle filtering and clips to AOI.
    
    Returns:
        ee.Image with bands: ['VV', 'VH', 'VV_VH_ratio']
    """
    collection = (
        ee.ImageCollection("COPERNICUS/S1_GRD")
        .filterBounds(geometry)
        .filterDate(start_date, end_date)
        .filter(ee.Filter.eq("instrumentMode", "IW"))         # Interferometric Wide Swath
        .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VV"))
        .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VH"))
        .filter(ee.Filter.eq("orbitProperties_pass", "DESCENDING"))
        .select(["VV", "VH"])
    )

    # Median composite (reduces speckle)
    composite = collection.median()

    # Add VV/VH ratio (forest structure proxy)
    vv_vh_ratio = composite.select("VV").subtract(composite.select("VH")).rename("VV_VH_ratio")
    composite = composite.addBands(vv_vh_ratio)

    return composite.clip(geometry)


# ── Sentinel-2 Optical Processing ─────────────────────────────

def mask_s2_clouds(image):
    """Apply cloud masking using Sentinel-2 QA60 band."""
    qa = image.select("QA60")
    cloud_bit_mask     = 1 << 10
    cirrus_bit_mask    = 1 << 11
    mask = (
        qa.bitwiseAnd(cloud_bit_mask).eq(0)
        .And(qa.bitwiseAnd(cirrus_bit_mask).eq(0))
    )
    return image.updateMask(mask).divide(10000)


def compute_indices(image):
    """Compute NDVI, EVI, and NDWI vegetation/water indices."""
    # NDVI
    ndvi = image.normalizedDifference(["B8", "B4"]).rename("NDVI")
    
    # EVI = 2.5 * (NIR - Red) / (NIR + 6*Red - 7.5*Blue + 1)
    evi = image.expression(
        "2.5 * ((NIR - RED) / (NIR + 6.0 * RED - 7.5 * BLUE + 1.0))",
        {
            "NIR" : image.select("B8"),
            "RED" : image.select("B4"),
            "BLUE": image.select("B2"),
        }
    ).rename("EVI")
    
    # NDWI (Gao 1996, uses NIR and SWIR for vegetation water content)
    ndwi = image.normalizedDifference(["B8", "B11"]).rename("NDWI")
    
    return image.addBands([ndvi, evi, ndwi])


def get_sentinel2_composite(geometry, start_date, end_date):
    """
    Fetch a cloud-free Sentinel-2 SR median composite for a given geometry and date range.
    
    Returns:
        ee.Image with bands: ['B2','B3','B4','B8','B11','B12','NDVI','EVI','NDWI']
    """
    collection = (
        ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .filterBounds(geometry)
        .filterDate(start_date, end_date)
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 20))
        .map(mask_s2_clouds)
        .map(compute_indices)
    )

    # Select key bands: Blue, Green, Red, NIR, SWIR1, SWIR2 + indices
    selected_bands = ["B2", "B3", "B4", "B8", "B11", "B12", "NDVI", "EVI", "NDWI"]
    composite = collection.select(selected_bands).median()

    return composite.clip(geometry)


# ── Combined Multi-modal Stack ─────────────────────────────────

def build_multimodal_stack(geometry, start_date, end_date, period_label):
    """
    Build a combined SAR + Optical image stack for one time period.
    
    Returns:
        ee.Image with 12 bands total:
          SAR: VV, VH, VV_VH_ratio                          (3 bands)
          Optical: B2, B3, B4, B8, B11, B12, NDVI, EVI, NDWI  (9 bands)
    """
    s1 = get_sentinel1_composite(geometry, start_date, end_date)
    s2 = get_sentinel2_composite(geometry, start_date, end_date)

    # Rename bands to avoid collisions and make period explicit
    s1_bands = ["SAR_VV", "SAR_VH", "SAR_ratio"]
    s2_bands = ["OPT_B2", "OPT_B3", "OPT_B4", "OPT_B8", "OPT_B11", "OPT_B12",
                "NDVI", "EVI", "NDWI"]

    s1 = s1.rename(s1_bands)
    s2 = s2.rename(s2_bands)

    # Cast all bands to Float32 to prevent GEE export errors (Float64 vs Float32 mismatch)
    stack = s1.addBands(s2).toFloat()
    
    print(f"  [{period_label}] Built multimodal stack: {len(s1_bands) + len(s2_bands)} bands")
    return stack


# ── Ground Truth / Label Generation ───────────────────────────

def generate_pseudo_labels(t1_image, t2_image, geometry):
    """
    Generate change detection pseudo-labels using:
    1. NDVI thresholding → Forest / Non-Forest classification
    2. NDVI difference → Changed / Unchanged classification
    
    Label encoding:
        0 = Unchanged Non-Forest
        1 = Unchanged Forest
        2 = Forest Loss (T1=Forest, T2=Non-Forest)  ← KEY CHANGE CLASS
        3 = Forest Gain (T1=Non-Forest, T2=Forest)
    
    Returns:
        ee.Image with single 'label' band (uint8)
    """
    NDVI_FOREST_THRESHOLD = 0.4    # NDVI > 0.4 → Forest
    NDVI_CHANGE_THRESHOLD = 0.1    # |NDVI_diff| > 0.1 → Changed

    ndvi_t1 = t1_image.select("NDVI")
    ndvi_t2 = t2_image.select("NDVI")

    forest_t1 = ndvi_t1.gt(NDVI_FOREST_THRESHOLD)   # 1=Forest at T1
    forest_t2 = ndvi_t2.gt(NDVI_FOREST_THRESHOLD)   # 1=Forest at T2

    ndvi_diff = ndvi_t2.subtract(ndvi_t1).abs()
    changed   = ndvi_diff.gt(NDVI_CHANGE_THRESHOLD)  # 1=Changed

    # Construct 4-class label
    label = (
        ee.Image(0)                                                         # base = 0
        .where(forest_t1.And(forest_t2).And(changed.Not()), 1)              # Stable Forest
        .where(forest_t1.And(forest_t2.Not()), 2)                           # Forest Loss
        .where(forest_t1.Not().And(forest_t2), 3)                           # Forest Gain
    )

    return label.rename("label").toUint8().clip(geometry)


# ── Temporal NDVI Series (for Analytics Phase) ─────────────────

def get_ndvi_timeseries(geometry, year_start=2018, year_end=2023):
    """
    Compute annual mean NDVI for a region from 2018–2023.
    Used later in analytics phase to generate temporal trend plots.
    
    Returns:
        List of dicts: [{'year': 2018, 'ndvi_mean': 0.62, 'ndvi_std': 0.07}, ...]
    """
    series = []
    for year in range(year_start, year_end + 1):
        start = f"{year}-01-01"
        end   = f"{year}-12-31"
        s2 = get_sentinel2_composite(geometry, start, end)
        
        stats = s2.select("NDVI").reduceRegion(
            reducer  = ee.Reducer.mean().combine(ee.Reducer.stdDev(), "", True),
            geometry = geometry,
            scale    = EXPORT_SCALE,
            maxPixels= MAX_PIXELS
        ).getInfo()
        
        series.append({
            "year"     : year,
            "ndvi_mean": round(stats.get("NDVI_mean", 0), 4),
            "ndvi_std" : round(stats.get("NDVI_stdDev", 0), 4),
        })
        print(f"    Year {year}: NDVI_mean={series[-1]['ndvi_mean']:.4f} ± {series[-1]['ndvi_std']:.4f}")
    
    return series


# ── Export Utilities ───────────────────────────────────────────

def export_to_drive(image, description, folder, region, scale=EXPORT_SCALE):
    """Submit a GEE export task to Google Drive."""
    task = ee.batch.Export.image.toDrive(
        image           = image,
        description     = description,
        folder          = folder,
        fileNamePrefix  = description,
        region          = region,
        scale           = scale,
        maxPixels       = int(MAX_PIXELS),
        fileFormat      = "GeoTIFF",
        formatOptions   = {"cloudOptimized": True},
    )
    task.start()
    print(f"  ✅ Export task submitted: {description}")
    return task


def check_tasks():
    """Print status of all running GEE tasks."""
    tasks = ee.batch.Task.list()
    for t in tasks[:10]:
        print(f"  [{t.state}] {t.config.get('description', 'unknown')}")


# ============================================================
#   MAIN ACQUISITION PIPELINE
# ============================================================

def run_acquisition():
    """
    Main pipeline:
    1. Authenticate GEE
    2. For each region:
       a. Build T1 and T2 multimodal stacks
       b. Generate pseudo-labels (change map)
       c. Compute NDVI time-series statistics
       d. Export all GeoTIFFs to Google Drive
    3. Save metadata JSON
    """

    # ── Step 1: Authenticate ─────────────────────────────────
    print("=" * 60)
    print("🔐 Authenticating with Google Earth Engine...")
    try:
        ee.Initialize(project=GEE_PROJECT)
        print("  ✅ Authenticated successfully\n")
    except Exception:
        ee.Authenticate()
        ee.Initialize(project=GEE_PROJECT)
        print("  ✅ Authenticated successfully\n")

    metadata = {}
    all_tasks = []

    for region_id, region_info in REGIONS.items():
        print("=" * 60)
        print(f"🌿 Processing Region: {region_info['name']}")
        print(f"   Forest Type : {region_info['forest_type']}")
        print(f"   BBox        : {region_info['bbox']}")
        print()

        geometry = get_study_geometry(region_info["bbox"])

        # ── Step 2a: Build Multimodal Stacks ─────────────────
        print("  📡 Building T1 (2021) multimodal stack...")
        t1_stack = build_multimodal_stack(geometry, T1_START, T1_END, "T1-2021")

        print("  📡 Building T2 (2023) multimodal stack...")
        t2_stack = build_multimodal_stack(geometry, T2_START, T2_END, "T2-2023")

        # ── Step 2b: Generate Pseudo-Labels ──────────────────
        print("  🏷️  Generating pseudo-labels (NDVI-threshold change map)...")
        label_map = generate_pseudo_labels(t1_stack, t2_stack, geometry)

        # ── Step 2c: NDVI Time-Series (2018–2023) ─────────────
        print("  📈 Computing NDVI time-series (2018–2023)...")
        ndvi_series = get_ndvi_timeseries(geometry, year_start=2018, year_end=2023)

        # ── Step 2d: Export to Drive ──────────────────────────
        print("  💾 Submitting export tasks to Google Drive...")

        prefix = f"{PROJECT_NAME}_{region_id}"

        t1_task = export_to_drive(
            image       = t1_stack,
            description = f"{prefix}_T1_2021_stack",
            folder      = DRIVE_FOLDER,
            region      = geometry.bounds().getInfo()["coordinates"],
        )
        t2_task = export_to_drive(
            image       = t2_stack,
            description = f"{prefix}_T2_2023_stack",
            folder      = DRIVE_FOLDER,
            region      = geometry.bounds().getInfo()["coordinates"],
        )
        lbl_task = export_to_drive(
            image       = label_map,
            description = f"{prefix}_labels",
            folder      = DRIVE_FOLDER,
            region      = geometry.bounds().getInfo()["coordinates"],
        )
        all_tasks.extend([t1_task, t2_task, lbl_task])

        # ── Store Metadata ────────────────────────────────────
        metadata[region_id] = {
            **region_info,
            "t1_period"   : {"start": T1_START, "end": T1_END},
            "t2_period"   : {"start": T2_START, "end": T2_END},
            "ndvi_series" : ndvi_series,
            "export_scale": EXPORT_SCALE,
            "bands_sar"   : ["SAR_VV", "SAR_VH", "SAR_ratio"],
            "bands_optical": ["OPT_B2", "OPT_B3", "OPT_B4", "OPT_B8",
                              "OPT_B11", "OPT_B12", "NDVI", "EVI", "NDWI"],
            "label_classes": {
                "0": "Unchanged Non-Forest",
                "1": "Unchanged Forest",
                "2": "Forest Loss",
                "3": "Forest Gain",
            },
            "exported_files": [
                f"{prefix}_T1_2021_stack.tif",
                f"{prefix}_T2_2023_stack.tif",
                f"{prefix}_labels.tif",
            ],
            "acquired_at": datetime.utcnow().isoformat() + "Z",
        }
        print(f"  ✅ Region '{region_id}' done.\n")

    # ── Step 3: Save Metadata JSON ─────────────────────────────
    os.makedirs("data", exist_ok=True)
    metadata_path = "data/acquisition_metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"📝 Metadata saved → {metadata_path}")

    # ── Step 4: Monitor Tasks ──────────────────────────────────
    print("\n⏳ All export tasks submitted. Current status:")
    check_tasks()
    print("\n💡 Tasks run in GEE background. Check progress at:")
    print("   https://code.earthengine.google.com/tasks")
    print("   Files will appear in Google Drive → ForestSense_Data/")

    return metadata


# ── Entry Point ────────────────────────────────────────────────
if __name__ == "__main__":
    # When running in Colab, call run_acquisition() directly after imports.
    # In a .py context, this ensures it doesn't auto-run on import.
    metadata = run_acquisition()
    print("\n✅ Phase 1 — Data Acquisition complete!")
