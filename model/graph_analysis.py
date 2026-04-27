"""
ForestSense — Regional Graph Analysis
======================================
Converts pixel-level prediction maps into Region Adjacency Graphs (RAG)
and computes spatial metrics for publication.

Pipeline:
  Prediction Map (H×W)
       ↓
  SLIC Superpixels   → semantically coherent regions
       ↓
  Region Adjacency Graph (RAG)  → nodes = regions, edges = adjacency
       ↓
  Spatial Metrics   → fragmentation, connectivity, patch size, loss area

Metrics computed (per patch, then aggregated per region):
  - n_forest_patches     : Number of connected forest regions
  - mean_patch_size_ha   : Mean forest patch size in hectares
  - largest_patch_idx    : Largest Patch Index = largest / total forest area
  - fragmentation_idx    : n_patches / total_forest_area (more = worse)
  - edge_density         : Forest perimeter / forest area (shape complexity)
  - connectivity         : Mean node degree in forest RAG
  - forest_loss_ha       : Forest pixels that changed to non-forest
  - forest_gain_ha       : Non-forest pixels that changed to forest
  - net_change_ha        : Gain - Loss (positive = net gain)
"""

import numpy as np
from skimage.segmentation import slic, mark_boundaries
from skimage.measure import label, regionprops
from scipy.spatial import cKDTree
import warnings
warnings.filterwarnings("ignore")

# ── Sentinel-2 resolution ───────────────────────────────────────────────────
PIXEL_SIZE_M  = 10.0          # 10m × 10m per pixel
PIXEL_AREA_HA = (PIXEL_SIZE_M ** 2) / 10_000   # = 0.01 ha per pixel

# Class indices
CLASS_BACKGROUND = 0
CLASS_STABLE_FOREST = 1
CLASS_FOREST_LOSS   = 2
CLASS_FOREST_GAIN   = 3


# ============================================================
#  SUPERPIXEL GENERATION
# ============================================================

def compute_slic(
    pred_map   : np.ndarray,
    rgb_image  : np.ndarray = None,
    n_segments : int = 200,
    compactness: float = 0.1,
) -> np.ndarray:
    """
    Apply SLIC superpixels to a prediction map.
    If rgb_image is provided, uses colour + spatial. Otherwise uses pred_map only.

    Args:
        pred_map   : (H, W) int array of class predictions [0-3]
        rgb_image  : (H, W, 3) float array in [0,1], optional
        n_segments : target number of superpixels
        compactness: trade-off between spatial and colour similarity

    Returns:
        (H, W) int array of superpixel labels (0-indexed)
    """
    if rgb_image is not None:
        # Use RGB for coherent superpixels
        image = rgb_image.astype(np.float32)
    else:
        # Use normalised prediction map as a single channel repeated to 3ch
        norm = pred_map.astype(np.float32) / 3.0
        image = np.stack([norm, norm, norm], axis=-1)

    segments = slic(
        image,
        n_segments  = n_segments,
        compactness = compactness,
        sigma       = 1.0,
        start_label = 0,
        enforce_connectivity = True,
    )
    return segments


# ============================================================
#  REGION ADJACENCY GRAPH
# ============================================================

class RegionAdjacencyGraph:
    """
    Lightweight RAG built from a superpixel segmentation map.

    Attributes:
        nodes : dict {seg_id: {'class': int, 'area': int, 'centroid': (r,c)}}
        edges : set of (id_a, id_b) pairs where a < b
    """

    def __init__(self, segments: np.ndarray, pred_map: np.ndarray):
        self.segments = segments
        self.pred_map = pred_map
        self.nodes    = {}
        self.edges    = set()
        self._build()

    def _build(self):
        """Build nodes and edges from segmentation map."""
        seg_ids = np.unique(self.segments)

        # ── Node attributes ───────────────────────────────────────────
        for sid in seg_ids:
            mask      = self.segments == sid
            classes   = self.pred_map[mask]
            # Dominant class (majority vote)
            dominant  = int(np.bincount(classes).argmax())
            area_px   = int(mask.sum())
            rows, cols = np.where(mask)
            centroid   = (float(rows.mean()), float(cols.mean()))
            self.nodes[sid] = {
                "class"    : dominant,
                "area_px"  : area_px,
                "area_ha"  : area_px * PIXEL_AREA_HA,
                "centroid" : centroid,
                "purity"   : float((classes == dominant).mean()),
            }

        # ── Edge detection (4-connectivity adjacency) ─────────────────
        # A pair of segments are adjacent if they share a pixel border
        H, W = self.segments.shape
        for r in range(H - 1):
            for c in range(W - 1):
                a = self.segments[r, c]
                right = self.segments[r, c + 1]
                down  = self.segments[r + 1, c]
                if a != right:
                    self.edges.add((min(a, right), max(a, right)))
                if a != down:
                    self.edges.add((min(a, down), max(a, down)))

    def forest_nodes(self):
        """Return node IDs whose dominant class is Stable Forest (1)."""
        return [sid for sid, attr in self.nodes.items()
                if attr["class"] == CLASS_STABLE_FOREST]

    def node_degrees(self):
        """Return {node_id: degree} for all nodes."""
        deg = {sid: 0 for sid in self.nodes}
        for a, b in self.edges:
            if a in deg: deg[a] += 1
            if b in deg: deg[b] += 1
        return deg


# ============================================================
#  SPATIAL METRICS
# ============================================================

def compute_patch_metrics(pred_map: np.ndarray) -> dict:
    """
    Compute landscape-level forest metrics from a prediction map.

    Uses 8-connectivity to identify contiguous forest patches.
    """
    H, W = pred_map.shape

    # ── Forest binary mask ────────────────────────────────────────────
    forest_mask = (pred_map == CLASS_STABLE_FOREST).astype(np.uint8)
    loss_mask   = (pred_map == CLASS_FOREST_LOSS).astype(np.uint8)
    gain_mask   = (pred_map == CLASS_FOREST_GAIN).astype(np.uint8)

    # ── Connected components (8-connectivity) ─────────────────────────
    labelled, n_patches = label(forest_mask, connectivity=2, return_num=True)
    props = regionprops(labelled)

    total_forest_px   = int(forest_mask.sum())
    total_forest_ha   = total_forest_px * PIXEL_AREA_HA

    if total_forest_px == 0 or n_patches == 0:
        return {
            "n_forest_patches"   : 0,
            "total_forest_ha"    : 0.0,
            "mean_patch_size_ha" : 0.0,
            "largest_patch_ha"   : 0.0,
            "largest_patch_idx"  : 0.0,
            "fragmentation_idx"  : 0.0,
            "edge_density"       : 0.0,
            "forest_loss_ha"     : float(loss_mask.sum() * PIXEL_AREA_HA),
            "forest_gain_ha"     : float(gain_mask.sum() * PIXEL_AREA_HA),
            "net_change_ha"      : float((gain_mask.sum() - loss_mask.sum()) * PIXEL_AREA_HA),
        }

    patch_areas_ha = [p.area * PIXEL_AREA_HA for p in props]
    perimeters     = [p.perimeter for p in props]

    largest_ha     = max(patch_areas_ha)
    mean_ha        = float(np.mean(patch_areas_ha))

    # Largest Patch Index
    lpi = largest_ha / total_forest_ha if total_forest_ha > 0 else 0.0

    # Fragmentation Index: more patches per unit area = more fragmented
    frag_idx = n_patches / (total_forest_ha + 1e-6)

    # Edge density: total perimeter / total area (pixels)
    total_perimeter = sum(perimeters)
    edge_density    = total_perimeter / (total_forest_px + 1e-6)

    return {
        "n_forest_patches"   : n_patches,
        "total_forest_ha"    : round(total_forest_ha, 4),
        "mean_patch_size_ha" : round(mean_ha, 4),
        "largest_patch_ha"   : round(largest_ha, 4),
        "largest_patch_idx"  : round(lpi, 4),
        "fragmentation_idx"  : round(frag_idx, 4),
        "edge_density"       : round(edge_density, 4),
        "forest_loss_ha"     : round(loss_mask.sum() * PIXEL_AREA_HA, 4),
        "forest_gain_ha"     : round(gain_mask.sum() * PIXEL_AREA_HA, 4),
        "net_change_ha"      : round((gain_mask.sum() - loss_mask.sum()) * PIXEL_AREA_HA, 4),
    }


def compare_metrics(t1_metrics: dict, t2_metrics: dict) -> dict:
    """Compute change between T1 and T2 metrics."""
    delta = {}
    for key in t1_metrics:
        v1 = t1_metrics[key]
        v2 = t2_metrics[key]
        if isinstance(v1, (int, float)):
            pct = ((v2 - v1) / (abs(v1) + 1e-8)) * 100
            delta[f"Δ{key}"] = round(v2 - v1, 4)
            delta[f"Δ{key}_pct"] = round(pct, 2)
    return delta


# ============================================================
#  GRAPH VISUALISATION HELPERS
# ============================================================

def draw_rag_on_map(
    pred_map  : np.ndarray,
    segments  : np.ndarray,
    rag       : RegionAdjacencyGraph,
    ax,
    label_cmap,
    title     : str = "",
    text_color: str = "#e8e8f0",
):
    """
    Draw prediction map with superpixel boundaries and RAG edges overlaid.

    Args:
        pred_map   : (H, W) int prediction
        segments   : (H, W) int superpixel labels
        rag        : RegionAdjacencyGraph instance
        ax         : matplotlib axis
        label_cmap : colourmap for 4 classes
    """
    import matplotlib.pyplot as plt

    # Prediction map base
    ax.imshow(pred_map, cmap=label_cmap, vmin=0, vmax=3,
              interpolation="nearest", alpha=0.85)

    # Superpixel boundaries
    boundary_img = mark_boundaries(
        np.zeros((*pred_map.shape, 3), dtype=np.float32),
        segments, color=(1, 1, 0.2), mode="outer"
    )
    boundary_mask = boundary_img.sum(axis=-1) > 0
    ax.imshow(boundary_mask, cmap="gray", alpha=0.3, interpolation="nearest")

    # Draw RAG edges (forest nodes only, subsampled for clarity)
    forest_ids = set(rag.forest_nodes())
    for a, b in list(rag.edges)[:300]:   # max 300 edges for readability
        if a in forest_ids and b in forest_ids:
            c_a = rag.nodes[a]["centroid"]
            c_b = rag.nodes[b]["centroid"]
            ax.plot([c_a[1], c_b[1]], [c_a[0], c_b[0]],
                    color="#2ecc71", lw=0.4, alpha=0.5)

    # Draw forest nodes as dots
    for sid in list(forest_ids)[:150]:
        c = rag.nodes[sid]["centroid"]
        ax.plot(c[1], c[0], "o", ms=2.0, color="#27ae60", alpha=0.7)

    ax.set_title(title, color=text_color, fontsize=9, pad=4)
    ax.axis("off")
