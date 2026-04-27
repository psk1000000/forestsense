"""
ForestSense — Phase 5 v2: Upgraded Gradio UI
- Visual metrics panel (bar chart figure)
- Change intensity heatmap (softmax probability)
- Both dropdown + file upload
Run after FIX CELL (siamese_model + vlm + processor already loaded)
"""

# ── CELL 1 ───────────────────────────────────────────────────────────────────
# !pip install -q gradio  (already installed)

# ── CELL 2 ───────────────────────────────────────────────────────────────────
import os, sys, io, warnings
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from PIL import Image
from skimage.segmentation import slic, mark_boundaries
from skimage.measure import label, regionprops
from qwen_vl_utils import process_vision_info
import gradio as gr
warnings.filterwarnings("ignore")

MANIFEST      = "/content/drive/MyDrive/forestsense/data/processed/manifest.csv"
PATCH_ROOT    = "/content/drive/MyDrive/forestsense"
PIXEL_AREA_HA = 0.01
DARK_BG       = "#0f0f1a"
PANEL_BG      = "#1a1a2e"
TEXT_C        = "#e8e8f0"
CMAP_COLORS   = ["#1a1a2e", "#2ecc71", "#e74c3c", "#3498db"]
label_cmap    = mcolors.ListedColormap(CMAP_COLORS)

df_manifest   = pd.read_csv(MANIFEST)
test_df       = df_manifest[df_manifest["split"] == "test"].reset_index(drop=True)
_session      = {"rag_pil": None, "metrics": None, "region": ""}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── helpers ───────────────────────────────────────────────────────────────────

def norm_img(a):
    p2, p98 = np.percentile(a, 2), np.percentile(a, 98)
    return np.clip((a - p2) / (p98 - p2 + 1e-8), 0, 1)

def make_rgb(s): return norm_img(np.stack([s[5], s[4], s[3]], axis=-1))

def make_sar(s):
    vv = norm_img(s[10] if s.shape[0] > 10 else s[0])
    return np.stack([vv, vv, vv], axis=-1)

def arr_to_pil(a): return Image.fromarray((a * 255).astype(np.uint8))

LEGEND_ITEMS = [
    ("#1a1a2e", "Non-Forest"),
    ("#2ecc71", "Stable Forest"),
    ("#e74c3c", "Forest Loss"),
    ("#3498db", "Forest Gain"),
]
import matplotlib.patches as mpatches

def _add_legend(ax):
    handles = [mpatches.Patch(color=c, label=l) for c, l in LEGEND_ITEMS]
    ax.legend(handles=handles, loc="lower center", ncol=2, fontsize=6,
              framealpha=0.4, facecolor=PANEL_BG,
              labelcolor=TEXT_C, edgecolor="#333355")

def pred_to_pil(pred, title="Change Map", add_legend=True):
    fig, ax = plt.subplots(figsize=(3.2, 3.5), dpi=100)
    fig.patch.set_facecolor(DARK_BG)
    ax.imshow(pred, cmap=label_cmap, vmin=0, vmax=3, interpolation="nearest")
    ax.set_title(title, color=TEXT_C, fontsize=8, pad=3)
    ax.axis("off")
    if add_legend:
        _add_legend(ax)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", facecolor=DARK_BG)
    plt.close(fig); buf.seek(0)
    return Image.open(buf).copy()

def comparison_panel_to_pil(t1_rgb, t2_rgb, pred, region_name):
    """3-panel side-by-side: T1 RGB | T2 RGB | Predicted Change Map with legend."""
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.8), dpi=110)
    fig.patch.set_facecolor(DARK_BG)
    fig.suptitle(f"ForestSense — {region_name}  (T1: 2021 → T2: 2023)",
                 color=TEXT_C, fontsize=11, fontweight="bold", y=1.02)

    axes[0].imshow(t1_rgb, interpolation="nearest")
    axes[0].set_title("T1 Optical RGB (2021)", color=TEXT_C, fontsize=9)
    axes[0].axis("off")

    axes[1].imshow(t2_rgb, interpolation="nearest")
    axes[1].set_title("T2 Optical RGB (2023)", color=TEXT_C, fontsize=9)
    axes[1].axis("off")

    axes[2].imshow(pred, cmap=label_cmap, vmin=0, vmax=3, interpolation="nearest")
    axes[2].set_title("Predicted Change Map", color=TEXT_C, fontsize=9)
    axes[2].axis("off")

    handles = [mpatches.Patch(color=c, label=l) for c, l in LEGEND_ITEMS]
    fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=9,
               framealpha=0.3, facecolor=PANEL_BG, labelcolor=TEXT_C,
               edgecolor="#333355", bbox_to_anchor=(0.5, -0.06))

    plt.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", facecolor=DARK_BG)
    plt.close(fig); buf.seek(0)
    return Image.open(buf).copy()

def heatmap_to_pil(probs):
    """
    probs: (4, H, W) softmax probabilities from model.
    Change intensity = P(loss) + P(gain) i.e. classes 2 and 3.
    """
    change_prob = probs[2] + probs[3]   # float32 (H, W)
    fig, ax = plt.subplots(figsize=(3, 3), dpi=100)
    fig.patch.set_facecolor(DARK_BG)
    im = ax.imshow(change_prob, cmap="hot", vmin=0, vmax=1, interpolation="bilinear")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04).ax.yaxis.set_tick_params(color=TEXT_C)
    ax.set_title("Change Probability", color=TEXT_C, fontsize=8)
    ax.axis("off")
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", facecolor=DARK_BG)
    plt.close(fig); buf.seek(0)
    return Image.open(buf).copy()

def metrics_to_pil(m):
    """Render spatial metrics as a styled bar + text figure."""
    TOTAL_PATCH_HA = 256 * 256 * PIXEL_AREA_HA   # 655.36 ha per 256×256 patch at 10m

    fig = plt.figure(figsize=(8, 3.5), facecolor=PANEL_BG)
    fig.suptitle(
        f"Spatial Metrics — {m.get('region','')}  (Patch = {TOTAL_PATCH_HA:.0f} ha total)",
        color=TEXT_C, fontsize=10, fontweight="bold"
    )

    gs   = fig.add_gridspec(1, 2, wspace=0.45)
    ax1  = fig.add_subplot(gs[0, 0])
    ax2  = fig.add_subplot(gs[0, 1])

    # ── Bar chart: area breakdown with % ─────────────────────────────────────
    labels = ["Stable\nForest", "Forest\nLoss", "Forest\nGain"]
    vals   = [
        max(m.get("total_forest_ha", 0) - m.get("forest_loss_ha", 0), 0),
        m.get("forest_loss_ha", 0),
        m.get("forest_gain_ha", 0),
    ]
    colors = ["#2ecc71", "#e74c3c", "#3498db"]
    bars   = ax1.bar(labels, vals, color=colors, edgecolor=DARK_BG, width=0.5)
    for bar, v in zip(bars, vals):
        pct = (v / TOTAL_PATCH_HA) * 100
        ax1.text(bar.get_x() + bar.get_width() / 2, v + max(vals) * 0.02,
                 f"{v:.1f} ha\n({pct:.1f}%)", ha="center", va="bottom",
                 color=TEXT_C, fontsize=7)
    ax1.set_facecolor(DARK_BG)
    ax1.set_title("Forest Area (ha)", color=TEXT_C, fontsize=9)
    ax1.tick_params(colors=TEXT_C, labelsize=8)
    for sp in ax1.spines.values(): sp.set_edgecolor("#333355")
    ax1.set_ylabel("ha", color=TEXT_C, fontsize=8)

    # ── Index gauges as horizontal bars ───────────────────────────────────────
    idx_labels = ["Fragmentation\nIndex", "Largest Patch\nIndex (LPI)"]
    idx_vals   = [m.get("fragmentation_idx", 0), m.get("largest_patch_idx", 0)]
    # Normalise fragmentation to 0-1 for display (cap at 5)
    idx_disp   = [min(idx_vals[0] / 5.0, 1.0), idx_vals[1]]
    idx_colors = ["#f39c12", "#9b59b6"]

    for i, (lbl, dv, rv, col) in enumerate(zip(idx_labels, idx_disp, idx_vals, idx_colors)):
        ax2.barh(i, dv, color=col, edgecolor=DARK_BG, height=0.4)
        ax2.text(dv + 0.02, i, f"{rv:.3f}", va="center", color=TEXT_C, fontsize=9)

    ax2.set_xlim(0, 1.25)
    ax2.set_yticks([0, 1])
    ax2.set_yticklabels(idx_labels, color=TEXT_C, fontsize=8)
    ax2.set_facecolor(DARK_BG)
    ax2.set_title("Spatial Indices", color=TEXT_C, fontsize=9)
    ax2.tick_params(colors=TEXT_C, labelsize=7)
    for sp in ax2.spines.values(): sp.set_edgecolor("#333355")

    # Net change annotation
    net = m.get("net_change_ha", 0)
    nc  = "#2ecc71" if net >= 0 else "#e74c3c"
    fig.text(0.5, -0.04, f"Net Change: {net:+.2f} ha  |  Forest Patches: {m.get('n_forest_patches',0)}",
             ha="center", color=nc, fontsize=10, fontweight="bold",
             transform=fig.transFigure)

    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", facecolor=PANEL_BG)
    plt.close(fig); buf.seek(0)
    return Image.open(buf).copy()

def rag_to_pil_and_metrics(pred, t1_rgb, t2_rgb):
    """
    Builds the 5-panel RAG figure matching Phase 3 output:
      T1 RGB | T2 RGB | Change Map | SLIC superpixels | RAG node-coloured overlay
    Also returns spatial metrics dict.
    """
    # Higher compactness (10) forces ~n_segments grid-like regions; 0.1 collapses to <10
    segments = slic(t1_rgb.astype(np.float32), n_segments=200,
                    compactness=10, sigma=1.5, start_label=0,
                    enforce_connectivity=True, min_size_factor=0.3)
    n_segs = len(np.unique(segments))

    # ── Build RAG node colours from dominant class of each superpixel ──────────
    node_class_map = np.zeros_like(pred)
    for sid in np.unique(segments):
        mask = segments == sid
        classes = pred[mask]
        dominant = int(np.bincount(classes).argmax())
        node_class_map[mask] = dominant

    # ── Spatial metrics (on raw pred map) ─────────────────────────────────────
    forest = (pred == 1).astype(np.uint8)
    lbl, n = label(forest, connectivity=2, return_num=True)
    props  = regionprops(lbl)
    tot_px = int(forest.sum()); tot_ha = tot_px * PIXEL_AREA_HA
    areas  = [p.area * PIXEL_AREA_HA for p in props] if props else [0]
    m = {
        "n_forest_patches"   : n,
        "total_forest_ha"    : round(tot_ha, 2),
        "mean_patch_size_ha" : round(float(np.mean(areas)), 4),
        "largest_patch_idx"  : round(max(areas) / (tot_ha + 1e-8), 3) if tot_ha > 0 else 0.0,
        "fragmentation_idx"  : round(n / (tot_ha + 1e-6), 3),
        "forest_loss_ha"     : round(float((pred == 2).sum()) * PIXEL_AREA_HA, 2),
        "forest_gain_ha"     : round(float((pred == 3).sum()) * PIXEL_AREA_HA, 2),
        "net_change_ha"      : round((float((pred==3).sum()) - float((pred==2).sum())) * PIXEL_AREA_HA, 2),
    }

    # ── 5-Panel Figure ────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 5, figsize=(18, 4), dpi=110)
    fig.patch.set_facecolor(DARK_BG)
    fig.suptitle(
        f"Region Adjacency Graph  —  Loss={m['forest_loss_ha']} ha  |  Gain={m['forest_gain_ha']} ha  |  Net={m['net_change_ha']:+.2f} ha",
        color=TEXT_C, fontsize=11, fontweight="bold"
    )

    # Panel 1: T1 RGB
    axes[0].imshow(t1_rgb, interpolation="nearest")
    axes[0].set_title("T1 True Colour (2021)", color=TEXT_C, fontsize=9)
    axes[0].axis("off")

    # Panel 2: T2 RGB
    axes[1].imshow(t2_rgb, interpolation="nearest")
    axes[1].set_title("T2 True Colour (2023)", color=TEXT_C, fontsize=9)
    axes[1].axis("off")

    # Panel 3: Prediction map
    axes[2].imshow(pred, cmap=label_cmap, vmin=0, vmax=3, interpolation="nearest")
    axes[2].set_title("Predicted Change Map", color=TEXT_C, fontsize=9)
    axes[2].axis("off")

    # Panel 4: SLIC superpixels on T1 RGB (yellow boundaries)
    slic_vis = mark_boundaries(t1_rgb.astype(np.float32), segments,
                               color=(1, 0.9, 0.1), mode="outer")
    axes[3].imshow(slic_vis, interpolation="nearest")
    axes[3].set_title(f"SLIC ({n_segs} superpixels)", color=TEXT_C, fontsize=9)
    axes[3].axis("off")

    # Panel 5: RAG — node class colours blended over T1 RGB + white boundaries
    axes[4].imshow(t1_rgb, interpolation="nearest")   # T1 RGB as background
    # Build RGBA class overlay
    norm = mcolors.Normalize(vmin=0, vmax=3)
    class_rgba = label_cmap(norm(node_class_map))      # (H,W,4)
    class_rgba[..., 3] = 0.6                           # 60% alpha
    axes[4].imshow(class_rgba, interpolation="nearest")
    # White superpixel boundaries
    bnd_white = mark_boundaries(t1_rgb.astype(np.float32), segments,
                                color=(1, 1, 1), mode="outer")
    bnd_mask  = bnd_white.sum(axis=-1) > 0
    axes[4].imshow(np.dstack([bnd_white, bnd_mask.astype(np.float32)]),
                   interpolation="nearest")
    axes[4].set_title(f"RAG (nodes={n_segs})", color=TEXT_C, fontsize=9)
    axes[4].axis("off")

    # Legend below
    handles = [mpatches.Patch(color=c, label=l) for c, l in LEGEND_ITEMS]
    fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=9,
               framealpha=0.3, facecolor=PANEL_BG, labelcolor=TEXT_C,
               edgecolor="#333355", bbox_to_anchor=(0.5, -0.08))

    plt.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", facecolor=DARK_BG)
    plt.close(fig); buf.seek(0)
    return Image.open(buf).copy(), m


def run_pipeline(t1_np, t2_np, label_np, region_name):
    from torch.amp import autocast
    import torch.nn.functional as F

    t1_t = torch.tensor(t1_np, dtype=torch.float32).unsqueeze(0).to(device)
    t2_t = torch.tensor(t2_np, dtype=torch.float32).unsqueeze(0).to(device)

    with torch.no_grad():
        with autocast("cuda"):
            logit = siamese_model(t1_t, t2_t)
        probs = F.softmax(logit, dim=1).squeeze(0).cpu().numpy()   # (4,H,W)
        pred  = probs.argmax(axis=0).astype(np.int64)              # (H,W)

    t1_rgb = make_rgb(t1_np); t2_rgb = make_rgb(t2_np)
    rag_pil, m    = rag_to_pil_and_metrics(pred, t1_rgb, t2_rgb)
    m["region"]   = region_name
    _session.update({"rag_pil": rag_pil, "metrics": m, "region": region_name})

    gt_pil      = pred_to_pil(label_np, title="Ground Truth") if label_np is not None else pred_to_pil(np.zeros_like(pred), title="Ground Truth")
    heat_pil    = heatmap_to_pil(probs)
    met_pil     = metrics_to_pil(m)

    return (
        arr_to_pil(t1_rgb), arr_to_pil(make_sar(t1_np)),
        arr_to_pil(t2_rgb), arr_to_pil(make_sar(t2_np)),
        pred_to_pil(pred, title="Predicted Change Map"), gt_pil,
        heat_pil, rag_pil, met_pil,
    )

def run_from_dropdown(choice):
    idx = int(choice.split("Patch ")[-1])
    row = test_df.iloc[idx]
    t1  = np.load(os.path.join(PATCH_ROOT, row["t1_path"])).astype(np.float32)
    t2  = np.load(os.path.join(PATCH_ROOT, row["t2_path"])).astype(np.float32)
    lbl = np.load(os.path.join(PATCH_ROOT, row["label_path"])).astype(np.int64)
    return run_pipeline(t1, t2, lbl, row["region"].replace("_", " ").title())

def run_from_upload(t1f, t2f, lblf):
    if t1f is None or t2f is None:
        raise gr.Error("Upload both T1 and T2 .npy files.")
    t1  = np.load(t1f.name).astype(np.float32)
    t2  = np.load(t2f.name).astype(np.float32)
    lbl = np.load(lblf.name).astype(np.int64) if lblf else None
    return run_pipeline(t1, t2, lbl, "Custom Upload")

def vlm_chat(message, history):
    if _session["rag_pil"] is None:
        return "⚠️ Run the pipeline first!"
    tmp = "/tmp/rag_ctx.png"; _session["rag_pil"].save(tmp)
    m   = _session["metrics"]; region = _session["region"]

    # Region-specific ecological context
    REGION_CONTEXT = {
        "Western Ghats": (
            "The Western Ghats is a UNESCO World Heritage Site and one of the world's eight "
            "biodiversity hotspots. It is home to over 5,000 species of flowering plants, 139 "
            "mammal species (including the Bengal Tiger and Asian Elephant), and is the source "
            "of major rivers like the Godavari and Krishna. Deforestation threats include "
            "plantation agriculture (tea, coffee, rubber), encroachment, and road construction. "
            "Forest fragmentation here is critical as it disrupts wildlife corridors between "
            "protected areas like Periyar and Mudumalai."
        ),
        "Sundarbans": (
            "The Sundarbans is the world's largest mangrove delta forest, a UNESCO World "
            "Heritage Site shared between India and Bangladesh. It is the primary habitat of "
            "the Bengal Tiger (Project Tiger reserve) and the Irrawaddy dolphin. Mangrove "
            "forests here act as a critical coastal buffer against cyclones and storm surges. "
            "Key threats include sea-level rise from climate change, salinity intrusion, "
            "cyclone damage (e.g. Amphan 2020), illegal logging, and aquaculture (shrimp farming) "
            "replacing mangroves. Forest gain here may represent mangrove regrowth or "
            "reforestation programmes."
        ),
        "Saranda": (
            "Saranda is the largest Sal (Shorea robusta) forest in Asia, located in Jharkhand, "
            "India. It is a critical habitat for the Asian Elephant and is classified as an "
            "Elephant Reserve. The forest is severely threatened by iron ore mining operations "
            "(over 20 active mines), left-wing extremism which limits conservation enforcement, "
            "and encroachment by tribal communities for subsistence agriculture. Forest loss "
            "here is closely linked to mining lease expansion and road construction for ore "
            "transportation. The tribal Adivasi communities (Ho and Munda people) depend "
            "entirely on this forest for their livelihoods."
        ),
    }

    ctx = REGION_CONTEXT.get(region, f"{region} forest region in India.")

    prompt = (
        f"You are an expert forest ecologist and remote sensing analyst specialising in {region}.\n\n"
        f"ECOLOGICAL CONTEXT:\n{ctx}\n\n"
        f"The attached image shows the 5-panel RAG: "
        f"T1 RGB (2021) | T2 RGB (2023) | Change Map | SLIC superpixels | RAG overlay.\n"
        f"Colours: Green=Stable Forest, Red=Forest Loss, Blue=Forest Gain, Black=Non-Forest.\n\n"
        f"USER QUESTION: {message}\n\n"
        f"Answer in 4-6 sentences. "
        f"Focus heavily on DESCRIBING WHAT YOU SEE visually in the image panels (e.g., the spatial distribution of loss/gain, the shape of the river/terrain in the RGB panels, how fragmentation looks in the RAG overlay). "
        f"Relate these visual observations to the ecology and threats of {region}. "
        f"Do NOT mention or invent any quantitative metrics or hectare numbers. Rely solely on visual and qualitative assessment."
    )

    msgs = [{"role": "user", "content": [
        {"type": "image", "image": tmp},
        {"type": "text",  "text": prompt},
    ]}]
    text = processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    img_in, vid_in = process_vision_info(msgs)
    inp = processor(text=[text], images=img_in, videos=vid_in,
                    padding=True, return_tensors="pt").to(vlm.device)
    with torch.no_grad():
        gen = vlm.generate(**inp, max_new_tokens=400)
    trimmed = [o[len(i):] for i, o in zip(inp.input_ids, gen)]
    return processor.batch_decode(trimmed, skip_special_tokens=True)[0]

dropdown_choices = [
    f"{row['region'].replace('_',' ').title()} — Patch {i}"
    for i, row in test_df.iterrows()
]

DARK_CSS = """
body,.gradio-container{background:#0f0f1a!important;color:#e8e8f0}
.gr-button-primary{background:linear-gradient(135deg,#7c3aed,#2ecc71)!important;
  color:#fff!important;font-weight:700;border-radius:10px;border:none}
h1{text-align:center;background:linear-gradient(90deg,#7c3aed,#2ecc71);
  -webkit-background-clip:text;-webkit-text-fill-color:transparent}
.dark{--background-fill-primary:#1a1a2e;--background-fill-secondary:#0f0f1a}
"""

with gr.Blocks(css=DARK_CSS, title="ForestSense") as demo:
    gr.Markdown("# 🌲 ForestSense — AI Forest Change Detection")

    gr.Markdown("### 📋 Select Test Patch")
    dropdown = gr.Dropdown(choices=dropdown_choices, label="Choose patch",
                           value=dropdown_choices[0])
    btn_dd   = gr.Button("🚀 Run ForestSense Pipeline", variant="primary")

    gr.Markdown("### 🛰️ Input Imagery")
    with gr.Row():
        o_t1rgb = gr.Image(label="T1 Optical RGB (2021)", height=250)
        o_t1sar = gr.Image(label="T1 SAR VV (2021)",      height=250)
        o_t2rgb = gr.Image(label="T2 Optical RGB (2023)", height=250)
        o_t2sar = gr.Image(label="T2 SAR VV (2023)",      height=250)

    gr.Markdown("### 🧠 Model Outputs")
    with gr.Row():
        o_pred = gr.Image(label="Predicted Change Map + Legend", height=280)
        o_gt   = gr.Image(label="Ground Truth Mask + Legend",    height=280)
        o_heat = gr.Image(label="🔥 Change Heatmap",             height=280)

    gr.Markdown("### 🖼️ RAG Graph — T1 | T2 | Change Map | SLIC | RAG")
    o_rag = gr.Image(label="5-Panel Region Adjacency Graph", height=380)

    gr.Markdown("### 📊 Spatial Metrics")
    o_met = gr.Image(label="Spatial Metrics Panel", height=320)

    gr.Markdown("### 🤖 AI Forest Ranger Chat")
    gr.Markdown("*Ask anything — deforestation causes, wildlife, conservation, ecological risk...*")
    chatbot = gr.ChatInterface(
        fn=vlm_chat, type="messages",
        examples=[
            "Generate an ecology report for this patch.",
            "Why might forest loss be occurring here?",
            "What does the fragmentation index tell us about wildlife corridors?",
            "Is this region ecologically vulnerable?",
            "What conservation actions would you recommend?",
        ],
        submit_btn="Ask AI Ranger 🌿",
    )

    outs = [o_t1rgb, o_t1sar, o_t2rgb, o_t2sar,
            o_pred, o_gt, o_heat, o_rag, o_met]

    btn_dd.click(fn=run_from_dropdown, inputs=[dropdown], outputs=outs)

# ── CELL 3: Launch ────────────────────────────────────────────────────────────
demo.launch(share=True, debug=False)
