"""
ForestSense — Phase 4: VLM Integration (Qwen2-VL-2B)
=====================================================
Run this in Google Colab using a GPU runtime (T4 or L4).
This script loads the state-of-the-art open-source Vision-Language Model
to interpret the Region Adjacency Graphs (RAG) and spatial metrics generated in Phase 3.

CELL STRUCTURE:
  Cell 1 — Install dependencies
  Cell 2 — Load Qwen2-VL-2B-Instruct to GPU
  Cell 3 — Generate AI Ecology Reports for each region
"""

# ── CELL 1: Install Dependencies ──────────────────────────────────────────────
!pip install -q git+https://github.com/huggingface/transformers accelerate
!pip install -q qwen-vl-utils torchvision


# ── CELL 2: Load Qwen2-VL Model ───────────────────────────────────────────────
import torch
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor

print("⏳ Downloading and loading Qwen2-VL-2B-Instruct (this takes ~1-2 mins)...")

# Load the model directly to the GPU in bfloat16 for fast inference
model_id = "Qwen/Qwen2-VL-2B-Instruct"

model = Qwen2VLForConditionalGeneration.from_pretrained(
    model_id, 
    torch_dtype=torch.bfloat16, 
    device_map="auto"
)
processor = AutoProcessor.from_pretrained(model_id)

print(f"✅ VLM loaded successfully on {model.device}!")


# ── CELL 3: Generate Ecology Reports ──────────────────────────────────────────
import os
import json
from qwen_vl_utils import process_vision_info

# Paths to Phase 3 outputs
METRICS_PATH = "/content/drive/MyDrive/forestsense/outputs/phase3_graphs/regional_graph_metrics.json"
GRAPH_DIR    = "/content/drive/MyDrive/forestsense/outputs/phase3_graphs"

if not os.path.exists(METRICS_PATH):
    raise FileNotFoundError("Run Phase 3 first! Could not find regional_graph_metrics.json")

with open(METRICS_PATH, "r") as f:
    metrics = json.load(f)

print("🌲 ForestSense AI Ranger — Generating Reports...\n")

for region, data in metrics.items():
    image_path = f"{GRAPH_DIR}/rag_{region}.png"
    if not os.path.exists(image_path):
        print(f"⚠️ Image not found for {region}, skipping.")
        continue
    
    region_name = region.replace('_', ' ').title()
    
    # ── Construct the Multimodal Prompt ───────────────────────────────────────
    prompt = f"""You are an expert forest ecologist and remote sensing analyst.
Analyze the provided Region Adjacency Graph (RAG) overlay image and the exact spatial metrics for the {region_name} region between 2021 and 2023.

Quantitative Metrics (from SiameseFusionNet):
- Total Forest Area: {data['total_forest_ha']:.1f} ha
- Forest Loss: {data['forest_loss_ha']:.1f} ha
- Forest Gain: {data['forest_gain_ha']:.1f} ha
- Net Change: {data['net_change_ha']:.1f} ha
- Fragmentation Index: {data['fragmentation_idx']:.3f} (higher = more fragmented)
- Largest Patch Index (LPI): {data['largest_patch_idx']:.3f} (closer to 1.0 = contiguous)

Task:
1. Interpret the visual RAG graph in the image (Panel 5). What do the nodes and edges tell you about the spatial structure of the forest change?
2. Explain what the quantitative metrics mean for the ecological health of this region (e.g. is it highly fragmented or stable?).
3. Provide a brief 2-sentence executive summary of the forest state.

Format your response clearly with headings. Do not invent any numbers, use only the metrics provided above.
"""
    
    # Prepare inputs for Qwen2-VL
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_path},
                {"type": "text", "text": prompt},
            ],
        }
    ]
    
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to(model.device)
    
    print(f"⏳ Generating report for {region_name}...")
    
    # Generate
    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=400)
        
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]
    
    # Print the report
    print("\n" + "="*70)
    print(f" 📄 AI ECOLOGY REPORT: {region_name.upper()}")
    print("="*70)
    print(output_text)
    print("="*70 + "\n\n")

    # Optionally save to a text file
    report_path = f"{GRAPH_DIR}/report_{region}.txt"
    with open(report_path, "w") as f:
        f.write(output_text)

print("✅ All AI reports generated and saved to Drive!")
