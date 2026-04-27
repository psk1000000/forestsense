# Technical Deep Dive: ForestSense Architecture & Implementation

This document provides a highly detailed, technical breakdown of the ForestSense pipeline. It is intended for researchers, data scientists, and engineers who wish to understand the exact methodologies, models, and algorithms used in this project.

---

## 1. Data Acquisition (Google Earth Engine)

The foundation of ForestSense relies on multi-temporal, multimodal satellite imagery acquired via the Google Earth Engine (GEE) Python API.

### 1.1 Modalities
We combine two fundamentally different types of remote sensing data to ensure robustness against tropical cloud cover and to capture both biochemical and structural forest properties.

*   **Sentinel-2 (Multispectral Optical):** We extract 10 bands (`B2`, `B3`, `B4` (RGB), `B5`, `B6`, `B7`, `B8`, `B8A` (NIR), `B11`, `B12` (SWIR)). Resolution varies natively between 10m and 20m, but all bands are resampled to 10m. Optical data is excellent for detecting chlorophyll loss.
*   **Sentinel-1 (Synthetic Aperture Radar - SAR):** We use C-band Ground Range Detected (GRD) imagery, extracting `VV` (vertical transmit/vertical receive) and `VH` (vertical transmit/horizontal receive) polarisations. SAR actively penetrates cloud cover and responds strongly to the physical structure (biomass/canopy geometry) of the forest.

### 1.2 Temporal Compositing
For each region, we define two timeframes: **T1 (Jan 1, 2021 – Dec 31, 2021)** and **T2 (Jan 1, 2023 – Dec 31, 2023)**. To remove clouds and seasonal anomalies, we generate an annual median composite for both Optical and SAR collections over the defined geometry.

### 1.3 Automated Pseudo-Label Generation
Instead of manual annotations, we use spectral indices to generate ground truth labels programmatically:
1.  **NDVI** (Normalized Difference Vegetation Index): $(NIR - Red) / (NIR + Red)$. Identifies healthy vegetation.
2.  **NDWI** (Normalized Difference Water Index): $(Green - NIR) / (Green + NIR)$. Masks out water bodies (e.g., Sundarbans rivers).

By calculating the difference ($\Delta NDVI = NDVI_{T2} - NDVI_{T1}$), we classify pixels:
*   `Class 0` (Non-Forest): Low NDVI in both T1 and T2.
*   `Class 1` (Stable Forest): High NDVI in both T1 and T2.
*   `Class 2` (Forest Loss): High NDVI in T1, dropped significantly in T2 ($\Delta NDVI < -0.15$).
*   `Class 3` (Forest Gain): Low NDVI in T1, rose significantly in T2 ($\Delta NDVI > +0.15$).

---

## 2. Data Processing & Patch Extraction

Remote sensing regions are too large to process in memory. We implemented a spatial patching pipeline:

1.  **Tensor Stacking:** The 10 optical bands and 2 SAR bands are stacked into a 12-band tensor.
2.  **Slicing:** Using a sliding window approach, the large regional GeoTIFFs are sliced into `256 × 256` pixel patches. At 10m/pixel resolution, each patch represents exactly **655.36 hectares** ($2.56 \text{ km} \times 2.56 \text{ km}$).
3.  **Filtering:** Patches with more than 80% empty pixels (borders) are discarded.
4.  **Dataset Splits:** The extracted `.npy` patches are split into training (70%), validation (15%), and testing (15%) sets, tracked via a `manifest.csv` file.

---

## 3. SiameseFusionNet: Deep Learning Architecture

The core change detection model is a custom PyTorch architecture called **SiameseFusionNet**, designed specifically for spatio-temporal segmentation.

### 3.1 Siamese Encoders
The model employs a "Siamese" architecture, meaning it uses two identical feature extraction sub-networks that share the exact same weights. 
*   **Backbone:** We use a modified `ResNet-18` architecture. We replace the first convolutional layer to accept 12 input channels (instead of standard 3-channel RGB) and train from scratch (no ImageNet pre-training, as satellite bands do not map to standard RGB photographs).
*   **Mechanism:** `Branch A` processes the T1 patch; `Branch B` processes the T2 patch. They map the temporal states into the same high-dimensional feature space.

### 3.2 Feature Fusion & Decoding
*   **Fusion:** The abstract spatial feature maps from both branches are concatenated along the channel dimension.
*   **Decoder:** A series of Transposed Convolutional layers (Deconv) and upsampling layers decode the fused features back to the original spatial resolution (`256 × 256`).
*   **Output Head:** The final layer uses a 2D convolution mapped to 4 output channels (our 4 classes), followed by a Softmax activation to generate per-pixel class probabilities.

---

## 4. Training and Evaluation Metrics

### 4.1 Loss Function
Forest change datasets are heavily imbalanced (e.g., 85% stable forest, 10% non-forest, 5% actual change). Standard Cross-Entropy fails here. We use a combination loss:
$$Loss = \lambda_1 \cdot \text{DiceLoss} + \lambda_2 \cdot \text{WeightedBCE}$$
The **Dice Loss** maximizes the overlap (Intersection over Union) of the minority classes (Loss/Gain), preventing the model from lazily predicting "Stable Forest" everywhere.

### 4.2 Evaluation Metrics
During validation and testing, the model is evaluated using rigorous semantic segmentation metrics:
*   **Overall Accuracy (OA):** Total correctly classified pixels.
*   **Precision & Recall (per class):** Crucial for evaluating the model's sensitivity to deforestation.
*   **F1-Score:** Harmonic mean of Precision and Recall.
*   **Mean Intersection over Union (mIoU):** The standard metric for segmentation, evaluating the exact spatial overlap of predictions versus ground truth.

---

## 5. Spatial Analytics: SLIC & Region Adjacency Graphs (RAG)

While pixel-wise accuracy is great for computer science, ecologists require structural landscape metrics. 

### 5.1 SLIC Superpixels
We apply the **Simple Linear Iterative Clustering (SLIC)** algorithm to the T1 RGB imagery. SLIC groups adjacent pixels with similar spectral colours and textures into larger, coherent polygons called "superpixels". This naturally isolates contiguous forest stands from rivers, roads, and bare land.

### 5.2 Region Adjacency Graphs (RAG)
Using the superpixels, we build a mathematical graph structure:
*   **Nodes:** Each superpixel is a node. We assign the node a class (Loss, Gain, Stable) based on the mathematical mode (most frequent prediction) of the SiameseFusionNet output within that superpixel boundary.
*   **Edges:** Nodes are connected if they physically touch in the spatial map.

### 5.3 Ecological Metrics
Using `skimage.measure.regionprops`, we extract structural data:
*   **Total/Loss/Gain Hectares:** Calculated by multiplying pixel counts by $0.01$ (since $10m \times 10m = 100m^2 = 0.01$ ha).
*   **Fragmentation Index:** $N_{patches} / Total Area$. A higher index indicates a broken, disconnected ecosystem, which is devastating for wildlife corridors.
*   **Largest Patch Index (LPI):** The percentage of the landscape comprised by the single largest contiguous forest patch.

---

## 6. Vision-Language Model (VLM) Strategy

To interpret the visual output, we integrated **Qwen2-VL-2B-Instruct**, a quantized Vision-Language Model capable of complex spatial reasoning.

### 6.1 Context Injection
A generic VLM does not know the difference between the Sundarbans and the Western Ghats. We engineered a dynamic system prompt that injects hardcoded, deeply specific ecological context into the model's memory *before* it answers. For example, if evaluating Saranda, the model is explicitly informed about *iron-ore mining* and *Sal forest ecology*.

### 6.2 Visual Forcing (Zero-Numeric Prompting)
Initially, VLMs tend to lazily read the metric numbers on the screen and regurgitate them to the user. We solved this via strict prompt engineering:
> *"Do NOT mention or invent any quantitative metrics or hectare numbers. Rely solely on visual and qualitative assessment."*

This forces Qwen2-VL to actually process the image. It looks at the RAG 5-panel figure, identifies the spatial distribution of the red nodes (loss), correlates them with visible features (like rivers or roads in the RGB panels), and synthesises an ecological hypothesis (e.g., *"The red loss clusters are highly concentrated along the riverbanks, suggesting severe erosion or targeted logging accessing the waterway..."*).

---

## 7. The Gradio Application Layer

The entire pipeline is packaged into a user-friendly Gradio web application.
*   **Asynchronous Processing:** Tensors are loaded from disk, pushed to the GPU, passed through the SiameseFusionNet (using mixed precision `torch.autocast` for speed), and processed by the CPU for SLIC/RAG generation.
*   **Interactive Chat:** A stateful `gr.ChatInterface` maintains conversational history, allowing the user to seamlessly ask follow-up questions to the VLM about the currently active spatial patch.
