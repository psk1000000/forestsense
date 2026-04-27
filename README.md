## 🚀 How to Run the forestsense Dashboard (Google Colab)

Follow these steps to set up and run the project using Google Colab.

---

### 📥 Step 1: Clone the Repository

```bash id="u9q2xj"
git clone https://github.com/psk1000000/forestsense.git
cd forestsense
```

---

### ⚙️ Step 2: Setup Google Colab

* Open https://colab.research.google.com/
* Create a **New Notebook**
* Go to **Runtime → Change runtime type → Select T4 GPU**

---

### 🔗 Step 3: Mount Google Drive

```python id="s5v0fp"
from google.colab import drive
drive.mount('/content/drive')
```

---

### 📦 Step 4: Install Dependencies

```bash id="2m6qk2"
pip install -r requirements.txt
pip install gdown gradio qwen-vl-utils scikit-image
```

---

### 🤖 Step 5: Download Model Weights

```bash id="vfxk5h"
gdown https://drive.google.com/uc?id=1f0oTbxYaSuEyP2SHmQbwZHMW6Es7RLFX -O /content/drive/MyDrive/forestsense/best_model.pth
```

📁 Model file must be located at:

```
/content/drive/MyDrive/forestsense/best_model.pth
```

---

### 🧠 Step 6: Load AI Models

```python id="r2lq8d"
import sys, torch
import pandas as pd
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor

PROJECT_ROOT = "/content/drive/MyDrive/forestsense"
sys.path.insert(0, PROJECT_ROOT)

from model.architecture import build_model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("⏳ Loading SiameseNet...")
siamese_model = build_model(backbone="resnet18", pretrained=False).to(device)

ckpt = torch.load(f"{PROJECT_ROOT}/best_model.pth", map_location=device)
siamese_model.load_state_dict(ckpt["model_state_dict"])
siamese_model.eval()

print("⏳ Loading Qwen2-VL-2B...")
vlm = Qwen2VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2-VL-2B-Instruct",
    torch_dtype=torch.float16,
    device_map="auto"
)

processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-2B-Instruct")

print("✅ All Models Ready!")
```

---

### 🌐 Step 7: Launch the Dashboard

I have included a file called `05_interactive_demo.py` in the repository.

* Open that file
* Copy all the code starting from **Line 13** (`import os, sys, io...`) to the end
* Paste it into a new Colab cell
* Run the cell

---

### 🎉 Done!

A **Gradio public link** will appear. Open it to use the forestsense dashboard.

---

## ⚠️ Notes

* Ensure GPU runtime is enabled
* First run may take ~1–2 minutes
