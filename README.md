# Brain Tumour Image Classification using CNN

> **67% test accuracy** on 1,311 unseen MRI scans across 4 tumour classes — built from scratch using TensorFlow/Keras.

---

## Overview

This project builds a custom Convolutional Neural Network (CNN) to automatically classify brain MRI scans into four categories:

| Class | Description |
|-------|-------------|
| Glioma | Arises from glial cells; often malignant; appears as irregular, poorly defined masses |
| Meningioma | Arises from the meninges; typically benign; well-defined rounded masses |
| Pituitary Tumour | Affects the pituitary gland; appears in the sellar region at the skull base |
| No Tumor | Healthy brain tissue; clean, uniform appearance |

Manual MRI diagnosis is slow, subjective, and prone to fatigue errors. This model provides fast, consistent preliminary screening to assist radiologists.

---

## Dataset

**Source:** [Brain Tumour Classification — Kaggle](https://www.kaggle.com/datasets/rishiksaisanthosh/brain-tumour-classification)

| Split | Images |
|-------|--------|
| Training | 22,848 |
| Testing | 1,311 |
| **Total** | **24,159** |

Class distribution (training): Glioma 5,284 · Meningioma 5,356 · No Tumor 6,380 · Pituitary 5,828

The dataset includes augmented images (horizontal/vertical flips, rotation, zoom, brightness variation) to improve robustness across different scanner orientations.

---

## Model Architecture

Custom Sequential CNN built in Keras — no pre-trained backbone.

```
Input: 128×128×3 MRI image
→ Conv2D (32 filters, 3×3) + MaxPooling2D
→ Conv2D (64 filters, 3×3) + MaxPooling2D
→ Conv2D (128 filters, 3×3) + MaxPooling2D
→ Flatten (25,088 values)
→ Dense (128 units, ReLU)
→ Dropout (0.5)
→ Dense (4 units, Softmax)

Total parameters: 3,305,156 (12.61 MB)
```

**Training config:**
- Optimiser: Adam
- Loss: Categorical Cross-Entropy
- Epochs: 10 | Batch size: 32
- Images streamed via `ImageDataGenerator` (no full dataset loaded into RAM)

---

## Results

| Metric | Value |
|--------|-------|
| Test Accuracy | **67%** |

The model achieved 67% test accuracy over 10 epochs. The confusion matrix analysis reveals the main failure mode was predicting "No Tumor" when uncertain, likely due to class imbalance and low input resolution (128×128).

### Confusion Matrix Highlights

- **Best class:** No Tumor (137/~405 correct) — healthy tissue is visually distinct
- **Hardest class:** Pituitary (65/~300 correct) — small sellar region is hard to resolve at 128×128
- **Main failure mode:** All classes get confused with No Tumor when model confidence is low, likely due to class imbalance (No Tumor has the largest training set at 6,380 images)
- **Secondary confusion:** Glioma ↔ Meningioma — both appear as large irregular masses in 2D axial slices; 3D positional cues are lost when volumetric data is flattened

---

## How to Run

### 1. Clone the repo
```bash
git clone https://github.com/ApurvaSunil/brain-tumour-classification.git
cd brain-tumour-classification
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Download the dataset
Download from [Kaggle](https://www.kaggle.com/datasets/sartajbhuvaji/brain-tumor-classification-mri) and place it as:
```
data/
├── Training/
│   ├── glioma/
│   ├── meningioma/
│   ├── notumor/
│   └── pituitary/
└── Testing/
    ├── glioma/
    ├── meningioma/
    ├── notumor/
    └── pituitary/
```

### 4. Run the notebook
Open `brain_tumour_classification.ipynb` in Google Colab or Jupyter and run all cells.

---

## Future Improvements

1. **Transfer Learning** — Fine-tune ResNet50 / EfficientNet to push accuracy beyond 99%
2. **Higher resolution** — Increase input from 128×128 to 224×224 to preserve fine structural details
3. **Grad-CAM** — Visualise which MRI regions drove each prediction (critical for clinical trust)
4. **Multi-modal input** — Combine T1, T2, FLAIR MRI modalities as multi-channel inputs
5. **Larger datasets** — Train across multiple hospitals/scanners for real-world robustness

---

## Team

| Name | Registration No. |
|------|-----------------|
| Karra Apurva Sunil | 2023006873 |
| Kinnera Chandhan Sai | 2023004196 |
| Joshika Yadlapalli | 2023003427 |

GITAM School of Technology — Department of Computer Science and Engineering

---

## Repository Structure

```
brain-tumour-classification/
├── brain_tumour_classification.ipynb   # Full pipeline: preprocessing → training → evaluation
├── report.pdf                          # Detailed academic project report
├── requirements.txt
└── README.md
```
