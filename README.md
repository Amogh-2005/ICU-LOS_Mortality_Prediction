# 🏥 A Multimodal Deep Learning Framework for ICU Outcome Prediction

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch&logoColor=white" />
  <img src="https://img.shields.io/badge/PyTorch%20Lightning-2.4.0-792EE5?logo=lightning&logoColor=white" />
  <img src="https://img.shields.io/badge/Streamlit-1.40.2-FF4B4B?logo=streamlit&logoColor=white" />
  <img src="https://img.shields.io/badge/Dataset-MIMIC--III-0078d7" />
  <a href="https://doi.org/10.1109/ICSIT70359.2026.11766079"><img src="https://img.shields.io/badge/IEEE%20Xplore-ICSIT%202026-00629B?logo=ieee&logoColor=white" /></a>
  <a href="https://doi.org/10.1109/ICSIT70359.2026.11766079"><img src="https://img.shields.io/badge/DOI-10.1109%2FICSIT70359.2026.11766079-blue" /></a>
</p>

> **Published in IEEE Xplore — 2026 International Conference on Sustainability, Innovation & Technology (ICSIT)**  
> DOI: **[10.1109/ICSIT70359.2026.11766079](https://doi.org/10.1109/ICSIT70359.2026.11766079)** · Publisher: IEEE · Added to IEEE Xplore: November 13, 2026

---

## 📌 Table of Contents

1. [Overview](#overview)
2. [Research Paper & Conference](#research-paper--conference)
3. [Architecture](#architecture)
4. [Project Structure](#project-structure)
5. [Dataset](#dataset)
6. [Feature Engineering](#feature-engineering)
7. [Model Components](#model-components)
8. [Training Strategy](#training-strategy)
9. [Loss Functions](#loss-functions)
10. [Evaluation Metrics](#evaluation-metrics)
11. [Explainability](#explainability)
12. [Streamlit Demo App](#streamlit-demo-app)
13. [Installation](#installation)
14. [Usage](#usage)
15. [Results](#results)
16. [Team](#team)

---

## Overview

This repository presents a **multimodal deep learning framework** for **early ICU outcome prediction** using the first 24 hours of patient data from MIMIC-III. The system simultaneously predicts:

| Task | Type | Output |
|------|------|--------|
| **Length of Stay (LOS)** | Ordinal 3-class | Band 0: < 24 h · Band 1: 24–72 h · Band 2: > 72 h |
| **In-Hospital Mortality** | Binary | Survival / Non-survival probability |

The model fuses two clinically heterogeneous data streams — **vital sign time-series** (temporal) and **tabular clinical features** (static + lab + diagnostic) — through a **Cross-Modal Attention Fusion** layer, enabling each modality to condition on the other for richer, context-aware representations.

---

## Research Paper & Conference

Our work has been published in **IEEE Xplore**:

> **2026 International Conference on Sustainability, Innovation & Technology (ICSIT)**  
> 📋 **Paper Title:** *A Multimodal Deep Learning Framework for ICU Outcome Prediction*  
> 🏛️ **Publisher:** IEEE  
> 🔗 **DOI:** 10.1109/ICSIT70359.2026.11766079
> 🗓️ **Date of Conference:** 26–27 September 2026  
> 📅 **Date Added to IEEE Xplore:** 13 November 2026  

- 🔗 [View on IEEE Xplore](https://doi.org/10.1109/ICSIT70359.2026.11766079)
- 📄 [Read the Research Paper PDF](./research_paper/final%20icu%20prediction%20paper.pdf)

The conference focuses on cutting-edge advances in Sustainability, Innovation, and Technology, including applications of AI and machine learning in healthcare and clinical decision support systems.
---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                             MultimodalICUModel                              │
│                                                                             │
│          Vitals (B, 24, 5)               Tabular (Clinical + Lab) (B, 156)  │
│                 │                                        │                  │
│                 ▼                                        ▼                  │
│    ┌───────────────────────────┐           ┌───────────────────────────┐    │
│    │        LSTMEncoder        │           │      TabularEncoder       │    │
│    │    Bidirectional LSTM     │           │ MLP [156→512→256→128→256] │    │
│    │   + Temporal Attention    │           │  BatchNorm + GELU + Drop  │    │
│    └─────────────┬─────────────┘           └─────────────┬─────────────┘    │
│                  │ vitals_emb (B, 256)                   │ tabular_emb      │
│                  │ (Query)                               │ (Key, Value)     │
│                  └───────────────────┬───────────────────┘ (B, 256)         │
│                                      │                                      │
│                                      ▼                                      │
│                        ┌───────────────────────────┐                        │
│                        │ CrossModalAttentionFusion │                        │
│                        │  8-head cross-attention   │                        │
│                        └─────────────┬─────────────┘                        │
│                                      │ fused (B, 256)                       │
│                                      ▼                                      │
│                        ┌───────────────────────────┐                        │
│                        │       MultiTaskHead       │                        │
│                        │  ┌─────────────┐ ┌─────┐  │                        │
│                        │  │ LOS Private │ │Mort │  │                        │
│                        │  │ Encoder(MLP)│ │Priv │  │                        │
│                        │  └──────┬──────┘ └─┬───┘  │                        │
│                        │         ▼          ▼      │                        │
│                        │    los_logits   mort_logit│                        │
│                        │      (B, 2)       (B, 1)  │                        │
│                        └───────────────────────────┘                        │
└─────────────────────────────────────────────────────────────────────────────┘
```

**Combined Loss:**
```
L = α · L_ordinal_LOS  +  (1-α) · L_binary_mortality
      + 0.15 · band1_recall_penalty
      + 0.10 · ordinal_consistency_loss
```
where `α = 0.65`, balancing both tasks with a stronger emphasis on LOS prediction.

---

## Project Structure

```
ICU-Predictor/
├── app/
│   └── demo.py                     # Streamlit interactive demo application
│
├── src/
│   ├── models/
│   │   ├── multimodal_model.py     # PyTorch Lightning LightningModule (full pipeline)
│   │   ├── lstm_encoder.py         # Bidirectional LSTM + Bahdanau temporal attention
│   │   ├── tabular_encoder.py      # MLP encoder for 156-dim tabular features
│   │   ├── fusion.py               # Cross-modal attention fusion layer
│   │   └── multitask_head.py       # Dual private encoders + LOS & mortality heads
│   │
│   ├── training/
│   │   ├── trainer.ipynb           # End-to-end training notebook  
│   │   ├── losses.py               # Ordinal LOS loss, mortality BCE, combined loss
│   │   └── metrics.py              # AUROC, F1, per-band accuracy, classification report
│   │
│   ├── explainability/
│   │   └── explainability.ipynb    # SHAP feature importance & model explainability analysis
│   │
│   └── data_load/
│       ├── dataset.py              # ICUDataset — PyTorch Dataset with split support
│       └── dataloader.py           # DataLoader factory (WeightedRandomSampler for train)
│
├── data/
│   └── new2.pkl                    # Preprocessed MIMIC-III dataset (train/val/test splits)
│
├── outputs/
│   ├── checkpoints/                # Saved model weights & calibration artifacts
│   ├── figures/                    # Training curves, confusion matrices, SHAP plots
│   └── SHAP_values/                # Saved SHAP value arrays
│
├── requirements.txt                # Python dependencies
└── README.md                       # This file
```

---

## Dataset

This project uses the **MIMIC-III Clinical Database** (Medical Information Mart for Intensive Care III).

- **Source:** Beth Israel Deaconess Medical Center, Boston, MA
- **Access:** PhysioNet (requires credentialed access and signed DUA)
- **Cohort:** Adult ICU admissions; first 24-hour window
- **Split:** Train / Validation / Test (stratified by LOS band and mortality)
- **Pickle file:** `data/new2.pkl` contains preprocessed tensors, split indices, normalisation statistics, label encoders, and class-weighting metadata

> ⚠️ **Note:** The MIMIC-III dataset requires IRB-approved PhysioNet credentials. The `new2.pkl` file is **not** included in this repository. Users must independently obtain and preprocess the raw MIMIC-III data.

---

## Feature Engineering

### Time-Series Modality — Vitals `(B, 24, 5)`

| Channel | Description |
|---------|-------------|
| `heart_rate` | Heart rate (bpm) — hourly bin mean |
| `resp_rate` | Respiratory rate (breaths/min) |
| `sbp` | Systolic blood pressure (mmHg) |
| `dbp` | Diastolic blood pressure (mmHg) |
| `spo2` | Peripheral oxygen saturation (%) |

All values are **z-score normalised** using training-set statistics.

---

### Tabular Modality — 156-dimensional Feature Vector `(B, 156)`

| Index Range | Feature Group | # Features |
|-------------|---------------|------------|
| `[0–6]` | Static (age, gender, ethnicity, admission type, insurance, care unit, prior ICU count) | 7 |
| `[7–14]` | Lab means (creatinine, BUN, lactate, haemoglobin, WBC, sodium, potassium, platelets) | 8 |
| `[15–22]` | Lab minimums | 8 |
| `[23–30]` | Lab maximums | 8 |
| `[31–38]` | Lab trend slopes | 8 |
| `[39–43]` | Vital sign trend slopes | 5 |
| `[44–46]` | Composite clinical scores (shock index, BUN-to-creatinine ratio, sepsis proxy) | 3 |
| `[47–54]` | Lab missingness indicators (binary) | 8 |
| `[55–154]` | CCS diagnosis multi-hot encoding (100 categories) | 100 |
| `[155]` | Mechanical ventilation flag (binary) | 1 |
| **Total** | | **156** |

---

## Model Components

### 1. `LSTMEncoder` — Vital Sign Temporal Encoder

- **Architecture:** Bidirectional 2-layer LSTM
- **Input:** `(B, 24, 5)` — 24 hourly bins × 5 vital channels
- **Pooling:** Bahdanau-style additive **temporal attention** over all 24 LSTM output states (replaces naive final-state pooling)
- **Output:** `(B, 256)` — attention-weighted context vector

### 2. `TabularEncoder` — Clinical & Lab Data MLP

- **Input:** `(B, 156)` — 156-dimensional vector comprising clinical (demographics, admin, diagnoses) and lab test data (means, minimums, maximums, slopes, missingness)
- **Architecture:** MLP with layers `[156 → 512 → 256 → 128 → 256]`
- **Regularisation:** BatchNorm1d + GELU activation + Dropout (p=0.3) after each hidden layer
- **Output:** `(B, 256)`

### 3. `CrossModalAttentionFusion` — Multimodal Fusion

- **Mechanism:** 8-head **cross-modal attention**
  - Vitals embedding acts as **Query**
  - Tabular embedding acts as **Key** and **Value**
- **Post-attention:** Residual connection → LayerNorm → Feed-Forward sublayer → LayerNorm
- **Output:** `(B, 256)` — fused multimodal representation

### 4. `MultiTaskHead` — Dual Prediction Heads

Two **independent private encoders** (2-layer MLPs) prevent task gradient interference:

| Head | Task | Output | Decoding |
|------|------|--------|----------|
| LOS Head | Ordinal 3-band LOS | `(B, 2)` boundary logits | K-1 binary decomposition with Platt scaling |
| Mortality Head | Binary mortality | `(B, 1)` scalar logit | Sigmoid with calibrated threshold |

---

## Training Strategy

| Hyperparameter | Value |
|---------------|-------|
| Optimiser | AdamW |
| Learning rate | 2×10⁻⁴ |
| Weight decay | 1×10⁻⁴ |
| LR Scheduler | ReduceLROnPlateau (factor=0.5, patience=5) |
| Scheduler monitor | `val/composite` (↑ higher is better) |
| Batch size | 64 |
| Class balancing | WeightedRandomSampler on training set |
| Experiment tracking | Weights & Biases (W&B) |
| Hyperparameter search | Optuna |
| Post-training calibration | Platt temperature scaling |

**Composite validation metric:**
```
val/composite = (Band-2 AUROC + Mortality AUROC + Band-1 Recall) / 3
```
Used for model checkpointing and learning rate scheduling.

---

## Loss Functions

### Ordinal LOS Loss (`OrdinalLOSLoss`)
Converts 3-band ordinal labels into two independent binary boundary targets using **K-1 decomposition**:
- `b0_target = (LOS ≥ Band 1)` — boundary 0
- `b1_target = (LOS ≥ Band 2)` — boundary 1

Each boundary uses `BCEWithLogitsLoss` with class-frequency-derived `pos_weight`.

### Band-1 Recall Penalty
Penalises cases where true Band-1 patients are pushed toward Band-2 by the model. Computed as the mean ReLU of boundary-1 logits for Band-1 patients (weight: **0.15**).

### Ordinal Consistency Loss
Penalises violations of the ordinal constraint `P(≥ Band 1) ≥ P(≥ Band 2)` using `ReLU(b1_logit − b0_logit)` (weight: **0.10**).

### Mortality Loss (`MortalityLoss`)
Standard `BCEWithLogitsLoss` with `pos_weight` equal to the negative-to-positive ratio in the training set.

---

## Evaluation Metrics

### LOS Task
| Metric | Description |
|--------|-------------|
| `los_acc` | Overall 3-class accuracy |
| `los_macro_f1` | Macro-averaged F1 |
| `los_f1_band{0,1,2}` | Per-band F1 scores |
| `los_acc_band{0,1,2}` | Per-band accuracy |
| `los_band2_auroc` | AUROC for Band-2 vs. rest (binary) |

### Mortality Task
| Metric | Description |
|--------|-------------|
| `mort_auroc` | Area Under the ROC Curve |
| `mort_f1` | Binary F1 |
| `mort_acc` | Accuracy |
| `mort_precision` | Precision |
| `mort_recall` | Recall |

---

## Explainability

The framework provides three explainability modalities via the Streamlit demo and `src/explainability/explainability.ipynb`:

### 1. Integrated Gradients (IG)
Attribution of each of the 156 tabular features toward the predicted outcome. Uses a **zero baseline** with 50 interpolation steps. Features are ranked by absolute attribution magnitude.

### 2. Temporal Attention Visualisation
A 24-bar chart showing which **ICU hours** the LSTM's Bahdanau attention focused on, extracted via a forward hook on `lstm_encoder.attention.score_layer`.

### 3. Feature Group Importance
Aggregates mean |IG attribution| by feature group (Labs, Vital Trends, Composite Scores, Diagnoses, Static Features, etc.) for a high-level interpretability overview.

### 4. SHAP Values
Detailed SHAP-based analysis is explored in `src/explainability/explainability.ipynb` with stored outputs in `outputs/SHAP_values/`.

---

## Streamlit Demo App

An interactive clinical decision support interface is provided in `app/demo.py`.

**Run the app:**
```bash
streamlit run app/demo.py
```

**Input sections:**
- **Section 1 — Vital Signs:** Heart rate, respiratory rate, systolic/diastolic BP, SpO2
- **Section 2 — Laboratory Values:** 8 lab parameters (missing values supported)
- **Section 3 — Static Features:** Age, gender, ethnicity, admission type, insurance, care unit, prior ICU count
- **Section 4 — Diagnoses (CCS):** Multi-select from top-20 CCS diagnosis codes
- **Section 5 — Procedures:** Mechanical ventilation flag

**Output panels:**
- LOS prediction (band + per-band probabilities + stacked bar chart)
- In-hospital mortality probability (risk badge + calibrated threshold)
- Feature Attribution (IG) chart
- Vital sign temporal attention (24-hour bar chart)
- Feature group importance chart

---

## Installation

### Prerequisites
- Python 3.10 or higher
- CUDA-capable GPU recommended (CPU inference supported)

### Steps

```bash
# 1. Clone the repository
git clone https://github.com/Amogh-2005/los-mortalityprediction.git
cd los-mortalityprediction

# 2. Create and activate a virtual environment
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. (Optional) Install PyTorch with CUDA support
# Visit https://pytorch.org/get-started/locally/ and choose your CUDA version
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| `pytorch-lightning` | 2.4.0 | Training framework |
| `torchmetrics` | 1.4.3 | Metric computation |
| `numpy` | 1.26.4 | Numerical computing |
| `pandas` | 2.2.3 | Data manipulation |
| `scikit-learn` | 1.5.2 | Metrics, preprocessing |
| `scipy` | 1.13.1 | Scientific computing |
| `wandb` | 0.19.1 | Experiment tracking |
| `optuna` | 3.6.1 | Hyperparameter optimisation |
| `shap` | 0.46.0 | SHAP explainability |
| `lime` | 0.2.0.1 | LIME explainability |
| `matplotlib` | 3.9.2 | Plotting |
| `seaborn` | 0.13.2 | Statistical visualisation |
| `streamlit` | 1.40.2 | Demo web application |
| `tqdm` | 4.66.6 | Progress bars |

---

## Usage

### Training

Open and run the training notebook:
```bash
jupyter notebook src/training/trainer.ipynb
```

The notebook covers:
- Data loading with `ICUDataset` and `get_all_dataloaders()`
- Model instantiation with `MultimodalICUModel(config)`
- W&B experiment tracking
- Optuna hyperparameter sweeps
- Platt temperature scaling for post-hoc calibration
- Checkpoint saving to `outputs/checkpoints/`

### Inference (Standalone)

```python
import torch
from src.models.multimodal_model import MultimodalICUModel

# Load model
model = MultimodalICUModel(config={})
ckpt  = torch.load("outputs/checkpoints/best_calibrated.ckpt", map_location="cpu")
model.load_state_dict(ckpt["state_dict"], strict=False)
model.eval()

# vitals: (1, 24, 5) — z-score normalised
# tabular: (1, 156) — z-score normalised (continuous), binary as-is
los_logits, mortality_logit = model(vitals, tabular)
```

### Explainability Notebook

```bash
jupyter notebook src/explainability/explainability.ipynb
```

---

## Results

> Results based on the hold-out **test set** (2,250 patients) after Platt temperature scaling and calibrated decision thresholds.  
> Best checkpoint: **Epoch 23** · `val/composite = 0.7090`

### LOS Classification Performance (Calibrated Threshold)

| LOS Band | AUROC | Recall | Precision |
|----------|-------|--------|-----------|
| Band 0 (< 24h) | 0.8304 | 0.76 | 0.40 |
| Band 1 (24–72h) | 0.6096 | 0.25 | 0.61 |
| Band 2 (> 72h) | 0.8020 | 0.71 | 0.60 |
| **Macro average** | **0.7474** | **0.57** | **0.54** |

### Mortality Prediction Performance

| Metric | Value |
|--------|-------|
| AUROC | **0.8675** |
| Recall | 0.70 |
| Precision | 0.38 |


---

## Team

| Name | Role |
|------|------|
| **Dr. Kumar Dilip** | Principal Supervisor / Mentor |
| **Amogh A Kashyap** | Research & Development |
| **Abhay Shetty** | Research & Development |
| **Aditya Sujith Kalmadi** | Research & Development |
| **Rashmi S Gowda** | Research & Development |

---

## Citation

If you use this work, please cite:

```bibtex
@inproceedings{icsit2026_icu_multimodal,
  title     = {A Multimodal Deep Learning Framework for ICU Outcome Prediction},
  author    = {Dilip, Kumar and Kashyap, Amogh A. and Shetty, Abhay and Kalmadi, Aditya Sujith and Gowda, Rashmi S.},
  booktitle = {2026 International Conference on Sustainability, Innovation \& Technology (ICSIT)},
  year      = {2026},
  publisher = {IEEE},
  doi       = {10.1109/ICSIT70359.2026.11766079}
}
```

---

## License

This project is intended for academic and research use only. The MIMIC-III dataset is governed by its own [PhysioNet Credentialed Health Data License](https://physionet.org/content/mimiciii/). Please ensure compliance with all applicable data use agreements before using or adapting this code.

---

<p align="center">
  Made with ❤️ for advancing clinical AI · Published in IEEE Xplore · ICSIT 2026 · <a href="https://doi.org/10.1109/ICSIT70359.2026.11766079">DOI: 10.1109/ICSIT70359.2026.11766079</a>
</p>
