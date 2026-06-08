# =============================================================================
# ICU Outcome Predictor — Streamlit Demo
# app/streamlit_app.py
#
# Requirements (pip install):
#   streamlit>=1.32  torch  plotly  scikit-learn  numpy  pandas
#
# Run:
#   streamlit run app/streamlit_app.py
# =============================================================================

import sys, os, warnings, pickle
warnings.filterwarnings("ignore")

# Make src importable when running from project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import torch
import torch.nn.functional as F
import plotly.graph_objects as go
import streamlit as st

from src.models.multimodal_model import MultimodalICUModel

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PICKLE_PATH = os.path.join(BASE_DIR, "data", "new2.pkl")
CKPT_PATH   = os.path.join(BASE_DIR, "outputs", "checkpoints", "best_calibrated.ckpt")
CAL_PATH    = os.path.join(BASE_DIR, "outputs", "checkpoints", "calibration_artifacts.pt")

# ---------------------------------------------------------------------------
# Calibration constants (also loaded from file; kept here as fallback)
# ---------------------------------------------------------------------------
BAND2_THRESH  = 0.469675
MORT_THRESH   = 0.188724
B1_TEMP       = 0.849921
MORT_TEMP     = 1.706231

# ---------------------------------------------------------------------------
# Vital channel order (matches training)
# ---------------------------------------------------------------------------
VITAL_CHANNELS = ["heart_rate", "resp_rate", "sbp", "dbp", "spo2"]

# ---------------------------------------------------------------------------
# Lab variable order (matches feature_names in pickle)
# ---------------------------------------------------------------------------
LAB_VARS = ["creatinine", "bun", "lactate", "hemoglobin", "wbc",
            "sodium", "potassium", "platelets"]

# ---------------------------------------------------------------------------
# CCS codes present in the feature vector, in their exact column order.
# Derived directly from the pickle's feature_names list.
# ---------------------------------------------------------------------------
CCS_COL_ORDER = [
    255,101,97,55,59,99,219,50,227,135,258,54,3,157,148,105,141,160,149,231,
    259,230,100,158,68,233,103,16,232,67,131,102,48,98,237,201,65,130,82,79,
    134,147,216,132,81,133,69,142,207,146,129,83,5,226,84,203,66,205,80,206,
    8,159,161,242,197,145,1,52,225,17,236,244,12,245,104,199,243,13,239,238,
    204,96,202,20,176,18,15,51,234,251,85,250,4,78,19,87,10,254,248,53
]
# dict: ccs_number -> column position within the 100-dim CCS block (0-indexed)
CCS_CODE_TO_BLOCK_IDX = {code: i for i, code in enumerate(CCS_COL_ORDER)}

# Top-20 CCS codes to show in the multiselect (only those present in the feature vector)
CCS_TOP20 = {
    8:   "Septicemia",
    50:  "Diabetes mellitus",
    53:  "Disorders of lipid metabolism",
    55:  "Fluid/electrolyte disorders",
    83:  "Hypertension",
    84:  "Cardiac arrest",
    96:  "Heart valve disorders",
    97:  "Peri-/endo-/myocarditis",
    99:  "Hypertension with complications",
    100: "Acute myocardial infarction",
    101: "Coronary artery disease",
    135: "Post-op/procedural recovery",
    149: "Biliary tract disease",
    157: "Acute renal failure",
    203: "Osteoarthritis",
    219: "Cardiac catheterisation",
    254: "Rehabilitation care",
}
# Keep only codes actually in the feature vector
CCS_TOP20 = {k: v for k, v in CCS_TOP20.items() if k in CCS_CODE_TO_BLOCK_IDX}

# ---------------------------------------------------------------------------
# Label encoder classes (read from pickle; hardcoded here as fallback)
# ---------------------------------------------------------------------------
GENDER_CLASSES     = ["F", "M"]
ADMISSION_CLASSES  = ["ELECTIVE", "EMERGENCY", "NEWBORN", "URGENT"]
INSURANCE_CLASSES  = ["Government", "Medicaid", "Medicare", "Private", "Self Pay"]
CAREUNIT_CLASSES   = ["CCU", "CSRU", "MICU", "NICU", "SICU", "TSICU"]
# Ethnicity: map UI choices to closest MIMIC category
ETHNICITY_MAP = {
    "White":    "WHITE",
    "Black":    "BLACK/AFRICAN AMERICAN",
    "Hispanic": "HISPANIC OR LATINO",
    "Asian":    "ASIAN",
    "Other":    "OTHER",
}


# ===========================================================================
# Data loading (cached)
# ===========================================================================

@st.cache_data(show_spinner=False)
def load_dataset_artifacts():
    """Load stats, feature_names and label_encoders from the pickle."""
    with open(PICKLE_PATH, "rb") as fh:
        data = pickle.load(fh)
    stats         = data.get("stats", {})
    feature_names = data.get("feature_names", [])
    label_encoders= data.get("label_encoders", {})
    return stats, feature_names, label_encoders


# ===========================================================================
# Model loading (cached)
# ===========================================================================

@st.cache_resource(show_spinner=False)
def load_model():
    """
    Instantiate MultimodalICUModel and load weights from the raw state-dict
    checkpoint.  Also loads calibration artifacts.
    Returns (model, cal) where cal is the calibration dict.
    """
    # Load calibration
    try:
        cal = torch.load(CAL_PATH, map_location="cpu")
    except Exception as e:
        st.error(f"Cannot load calibration artifacts: {e}")
        cal = {
            "band2_threshold": BAND2_THRESH,
            "mortality_threshold": MORT_THRESH,
            "b1_temperature": B1_TEMP,
            "mort_temperature": MORT_TEMP,
        }

    # Build model
    model = MultimodalICUModel(config={})
    model.eval()

    # Load checkpoint
    try:
        ckpt = torch.load(CKPT_PATH, map_location="cpu")
        # The checkpoint is already a plain OrderedDict of weights
        if isinstance(ckpt, dict) and "state_dict" in ckpt:
            state_dict = ckpt["state_dict"]
        else:
            state_dict = ckpt
        # Strip 'model.' prefix if present
        cleaned = {}
        for k, v in state_dict.items():
            cleaned[k[6:] if k.startswith("model.") else k] = v
        model.load_state_dict(cleaned, strict=False)
    except Exception as e:
        st.error(f"Cannot load model checkpoint: {e}")

    return model, cal

# ===========================================================================
# Feature vector construction
# ===========================================================================

def zscore(value, mean, std, eps=1e-8):
    return (value - mean) / (std + eps)


def build_vitals_tensor(hr, rr, sbp, dbp, spo2, stats):
    """
    Build vitals tensor of shape (1, 24, 5).
    Tiles the entered values across all 24 hourly bins.
    Applies z-score normalisation using training-set vitals stats.
    stats['vitals'] = {'heart_rate': {'mean': .., 'std': ..}, ...}
    """
    vs = stats.get("vitals", {})
    channels = []
    for val, key in zip(
        [hr, rr, sbp, dbp, spo2],
        ["heart_rate", "resp_rate", "sbp", "dbp", "spo2"]
    ):
        m = vs.get(key, {}).get("mean", 0.0)
        s = vs.get(key, {}).get("std",  1.0)
        norm_val = zscore(val, m, s)
        channels.append(np.full(24, norm_val, dtype=np.float32))

    vitals_np = np.stack(channels, axis=1)          # (24, 5)
    vitals_t  = torch.tensor(vitals_np).unsqueeze(0) # (1, 24, 5)
    return vitals_t


def build_tabular_vector(
    # Vitals (raw)
    hr, rr, sbp, dbp, spo2,
    # Labs (raw or None)
    creatinine, bun, lactate, hemoglobin, wbc, sodium, potassium, platelets,
    # Static
    age, gender_idx, ethnicity_idx, admission_idx, insurance_idx, careunit_idx,
    prior_icu,
    # Diagnoses
    selected_ccs_codes,
    # Procedures
    mech_vent_flag,
    # stats dict
    stats,
):
    """
    Assembles the 156-dim tabular feature vector in the exact order
    used during training (read from feature_names in the pickle):

    [0-6]   age, gender, ethnicity, admission_type, insurance,
            first_careunit, prior_icu_count
    [7-14]  mean_creatinine .. mean_platelets
    [15-22] min_creatinine  .. min_platelets
    [23-30] max_creatinine  .. max_platelets
    [31-38] trend_creatinine .. trend_platelets  (lab trends, all 0)
    [39-43] trend_heart_rate .. trend_spo2       (vital trends, all 0)
    [44-46] shock_index, bun_cr_ratio, sepsis_proxy
    [47-54] miss_creatinine .. miss_platelets
    [55-154] ccs_* (100-dim binary, in CCS_COL_ORDER)
    [155]   mech_vent
    """
    feat = np.zeros(156, dtype=np.float32)

    # ----- [0-6] Static features ----------------------------------------
    feat[0] = zscore(age, stats.get("age", {}).get("mean", 54.0),
                         stats.get("age", {}).get("std",  26.0))
    feat[1] = float(gender_idx)
    feat[2] = float(ethnicity_idx)
    feat[3] = float(admission_idx)
    feat[4] = float(insurance_idx)
    feat[5] = float(careunit_idx)
    feat[6] = zscore(prior_icu,
                     stats.get("prior_icu_count", {}).get("mean", 0.54),
                     stats.get("prior_icu_count", {}).get("std",  1.64))

    # ----- Lab values (raw; None = missing) --------------------------------
    lab_vals = [creatinine, bun, lactate, hemoglobin, wbc, sodium, potassium, platelets]
    lab_keys = ["creatinine", "bun", "lactate", "hemoglobin", "wbc",
                "sodium", "potassium", "platelets"]

    mean_raw = []
    missingness = []
    for i, (val, key) in enumerate(zip(lab_vals, lab_keys)):
        m_key  = f"mean_{key}"
        mn_key = f"min_{key}"
        mx_key = f"max_{key}"
        m_stats = stats.get(m_key, {"mean": 0.0, "std": 1.0})
        mn_stats= stats.get(mn_key,{"mean": 0.0, "std": 1.0})
        mx_stats= stats.get(mx_key,{"mean": 0.0, "std": 1.0})

        if val is None:
            # Use mean as imputation; flag missing
            raw_mean = m_stats["mean"]
            miss = 1.0
        else:
            raw_mean = float(val)
            miss = 0.0

        # mean
        feat[7  + i] = zscore(raw_mean, m_stats["mean"],  m_stats["std"])
        # min (same single observation)
        feat[15 + i] = zscore(raw_mean, mn_stats["mean"], mn_stats["std"])
        # max
        feat[23 + i] = zscore(raw_mean, mx_stats["mean"], mx_stats["std"])

        mean_raw.append(raw_mean if val is not None else None)
        missingness.append(miss)

    # ----- [31-43] Trends — all zeros (no trajectory) ----------------------
    # feat[31:44] already zero

    # ----- [44-46] Composite clinical scores --------------------------------
    cr_raw  = creatinine if creatinine else stats.get("mean_creatinine", {}).get("mean", 1.0)
    bun_raw = bun        if bun        else stats.get("mean_bun",        {}).get("mean", 34.0)
    lac_raw = lactate    if lactate    else stats.get("mean_lactate",     {}).get("mean", 132.0)
    wbc_raw = wbc        if wbc        else stats.get("mean_wbc",         {}).get("mean", 48.0)

    shock_idx   = hr / sbp if sbp > 0 else 0.0
    bun_cr      = bun_raw / cr_raw if cr_raw > 0 else 0.0
    sepsis_prx  = lac_raw * wbc_raw

    si_stats  = stats.get("shock_index",  {"mean": 0.758, "std": 0.213})
    bc_stats  = stats.get("bun_cr_ratio", {"mean": 18.93, "std": 14.38})
    sp_stats  = stats.get("sepsis_proxy", {"mean": 54634.0, "std": 2462788.0})

    feat[44] = zscore(shock_idx,  si_stats["mean"], si_stats["std"])
    feat[45] = zscore(bun_cr,     bc_stats["mean"], bc_stats["std"])
    feat[46] = zscore(sepsis_prx, sp_stats["mean"], sp_stats["std"])

    # ----- [47-54] Missingness indicators -----------------------------------
    for i, m in enumerate(missingness):
        feat[47 + i] = m

    # ----- [55-154] CCS multi-hot (100 dims, in CCS_COL_ORDER) -------------
    for code in selected_ccs_codes:
        block_idx = CCS_CODE_TO_BLOCK_IDX.get(code)
        if block_idx is not None:
            feat[55 + block_idx] = 1.0

    # ----- [155] Mechanical ventilation flag --------------------------------
    feat[155] = float(mech_vent_flag)

    return torch.tensor(feat, dtype=torch.float32).unsqueeze(0)  # (1, 156)

# ===========================================================================
# Inference + attention extraction + Integrated Gradients
# ===========================================================================

def run_inference(model, vitals_t, tabular_t, cal):
    """
    Forward pass with attention hook capture.
    Returns dict with prediction results and attention weights.

    Temporal attention is captured by hooking lstm_encoder.attention.score_layer
    output, then applying softmax to get the 24 weights.

    Cross-modal attention is a 1x1 scalar (seq_len=1 for both Q and K/V),
    so instead we return None and use IG magnitudes for group attribution.
    """
    temporal_scores_buf = []

    def _temporal_hook(module, inp, out):
        # out: (B, T, 1) — raw scores before softmax
        temporal_scores_buf.append(out.detach().cpu())

    hook = model.lstm_encoder.attention.score_layer.register_forward_hook(
        _temporal_hook
    )

    model.eval()
    with torch.no_grad():
        los_logits, mortality_logit = model(vitals_t, tabular_t)

    hook.remove()

    # --- Temporal attention weights (shape 24,) ---
    if temporal_scores_buf:
        scores = temporal_scores_buf[0]          # (B, 24, 1)
        weights = F.softmax(scores, dim=1)        # (B, 24, 1)
        temporal_attn = weights.squeeze().numpy() # (24,)
    else:
        temporal_attn = np.ones(24) / 24.0

    # --- LOS prediction with Platt temperature scaling ---
    b1_temp  = cal.get("b1_temperature",   B1_TEMP)
    m_temp   = cal.get("mort_temperature", MORT_TEMP)
    b2_thr   = cal.get("band2_threshold",  BAND2_THRESH)
    m_thr    = cal.get("mortality_threshold", MORT_THRESH)

    los_scaled = los_logits / b1_temp                          # (1, 2)
    p = torch.sigmoid(los_scaled).squeeze()                    # (2,)
    p_ge1, p_ge2 = p[0].item(), p[1].item()

    p_band0 = 1.0 - p_ge1
    p_band1 = p_ge1 - p_ge2
    p_band2 = p_ge2

    # Clamp band probs to [0,1]
    p_band0 = max(0.0, p_band0)
    p_band1 = max(0.0, p_band1)
    p_band2 = max(0.0, p_band2)

    los_pred = int(np.argmax([p_band0, p_band1, p_band2]))
    if p_band2 > b2_thr:
        los_pred = 2

    # --- Mortality prediction ---
    mort_logit  = mortality_logit.squeeze() / m_temp
    p_mort      = torch.sigmoid(mort_logit).item()
    mort_pred   = int(p_mort > m_thr)

    return {
        "los_pred":      los_pred,
        "p_band0":       p_band0,
        "p_band1":       p_band1,
        "p_band2":       p_band2,
        "p_mort":        p_mort,
        "mort_pred":     mort_pred,
        "temporal_attn": temporal_attn,
        "los_logits_raw": los_logits,
        "mort_logit_raw": mortality_logit,
    }


def compute_integrated_gradients(model, vitals_t, tabular_t, target="los_band2", steps=50):
    """
    Integrated Gradients for the 156-dim tabular input.
    Baseline = zeros tensor.
    target options: 'los_band2', 'los_band0', 'mortality'
    Returns ig_attributions: np.ndarray of shape (156,)
    """
    model.eval()
    baseline = torch.zeros_like(tabular_t)
    alphas = torch.linspace(0, 1, steps + 1)
    integrated_grads = torch.zeros_like(tabular_t)

    for alpha in alphas:
        interp = (baseline + alpha * (tabular_t - baseline)).detach().requires_grad_(True)
        los_logits, mort_logit = model(vitals_t, interp)

        if target == "los_band2":
            score = los_logits[:, 1].sum()
        elif target == "los_band0":
            score = -los_logits[:, 0].sum()
        else:  # mortality
            score = mort_logit.sum()

        score.backward()
        if interp.grad is not None:
            integrated_grads = integrated_grads + interp.grad.detach()

    avg_grads = integrated_grads / (steps + 1)
    ig = ((tabular_t - baseline) * avg_grads).squeeze().detach().numpy()  # (156,)
    return ig

# ===========================================================================
# Streamlit UI
# ===========================================================================

def _band_label(band):
    return {0: "Short Stay  < 24 h", 1: "Moderate Stay  24–72 h", 2: "Prolonged Stay  > 72 h"}[band]

def _band_color(band):
    return {0: "#22c55e", 1: "#f59e0b", 2: "#ef4444"}[band]


def make_los_bar(p0, p1, p2):
    fig = go.Figure()
    for p, lbl, col in [
        (p0, "Band 0 < 24 h", "#22c55e"),
        (p1, "Band 1 24–72 h", "#f59e0b"),
        (p2, "Band 2 > 72 h", "#ef4444"),
    ]:
        fig.add_trace(go.Bar(
            x=[p], y=["LOS"], orientation="h",
            name=lbl, marker_color=col,
            text=f"{p:.1%}", textposition="inside",
            hovertemplate=f"{lbl}: {p:.1%}<extra></extra>",
        ))
    fig.update_layout(
        barmode="stack", height=90, margin=dict(l=0, r=0, t=0, b=0),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        showlegend=True, legend=dict(orientation="h", y=-0.5),
        xaxis=dict(range=[0, 1], showticklabels=False, showgrid=False),
        yaxis=dict(showticklabels=False),
        font=dict(color="#e2e8f0"),
    )
    return fig


def make_ig_chart(ig_vals, feature_names, top_n=15):
    abs_ig = np.abs(ig_vals)
    top_idx = np.argsort(abs_ig)[::-1][:top_n]
    top_idx = top_idx[np.argsort(ig_vals[top_idx])]  # sort by value for readability

    vals  = ig_vals[top_idx]
    names = [feature_names[i] if i < len(feature_names) else f"feat_{i}" for i in top_idx]
    colors = ["#ef4444" if v >= 0 else "#3b82f6" for v in vals]

    fig = go.Figure(go.Bar(
        x=vals, y=names, orientation="h",
        marker_color=colors,
        hovertemplate="%{y}: %{x:.4f}<extra></extra>",
    ))
    fig.update_layout(
        height=420, margin=dict(l=0, r=10, t=10, b=10),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(17,24,39,0.6)",
        xaxis=dict(title="Attribution", color="#94a3b8", gridcolor="#1e293b"),
        yaxis=dict(color="#94a3b8"),
        font=dict(color="#e2e8f0", size=12),
    )
    return fig


def make_temporal_chart(attn_weights):
    hours = list(range(1, 25))
    top3  = np.argsort(attn_weights)[::-1][:3]
    colors = ["#f59e0b" if i in top3 else "#3b82f6" for i in range(24)]

    fig = go.Figure(go.Bar(
        x=hours, y=attn_weights.tolist(),
        marker_color=colors,
        hovertemplate="Hour %{x}: %{y:.4f}<extra></extra>",
    ))
    fig.update_layout(
        title="Which ICU hours most influenced the prediction",
        height=340, margin=dict(l=10, r=10, t=40, b=10),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(17,24,39,0.6)",
        xaxis=dict(title="Hour of ICU stay", color="#94a3b8", gridcolor="#1e293b",
                   tickvals=list(range(1, 25))),
        yaxis=dict(title="Attention weight", color="#94a3b8", gridcolor="#1e293b"),
        font=dict(color="#e2e8f0"),
    )
    return fig


def make_group_chart(ig_vals):
    groups = {
        "Labs (mean/min/max)":   list(range(7, 31)),
        "Trends":                list(range(31, 44)),
        "Composite Scores":      list(range(44, 47)),
        "Missingness":           list(range(47, 55)),
        "Diagnoses (CCS)":       list(range(55, 155)),
        "Mech. Ventilation":     [155],
        "Static":                list(range(0, 7)),
    }
    group_names = list(groups.keys())
    group_vals  = [float(np.abs(ig_vals[idxs]).mean()) for idxs in groups.values()]

    fig = go.Figure(go.Bar(
        x=group_vals, y=group_names, orientation="h",
        marker_color="#6366f1",
        hovertemplate="%{y}: %{x:.4f}<extra></extra>",
    ))
    fig.update_layout(
        title="Mean |attribution| per feature group",
        height=340, margin=dict(l=0, r=10, t=40, b=10),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(17,24,39,0.6)",
        xaxis=dict(title="Mean |IG attribution|", color="#94a3b8", gridcolor="#1e293b"),
        yaxis=dict(color="#94a3b8"),
        font=dict(color="#e2e8f0"),
    )
    return fig


# ===========================================================================
# Main app
# ===========================================================================

def main():
    st.set_page_config(
        page_title="ICU Outcome Predictor",
        page_icon="🏥",
        layout="wide",
        initial_sidebar_state="collapsed",
    )

    # --- Global CSS ---
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    .main { background: #0f172a; }
    .block-container { padding-top: 1.5rem; }
    .badge {
        display: inline-block; padding: 4px 14px; border-radius: 20px;
        font-weight: 600; font-size: 0.85rem; letter-spacing: 0.04em;
    }
    .badge-red    { background: #450a0a; color: #ef4444; border: 1px solid #ef4444; }
    .badge-yellow { background: #451a03; color: #f59e0b; border: 1px solid #f59e0b; }
    .badge-green  { background: #052e16; color: #22c55e; border: 1px solid #22c55e; }
    .result-card {
        background: #1e293b; border: 1px solid #334155;
        border-radius: 12px; padding: 1.2rem 1.5rem; margin-bottom: 1rem;
    }
    .metric-big { font-size: 2rem; font-weight: 700; }
    .section-label {
        font-size: 0.7rem; font-weight: 600; letter-spacing: 0.1em;
        color: #64748b; text-transform: uppercase; margin-bottom: 0.5rem;
    }
    div[data-testid="stNumberInput"] label { color: #94a3b8; font-size: 0.85rem; }
    div[data-testid="stSelectbox"] label  { color: #94a3b8; font-size: 0.85rem; }
    div[data-testid="stMultiSelect"] label{ color: #94a3b8; font-size: 0.85rem; }
    div[data-testid="stCheckbox"] label   { color: #94a3b8; font-size: 0.85rem; }
    </style>
    """, unsafe_allow_html=True)

    # --- Header ---
    st.markdown("""
    <div style='border-bottom:1px solid #1e293b; padding-bottom:1rem; margin-bottom:1.5rem;'>
      <h1 style='color:#64748b; margin:0; font-size:1.8rem; font-weight:700;'>
        ICU Outcome Predictor
      </h1>
      <p style='color:#64748b; margin:0.3rem 0 0; font-size:0.9rem;'>
        Early risk stratification from first 24 hours of ICU admission —
        MIMIC-III multimodal deep learning model
      </p>
    </div>
    """, unsafe_allow_html=True)

    # Load artifacts
    with st.spinner("Loading model and dataset artifacts…"):
        try:
            stats, feature_names, label_encoders = load_dataset_artifacts()
        except Exception as e:
            st.error(f"Failed to load dataset pickle: {e}")
            return
        try:
            model, cal = load_model()
        except Exception as e:
            st.error(f"Failed to load model: {e}")
            return

    # Extract label encoder classes (fallback to hardcoded)
    def _le_classes(key, fallback):
        le = label_encoders.get(key)
        return list(le.classes_) if le and hasattr(le, "classes_") else fallback

    gender_cls    = _le_classes("gender",         GENDER_CLASSES)
    ethnicity_cls_full = _le_classes("ethnicity", list(ETHNICITY_MAP.values()))
    admission_cls = _le_classes("admission_type", ADMISSION_CLASSES)
    insurance_cls = _le_classes("insurance",      INSURANCE_CLASSES)
    careunit_cls  = _le_classes("first_careunit", CAREUNIT_CLASSES)

    # Ethnicity UI choices -> full MIMIC string -> index
    eth_ui_choices = list(ETHNICITY_MAP.keys())

    # -----------------------------------------------------------------------
    # Layout: 40 % left (form) | 60 % right (results)
    # -----------------------------------------------------------------------
    left, right = st.columns([4, 6], gap="large")

    # ===== LEFT COLUMN — Input form ========================================
    with left:
        st.markdown('<div class="section-label">Section 1 — Vital Signs</div>',
                    unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        hr  = c1.number_input("Heart Rate (bpm)",   min_value=20.0, max_value=250.0,
                               value=85.0, step=1.0, help="Normal 60–100")
        rr  = c2.number_input("Respiratory Rate (br/min)", min_value=4.0,  max_value=60.0,
                               value=18.0, step=1.0, help="Normal 12–20")
        sbp = c1.number_input("Systolic BP (mmHg)", min_value=40.0, max_value=260.0,
                               value=120.0, step=1.0, help="Normal 90–140")
        dbp = c2.number_input("Diastolic BP (mmHg)",min_value=20.0, max_value=180.0,
                               value=75.0, step=1.0, help="Normal 60–90")
        spo2= c1.number_input("SpO2 (%)",           min_value=50.0, max_value=100.0,
                               value=97.0, step=0.5, help="Normal 94–100")

        st.markdown('<div class="section-label" style="margin-top:1rem;">Section 2 — Laboratory Values</div>',
                    unsafe_allow_html=True)
        st.caption("Leave blank / -1 to mark as missing")

        def _lab(label, default, lo, hi, help_txt):
            raw = st.number_input(label, min_value=-1.0, max_value=float(hi),
                                  value=float(default), step=0.1, help=help_txt)
            return None if raw < 0 else raw

        c3, c4 = st.columns(2)
        with c3:
            creatinine = _lab("Creatinine (mg/dL)",   1.0, -1, 50,  "Normal 0.6–1.2")
            lactate    = _lab("Lactate (mmol/L)",      1.2, -1, 30,  "Normal 0.5–2.0")
            wbc        = _lab("WBC (10³/µL)",         10.0, -1, 500, "Normal 4–11")
            potassium  = _lab("Potassium (mEq/L)",     4.0, -1, 20,  "Normal 3.5–5.0")
        with c4:
            bun        = _lab("BUN (mg/dL)",          20.0, -1, 300, "Normal 7–25")
            hemoglobin = _lab("Haemoglobin (g/dL)",   12.0, -1, 25,  "Normal 12–17")
            sodium     = _lab("Sodium (mEq/L)",       138.0, -1, 180,"Normal 135–145")
            platelets  = _lab("Platelet Count (10³/µL)",200.0,-1,1500,"Normal 150–400")

        st.markdown('<div class="section-label" style="margin-top:1rem;">Section 3 — Static Features</div>',
                    unsafe_allow_html=True)
        c5, c6 = st.columns(2)
        age       = c5.number_input("Age (years)", min_value=0, max_value=120, value=65)
        prior_icu = c6.number_input("Prior ICU Admissions", min_value=0, max_value=50, value=0)

        c7, c8 = st.columns(2)
        gender_ui    = c7.selectbox("Gender",     ["Male", "Female"])
        gender_idx   = gender_cls.index("M") if gender_ui == "Male" else gender_cls.index("F")

        eth_choice   = c8.selectbox("Ethnicity",  eth_ui_choices)
        eth_full     = ETHNICITY_MAP[eth_choice]
        # find closest match in full list
        if eth_full in ethnicity_cls_full:
            ethnicity_idx = ethnicity_cls_full.index(eth_full)
        else:
            # fallback: find any match containing key substring
            matches = [i for i, c in enumerate(ethnicity_cls_full) if eth_choice.upper() in c]
            ethnicity_idx = matches[0] if matches else 0

        c9, c10 = st.columns(2)
        adm_choice   = c9.selectbox("Admission Type", [x for x in admission_cls if x != "NEWBORN"])
        admission_idx= admission_cls.index(adm_choice)
        ins_choice   = c10.selectbox("Insurance", insurance_cls)
        insurance_idx= insurance_cls.index(ins_choice)

        c11, c12 = st.columns(2)
        cu_choice    = c11.selectbox("First Care Unit", careunit_cls)
        careunit_idx = careunit_cls.index(cu_choice)

        st.markdown('<div class="section-label" style="margin-top:1rem;">Section 4 — Diagnoses (CCS)</div>',
                    unsafe_allow_html=True)
        ccs_options = {v: k for k, v in CCS_TOP20.items()}  # label -> code
        selected_labels = st.multiselect(
            "Select primary diagnoses",
            options=list(ccs_options.keys()),
            help="Select all that apply",
        )
        selected_ccs = [ccs_options[lbl] for lbl in selected_labels]

        st.markdown('<div class="section-label" style="margin-top:1rem;">Section 5 — Procedures</div>',
                    unsafe_allow_html=True)
        mech_vent = st.checkbox("Mechanical ventilation initiated within 24 h")

        st.markdown("<br>", unsafe_allow_html=True)
        predict_btn = st.button("Predict Outcome", type="primary", use_container_width=True)

    # ===== RIGHT COLUMN — Results ==========================================
    with right:
        if not predict_btn:
            st.markdown("""
            <div style='height:300px; display:flex; align-items:center; justify-content:center;
                        color:#334155; font-size:1.1rem; border:1px dashed #1e293b;
                        border-radius:12px;'>
              Complete the form and click <strong style="color:#6366f1;">&nbsp;Predict Outcome</strong>
            </div>
            """, unsafe_allow_html=True)
            return

        with st.spinner("Running inference…"):
            vitals_t  = build_vitals_tensor(hr, rr, sbp, dbp, spo2, stats)
            tabular_t = build_tabular_vector(
                hr=hr, rr=rr, sbp=sbp, dbp=dbp, spo2=spo2,
                creatinine=creatinine, bun=bun, lactate=lactate,
                hemoglobin=hemoglobin, wbc=wbc, sodium=sodium,
                potassium=potassium, platelets=platelets,
                age=age, gender_idx=gender_idx, ethnicity_idx=ethnicity_idx,
                admission_idx=admission_idx, insurance_idx=insurance_idx,
                careunit_idx=careunit_idx, prior_icu=prior_icu,
                selected_ccs_codes=selected_ccs,
                mech_vent_flag=int(mech_vent),
                stats=stats,
            )
            result = run_inference(model, vitals_t, tabular_t, cal)

        # --- PANEL 1: Predictions ---
        los_band  = result["los_pred"]
        los_label = _band_label(los_band)
        los_color = _band_color(los_band)
        badge_cls = {0: "badge-green", 1: "badge-yellow", 2: "badge-red"}[los_band]

        mort_pct  = result["p_mort"] * 100
        mort_high = result["mort_pred"] == 1
        mort_badge_cls = "badge-red" if mort_high else "badge-green"
        mort_badge_txt = "HIGH RISK" if mort_high else "LOW RISK"
        mort_icon      = "⚠" if mort_high else "✓"

        st.markdown("### Prediction Results")

        mcol1, mcol2 = st.columns(2)
        with mcol1:
            st.markdown(f"""
            <div class="result-card">
              <div class="section-label">Length of Stay</div>
              <div class="metric-big" style="color:{los_color};">{los_label}</div>
              <div style="margin-top:0.5rem;">
                <span class="badge {badge_cls}">{los_label.split()[0]} stay</span>
              </div>
              <div style="margin-top:0.8rem; font-size:0.8rem; color:#64748b;">
                P(Band 0): {result["p_band0"]:.1%} &nbsp;|&nbsp;
                P(Band 1): {result["p_band1"]:.1%} &nbsp;|&nbsp;
                P(Band 2): {result["p_band2"]:.1%}
              </div>
            </div>
            """, unsafe_allow_html=True)

        with mcol2:
            st.markdown(f"""
            <div class="result-card">
              <div class="section-label">In-Hospital Mortality</div>
              <div class="metric-big" style="color:{'#ef4444' if mort_high else '#22c55e'};">
                {mort_pct:.1f}%
              </div>
              <div style="margin-top:0.5rem;">
                <span class="badge {mort_badge_cls}">{mort_icon} {mort_badge_txt}</span>
              </div>
              <div style="margin-top:0.8rem; font-size:0.8rem; color:#64748b;">
                Threshold: {cal.get('mortality_threshold', MORT_THRESH):.1%}
              </div>
            </div>
            """, unsafe_allow_html=True)

        st.plotly_chart(
            make_los_bar(result["p_band0"], result["p_band1"], result["p_band2"]),
            use_container_width=True, config={"displayModeBar": False},
        )

        # --- PANEL 2: Explainability ---
        st.markdown("### Explainability")
        tab1, tab2, tab3 = st.tabs([
            "Feature Attribution (IG)",
            "Vital Sign Attention (24 h)",
            "Feature Group Importance",
        ])

        with tab1:
            # IG target matches predicted LOS band
            ig_target = {0: "los_band0", 1: "los_band2", 2: "los_band2"}.get(los_band, "los_band2")
            with st.spinner("Computing Integrated Gradients (~1 s)…"):
                ig = compute_integrated_gradients(model, vitals_t, tabular_t, target=ig_target)

            st.caption(
                "Features pushing **toward** this prediction (red) vs "
                "features pushing **away** (blue). Top 15 by absolute attribution."
            )
            st.plotly_chart(
                make_ig_chart(ig, feature_names, top_n=15),
                use_container_width=True,
                config={"displayModeBar": False},
            )

        with tab2:
            st.caption(
                "Attention weights learned over the 24-hour ICU stay. "
                "Orange bars = top-3 attended hours."
            )
            st.plotly_chart(
                make_temporal_chart(result["temporal_attn"]),
                use_container_width=True,
                config={"displayModeBar": False},
            )

        with tab3:
            st.caption("Mean absolute IG attribution aggregated by feature group.")
            # Re-use IG from tab1 if computed, else compute quickly
            if 'ig' not in dir():
                ig = compute_integrated_gradients(model, vitals_t, tabular_t,
                                                  target=ig_target, steps=30)
            st.plotly_chart(
                make_group_chart(ig),
                use_container_width=True,
                config={"displayModeBar": False},
            )


if __name__ == "__main__":
    main()
