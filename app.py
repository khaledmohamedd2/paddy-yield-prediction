import re
import pickle

import joblib
import numpy as np
import pandas as pd
import streamlit as st

# ----------------------------------------------------------------------------
# Page config
# ----------------------------------------------------------------------------
st.set_page_config(
    page_title="Paddy Yield Prediction",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded",
)

MODEL_PATH = "random_forest_model.pkl"
PREPROCESSOR_PATH = "preprocessor.pkl"
DATA_PATH = "paddydataset.csv"
TARGET_NORM = "paddyyieldinkg"


def norm(name: str) -> str:
    """Normalise a column name so matching ignores spaces/case/symbols."""
    return re.sub(r"[^a-z0-9]", "", str(name).lower())


# ----------------------------------------------------------------------------
# Styling
# ----------------------------------------------------------------------------
st.markdown(
    """
    <style>
    .stApp { background: #f4f8f1; }
    .block-container { padding-top: 1.5rem; padding-bottom: 3rem; max-width: 1200px; }

    .hero {
        background: linear-gradient(120deg, #1b5e20 0%, #2e7d32 50%, #7cb342 100%);
        padding: 2.2rem 2.5rem;
        border-radius: 18px;
        color: #ffffff;
        box-shadow: 0 8px 24px rgba(27, 94, 32, 0.25);
        margin-bottom: 1.5rem;
    }
    .hero h1 { color: #ffffff; margin: 0; font-size: 2.4rem; font-weight: 800; }
    .hero p  { color: #e8f5e9; margin: 0.4rem 0 0 0; font-size: 1.05rem; }

    .section-header {
        display: flex; align-items: center; gap: 0.75rem;
        background: #ffffff;
        border-left: 6px solid #2e7d32;
        border-radius: 10px;
        padding: 0.8rem 1.1rem;
        margin: 1.6rem 0 0.8rem 0;
        box-shadow: 0 2px 8px rgba(0,0,0,0.06);
    }
    .section-header .icon { font-size: 1.6rem; }
    .section-header .title { font-size: 1.2rem; font-weight: 700; color: #1b5e20; line-height: 1.2; }
    .section-header .desc  { font-size: 0.85rem; color: #6b7a6b; }

    div[data-testid="stForm"] {
        background: transparent; border: none; padding: 0;
    }

    div[data-testid="stFormSubmitButton"] button {
        width: 100%;
        background: linear-gradient(90deg, #2e7d32, #66bb6a);
        color: #ffffff;
        font-size: 1.25rem;
        font-weight: 700;
        padding: 0.85rem 1rem;
        border: none;
        border-radius: 12px;
        box-shadow: 0 6px 16px rgba(46, 125, 50, 0.35);
        transition: all 0.2s ease;
    }
    div[data-testid="stFormSubmitButton"] button:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 22px rgba(46, 125, 50, 0.45);
        color: #ffffff;
        border: none;
    }

    .result-card {
        background: linear-gradient(135deg, #ffffff 0%, #e8f5e9 100%);
        border: 2px solid #66bb6a;
        border-radius: 20px;
        padding: 2rem 1.5rem;
        text-align: center;
        margin-top: 1.5rem;
        box-shadow: 0 10px 28px rgba(46, 125, 50, 0.2);
    }
    .result-card .label {
        font-size: 1rem; letter-spacing: 0.12em; text-transform: uppercase;
        color: #558b2f; font-weight: 600;
    }
    .result-card .value {
        font-size: 3.8rem; font-weight: 800; color: #1b5e20; margin: 0.4rem 0;
    }
    .result-card .note { font-size: 0.9rem; color: #6b7a6b; }

    section[data-testid="stSidebar"] { background: #e8f5e9; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ----------------------------------------------------------------------------
# Loaders (cached) - model / preprocessor / data are NOT modified
# ----------------------------------------------------------------------------
def _load_pickle(path):
    try:
        return joblib.load(path)
    except Exception:
        with open(path, "rb") as f:
            return pickle.load(f)


@st.cache_resource
def load_artifacts():
    model = _load_pickle(MODEL_PATH)
    preprocessor = _load_pickle(PREPROCESSOR_PATH)
    return model, preprocessor


@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)


try:
    model, preprocessor = load_artifacts()
    data = load_data()
except FileNotFoundError as e:
    st.error(
        f"Required file not found: {e.filename}. Make sure `{MODEL_PATH}`, "
        f"`{PREPROCESSOR_PATH}` and `{DATA_PATH}` are in the same folder as app.py."
    )
    st.stop()
except Exception as e:
    st.error(f"Could not load the model files: {e}")
    st.stop()

# Feature columns = every CSV column except the target (original names kept)
feature_cols = [c for c in data.columns if norm(c) != TARGET_NORM]
lookup = {norm(c): c for c in feature_cols}


# ----------------------------------------------------------------------------
# Section layout (44 inputs)
# ----------------------------------------------------------------------------
SECTIONS = [
    {
        "icon": "🏡",
        "title": "1. Farm Information",
        "desc": "Field size and location of the farm.",
        "cols": 2,
        "fields": ["Hectares", "Agriblock"],
    },
    {
        "icon": "🌱",
        "title": "2. Soil & Seed Information",
        "desc": "Soil type, rice variety, seed rate, nursery and land preparation.",
        "cols": 3,
        "fields": [
            "Variety", "Soil Types", "Seedrate(in Kg)", "Nursery",
            "Nursery area (Cents)", "LP_Mainfield(in Tonnes)",
            "LP_nurseryarea(in Tonnes)",
        ],
    },
    {
        "icon": "🧪",
        "title": "3. Fertilizer & Pest Management",
        "desc": "Fertilizer, herbicide, micronutrient and pesticide applications.",
        "cols": 3,
        "fields": [
            "DAP_20days", "Weed28D_thiobencarb", "Urea_40Days",
            "Potassh_50Days", "Micronutrients_70Days", "Pest_60Day(in ml)",
        ],
    },
    {
        "icon": "🌧️",
        "title": "4. Rainfall & Irrigation",
        "desc": "Rainfall and artificial irrigation (AI) across the growth stages.",
        "cols": 4,
        "fields": [
            "30DRain( in mm)", "30DAI(in mm)",
            "30_50DRain( in mm)", "30_50DAI(in mm)",
            "51_70DRain(in mm)", "51_70AI(in mm)",
            "71_105DRain(in mm)", "71_105DAI(in mm)",
        ],
    },
    {
        "icon": "🌡️",
        "title": "5. Temperature",
        "desc": "Minimum and maximum temperature in each 30-day window.",
        "cols": 4,
        "fields": [
            "Min temp_D1_D30", "Max temp_D1_D30",
            "Min temp_D31_D60", "Max temp_D31_D60",
            "Min temp_D61_D90", "Max temp_D61_D90",
            "Min temp_D91_D120", "Max temp_D91_D120",
        ],
    },
    {
        "icon": "💨",
        "title": "6. Wind & Humidity",
        "desc": "Wind speed, wind direction and relative humidity per 30-day window.",
        "cols": 4,
        "fields": [
            "Inst Wind Speed_D1_D30(in Knots)",
            "Inst Wind Speed_D31_D60(in Knots)",
            "Inst Wind Speed_D61_D90(in Knots)",
            "Inst Wind Speed_D91_D120(in Knots)",
            "Wind Direction_D1_D30", "Wind Direction_D31_D60",
            "Wind Direction_D61_D90", "Wind Direction_D91_D120",
            "Relative Humidity_D1_D30", "Relative Humidity_D31_D60",
            "Relative Humidity_D61_D90", "Relative Humidity_D91_D120",
        ],
    },
    {
        "icon": "📦",
        "title": "7. Other Information",
        "desc": "Additional harvest-related information.",
        "cols": 2,
        "fields": ["Trash(in bundles)"],
    },
]

# Tooltips for selected fields
HELP = {
    "hectares": "Total cultivated area of the field, in hectares.",
    "agriblock": "Agricultural block (administrative zone) where the farm is located.",
    "variety": "Rice variety grown in the field.",
    "soiltypes": "Soil type of the field.",
    "seedrateinkg": "Quantity of seed used, in kilograms.",
    "nursery": "Type of nursery used to raise the seedlings.",
    "nurseryareacents": "Nursery area in cents (1 cent = 40.5 m²).",
    "lpmainfieldintonnes": "Land preparation input for the main field, in tonnes.",
    "lpnurseryareaintonnes": "Land preparation input for the nursery area, in tonnes.",
    "dap20days": "DAP fertilizer applied around day 20.",
    "weed28dthiobencarb": "Thiobencarb herbicide applied around day 28.",
    "urea40days": "Urea fertilizer applied around day 40.",
    "potassh50days": "Potash fertilizer applied around day 50.",
    "micronutrients70days": "Micronutrients applied around day 70.",
    "pest60dayinml": "Pesticide applied around day 60, in millilitres.",
    "trashinbundles": "Trash (crop residue) measured in bundles.",
}


def help_for(col: str) -> str:
    key = norm(col)
    if key in HELP:
        return HELP[key]
    low = col.lower()
    if "rain" in low:
        return "Rainfall during this growth stage, in mm."
    if "ai(" in low.replace(" ", "") or "dai" in low:
        return "Artificial irrigation during this growth stage, in mm."
    if "min temp" in low:
        return "Minimum temperature during this period (°C)."
    if "max temp" in low:
        return "Maximum temperature during this period (°C)."
    if "wind speed" in low:
        return "Instantaneous wind speed during this period, in knots."
    if "wind direction" in low:
        return "Dominant wind direction during this period."
    if "humidity" in low:
        return "Relative humidity during this period (%)."
    return None


def nice_label(col: str) -> str:
    return " ".join(str(col).replace("_", " ").split())


# ----------------------------------------------------------------------------
# Input widgets
# ----------------------------------------------------------------------------
def render_input(col_name: str, key_prefix: str = "in"):
    """Create the right widget for a feature and return its value."""
    series = data[col_name]
    label = nice_label(col_name)
    tip = help_for(col_name)
    key = f"{key_prefix}_{norm(col_name)}"

    if pd.api.types.is_numeric_dtype(series):
        s = series.dropna().astype(float)
        dmin, dmax = float(s.min()), float(s.max())
        median = float(s.median())
        lower = min(0.0, dmin)
        span = max(dmax - dmin, 1e-6)
        step = float(10 ** np.floor(np.log10(span / 100))) if span > 0 else 1.0
        step = max(step, 0.01)
        return st.number_input(
            label,
            min_value=float(lower),
            value=float(median),
            step=float(step),
            format="%.2f",
            help=(tip + " " if tip else "") + f"Typical range: {dmin:,.2f} – {dmax:,.2f}",
            key=key,
        )

    options = sorted(series.dropna().astype(str).unique().tolist())
    mode = series.dropna().astype(str).mode()
    default_idx = options.index(mode.iloc[0]) if len(mode) else 0
    return st.selectbox(label, options, index=default_idx, help=tip, key=key)


def section_header(icon, title, desc):
    st.markdown(
        f"""
        <div class="section-header">
            <div class="icon">{icon}</div>
            <div>
                <div class="title">{title}</div>
                <div class="desc">{desc}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ----------------------------------------------------------------------------
# Sidebar
# ----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("## 🌾 About")
    st.write(
        "This app predicts the **paddy (rice) yield in Kg** from farm, soil, "
        "fertilizer, rainfall, temperature, wind and humidity information "
        "using a trained Random Forest model."
    )
    st.markdown("### How to use")
    st.write(
        "1. Fill in each section.\n"
        "2. Click **Predict Yield** at the bottom.\n"
        "3. Read the predicted yield in the result card."
    )
    st.markdown("### Model")
    st.caption("Random Forest Regressor")
    st.caption(f"Inputs required: {len(feature_cols)}")

# ----------------------------------------------------------------------------
# Hero
# ----------------------------------------------------------------------------
st.markdown(
    """
    <div class="hero">
        <h1>🌾 Paddy Yield Prediction</h1>
        <p>Estimate the expected paddy yield (in Kg) from your farm conditions,
        inputs and weather data using a machine learning model.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ----------------------------------------------------------------------------
# Form with all inputs
# ----------------------------------------------------------------------------
values = {}
used = set()

with st.form("prediction_form"):
    for sec in SECTIONS:
        fields = []
        for f in sec["fields"]:
            actual = lookup.get(norm(f))
            if actual is not None and actual not in used:
                fields.append(actual)
                used.add(actual)

        # Put the last section's leftovers (any unmapped feature) in "Other"
        if sec["title"].startswith("7."):
            for c in feature_cols:
                if c not in used:
                    fields.append(c)
                    used.add(c)

        if not fields:
            continue

        section_header(sec["icon"], sec["title"], sec["desc"])
        n = sec["cols"]
        for start in range(0, len(fields), n):
            row = fields[start:start + n]
            cols = st.columns(n)
            for col_ui, name in zip(cols, row):
                with col_ui:
                    values[name] = render_input(name)

    st.markdown("<br>", unsafe_allow_html=True)
    submitted = st.form_submit_button("🌾 Predict Yield")

# ----------------------------------------------------------------------------
# Prediction (same pipeline: preprocessor.transform -> model.predict)
# ----------------------------------------------------------------------------
if submitted:
    try:
        input_df = pd.DataFrame([values], columns=feature_cols)

        # If the preprocessor remembers its training column names, match them
        expected = getattr(preprocessor, "feature_names_in_", None)
        if expected is not None:
            by_norm = {norm(c): c for c in input_df.columns}
            rename = {}
            for e in expected:
                src = by_norm.get(norm(e))
                if src is not None:
                    rename[src] = e
            input_df = input_df.rename(columns=rename)
            input_df = input_df[[e for e in expected if e in input_df.columns]]

        processed = preprocessor.transform(input_df)
        prediction = float(np.ravel(model.predict(processed))[0])

        st.markdown(
            f"""
            <div class="result-card">
                <div class="label">Predicted Paddy Yield</div>
                <div class="value">🌾 {prediction:,.0f} Kg</div>
                <div class="note">Estimated total yield based on the values you entered.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    except Exception as e:
        st.error(f"Prediction failed: {e}")

st.markdown("---")
st.caption("Paddy Yield Prediction · Built with Streamlit and scikit-learn")