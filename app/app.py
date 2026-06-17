
import streamlit as st
import pandas as pd
import joblib
import requests
import shap
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from security import (
    apply_security, validate_all_inputs,
    check_rate_limit, safe_error_message
)

# ─────────────────────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Crop Yield Predictor",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Activate firewall
apply_security()

# ─────────────────────────────────────────────────────────────
# Premium CSS
# ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* ── Google Font ─────────────────────────────────────── */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    /* ── Global ──────────────────────────────────────────── */
    html, body, [class*="st-"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    .main .block-container {
        max-width: 1200px;
        padding-top: 2rem;
        padding-bottom: 4rem;
    }

    /* ── Header ──────────────────────────────────────────── */
    .app-header {
        padding: 1.5rem 0 1rem 0;
        border-bottom: 1px solid #e8e8ed;
        margin-bottom: 2rem;
    }
    .app-header h1 {
        font-size: 1.75rem;
        font-weight: 700;
        color: #1d1d1f;
        margin: 0;
        letter-spacing: -0.02em;
    }
    .app-header p {
        font-size: 0.9rem;
        color: #86868b;
        margin: 0.25rem 0 0 0;
        font-weight: 400;
    }

    /* ── Section titles ──────────────────────────────────── */
    .section-title {
        font-size: 1.1rem;
        font-weight: 600;
        color: #1d1d1f;
        letter-spacing: -0.01em;
        margin: 2rem 0 1rem 0;
        padding-bottom: 0.5rem;
        border-bottom: 1px solid #e8e8ed;
    }

    /* ── Cards ────────────────────────────────────────────── */
    div[data-testid="stVerticalBlock"] > div[data-testid="stVerticalBlock"] {
        background: #ffffff;
        border: 1px solid #e8e8ed;
        border-radius: 12px;
        padding: 0.25rem;
    }

    /* ── Metric styling ──────────────────────────────────── */
    [data-testid="stMetric"] {
        background: #fafafa;
        border: 1px solid #f0f0f5;
        border-radius: 10px;
        padding: 1rem 1.25rem;
    }
    [data-testid="stMetricLabel"] {
        font-size: 0.75rem !important;
        font-weight: 500;
        color: #86868b !important;
        text-transform: uppercase;
        letter-spacing: 0.04em;
    }
    [data-testid="stMetricValue"] {
        font-size: 1.5rem !important;
        font-weight: 600;
        color: #1d1d1f !important;
    }

    /* ── Buttons ──────────────────────────────────────────── */
    .stButton > button {
        background: #1d1d1f;
        color: #ffffff;
        border: none;
        border-radius: 8px;
        padding: 0.6rem 1.5rem;
        font-size: 0.85rem;
        font-weight: 500;
        letter-spacing: 0.01em;
        transition: all 0.2s ease;
    }
    .stButton > button:hover {
        background: #424245;
        color: #ffffff;
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.08);
    }
    .stButton > button:active {
        transform: translateY(0);
    }

    /* ── Sidebar ──────────────────────────────────────────── */
    [data-testid="stSidebar"] {
        background: #fafafa;
        border-right: 1px solid #e8e8ed;
    }
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h2 {
        font-size: 0.85rem;
        font-weight: 600;
        color: #86868b;
        text-transform: uppercase;
        letter-spacing: 0.06em;
    }

    /* ── Dataframe ────────────────────────────────────────── */
    [data-testid="stDataFrame"] {
        border-radius: 8px;
        overflow: hidden;
    }

    /* ── Dividers ─────────────────────────────────────────── */
    hr {
        border: none;
        border-top: 1px solid #e8e8ed;
        margin: 1.5rem 0;
    }

    /* ── Success box ──────────────────────────────────────── */
    .stSuccess {
        background: #f5f9f5 !important;
        border-left: 3px solid #34c759 !important;
        border-radius: 8px;
        color: #1d1d1f !important;
    }

    /* ── Warning box ──────────────────────────────────────── */
    .stWarning {
        border-radius: 8px;
    }

    /* ── Caption ──────────────────────────────────────────── */
    .stCaption {
        color: #86868b !important;
        font-size: 0.8rem !important;
    }

    /* ── Footer ──────────────────────────────────────────── */
    .app-footer {
        text-align: center;
        padding: 3rem 0 1.5rem 0;
        border-top: 1px solid #e8e8ed;
        margin-top: 4rem;
    }
    .app-footer p {
        font-size: 0.8rem;
        color: #86868b;
        font-weight: 400;
        letter-spacing: 0.02em;
        margin: 0;
    }

    /* ── Hide Streamlit branding ──────────────────────────── */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header[data-testid="stHeader"] {
        background: transparent;
    }

    /* ── Force-load Material Symbols + hide ligature text ── */
    @import url('https://fonts.googleapis.com/css2?family=Material+Symbols+Rounded:opsz,wght,FILL,GRAD@20..48,100..700,0..1,-50..200&display=swap');

    /* Hide the collapsed sidebar expand button text */
    [data-testid="collapsedControl"] {
        visibility: hidden !important;
        width: 0 !important;
        height: 0 !important;
        overflow: hidden !important;
        position: absolute !important;
    }

    /* Style the collapse button inside sidebar */
    [data-testid="stSidebarCollapseButton"] button {
        font-size: 0 !important;
        color: transparent !important;
        overflow: hidden;
        width: 32px;
        height: 32px;
    }
    [data-testid="stSidebarCollapseButton"] button span {
        font-size: 0 !important;
        visibility: hidden !important;
    }
    [data-testid="stSidebarCollapseButton"] button svg {
        visibility: visible !important;
        width: 20px;
        height: 20px;
    }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# Paths & model
# ─────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "models" / "crop_yield_pipeline.pkl"


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_data
def fetch_nasa_weather(lat, lon, refresh_token, start="20200101", end="20241231"):
    url = "https://power.larc.nasa.gov/api/temporal/daily/point"
    params = {
        "parameters": "T2M_MAX,T2M_MIN,PRECTOTCORR,RH2M,ALLSKY_SFC_SW_DWN",
        "community": "AG",
        "longitude": lon,
        "latitude": lat,
        "start": start,
        "end": end,
        "format": "JSON"
    }
    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()
    data = r.json()["properties"]["parameter"]
    df = pd.DataFrame(data)
    df.index = pd.to_datetime(df.index, format="%Y%m%d")
    df.index.name = "date"
    return df


def build_input_frame(
    soil_ph, soil_moisture, avg_temperature, total_rainfall,
    fertilizer_amount, pesticide_usage, sunlight_hours,
    nitrogen_content, phosphorus_content, potassium_content,
    irrigation_frequency, crop_type, region, season
):
    return pd.DataFrame([{
        "soil_ph": soil_ph,
        "soil_moisture": soil_moisture,
        "avg_temperature": avg_temperature,
        "total_rainfall": total_rainfall,
        "fertilizer_amount": fertilizer_amount,
        "pesticide_usage": pesticide_usage,
        "sunlight_hours": sunlight_hours,
        "nitrogen_content": nitrogen_content,
        "phosphorus_content": phosphorus_content,
        "potassium_content": potassium_content,
        "irrigation_frequency": irrigation_frequency,
        "crop_type": crop_type,
        "region": region,
        "season": season
    }])


def get_preprocessor_and_estimator(model):
    if hasattr(model, "named_steps"):
        steps = list(model.named_steps.keys())
        if len(steps) >= 2:
            preprocessor = model.named_steps[steps[0]]
            estimator = model.named_steps[steps[-1]]
            return preprocessor, estimator
    return None, model


@st.cache_data
def load_training_data():
    """Load training CSV to use as SHAP background — gives the explainer
    a realistic distribution of feature values."""
    csv_path = PROJECT_ROOT / "data" / "raw" / "crop_yield_train.csv"
    if csv_path.exists():
        df = pd.read_csv(csv_path)
        drop = [c for c in ["id", "harvest_date", "field_id", "yield_tpha"] if c in df.columns]
        return df.drop(columns=drop, errors="ignore")
    return None


def prettify_feature_names(names):
    """Map sklearn transformer output names back to readable labels."""
    clean = []
    for name in names:
        label = name
        if "__" in label:
            label = label.split("__", 1)[1]

        for cat in ["crop_type", "region", "season"]:
            if label.startswith(cat + "_"):
                value = label[len(cat) + 1:]
                label = cat.replace("_", " ").title() + ": " + value
                break
        else:
            label = label.replace("_", " ").title()

        label = label.replace(" Ph", " pH").replace("Ndvi", "NDVI")
        clean.append(label)
    return clean


def build_shap_explanation(model, input_df):
    """Return (shap_values_row, feature_names) for a single input row."""
    preprocessor, estimator = get_preprocessor_and_estimator(model)

    if preprocessor is not None:
        transformed = preprocessor.transform(input_df)
        if hasattr(transformed, "toarray"):
            transformed = transformed.toarray()

        if hasattr(preprocessor, "get_feature_names_out"):
            raw_names = list(preprocessor.get_feature_names_out())
        else:
            raw_names = [f"feature_{i}" for i in range(transformed.shape[1])]

        feature_names = prettify_feature_names(raw_names)
        transformed_input = pd.DataFrame(transformed, columns=feature_names)

        train_df = load_training_data()
        if train_df is not None:
            bg_raw = train_df.sample(min(len(train_df), 100), random_state=42)
            bg_transformed = preprocessor.transform(bg_raw)
            if hasattr(bg_transformed, "toarray"):
                bg_transformed = bg_transformed.toarray()
            background = pd.DataFrame(bg_transformed, columns=feature_names)
        else:
            rng = np.random.default_rng(42)
            noise = rng.normal(0, 0.1, size=(20, transformed.shape[1]))
            background = pd.DataFrame(
                np.clip(transformed + noise, 0, None), columns=feature_names
            )

        explainer = shap.Explainer(estimator.predict, background)
        shap_values = explainer(transformed_input)
        return shap_values[0], feature_names

    train_df = load_training_data()
    if train_df is not None:
        background = train_df.sample(min(len(train_df), 100), random_state=42)
    else:
        background = input_df.copy()
    explainer = shap.Explainer(model.predict, background)
    shap_values = explainer(input_df)
    return shap_values[0], prettify_feature_names(list(input_df.columns))


# ─────────────────────────────────────────────────────────────
# Load model
# ─────────────────────────────────────────────────────────────
model = load_model()

if "weather_refresh" not in st.session_state:
    st.session_state.weather_refresh = 0

# ─────────────────────────────────────────────────────────────
# Header
# ─────────────────────────────────────────────────────────────
st.markdown("""
<div class="app-header">
    <h1>Crop Yield Predictor</h1>
    <p>ML-powered yield estimation using soil, climate, and agricultural inputs</p>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## Parameters")
    st.caption("Adjust the input features below.")

    st.markdown("---")
    st.markdown("**Soil**")
    soil_ph = st.slider("pH Level", 3.0, 10.0, 6.5)
    soil_moisture = st.slider("Moisture (%)", 0.0, 100.0, 45.0)

    st.markdown("---")
    st.markdown("**Climate**")
    avg_temperature = st.slider("Avg. Temperature (°C)", 0.0, 50.0, 28.0)
    total_rainfall = st.number_input("Total Rainfall (mm)", value=1200.0)
    sunlight_hours = st.slider("Sunlight (hrs/day)", 0.0, 24.0, 8.0)

    st.markdown("---")
    st.markdown("**Farm Management**")
    fertilizer_amount = st.number_input("Fertilizer", value=50.0)
    pesticide_usage = st.number_input("Pesticide Usage", value=2.0)
    nitrogen_content = st.number_input("Nitrogen", value=30.0)
    phosphorus_content = st.number_input("Phosphorus", value=20.0)
    potassium_content = st.number_input("Potassium", value=25.0)
    irrigation_frequency = st.slider("Irrigation Frequency (days)", 0, 30, 10)

    st.markdown("---")
    st.markdown("**Classification**")
    crop_type = st.selectbox("Crop", ["Rice", "Wheat", "Maize", "Soybean"])
    region = st.selectbox("Region", ["Selangor", "Johor", "Penang", "Perak"])
    season = st.selectbox("Season", ["Wet", "Dry", "Monsoon"])

    st.markdown("---")
    if st.button("Refresh Weather Data", use_container_width=True):
        st.session_state.weather_refresh += 1

# ─────────────────────────────────────────────────────────────
# Build input
# ─────────────────────────────────────────────────────────────
input_df = build_input_frame(
    soil_ph, soil_moisture, avg_temperature, total_rainfall,
    fertilizer_amount, pesticide_usage, sunlight_hours,
    nitrogen_content, phosphorus_content, potassium_content,
    irrigation_frequency, crop_type, region, season
)

# ─────────────────────────────────────────────────────────────
# Key Metrics
# ─────────────────────────────────────────────────────────────
st.markdown('<div class="section-title">Overview</div>', unsafe_allow_html=True)

m1, m2, m3, m4 = st.columns(4)
with m1:
    st.metric("Soil pH", f"{soil_ph:.1f}")
with m2:
    st.metric("Temperature", f"{avg_temperature:.1f}°C")
with m3:
    st.metric("Rainfall", f"{total_rainfall:,.0f} mm")
with m4:
    st.metric("Sunlight", f"{sunlight_hours:.1f} hrs")

# ─────────────────────────────────────────────────────────────
# Input Summary & Weather
# ─────────────────────────────────────────────────────────────
st.markdown('<div class="section-title">Data</div>', unsafe_allow_html=True)

col_left, col_right = st.columns([1, 1], gap="large")

with col_left:
    with st.container(border=True):
        st.markdown("**Input Summary**")
        st.dataframe(input_df, use_container_width=True, hide_index=True)

        if st.button("Run Prediction", use_container_width=True):
            check_rate_limit()
            try:
                validate_all_inputs(
                    soil_ph, soil_moisture, avg_temperature, total_rainfall,
                    fertilizer_amount, pesticide_usage, sunlight_hours,
                    nitrogen_content, phosphorus_content, potassium_content,
                    irrigation_frequency, crop_type, region, season
                )
                predicted_yield = model.predict(input_df)[0]
                st.session_state["predicted_yield"] = float(predicted_yield)
                st.metric("Predicted Yield", f"{predicted_yield:.3f} t/ha")
            except ValueError as ve:
                st.error(str(ve))
            except Exception as e:
                st.error(safe_error_message(e))

with col_right:
    with st.container(border=True):
        st.markdown("**NASA POWER Weather**")
        st.caption("Petaling Jaya area — daily climate observations")
        lat, lon = 3.0738, 101.5183
        try:
            weather_df = fetch_nasa_weather(lat, lon, st.session_state.weather_refresh)
            st.dataframe(weather_df.head(), use_container_width=True)
        except Exception as e:
            st.error(f"Could not load weather data: {e}")

# ─────────────────────────────────────────────────────────────
# Model Explanation
# ─────────────────────────────────────────────────────────────
st.markdown('<div class="section-title">Model Explanation</div>', unsafe_allow_html=True)

with st.container(border=True):

    if st.button("Explain Prediction", use_container_width=True):
        check_rate_limit()
        with st.spinner("Analysing feature contributions..."):
            try:
                pred = float(model.predict(input_df)[0])

                explanation, feat_names = build_shap_explanation(model, input_df)
                shap_vals = np.array(explanation.values).flatten()

                if shap_vals.size == 0 or np.all(np.isnan(shap_vals)):
                    raise ValueError("SHAP returned empty or NaN values.")

                impact_df = pd.DataFrame({
                    "feature": feat_names, "impact": shap_vals
                }).sort_values("impact", key=abs, ascending=False)

                top_pos = impact_df[impact_df["impact"] > 0].head(1)
                top_neg = impact_df[impact_df["impact"] < 0].head(1)

                pos_name = top_pos.iloc[0]["feature"] if len(top_pos) else "None"
                pos_val  = float(top_pos.iloc[0]["impact"]) if len(top_pos) else 0.0
                neg_name = top_neg.iloc[0]["feature"] if len(top_neg) else "None"
                neg_val  = float(top_neg.iloc[0]["impact"]) if len(top_neg) else 0.0

                # ── Summary metrics
                r1, r2, r3 = st.columns(3)
                with r1:
                    st.metric("Predicted Yield", f"{pred:.3f} t/ha")
                with r2:
                    st.metric("Top Positive Driver", pos_name,
                              delta=f"+{pos_val:.4f}" if pos_name != "None" else None)
                with r3:
                    st.metric("Top Negative Driver", neg_name,
                              delta=f"{neg_val:.4f}" if neg_name != "None" else None)

                st.caption(
                    f"Inputs: {crop_type}  ·  {region}  ·  {season}  ·  "
                    f"pH {soil_ph:.1f}  ·  Moisture {soil_moisture:.0f}%  ·  "
                    f"Temp {avg_temperature:.1f}°C  ·  Rainfall {total_rainfall:,.0f}mm  ·  "
                    f"Sunlight {sunlight_hours:.1f}h"
                )

                # ── Plain-English summary
                if pos_name != "None" and neg_name != "None":
                    st.success(
                        f"The model predicts **{pred:.3f} t/ha** for **{crop_type}** "
                        f"in **{region}** during the **{season}** season. "
                        f"**{pos_name}** had the strongest positive effect "
                        f"(+{pos_val:.4f}), while **{neg_name}** had the largest "
                        f"negative effect ({neg_val:.4f})."
                    )
                elif pos_name != "None":
                    st.success(
                        f"The model predicts **{pred:.3f} t/ha**. "
                        f"**{pos_name}** was the strongest contributing factor "
                        f"(+{pos_val:.4f})."
                    )
                else:
                    st.success(f"The model predicts **{pred:.3f} t/ha**.")

                # ── Feature impact chart
                st.markdown("---")
                st.markdown("**Feature Impact Breakdown**")
                st.caption("Red indicates a positive effect on yield; blue indicates negative.")

                matplotlib.use("Agg")
                plt.close("all")

                chart_df = pd.DataFrame({
                    "Feature": feat_names, "Impact": shap_vals
                })
                chart_df["Abs"] = chart_df["Impact"].abs()
                chart_df = chart_df.sort_values("Abs", ascending=True).tail(14)
                chart_df = chart_df.drop(columns="Abs")

                fig, ax = plt.subplots(figsize=(10, 6), dpi=120)
                fig.patch.set_facecolor("#ffffff")
                ax.set_facecolor("#ffffff")

                colors = ["#d63031" if v > 0 else "#0984e3" for v in chart_df["Impact"]]
                bars = ax.barh(
                    chart_df["Feature"], chart_df["Impact"],
                    color=colors, height=0.55, edgecolor="none"
                )

                for bar, val in zip(bars, chart_df["Impact"]):
                    x_pos = bar.get_width()
                    ax.text(
                        x_pos + (0.008 if val >= 0 else -0.008),
                        bar.get_y() + bar.get_height() / 2,
                        f"{val:+.4f}",
                        ha="left" if val >= 0 else "right",
                        va="center",
                        fontsize=8.5,
                        fontfamily="Inter",
                        color="#636e72"
                    )

                ax.set_xlabel("SHAP value (impact on yield)", fontsize=9, color="#86868b")
                ax.axvline(0, color="#d1d1d6", linewidth=0.6)
                ax.tick_params(axis="both", labelsize=9, colors="#636e72")
                ax.spines["top"].set_visible(False)
                ax.spines["right"].set_visible(False)
                ax.spines["bottom"].set_color("#d1d1d6")
                ax.spines["left"].set_color("#d1d1d6")
                fig.tight_layout(pad=1.5)
                st.pyplot(fig)
                plt.close(fig)

            except Exception as e:
                st.warning(safe_error_message(e))

                try:
                    pred = float(model.predict(input_df)[0])
                    st.metric("Predicted Yield", f"{pred:.3f} t/ha")

                    try:
                        vals = np.array(explanation.values).flatten()
                        names = feat_names

                        bar_df = pd.DataFrame({"Feature": names, "Impact": vals})
                        bar_df["abs"] = bar_df["Impact"].abs()
                        bar_df = bar_df.sort_values("abs", ascending=False).head(12)
                        bar_df = bar_df.drop(columns="abs")

                        fig2, ax2 = plt.subplots(figsize=(8, 4), dpi=120)
                        fig2.patch.set_facecolor("#ffffff")
                        colors = ["#d63031" if v > 0 else "#0984e3" for v in bar_df["Impact"]]
                        ax2.barh(bar_df["Feature"], bar_df["Impact"], color=colors)
                        ax2.set_xlabel("SHAP value (impact on yield)")
                        ax2.invert_yaxis()
                        ax2.spines["top"].set_visible(False)
                        ax2.spines["right"].set_visible(False)
                        fig2.tight_layout()
                        st.pyplot(fig2)
                        plt.close(fig2)
                    except Exception:
                        st.info("Could not extract individual feature impacts.")

                    with st.expander("Error details"):
                        st.code(str(e))

                except Exception as pred_err:
                    st.error(f"Prediction also failed: {pred_err}")

# ─────────────────────────────────────────────────────────────
# Footer
# ─────────────────────────────────────────────────────────────
st.markdown("""
<div class="app-footer">
    <p>Build by Huzaib Wadoo</p>
</div>
""", unsafe_allow_html=True)