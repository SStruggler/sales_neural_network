"""
app.py
======
Retail Sales Prediction — Interactive Streamlit Web Application

Users enter business parameters via a sidebar, and the trained ANN
returns an Expected Sales Amount with contextual business insights.

Run:
    streamlit run app/app.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
import streamlit as st
import joblib
import warnings
warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ──────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="RetailMind · Sales Predictor",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────────────────────────────────────
# CUSTOM CSS — Dark luxury theme
# ──────────────────────────────────────────────────────────────────────────────

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;600;700&family=DM+Mono:wght@400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'Space Grotesk', sans-serif;
    background-color: #0a0c14;
    color: #e2e8f0;
}
.stApp { background-color: #0a0c14; }

/* Sidebar */
section[data-testid="stSidebar"] {
    background: linear-gradient(160deg, #0d1020 0%, #111827 100%);
    border-right: 1px solid #1e293b;
}

/* Metric cards */
.metric-card {
    background: linear-gradient(135deg, #0f1729 0%, #131d35 100%);
    border: 1px solid #1e3a5f;
    border-radius: 16px;
    padding: 24px 28px;
    text-align: center;
    box-shadow: 0 8px 32px rgba(0,212,255,0.05);
}
.metric-value {
    font-family: 'DM Mono', monospace;
    font-size: 2.8rem;
    font-weight: 700;
    background: linear-gradient(135deg, #00d4ff 0%, #0099ff 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    line-height: 1.1;
}
.metric-label {
    color: #64748b;
    font-size: 0.85rem;
    font-weight: 600;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    margin-top: 6px;
}

/* Insight cards */
.insight-card {
    background: #0f1729;
    border: 1px solid #1e293b;
    border-radius: 12px;
    padding: 16px 20px;
    margin: 8px 0;
    border-left: 3px solid;
}
.insight-positive { border-left-color: #10b981; }
.insight-warning  { border-left-color: #f59e0b; }
.insight-info     { border-left-color: #3b82f6; }

/* Section headers */
.section-title {
    font-size: 1.1rem;
    font-weight: 700;
    color: #94a3b8;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    margin-bottom: 12px;
    padding-bottom: 8px;
    border-bottom: 1px solid #1e293b;
}

/* Predict button */
div[data-testid="stButton"] > button {
    width: 100%;
    background: linear-gradient(135deg, #0099ff 0%, #00d4ff 100%);
    color: #000;
    font-weight: 700;
    font-size: 1rem;
    border: none;
    border-radius: 10px;
    padding: 14px 0;
    letter-spacing: 0.05em;
    transition: all 0.2s;
    text-transform: uppercase;
}
div[data-testid="stButton"] > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 24px rgba(0,153,255,0.4);
}
</style>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────────────────────
# MODEL LOADING (cached for performance)
# ──────────────────────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner="Loading AI model…")
def load_artifacts():
    """Load Keras model and StandardScaler from disk."""
    import tensorflow as tf
    scaler = joblib.load("models/scaler.joblib")
    model  = tf.keras.models.load_model("models/sales_model.keras")
    return scaler, model

@st.cache_data
def load_training_data():
    """Load raw data for reference stats."""
    df = pd.read_csv("data/raw/sales.csv", encoding="latin-1")
    return df


# ──────────────────────────────────────────────────────────────────────────────
# FEATURE ENGINEERING (must mirror preprocessing.py exactly)
# ──────────────────────────────────────────────────────────────────────────────

def build_input_vector(
    region: str,
    segment: str,
    category: str,
    quantity: int,
    discount: float,
    profit: float,
    month: int,
    year: int,
    day_of_week: int,
    scaler,
    training_feature_names: list,
) -> np.ndarray:
    """
    Replicate preprocessing.py feature engineering for a single prediction.
    Returns a scaled 2D numpy array ready for model.predict().
    """
    is_weekend         = 1 if day_of_week >= 5 else 0
    discount_frac      = discount / 100.0
    discount_pct       = float(discount)
    promo_active       = 1 if discount > 0 else 0
    month_sin          = np.sin(2 * np.pi * month / 12)
    month_cos          = np.cos(2 * np.pi * month / 12)
    day_sin            = np.sin(2 * np.pi * day_of_week / 7)
    day_cos            = np.cos(2 * np.pi * day_of_week / 7)

    # Build base row as dict (numerical + OHE flags = 0 by default)
    row = {f: 0.0 for f in training_feature_names}

    # Numericals
    row["Quantity"]            = float(quantity)
    row["Discount"]            = discount_frac
    row["Profit"]              = profit
    row["discount_percentage"] = discount_pct
    row["promo_active"]        = float(promo_active)
    row["month"]               = float(month)
    row["year"]                = float(year)
    row["day_of_week"]         = float(day_of_week)
    row["is_weekend"]          = float(is_weekend)
    row["month_sin"]           = month_sin
    row["month_cos"]           = month_cos
    row["day_sin"]             = day_sin
    row["day_cos"]             = day_cos

    # OHE: Region (drop_first=True drops first category alphabetically)
    # Actual dummies depend on training data; we map known values
    region_map = {
        "Central": "Region_East",   # Central is the dropped base
        "East":    "Region_East",
        "South":   "Region_South",
        "West":    "Region_West",
    }
    # Safer approach: set whatever OHE columns exist
    for col in training_feature_names:
        if col.startswith("Region_") and region in col:
            row[col] = 1.0
        if col.startswith("Segment_") and segment in col:
            row[col] = 1.0
        if col.startswith("Category_") and category in col:
            row[col] = 1.0

    X = pd.DataFrame([row])[training_feature_names]
    return scaler.transform(X)


# ──────────────────────────────────────────────────────────────────────────────
# INSIGHT GENERATOR
# ──────────────────────────────────────────────────────────────────────────────

def generate_insights(prediction: float, discount: float, category: str, segment: str) -> list:
    """Generate contextual business insights based on prediction and inputs."""
    insights = []

    if prediction > 1000:
        insights.append(("positive", "🚀 High-value transaction detected. Consider upselling complementary products."))
    elif prediction > 500:
        insights.append(("info", "📦 Mid-tier sale. Bundle offers could increase order value."))
    else:
        insights.append(("warning", "💡 Lower value sale. Check if discount is eroding margin unnecessarily."))

    if discount > 30:
        insights.append(("warning", f"⚠️ Deep discount ({discount:.0f}%) applied. Monitor profit margin carefully."))
    elif discount > 0:
        insights.append(("info", f"🏷️ Moderate discount ({discount:.0f}%) active — within healthy promotional range."))
    else:
        insights.append(("positive", "✅ No discount applied. Full-price sale maximizes margin."))

    if category == "Technology":
        insights.append(("positive", "💻 Tech products carry strong margins. Ideal for Q4 push."))
    elif category == "Furniture":
        insights.append(("info", "🪑 Furniture has longer sales cycles. Regional targeting recommended."))
    else:
        insights.append(("info", "📎 Office Supplies show consistent demand. Great for repeat customer programs."))

    if segment == "Corporate":
        insights.append(("positive", "🏢 Corporate buyers tend to make larger, recurring purchases."))
    elif segment == "Consumer":
        insights.append(("info", "👤 Consumer segment benefits most from seasonal promotions."))

    return insights


# ──────────────────────────────────────────────────────────────────────────────
# SIDEBAR — USER INPUTS
# ──────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## 📊 RetailMind")
    st.markdown("*Retail Sales Prediction Engine*")
    st.markdown("---")

    st.markdown('<div class="section-title">Transaction Details</div>', unsafe_allow_html=True)

    region = st.selectbox(
        "🌍 Region",
        ["West", "East", "Central", "South"],
        help="Geographic sales region"
    )
    segment = st.selectbox(
        "👥 Customer Segment",
        ["Consumer", "Corporate", "Home Office"],
    )
    category = st.selectbox(
        "📦 Product Category",
        ["Office Supplies", "Technology", "Furniture"],
    )

    st.markdown("---")
    st.markdown('<div class="section-title">Order Parameters</div>', unsafe_allow_html=True)

    quantity = st.slider("📦 Quantity", min_value=1, max_value=20, value=3)
    discount = st.slider("🏷️ Discount (%)", min_value=0, max_value=80, value=10, step=5)
    profit   = st.number_input("💰 Expected Profit ($)", min_value=-500.0, max_value=5000.0,
                                value=50.0, step=10.0,
                                help="Estimated profit margin for this transaction")

    st.markdown("---")
    st.markdown('<div class="section-title">Time & Seasonality</div>', unsafe_allow_html=True)

    month = st.select_slider(
        "📅 Month",
        options=list(range(1, 13)),
        format_func=lambda m: ["Jan","Feb","Mar","Apr","May","Jun",
                                "Jul","Aug","Sep","Oct","Nov","Dec"][m-1],
        value=11
    )
    year = st.selectbox("📆 Year", [2024, 2025, 2026], index=1)
    day_of_week = st.selectbox(
        "📅 Day of Week",
        [0, 1, 2, 3, 4, 5, 6],
        format_func=lambda d: ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"][d],
        index=1
    )

    st.markdown("---")
    predict_btn = st.button("🔮 Predict Sales", use_container_width=True)


# ──────────────────────────────────────────────────────────────────────────────
# MAIN PANEL
# ──────────────────────────────────────────────────────────────────────────────

# Header
st.markdown("# 🏪 Retail Sales Intelligence Platform")
st.markdown("*Deep learning–powered sales prediction for smarter business decisions*")
st.markdown("---")

# Load model
try:
    scaler, model = load_artifacts()
    df_raw = load_training_data()

    # Get feature names from scaler (stored during training)
    # We reconstruct them from the training run
    # Load the actual feature names from the processed predictions file
    import os
    feature_names_path = "data/processed/feature_names.txt"
    if os.path.exists(feature_names_path):
        with open(feature_names_path) as f:
            training_feature_names = [line.strip() for line in f.readlines()]
    else:
        # Fallback: rebuild feature names by running preprocessing
        from src.preprocessing import (
            clean_data, engineer_datetime_features,
            engineer_business_features, encode_categoricals,
            build_feature_matrix
        )
        df_c = clean_data(df_raw)
        df_f = engineer_datetime_features(df_c)
        df_f = engineer_business_features(df_f)
        df_e = encode_categoricals(df_f)
        X, _, names = build_feature_matrix(df_e)
        training_feature_names = names
        with open(feature_names_path, "w") as fw:
            fw.write("\n".join(names))

    model_loaded = True

except Exception as e:
    st.error(f"⚠️ Could not load model: {e}. Please run `python src/train_pipeline.py` first.")
    model_loaded = False


# KPI summary from training data
if model_loaded:
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-value">${df_raw['Sales'].sum()/1e6:.1f}M</div>
            <div class="metric-label">Total Historical Revenue</div>
        </div>""", unsafe_allow_html=True)
    with col2:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-value">{len(df_raw):,}</div>
            <div class="metric-label">Total Transactions</div>
        </div>""", unsafe_allow_html=True)
    with col3:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-value">${df_raw['Sales'].mean():.0f}</div>
            <div class="metric-label">Avg Order Value</div>
        </div>""", unsafe_allow_html=True)
    with col4:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-value">{df_raw['Discount'].mean()*100:.1f}%</div>
            <div class="metric-label">Avg Discount Rate</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Prediction section
    if predict_btn:
        with st.spinner("⚙️ Running neural network inference…"):
            try:
                X_input = build_input_vector(
                    region, segment, category, quantity, discount, profit,
                    month, year, day_of_week, scaler, training_feature_names
                )
                prediction = float(model.predict(X_input, verbose=0)[0][0])
                prediction = max(prediction, 0.0)

                # Result display
                st.markdown("---")
                st.markdown("## 🎯 Prediction Results")

                res_col1, res_col2 = st.columns([1, 2])

                with res_col1:
                    st.markdown(f"""<div class="metric-card" style="padding: 40px;">
                        <div style="color:#64748b;font-size:0.8rem;letter-spacing:0.1em;text-transform:uppercase;margin-bottom:8px;">Expected Sales Amount</div>
                        <div class="metric-value" style="font-size:3.5rem;">${prediction:,.2f}</div>
                        <div style="color:#475569;font-size:0.85rem;margin-top:12px;">
                            {region} · {category} · {segment}
                        </div>
                        <div style="color:#334155;font-size:0.8rem;margin-top:4px;">
                            Qty: {quantity} · Discount: {discount}% · Month: {month}
                        </div>
                    </div>""", unsafe_allow_html=True)

                    # Confidence band
                    low  = prediction * 0.85
                    high = prediction * 1.15
                    st.markdown(f"""<div style="margin-top:16px;padding:16px;background:#0f1729;border-radius:12px;border:1px solid #1e293b;text-align:center;">
                        <div style="color:#64748b;font-size:0.75rem;text-transform:uppercase;letter-spacing:0.1em;">Prediction Range (±15%)</div>
                        <div style="font-family:'DM Mono',monospace;color:#94a3b8;font-size:1.05rem;margin-top:6px;">
                            ${low:,.0f} — ${high:,.0f}
                        </div>
                    </div>""", unsafe_allow_html=True)

                with res_col2:
                    st.markdown("### 💡 Business Insights")
                    insights = generate_insights(prediction, discount, category, segment)
                    for style_class, text in insights:
                        st.markdown(
                            f'<div class="insight-card insight-{style_class}">{text}</div>',
                            unsafe_allow_html=True
                        )

                # Comparison against historical average
                st.markdown("---")
                avg_sales = df_raw["Sales"].mean()
                delta = ((prediction - avg_sales) / avg_sales) * 100
                arrow = "▲" if delta >= 0 else "▼"
                color = "#10b981" if delta >= 0 else "#ef4444"

                st.markdown(f"""
                <div style="background:#0f1729;border:1px solid #1e293b;border-radius:12px;padding:20px;text-align:center;">
                    <span style="color:#64748b;font-size:0.85rem;">vs. Historical Average (${avg_sales:,.0f})  </span>
                    <span style="color:{color};font-size:1.2rem;font-weight:700;font-family:'DM Mono';">
                        {arrow} {abs(delta):.1f}%
                    </span>
                    <span style="color:#475569;font-size:0.85rem;">  above average</span>
                </div>
                """, unsafe_allow_html=True)

            except Exception as e:
                st.error(f"Prediction failed: {e}")
                import traceback
                st.code(traceback.format_exc())

    else:
        # Placeholder state
        st.markdown("""
        <div style="text-align:center;padding:80px 40px;background:#0d1120;border:2px dashed #1e293b;border-radius:20px;margin-top:20px;">
            <div style="font-size:4rem;margin-bottom:16px;">🔮</div>
            <div style="font-size:1.4rem;font-weight:700;color:#94a3b8;margin-bottom:8px;">Configure & Predict</div>
            <div style="color:#475569;font-size:0.95rem;">
                Set your transaction parameters in the sidebar,<br>then click <strong>Predict Sales</strong> to get an AI-powered forecast.
            </div>
        </div>
        """, unsafe_allow_html=True)

    # Feature importance proxy (based on discount and quantity)
    st.markdown("---")
    st.markdown("### 📈 Historical Sales Distribution")
    import altair as alt

    chart_df = df_raw.groupby("Category")["Sales"].mean().reset_index()
    chart_df.columns = ["Category", "Avg Sales"]
    chart = alt.Chart(chart_df).mark_bar(
        cornerRadiusTopLeft=6, cornerRadiusTopRight=6, color="#0099ff"
    ).encode(
        x=alt.X("Category:N", axis=alt.Axis(labelColor="#94a3b8", titleColor="#64748b")),
        y=alt.Y("Avg Sales:Q", axis=alt.Axis(labelColor="#94a3b8", titleColor="#64748b")),
        tooltip=["Category", alt.Tooltip("Avg Sales:Q", format="$.2f")]
    ).properties(
        background="#0f1729", height=280
    ).configure_view(
        strokeOpacity=0
    ).configure_axis(
        gridColor="#1e293b", domainColor="#1e293b"
    )
    st.altair_chart(chart, use_container_width=True)
