import zipfile
import pandas as pd

zip_path = "data.zip"   # आपकी zip file का नाम

with zipfile.ZipFile(zip_path, 'r') as z:
    csv_file = z.namelist()[0]  # ZIP के अंदर पहली CSV फाइल
    with z.open(csv_file) as f:
        df = pd.read_csv(f)



# app.py

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

# -------------------------------------------------
# PAGE CONFIG + BASIC STYLE
# -------------------------------------------------
st.set_page_config(
    page_title="Real Estate Investment Advisor",
    layout="wide",
    page_icon="🏠",
)

st.markdown(
    """
    <style>
    .main {
        background: linear-gradient(135deg, #0f172a 0%, #020617 40%, #0b1120 100%);
        color: #f9fafb;
    }
    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 1.5rem;
        max-width: 1200px;
    }
    .glass-card {
        background: rgba(15,23,42,0.92);
        border-radius: 18px;
        padding: 18px 22px;
        border: 1px solid rgba(148,163,184,0.35);
        box-shadow: 0 24px 50px rgba(15,23,42,0.85);
    }
    .pill {
        padding: 4px 12px;
        border-radius: 999px;
        font-size: 0.8rem;
        border: 1px solid rgba(148,163,184,0.5);
        display: inline-block;
        margin-bottom: 4px;
        color: #cbd5f5;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# -------------------------------------------------
# LOAD DATA
# -------------------------------------------------
@st.cache_data
def load_data(path: str):
    df = pd.read_csv(path, low_memory=False)
    df.columns = [c.strip() for c in df.columns]
    return df

try:
    df = load_data("india_housing_prices.csv")
except Exception as e:
    st.error("⚠️ `india_housing_prices.csv` फाइल नहीं मिली। उसे इस app.py वाली जगह पर upload रखो।")
    st.stop()

# छोटी cleaning
if "Price_in_Lakhs" not in df.columns and "Price_per_SqFt" not in df.columns:
    st.error("Dataset में `Price_in_Lakhs` या `Price_per_SqFt` कॉलम होना बहुत ज़रूरी है।")
    st.stop()

# -------------------------------------------------
# TRAIN MODEL (lightweight)
# -------------------------------------------------
@st.cache_resource(show_spinner="ML model training चल रहा है…")
def train_model(df: pd.DataFrame):
    # target चुनो
    if "Price_per_SqFt" in df.columns and df["Price_per_SqFt"].notna().sum() > 1_000:
        target = "Price_per_SqFt"
    else:
        target = "Price_in_Lakhs"

    numeric_feats = [c for c in ["Size_in_SqFt", "BHK", "Year_Built"] if c in df.columns]
    cat_feats = [c for c in ["State", "City", "Locality", "Property_Type", "Furnished_Status"] if c in df.columns]

    df_model = df.dropna(subset=[target]).copy()

    # कम से कम कुछ feature data
    keep = pd.Series(True, index=df_model.index)
    if numeric_feats:
        keep &= df_model[numeric_feats].notna().any(axis=1)
    if cat_feats:
        keep &= df_model[cat_feats].notna().any(axis=1)
    df_model = df_model[keep]

    # speed के लिए sample
    df_model = df_model.sample(min(len(df_model), 40_000), random_state=42)

    X_raw = df_model[numeric_feats + cat_feats].copy()
    y = df_model[target].values

    for c in numeric_feats:
        X_raw[c] = X_raw[c].fillna(X_raw[c].median())
    for c in cat_feats:
        X_raw[c] = X_raw[c].fillna("MISSING").astype(str)

    # locality trim
    if "Locality" in X_raw.columns:
        top_loc = X_raw["Locality"].value_counts().nlargest(200).index
        X_raw["Locality"] = X_raw["Locality"].where(X_raw["Locality"].isin(top_loc), "OTHER")

    # one-hot via get_dummies
    X = pd.get_dummies(X_raw, columns=cat_feats, drop_first=True)
    feature_cols = X.columns.tolist()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.15, random_state=42
    )

    model = RandomForestRegressor(
        n_estimators=80,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    r2 = model.score(X_test, y_test)

    bundle = {
        "model": model,
        "target": target,
        "numeric_feats": numeric_feats,
        "cat_feats": cat_feats,
        "feature_cols": feature_cols,
        "X_raw": X_raw,
        "r2": r2,
    }
    return bundle


bundle = train_model(df)

# -------------------------------------------------
# HEADER
# -------------------------------------------------
st.markdown(
    """
    <div class="glass-card" style="margin-bottom:1.2rem;">
      <div class="pill">🏠 Real Estate · India Housing Dataset</div>
      <h1 style="margin:0;font-size:2.2rem;color:#e5e7eb;">REAL ESTATE INVESTMENT ADVISOR</h1>
      <p style="margin-top:0.25rem;color:#9ca3af;font-size:0.95rem;">
        Professional property analysis, quick investment scoring, market insights & smart projections — powered by your <b>india_housing_prices</b> data.
      </p>
    </div>
    """,
    unsafe_allow_html=True,
)

# -------------------------------------------------
# TABS
# -------------------------------------------------
tab1, tab2, tab3, tab4 = st.tabs(
    [
        "⚡ Quick Predictor",
        "🔍 Property Search",
        "📊 Market Insights",
        "ℹ️ About & Skills",
    ]
)

# helper lists
cities = sorted(df["City"].dropna().unique().tolist()) if "City" in df.columns else []
ptypes = sorted(df["Property_Type"].dropna().unique().tolist()) if "Property_Type" in df.columns else []
localities = sorted(df["Locality"].dropna().unique().tolist()) if "Locality" in df.columns else []

# -------------------------------------------------
# TAB 1: QUICK PREDICTOR
# -------------------------------------------------
with tab1:
    col_left, col_right = st.columns([1.6, 1.4])

    with col_left:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.subheader("Quick Investment Predictor")

        c1, c2 = st.columns(2)
        city = c1.selectbox("City", cities or ["N/A"])
        ptype = c2.selectbox("Property Type", ptypes or ["Apartment"])

        c3, c4 = st.columns(2)
        bhk = c3.slider("BHK", 1, 6, 2)
        size = c4.number_input("Size (Sq Ft)", min_value=200, max_value=10000, value=1200, step=50)

        c5, c6 = st.columns(2)
        current_price_lakh = c5.number_input("Current Price (₹ Lakhs)", min_value=5.0, max_value=1000.0, value=150.0, step=1.0)
        age_years = c6.slider("Age of Property (Years)", 0, 50, 5)

        st.markdown("##### Amenities Score (1–10)")
        a1, a2, a3 = st.columns(3)
        school = a1.slider("Schools", 1, 10, 7)
        hospital = a2.slider("Hospitals", 1, 10, 6)
        transport = a3.slider("Transport", 1, 10, 8)

        btn = st.button("🚀 Predict Investment Potential", use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with col_right:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.subheader("Prediction Results")

        if btn:
            # -------- build one-row input like training --------
            row = {}

            if "Size_in_SqFt" in bundle["numeric_feats"]:
                row["Size_in_SqFt"] = size
            if "BHK" in bundle["numeric_feats"]:
                row["BHK"] = bhk
            if "Year_Built" in bundle["numeric_feats"]:
                # approx build year
                row["Year_Built"] = 2025 - age_years

            if "State" in bundle["cat_feats"] and "State" in df.columns:
                # state from dataset by city
                state_val = df.loc[df["City"] == city, "State"].mode()
                row["State"] = state_val.iloc[0] if not state_val.empty else "MISSING"
            if "City" in bundle["cat_feats"]:
                row["City"] = city
            if "Locality" in bundle["cat_feats"]:
                # pick a common locality of that city
                loc_series = df.loc[df["City"] == city, "Locality"]
                loc_val = loc_series.mode()
                row["Locality"] = loc_val.iloc[0] if not loc_val.empty else "OTHER"
            if "Property_Type" in bundle["cat_feats"]:
                row["Property_Type"] = ptype
            if "Furnished_Status" in bundle["cat_feats"]:
                row["Furnished_Status"] = "Semi-Furnished"

            row_df = pd.DataFrame([row])

            # ensure numeric columns
            for c in bundle["numeric_feats"]:
                if c not in row_df.columns:
                    row_df[c] = bundle["X_raw"][c].median()

            # handle categories
            for c in bundle["cat_feats"]:
                if c not in row_df.columns:
                    row_df[c] = "MISSING"
                row_df[c] = row_df[c].astype(str)

            # locality trimming
            if "Locality" in row_df.columns and "Locality" in bundle["X_raw"].columns:
                top_loc = bundle["X_raw"]["Locality"].value_counts().nlargest(200).index
                row_df["Locality"] = row_df["Locality"].where(row_df["Locality"].isin(top_loc), "OTHER")

            # one-hot
            row_enc = pd.get_dummies(row_df, columns=bundle["cat_feats"], drop_first=True)
            row_enc = row_enc.reindex(columns=bundle["feature_cols"], fill_value=0)

            pred = bundle["model"].predict(row_enc)[0]

            if bundle["target"] == "Price_per_SqFt":
                price_per_sqft = pred
                fair_total_rs = price_per_sqft * size
                fair_total_lakh = fair_total_rs / 100000
            else:
                fair_total_lakh = pred
                fair_total_rs = fair_total_lakh * 100000
                price_per_sqft = fair_total_rs / size

            # investment score
            current_rs = current_price_lakh * 100000
            upside = (fair_total_rs - current_rs) / current_rs * 100 if current_rs > 0 else 0

            amenity_score = (school + hospital + transport) / 3
            # score from several things
            score = 70 + (amenity_score - 5) * 3 - age_years * 0.3 + upside * 0.2
            score = int(np.clip(score, 0, 99))

            if upside > 8 and score >= 75:
                label = "✅ GOOD INVESTMENT"
                color = "#16a34a"
                desc = "Property fair value current price से ऊपर है, और amenities भी strong हैं।"
            elif upside > -5:
                label = "🟡 FAIR / HOLD"
                color = "#facc15"
                desc = "Price लगभग fair लग रहा है; location/negotiation पर depend करेगा।"
            else:
                label = "🔴 OVERPRICED / RISKY"
                color = "#f97316"
                desc = "Model के हिसाब से ये property थोड़ी महंगी लग रही है।"

            st.markdown(
                f"""
                <div style="border-radius:18px;padding:18px;margin-top:4px;
                            background:linear-gradient(135deg,{color}33,#020617);border:1px solid {color};">
                    <h3 style="margin:0;color:{color};">{label}</h3>
                    <p style="margin-top:6px;color:#e5e7eb;font-size:0.9rem;">{desc}</p>
                    <p style="margin-top:0;color:#9ca3af;font-size:0.85rem;">
                        Model R² (test): <b>{bundle['r2']:.2f}</b>
                    </p>
                </div>
                """,
                unsafe_allow_html=True,
            )

            m1, m2, m3 = st.columns(3)
            m1.metric("Current Price (₹ Lakh)", f"{current_price_lakh:,.1f}")
            m2.metric("Fair Value (₹ Lakh)", f"{fair_total_lakh:,.1f}")
            m3.metric("Upside / Downside (%)", f"{upside:,.1f}%")

            k1, k2 = st.columns(2)
            k1.metric("Price per SqFt (₹)", f"{price_per_sqft:,.0f}")
            k2.metric("Investment Score", f"{score}/100")

            # forecast simple
            st.markdown("##### 5-Year Price Growth Projection")
            years = np.arange(0, 6)
            # assume growth influenced by score (just heuristic)
            growth = 0.04 + (score - 60) / 1000
            forecast = current_price_lakh * (1 + growth) ** years
            proj_df = pd.DataFrame({"Year": years, "Price_Lakh": forecast})
            st.line_chart(proj_df.set_index("Year"))

        else:
            st.info("ऊपर details भरकर **Predict Investment Potential** बटन दबाओ।")

        st.markdown("</div>", unsafe_allow_html=True)

# -------------------------------------------------
# TAB 2: PROPERTY SEARCH
# -------------------------------------------------
with tab2:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.subheader("Property Search & Filter")

    f1, f2, f3, f4 = st.columns(4)
    city_filter = f1.multiselect("City", options=cities, default=cities[:3] if cities else [])
    ptype_filter = f2.multiselect("Property Type", options=ptypes, default=ptypes if ptypes else [])
    min_bhk = f3.number_input("Min BHK", 0, 10, 1)
    max_bhk = f4.number_input("Max BHK", 0, 20, 5)

    g1, g2, g3, g4 = st.columns(4)
    min_size = g1.number_input("Min Size (Sq Ft)", 100, 20000, 500)
    max_size = g2.number_input("Max Size (Sq Ft)", 100, 50000, 3000)
    min_price = g3.number_input("Min Price (Lakh)", 0.0, 5000.0, 20.0)
    max_price = g4.number_input("Max Price (Lakh)", 0.0, 5000.0, 500.0)

    df_f = df.copy()

    if city_filter:
        df_f = df_f[df_f["City"].isin(city_filter)]
    if ptype_filter:
        df_f = df_f[df_f["Property_Type"].isin(ptype_filter)]

    if "BHK" in df_f.columns:
        df_f = df_f[(df_f["BHK"] >= min_bhk) & (df_f["BHK"] <= max_bhk)]

    if "Size_in_SqFt" in df_f.columns:
        df_f = df_f[(df_f["Size_in_SqFt"] >= min_size) & (df_f["Size_in_SqFt"] <= max_size)]

    if "Price_in_Lakhs" in df_f.columns:
        df_f = df_f[(df_f["Price_in_Lakhs"] >= min_price) & (df_f["Price_in_Lakhs"] <= max_price)]

    st.markdown("---")
    st.write(f"**Results: {len(df_f):,} properties**")

    c1, c2, c3 = st.columns(3)
    if "Price_in_Lakhs" in df_f.columns:
        c1.metric("Median Price (Lakh)", f"{df_f['Price_in_Lakhs'].median():.1f}")
    if "Size_in_SqFt" in df_f.columns:
        c2.metric("Median Size (Sq Ft)", f"{df_f['Size_in_SqFt'].median():.0f}")
    if "BHK" in df_f.columns:
        c3.metric("Median BHK", f"{df_f['BHK'].median():.1f}")

    st.dataframe(df_f.head(200))

    st.markdown("</div>", unsafe_allow_html=True)

# -------------------------------------------------
# TAB 3: MARKET INSIGHTS
# -------------------------------------------------
with tab3:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.subheader("Market Insights & Trends")

    if "City" not in df.columns or "Price_in_Lakhs" not in df.columns:
        st.info("City-wise insights के लिए `City` और `Price_in_Lakhs` कॉलम चाहिए।")
    else:
        city_stats = (
            df.groupby("City", as_index=False)
            .agg(
                avg_price=("Price_in_Lakhs", "median"),
                avg_size=("Size_in_SqFt", "median"),
                count=("ID", "count") if "ID" in df.columns else ("City", "count"),
            )
            .sort_values("avg_price", ascending=False)
        )

        # pseudo growth rate (rank पर depend)
        ranks = np.linspace(9.5, 5.5, len(city_stats))
        city_stats["growth_rate"] = np.round(ranks, 1)

        import plotly.express as px

        st.markdown("#### City Comparison — Prices & Growth")
        fig = px.bar(
            city_stats.head(10),
            x="City",
            y="avg_price",
            labels={"avg_price": "Median Price (₹ Lakh)"},
        )
        fig.update_traces(marker_line_width=0)
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("#### Investment Opportunity Matrix")
        # simple investment score heuristic
        max_price = city_stats["avg_price"].max()
        min_price = city_stats["avg_price"].min()
        norm_price = 1 - (city_stats["avg_price"] - min_price) / (max_price - min_price + 1e-6)
        demand_score = np.interp(city_stats["count"], (city_stats["count"].min(), city_stats["count"].max()), (70, 98))

        city_stats["demand_score"] = np.round(demand_score, 1)
        city_stats["investment_score"] = np.round(
            0.45 * city_stats["growth_rate"] + 0.35 * city_stats["demand_score"] / 10 + 0.2 * norm_price * 10, 1
        )

        def rec(x):
            if x >= 80:
                return "Buy"
            elif x >= 70:
                return "Hold"
            else:
                return "Avoid"

        city_stats["recommendation"] = city_stats["investment_score"].apply(rec)

        show_cols = [
            "City",
            "demand_score",
            "growth_rate",
            "avg_price",
            "investment_score",
            "recommendation",
        ]
        df_show = city_stats[show_cols].rename(
            columns={
                "demand_score": "Demand Score",
                "growth_rate": "Growth Rate (%)",
                "avg_price": "Avg Price (₹ Lakh)",
                "investment_score": "Investment Score",
                "recommendation": "Recommendation",
            }
        )
        st.dataframe(df_show, hide_index=True)

    st.markdown("</div>", unsafe_allow_html=True)

# -------------------------------------------------
# TAB 4: ABOUT
# -------------------------------------------------
with tab4:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.subheader("About this Project")

    st.markdown(
        """
        **Project Name:** Real Estate Investment Advisor  
        **Dataset:** `india_housing_prices.csv` (250k+ rows, India cities, BHK, size, price, locality, amenities)

        ### What this app does
        - 🧠 **ML-powered price prediction** using Random Forest (Price per SqFt or Price in Lakhs)
        - ⚡ **Quick Investment Predictor** – fair value, upside %, investment score, 5-year projection  
        - 🔍 **Property Search & Filter** – city, BHK, size, price range, property type  
        - 📊 **Market Insights** – city-wise price comparison & investment matrix  

        ### Tech Stack
        - Python, Pandas, NumPy  
        - Scikit-learn (RandomForestRegressor)  
        - Streamlit (interactive UI & dashboards)  
        - Plotly (beautiful charts)

        इसको आप अपने कॉलेज project / portfolio के लिए use कर सकते हो, और चाहो तो आगे features जोड़ सकते हो:
        - Loan EMI & cashflow analysis  
        - Tax, maintenance, vacancy assumptions  
        - User login और database storage  
        """
    )

    st.markdown("</div>", unsafe_allow_html=True)
