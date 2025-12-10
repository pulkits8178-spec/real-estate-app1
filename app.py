import streamlit as st
import pandas as pd
import plotly.express as px
import glob
import os


# -------------------------------------------
# LOAD MERGED DATASET FROM MULTIPLE CSV FILES
# -------------------------------------------

path = "./"  # all CSVs are in repo root folder

all_parts = glob.glob(path + "india_part_*.csv")

if len(all_parts) == 0:
    raise FileNotFoundError("❌ No dataset found! Upload india_part_1.csv, part_2 ... first.")

df = pd.concat([pd.read_csv(f) for f in all_parts], ignore_index=True)

print("Dataset Loaded:", df.shape)


# -----------------------------
# LOGIN SYSTEM
# -----------------------------
def login_page():
    st.title("🔐 Login to Continue")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        if username == "admin" and password == "1234":
            st.session_state.logged_in = True
            st.success("Login Successful! 🎉")
        else:
            st.error("Invalid username or password")

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False


# -----------------------------
# FAVORITES SYSTEM
# -----------------------------
if "favorites" not in st.session_state:
    st.session_state.favorites = []

def save_favorite(property_data):
    st.session_state.favorites.append(property_data)
    st.success("Property saved ⭐")

def show_favorites():
    st.header("⭐ Your Saved Properties")
    if not st.session_state.favorites:
        st.info("No favorite properties saved yet.")
    else:
        for i, item in enumerate(st.session_state.favorites):
            st.subheader(f"Property #{i+1}")
            st.json(item)


# -----------------------------
# VISUALIZATION FUNCTIONS
# -----------------------------
def demand_heatmap(df):
    st.header("🔥 City Demand Heat Map")
    city_demand = df.groupby("City")["Price_in_Lakhs"].mean().reset_index()

    fig = px.density_heatmap(city_demand,
                             x="City",
                             y="Price_in_Lakhs",
                             title="City-wise Average Prices")
    st.plotly_chart(fig)


def price_trend_chart(df):
    st.header("📈 Price Trend Over Years")
    trend = df.groupby("Year_Built")["Price_in_Lakhs"].mean().reset_index()

    fig = px.line(trend,
                  x="Year_Built",
                  y="Price_in_Lakhs",
                  markers=True,
                  title="Yearly Price Trend")
    st.plotly_chart(fig)


def investment_visual(score):
    st.header("📊 Investment Score Visualization")
    fig = px.pie(values=[score, 100 - score],
                 names=["Score", "Remaining"],
                 hole=0.5)
    st.plotly_chart(fig)


# ------------------------------------------------------------------------------
# MAIN APP
# ------------------------------------------------------------------------------
if not st.session_state.logged_in:
    login_page()
    st.stop()

# Load your dataset
#df = pd.read_csv("india_housing_prices.csv")

# Sidebar Menu
st.sidebar.title("🏠 Menu")
option = st.sidebar.radio("Navigation",
    [
        "🏡 Home",
        "🔎 Property Search",
        "⭐ Favorites",
        "📈 Price Trends",
        "🔥 Demand Heat Map",
        "📊 Investment Score"
    ]
)

# -----------------------------
# HOME PAGE
# -----------------------------
if option == "🏡 Home":
    st.title("🏠 Real Estate Investment Advisor")
    st.write("Welcome to the best property investment analysis tool in India!")


# -----------------------------
# PROPERTY SEARCH
# -----------------------------
elif option == "🔎 Property Search":
    st.title("🔎 Property Search")

    city = st.selectbox("Select City", df["City"].unique())
    bhk = st.number_input("BHK", 1, 10)
    size = st.slider("Size (SqFt)", 300, 5000)

    price_est = int(size * 12)  # simple estimation example

    if st.button("Predict Price"):
        st.success(f"Estimated Price: ₹ {price_est:,}")

        if st.button("⭐ Save to Favorites"):
            save_favorite({
                "City": city,
                "BHK": bhk,
                "Size": size,
                "Estimated_Price": price_est
            })


# -----------------------------
# SHOW FAVORITES
# -----------------------------
elif option == "⭐ Favorites":
    show_favorites()


# -----------------------------
# PRICE TREND CHART
# -----------------------------
elif option == "📈 Price Trends":
    price_trend_chart(df)


# -----------------------------
# DEMAND HEAT MAP
# -----------------------------
elif option == "🔥 Demand Heat Map":
    demand_heatmap(df)


# -----------------------------
# INVESTMENT SCORE VISUALIZATION
# -----------------------------
elif option == "📊 Investment Score":
    score = st.slider("Select Score", 1, 100, 85)
    investment_visual(score)
