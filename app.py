import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# ---------------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------------
st.set_page_config(page_title="India Housing Analytics", layout="wide")
st.title("🏠 India Housing Analytics — Full Interactive Dashboard")

# ---------------------------------------------------------
# AUTO-LOAD CSV (NO UPLOAD NEEDED)
# ---------------------------------------------------------

datapath = "india_housing_engineered.csv"  # <-- Your dataset name
data = pd.read_csv(datapath)    

# Clean numeric columns safely
numeric_cols = data.select_dtypes(include=['int64','float64']).columns

# Sidebar navigation
menu = st.sidebar.radio(
    "📌 Navigation",
    ["🏡 Home",
     "📊 Data Exploration",
     "📈 Investment Analysis",
     "🛠 Amenities & Property Features",
     "🗺 Maps & City Insights",
     "🤖 ML Prediction (Price & Investment)"]
)

# ---------------------------------------------------------
# PAGE 1 — HOME
# ---------------------------------------------------------
if menu == "🏡 Home":
    st.header("📌 Dataset Preview")
    st.dataframe(data.head())

    st.header("📊 Summary Statistics")
    st.dataframe(data.describe(include='all'))

    st.info("""
    This dashboard includes:
    - Full EDA  
    - Investment scoring  
    - Property & Amenities analysis  
    - Machine learning predictions  
    - City-wise mapping  
    """)

# ---------------------------------------------------------
# PAGE 2 — DATA EXPLORATION
# ---------------------------------------------------------
if menu == "📊 Data Exploration":
    st.header("📊 Exploratory Data Analysis")

    # Filters
    st.sidebar.subheader("Filters")

    if "Price_in_Lakhs" in data.columns:
        min_price, max_price = int(data["Price_in_Lakhs"].min()), int(data["Price_in_Lakhs"].max())
        price_range = st.sidebar.slider("Price Range (in Lakhs)", min_price, max_price, (min_price, max_price))
        data = data[(data["Price_in_Lakhs"] >= price_range[0]) & (data["Price_in_Lakhs"] <= price_range[1])]

    if "City" in data.columns:
        selected_city = st.sidebar.multiselect("City", data["City"].unique())
        if selected_city:
            data = data[data["City"].isin(selected_city)]

    if "Property_Type" in data.columns:
        selected_pt = st.sidebar.multiselect("Property Type", data["Property_Type"].unique())
        if selected_pt:
            data = data[data["Property_Type"].isin(selected_pt)]

    if "BHK" in data.columns:
        selected_bhk = st.sidebar.multiselect("BHK", data["BHK"].unique())
        if selected_bhk:
            data = data[data["BHK"].isin(selected_bhk)]

    if "Furnished_Status" in data.columns:
        selected_fs = st.sidebar.multiselect("Furnished Status", data["Furnished_Status"].unique())
        if selected_fs:
            data = data[data["Furnished_Status"].isin(selected_fs)]

    if "Ready_to_Move" in data.columns:
        selected_rtm = st.sidebar.multiselect("Ready to Move", data["Ready_to_Move"].unique())
        if selected_rtm:
            data = data[data["Ready_to_Move"].isin(selected_rtm)]


    # Price Distribution
    st.subheader("💰 Price Distribution")
    if "Price_in_Lakhs" in data.columns:
        fig = px.histogram(data, x="Price_in_Lakhs", nbins=40)
        st.plotly_chart(fig, use_container_width=True)

    # Size Distribution
    st.subheader("📏 Property Size Distribution")
    if "Size_in_SqFt" in data.columns:
        fig = px.histogram(data, x="Size_in_SqFt", nbins=40)
        st.plotly_chart(fig, use_container_width=True)

    # Price vs Size
    st.subheader("📈 Size vs Price Relationship")
    if "Size_in_SqFt" in data.columns:
        fig = px.scatter(data, x="Size_in_SqFt", y="Price_in_Lakhs", color="City")
        st.plotly_chart(fig, use_container_width=True)


# ---------------------------------------------------------
# PAGE 3 — INVESTMENT ANALYSIS
# ---------------------------------------------------------
if menu == "📈 Investment Analysis":
    st.header("💹 Investment Score Analysis")

    # Filters
    st.sidebar.subheader("Filters")

    if "Price_in_Lakhs" in data.columns:
        min_price, max_price = int(data["Price_in_Lakhs"].min()), int(data["Price_in_Lakhs"].max())
        price_range = st.sidebar.slider("Price Range (in Lakhs)", min_price, max_price, (min_price, max_price))
        data = data[(data["Price_in_Lakhs"] >= price_range[0]) & (data["Price_in_Lakhs"] <= price_range[1])]

    if "City" in data.columns:
        selected_city = st.sidebar.multiselect("City", data["City"].unique())
        if selected_city:
            data = data[data["City"].isin(selected_city)]

    if "Property_Type" in data.columns:
        selected_pt = st.sidebar.multiselect("Property Type", data["Property_Type"].unique())
        if selected_pt:
            data = data[data["Property_Type"].isin(selected_pt)]

    if "BHK" in data.columns:
        selected_bhk = st.sidebar.multiselect("BHK", data["BHK"].unique())
        if selected_bhk:
            data = data[data["BHK"].isin(selected_bhk)]

    if "Furnished_Status" in data.columns:
        selected_fs = st.sidebar.multiselect("Furnished Status", data["Furnished_Status"].unique())
        if selected_fs:
            data = data[data["Furnished_Status"].isin(selected_fs)]

    if "Ready_to_Move" in data.columns:
        selected_rtm = st.sidebar.multiselect("Ready to Move", data["Ready_to_Move"].unique())
        if selected_rtm:
            data = data[data["Ready_to_Move"].isin(selected_rtm)]

    # Investment Score Distribution
    if "Investment_Score" in data.columns:
        st.subheader("📈 Investment Score Distribution")
        fig = px.histogram(data, x="Investment_Score", nbins=30)
        st.plotly_chart(fig, use_container_width=True)

    # Good Investment Pie
    if "Good_Investment" in data.columns:
        st.subheader("🏆 Good Investment Classification")
        data["Good_Investment_Label"] = data["Good_Investment"].map({1: "Yes", 0: "No"})

        fig = px.pie(data, names="Good_Investment_Label", title="Good Investment Classification")
        st.plotly_chart(fig, use_container_width=True)

    # Top Investment Localities
    st.subheader("📍 Top 20 Localities for Investment")
    if "Locality" in data.columns and "Investment_Score" in data.columns:
        top_localities = (
            data.groupby("Locality")["Investment_Score"]
            .mean()
            .sort_values(ascending=False)
            .head(20)
        )
        st.dataframe(top_localities)

# ---------------------------------------------------------
# PAGE 4 — AMENITIES & PROPERTY FEATURES
# ---------------------------------------------------------
if menu == "🛠 Amenities & Property Features":
    st.header("🛠 Amenities & Features Analysis")


    # Filters
    st.sidebar.subheader("Filters")

    if "Price_in_Lakhs" in data.columns:
        min_price, max_price = int(data["Price_in_Lakhs"].min()), int(data["Price_in_Lakhs"].max())
        price_range = st.sidebar.slider("Price Range (in Lakhs)", min_price, max_price, (min_price, max_price))
        data = data[(data["Price_in_Lakhs"] >= price_range[0]) & (data["Price_in_Lakhs"] <= price_range[1])]

    if "City" in data.columns:
        selected_city = st.sidebar.multiselect("City", data["City"].unique())
        if selected_city:
            data = data[data["City"].isin(selected_city)]

    if "Property_Type" in data.columns:
        selected_pt = st.sidebar.multiselect("Property Type", data["Property_Type"].unique())
        if selected_pt:
            data = data[data["Property_Type"].isin(selected_pt)]

    if "BHK" in data.columns:
        selected_bhk = st.sidebar.multiselect("BHK", data["BHK"].unique())
        if selected_bhk:
            data = data[data["BHK"].isin(selected_bhk)]

    if "Furnished_Status" in data.columns:
        selected_fs = st.sidebar.multiselect("Furnished Status", data["Furnished_Status"].unique())
        if selected_fs:
            data = data[data["Furnished_Status"].isin(selected_fs)]

    if "Ready_to_Move" in data.columns:
        selected_rtm = st.sidebar.multiselect("Ready to Move", data["Ready_to_Move"].unique())
        if selected_rtm:
            data = data[data["Ready_to_Move"].isin(selected_rtm)]

    # Amenities
    if "Amenity_Count" in data.columns:
        st.subheader("📦 Amenities vs Price per SqFt")
        fig = px.scatter(data, x="Amenity_Count", y="Price_per_SqFt_new", color="City")
        st.plotly_chart(fig, use_container_width=True)

    # Property Age
    if "Age_of_Property" in data.columns:
        st.subheader("🏚 Property Age Distribution")
        fig = px.histogram(data, x="Age_of_Property", nbins=40)
        st.plotly_chart(fig, use_container_width=True)

    # Parking
    if "Parking_Space" in data.columns:
        st.subheader("🚗 Parking Space vs Price")
        fig = px.scatter(data, x="Parking_Space", y="Price_in_Lakhs")
        st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------------
# PAGE 5 — CITY & STATE INSIGHTS
# ---------------------------------------------------------
if menu == "🗺 City & State Insights":
    st.header("🗺 City & State Insights")

    # Filters
    st.sidebar.subheader("Filters")

    if "Price_in_Lakhs" in data.columns:
        min_price, max_price = int(data["Price_in_Lakhs"].min()), int(data["Price_in_Lakhs"].max())
        price_range = st.sidebar.slider("Price Range (in Lakhs)", min_price, max_price, (min_price, max_price))
        data = data[(data["Price_in_Lakhs"] >= price_range[0]) & (data["Price_in_Lakhs"] <= price_range[1])]

    if "City" in data.columns:
        selected_city = st.sidebar.multiselect("City", data["City"].unique())
        if selected_city:
            data = data[data["City"].isin(selected_city)]

    if "Property_Type" in data.columns:
        selected_pt = st.sidebar.multiselect("Property Type", data["Property_Type"].unique())
        if selected_pt:
            data = data[data["Property_Type"].isin(selected_pt)]

    if "BHK" in data.columns:
        selected_bhk = st.sidebar.multiselect("BHK", data["BHK"].unique())
        if selected_bhk:
            data = data[data["BHK"].isin(selected_bhk)]

    if "Furnished_Status" in data.columns:
        selected_fs = st.sidebar.multiselect("Furnished Status", data["Furnished_Status"].unique())
        if selected_fs:
            data = data[data["Furnished_Status"].isin(selected_fs)]

    if "Ready_to_Move" in data.columns:
        selected_rtm = st.sidebar.multiselect("Ready to Move", data["Ready_to_Move"].unique())
        if selected_rtm:
            data = data[data["Ready_to_Move"].isin(selected_rtm)]

    if "City" in data.columns:
        st.subheader("💰 Average Price by City")
        city_avg = data.groupby("City")["Price_in_Lakhs"].mean().reset_index()
        fig = px.bar(city_avg, x="City", y="Price_in_Lakhs")
        st.plotly_chart(fig, use_container_width=True)

    if "State" in data.columns:
        st.subheader("🏘 Average Properties by State")
        state_avg = data.groupby("State")["Price_in_Lakhs"].mean().reset_index()
        fig = px.bar(state_avg, x="State", y="Price_in_Lakhs")
        st.plotly_chart(fig, use_container_width=True)



# ---------------------------------------------------------
# PAGE 6 — MACHINE LEARNING
# ---------------------------------------------------------
if menu == "🤖 ML Prediction (Price & Investment)":
    st.header("🤖 Machine Learning Models")

     # Filters
    st.sidebar.subheader("Filters")

    if "Price_in_Lakhs" in data.columns:
        min_price, max_price = int(data["Price_in_Lakhs"].min()), int(data["Price_in_Lakhs"].max())
        price_range = st.sidebar.slider("Price Range (in Lakhs)", min_price, max_price, (min_price, max_price))
        data = data[(data["Price_in_Lakhs"] >= price_range[0]) & (data["Price_in_Lakhs"] <= price_range[1])]

    if "City" in data.columns:
        selected_city = st.sidebar.multiselect("City", data["City"].unique())
        if selected_city:
            data = data[data["City"].isin(selected_city)]

    if "Property_Type" in data.columns:
        selected_pt = st.sidebar.multiselect("Property Type", data["Property_Type"].unique())
        if selected_pt:
            data = data[data["Property_Type"].isin(selected_pt)]

    if "BHK" in data.columns:
        selected_bhk = st.sidebar.multiselect("BHK", data["BHK"].unique())
        if selected_bhk:
            data = data[data["BHK"].isin(selected_bhk)]

    if "Furnished_Status" in data.columns:
        selected_fs = st.sidebar.multiselect("Furnished Status", data["Furnished_Status"].unique())
        if selected_fs:
            data = data[data["Furnished_Status"].isin(selected_fs)]

    if "Ready_to_Move" in data.columns:
        selected_rtm = st.sidebar.multiselect("Ready to Move", data["Ready_to_Move"].unique())
        if selected_rtm:
            data = data[data["Ready_to_Move"].isin(selected_rtm)]

    # Label Encode
    df_ml = data.copy()
    le = LabelEncoder()

    for col in df_ml.select_dtypes(include='object').columns:
        df_ml[col] = le.fit_transform(df_ml[col].astype(str))

    # PRICE PREDICTION MODEL
    if "Price_in_Lakhs" in df_ml.columns:
        st.subheader("💰 Price Prediction")
        X = df_ml.drop("Price_in_Lakhs", axis=1)
        y = df_ml["Price_in_Lakhs"]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)
        model = RandomForestRegressor()
        model.fit(X_train, y_train)

        score = model.score(X_test, y_test)
        st.success(f"Model R² Score: {score:.3f}")

    # GOOD INVESTMENT CLASSIFICATION
    if "Good_Investment" in df_ml.columns:
        st.subheader("🏆 Good Investment Prediction (Classification)")
        X = df_ml.drop("Good_Investment", axis=1)
        y = df_ml["Good_Investment"]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)
        clf = RandomForestClassifier()
        clf.fit(X_train, y_train)

        acc = clf.score(X_test, y_test)
        st.success(f"Model Accuracy: {acc:.3f}")

