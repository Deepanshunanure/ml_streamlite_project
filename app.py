import streamlit as st
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

prediction_model = joblib.load(os.path.join(BASE_DIR, "best_notebook_model_Random_Forest_(Default).joblib"))
cluster_model = joblib.load(os.path.join(BASE_DIR, "notebook_kmeans_model.joblib"))
scaler = joblib.load(os.path.join(BASE_DIR, "notebook_scaler.joblib"))
df = pd.read_csv(os.path.join(BASE_DIR, "synthetic_household_power.csv"))

# ✅ Cluster Label Mapping
cluster_labels = {
    0: "Low Usage Household",
    1: "Moderate Usage Household",
    2: "High Usage Household"
}

# ---------------- Load Dataset for EDA ----------------
df = pd.read_csv("synthetic_household_power.csv")
df['datetime'] = pd.to_datetime(df['datetime'])
df['hour'] = df['datetime'].dt.hour
df['weekday'] = df['datetime'].dt.weekday

# ---------------- Page Style ----------------
st.set_page_config(page_title="Power Consumption App", page_icon="⚡", layout="wide")

st.markdown("""
    <style>
        .title { text-align:center; font-size:42px !important; color:#3d3d6b; font-weight:700; }
        .box {
            padding:22px; background:white; border-radius:14px;
            box-shadow:0 4px 15px rgba(0,0,0,0.07); margin-bottom: 28px;
        }
        .stButton>button {
            width:100%; padding:12px; border-radius:10px; font-size:18px;
            background:linear-gradient(90deg, #6a5acd, #836fff); color:white; border:none;
        }
    </style>
""", unsafe_allow_html=True)

st.markdown("<h1 class='title'>⚡ Power Consumption Analysis Dashboard</h1>", unsafe_allow_html=True)

# ---------------- Tabs ----------------
tab1, tab2, tab3 = st.tabs(["🔮 Prediction", "🔷 Clustering", "📊 Data Insights"])

# ---------------- TAB 1: Prediction ----------------
with tab1:
    st.markdown("<div class='box'>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        global_active_power = st.number_input("Global Active Power", min_value=0.0, step=0.01)
        global_reactive_power = st.number_input("Global Reactive Power", min_value=0.0, step=0.01)
        voltage = st.number_input("Voltage", min_value=0.0, step=0.1)
        global_intensity = st.number_input("Global Intensity", min_value=0.0, step=0.1)

    with col2:
        sub_metering_1 = st.number_input("Sub Metering 1", min_value=0.0, step=0.1)
        sub_metering_2 = st.number_input("Sub Metering 2", min_value=0.0, step=0.1)
        sub_metering_3 = st.number_input("Sub Metering 3", min_value=0.0, step=0.1)

        total_sub_metering = sub_metering_1 + sub_metering_2 + sub_metering_3
        st.info(f"**Auto Total Sub Metering:** {total_sub_metering:.2f}")

    hour = st.slider("Hour of Day", 0, 23, 12)
    weekday = st.slider("Weekday (0=Mon...6=Sun)", 0, 6, 3)

    features = pd.DataFrame([[global_active_power, global_reactive_power, voltage, global_intensity,
                              sub_metering_1, sub_metering_2, sub_metering_3, total_sub_metering,
                              hour, weekday]],
                            columns=[
                                "global_active_power", "global_reactive_power", "voltage", "global_intensity",
                                "sub_metering_1", "sub_metering_2", "sub_metering_3", "total_sub_metering",
                                "hour", "weekday"
                            ])

    scaled_features = scaler.transform(features)

    if st.button("Predict Power Consumption"):
        pred = prediction_model.predict(scaled_features)[0]
        st.success(f"**Predicted Next Hour Power Consumption:** {pred:.3f} kW ⚡")

    st.markdown("</div>", unsafe_allow_html=True)

# ---------------- TAB 2: Clustering ----------------
with tab2:
    st.markdown("<div class='box'>", unsafe_allow_html=True)

    if st.button("Find Usage Cluster Group"):
        cluster = cluster_model.predict(scaled_features)[0]

        # ✅ Use mapping instead of showing number
        label = cluster_labels[cluster]

        st.info(f"**Usage Category:** {label} 🔍")

        # Optional Explanation
        if label == "Low Usage Household":
            st.write("➤ Very low power consumption, possibly energy-efficient lifestyle or fewer appliances.")
        elif label == "Moderate Usage Household":
            st.write("➤ Average household consumption, normal daily activity.")
        elif label == "High Usage Household":
            st.write("➤ High power demand detected. Could indicate heavy appliance use or large household size.")

    st.markdown("</div>", unsafe_allow_html=True)

# ---------------- TAB 3: Data Insights ----------------
with tab3:
    st.markdown("<div class='box'>", unsafe_allow_html=True)

    st.write("### 🔍 Dataset Preview")
    st.dataframe(df.head())

    st.write("---")
    st.write("### ⚡ Power Consumption Over Time")
    fig1, ax1 = plt.subplots(figsize=(10, 3))
    ax1.plot(df['datetime'], df['global_active_power'])
    ax1.set_xlabel("Time")
    ax1.set_ylabel("Global Active Power (kW)")
    st.pyplot(fig1)

    st.write("---")
    st.write("### 🕒 Average Power Consumption by Hour")
    hourly_mean = df.groupby('hour')['global_active_power'].mean()
    fig2, ax2 = plt.subplots(figsize=(7, 3))
    ax2.plot(hourly_mean.index, hourly_mean.values, marker="o")
    st.pyplot(fig2)

    st.write("---")
    st.write("### 📅 Average Power Consumption by Weekday")
    weekday_mean = df.groupby('weekday')['global_active_power'].mean()
    fig3, ax3 = plt.subplots(figsize=(7, 3))
    ax3.bar(weekday_mean.index, weekday_mean.values)
    st.pyplot(fig3)

    st.write("---")
    st.write("### 🔥 Correlation Heatmap")
    corr = df[['global_active_power','global_reactive_power','voltage','global_intensity',
               'sub_metering_1','sub_metering_2','sub_metering_3','total_sub_metering']].corr()

    fig4, ax4 = plt.subplots(figsize=(8,5))
    cax = ax4.imshow(corr, cmap='coolwarm')
    fig4.colorbar(cax)
    st.pyplot(fig4)

    st.success("""
    ✅ Insights:
    • Evening hours show highest power usage.  
    • Weekends differ from weekdays in consumption patterns.  
    • Global Intensity strongly correlates with Active Power.  
    • Clustering identifies household consumption behavior groups.  
    """)

    st.markdown("</div>", unsafe_allow_html=True)
