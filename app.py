import streamlit as st
import pandas as pd
import numpy as np
import io
from datetime import datetime
import altair as alt
from db import supabase

st.set_page_config(page_title="Smart Irrigation Advisor", layout="wide")

# -----------------------
# Helper Functions
# -----------------------

def parse_date_column(df):
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    return df

def compute_etc_if_missing(df):
    if "etc" in df.columns and df["etc"].notna().sum() > 0:
        return df
    if "et0" in df.columns and "kc" in df.columns:
        df["etc"] = df["et0"] * df["kc"]
    return df

# -----------------------
# FETCH UNIQUE VALUES
# -----------------------

@st.cache_data
def get_unique_values(column):
    all_data = []
    start = 0
    batch = 1000

    while True:
        response = supabase.table("irrigation_data") \
            .select(column) \
            .range(start, start + batch - 1) \
            .execute()

        if not response.data:
            break

        all_data.extend(response.data)
        start += batch

    df = pd.DataFrame(all_data)
    return sorted(df[column].dropna().unique().tolist())

# -----------------------
# LOAD DATA
# -----------------------

@st.cache_data
def load_data(district, crop):
    query = supabase.table("irrigation_data") \
        .select("*") \
        .limit(1000)

    if district != "All":
        query = query.eq("district", district)

    if crop != "All":
        query = query.eq("crop", crop)

    response = query.execute()
    df = pd.DataFrame(response.data)

    if not df.empty:
        df.columns = df.columns.str.lower()
        df = parse_date_column(df)
        df = compute_etc_if_missing(df)

    return df

# -----------------------
# Soil Defaults
# -----------------------

SOIL_DEFAULTS = {
    "loam": {"fc": 0.30, "wp": 0.12},
    "sandy": {"fc": 0.10, "wp": 0.04},
    "clay": {"fc": 0.40, "wp": 0.20},
}

# -----------------------
# Simulation
# -----------------------

def swb_simulation(df, root_depth_m=1.0, soil_fc=0.30, soil_wp=0.12,
                   depletion_fraction=0.5, irrigation_efficiency=0.8,
                   max_irrigation_mm=50):

    df = df.copy().sort_values("date").reset_index(drop=True)

    root_depth_mm = root_depth_m * 1000
    awc_mm = (soil_fc - soil_wp) * root_depth_mm

    soil_mm = soil_fc * root_depth_mm
    threshold_mm = soil_mm - depletion_fraction * awc_mm

    rows = []

    for _, row in df.iterrows():
        rain = row.get("rain", 0) or 0
        etc = row.get("etc", 0) or 0

        soil_mm += rain
        soil_mm -= etc
        soil_mm = max(0, soil_mm)

        irrig = 0
        irrig_flag = False

        if soil_mm < threshold_mm:
            required = soil_fc * root_depth_mm - soil_mm
            irrig = min(max_irrigation_mm, required / irrigation_efficiency)
            soil_mm += irrig * irrigation_efficiency
            irrig_flag = True

        soil_mm = min(soil_mm, soil_fc * root_depth_mm)

        rows.append({
            "date": row["date"],
            "precip_mm": rain,
            "etc_mm": etc,
            "soil_mm": soil_mm,
            "irrigation_mm": irrig,
            "irrigated": irrig_flag
        })

    return pd.DataFrame(rows)

# -----------------------
# UI
# -----------------------

st.title("🌾 Smart Irrigation Advisor")

district_list = get_unique_values("district")
crop_list = get_unique_values("crop")

selected_district = st.selectbox("Select District", ["All"] + district_list)
selected_crop = st.selectbox("Select Crop", ["All"] + crop_list)

df = load_data(selected_district, selected_crop)

if df.empty:
    st.warning("No data found")
    st.stop()

soil = st.selectbox("Soil Type", list(SOIL_DEFAULTS.keys()))
soil_params = SOIL_DEFAULTS[soil]

root_depth = st.number_input("Root Depth (m)", value=1.0)
depletion = st.slider("Depletion Fraction", 0.1, 0.9, 0.5)
efficiency = st.slider("Irrigation Efficiency", 0.3, 0.95, 0.8)

# -----------------------
# RUN SIMULATION (🔥 UPDATED)
# -----------------------

if st.button("Run Simulation"):

    with st.spinner("Calculating irrigation schedule... 🌾"):

        sim_df = swb_simulation(
            df,
            root_depth_m=root_depth,
            soil_fc=soil_params["fc"],
            soil_wp=soil_params["wp"],
            depletion_fraction=depletion,
            irrigation_efficiency=efficiency
        )

    st.success("Simulation Complete")

    # Metrics
    total_irrig = sim_df["irrigation_mm"].sum()
    events = sim_df["irrigated"].sum()

    col1, col2 = st.columns(2)
    col1.metric("Total Irrigation (mm)", f"{total_irrig:.1f}")
    col2.metric("Irrigation Events", int(events))

    # Charts
    base = alt.Chart(sim_df).encode(x="date:T")

    st.subheader("🌧️ Precipitation vs ETc")
    st.altair_chart(
        alt.layer(
            base.mark_bar(color="blue").encode(y="precip_mm"),
            base.mark_line(color="orange").encode(y="etc_mm")
        ),
        use_container_width=True
    )

    st.subheader("🌱 Soil Moisture & Irrigation")
    st.altair_chart(
        alt.layer(
            base.mark_line(color="green").encode(y="soil_mm"),
            base.mark_bar(color="red").encode(y="irrigation_mm")
        ),
        use_container_width=True
    )

    # Download
    csv = sim_df.to_csv(index=False).encode("utf-8")
    st.download_button("Download Results", csv, "results.csv")

else:
    st.info("Click 'Run Simulation'")