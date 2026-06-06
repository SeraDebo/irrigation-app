# db.py
# Supabase (Postgres) data-access layer for app.py

import streamlit as st
import pandas as pd
from supabase import create_client

url = st.secrets["SUPABASE_URL"]
key = st.secrets["SUPABASE_KEY"]
supabase = create_client(url, key)


# Supabase returns at most 1000 rows per request, so paginate with .range().
def _fetch_all(table, columns="*", filters=None, page_size=1000):
    rows = []
    start = 0
    try:
        while True:
            q = supabase.table(table).select(columns)
            if filters:
                for col, val in filters.items():
                    q = q.eq(col, val)
            resp = q.range(start, start + page_size - 1).execute()
            batch = resp.data or []
            rows.extend(batch)
            if len(batch) < page_size:
                break
            start += page_size
    except Exception:
        # Missing/optional table -> return empty; app.py guards for this.
        return pd.DataFrame()
    return pd.DataFrame(rows)


@st.cache_data(ttl=3600)
def load_district_centroids():
    df = _fetch_all("district_centroids")
    if not df.empty:
        df.columns = df.columns.str.lower().str.strip()
    return df


@st.cache_data(ttl=3600)
def load_crop_kc_defaults():
    df = _fetch_all("crop_kc_defaults")
    if not df.empty:
        df.columns = df.columns.str.lower().str.strip()
    return df


@st.cache_data(ttl=3600)
def load_indian_crop_dataset():
    df = _fetch_all("indian_crop_dataset")
    if not df.empty:
        df.columns = df.columns.str.lower().str.strip()
    return df


@st.cache_data(ttl=3600)
def get_crop_list():
    df = _fetch_all("crop_production", columns="crop")
    if df.empty:
        return []
    col = "crop" if "crop" in df.columns else df.columns[0]
    return sorted(df[col].dropna().astype(str).str.strip().unique().tolist())


@st.cache_data(ttl=3600)
def get_crop_production(crop):
    df = _fetch_all("crop_production", filters={"crop": crop})
    if df.empty:
        return df
    rename = {
        "state_name": "State_Name",
        "district_name": "District_Name",
        "season": "Season",
        "yield": "Yield",
        "crop": "Crop",
        "crop_year": "Crop_Year",
        "area": "Area",
        "production": "Production",
    }
    df = df.rename(columns={c: rename.get(c.lower().strip(), c) for c in df.columns})
    if "Yield" in df.columns:
        df["Yield"] = pd.to_numeric(df["Yield"], errors="coerce")
    return df
