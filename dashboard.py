# dashboard.py
# Blackpool Low EPC Dashboard (Streamlit)
# - Upload CSV or use bundled example
# - Filters for flats-only, EPC D–G, address/postcode search, floor area, date range
# - Minimum units per building (fixes the pandas assignment error)
# - Optional merge of flats into one pin per building
# - Map + Street View preview; selection list drives both

import os
import io
import math
import json
from datetime import datetime, date
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st
import pydeck as pdk

# -----------------------------
# Helpers
# -----------------------------

EPC_ORDER = ["A", "B", "C", "D", "E", "F", "G", "U", "NA", ""]
LOW_EPC = ["D", "E", "F", "G"]

@st.cache_data(show_spinner=False)
def load_csv(file_bytes: bytes) -> pd.DataFrame:
    return pd.read_csv(io.BytesIO(file_bytes))

def coerce_date(s, fallback=None) -> Optional[date]:
    if pd.isna(s):
        return fallback
    for fmt in ("%Y-%m-%d", "%d/%m/%Y", "%Y/%m/%d", "%d-%m-%Y", "%m/%d/%Y"):
        try:
            return datetime.strptime(str(s), fmt).date()
        except Exception:
            continue
    try:
        return pd.to_datetime(s).date()
    except Exception:
        return fallback

def ensure_cols(df: pd.DataFrame) -> pd.DataFrame:
    """Create/standardize columns used by the app if missing in the input."""
    df = df.copy()

    # Address-ish display column
    if "__DISPLAY_ADDR__" not in df.columns:
        # try to build from known EPC headers
        parts = []
        for c in ["ADDRESS", "ADDRESS1", "ADDRESS_LINE_1", "ADDRESS_LINE_2", "BUILDING_NAME"]:
            if c in df.columns:
                parts.append(df[c].astype(str))
        if "POSTCODE" in df.columns:
            parts.append(df["POSTCODE"].astype(str))
        if parts:
            df["__DISPLAY_ADDR__"] = (
                parts[0].fillna("") if len(parts) == 1 else parts[0].fillna("")
            )
            for p in parts[1:]:
                df["__DISPLAY_ADDR__"] = df["__DISPLAY_ADDR__"].str.strip() + ", " + p.fillna("")
            df["__DISPLAY_ADDR__"] = df["__DISPLAY_ADDR__"].str.replace(r",\s*$", "", regex=True)
        else:
            df["__DISPLAY_ADDR__"] = df.index.astype(str)

    # Postcode normalization
    if "POSTCODE" not in df.columns:
        guess = None
        for c in ["POST_CODE", "POSTALCODE", "ZIP", "PCODE"]:
            if c in df.columns:
                guess = c
                break
        if guess:
            df.rename(columns={guess: "POSTCODE"}, inplace=True)
        else:
            df["POSTCODE"] = ""

    # EPC rating
    if "CURRENT_ENERGY_RATING" not in df.columns:
        alt = None
        for c in ["EPC", "ENERGY_RATING", "CURRENT_RATING", "EPC_RATING"]:
            if c in df.columns:
                alt = c
                break
        if alt:
            df.rename(columns={alt: "CURRENT_ENERGY_RATING"}, inplace=True)
        else:
            df["CURRENT_ENERGY_RATING"] = ""

    # Property type / flats flag
    if "PROPERTY_TYPE" not in df.columns:
        df["PROPERTY_TYPE"] = ""

    # Flat-level indicator (best effort)
    if "IS_FLAT" not in df.columns:
        df["IS_FLAT"] = (
            df["PROPERTY_TYPE"].astype(str).str.contains("flat|apartment|maisonette", case=False, na=False)
            | df["__DISPLAY_ADDR__"].astype(str).str.contains(r"\bflat\b|\bapt\b|\bapartment\b", case=False, na=False)
        )

    # Total floor area
    if "TOTAL_FLOOR_AREA" not in df.columns:
        alt = None
        for c in ["TOTAL_FLOOR_AREA_M2", "FLOOR_AREA", "FLOORAREA", "AREA_M2"]:
            if c in df.columns:
                alt = c
                break
        if alt:
            df.rename(columns={alt: "TOTAL_FLOOR_AREA"}, inplace=True)
        else:
            df["TOTAL_FLOOR_AREA"] = np.nan

    # Lodgement date
    if "LODGEMENT_DATE" not in df.columns:
        alt = None
        for c in ["LODGE_DATE", "INSPECTION_DATE", "DATE_LODGED", "DATE"]:
            if c in df.columns:
                alt = c
                break
        if alt:
            df.rename(columns={alt: "LODGEMENT_DATE"}, inplace=True)
        else:
            df["LODGEMENT_DATE"] = ""

    # Lat/Lon (optional but great for map)
    if "LATITUDE" not in df.columns:  df["LATITUDE"] = np.nan
    if "LONGITUDE" not in df.columns: df["LONGITUDE"] = np.nan

    # Building identifier to group units (choose best available)
    if "BUILDING_ID" not in df.columns:
        if "UPRN" in df.columns:
            df["BUILDING_ID"] = df["UPRN"].astype(str)
        else:
            # Fallback: normalized address without flat numbers
            df["BUILDING_ID"] = (
                df["__DISPLAY_ADDR__"]
                .str.replace(r"\bflat\s*\d+[A-Za-z]?\b", "", regex=True, case=False)
                .str.replace(r"\b(apartment|apt)\s*\d+[A-Za-z]?\b", "", regex=True, case=False)
                .str.replace(r"\s+", " ", regex=True)
                .str.strip()
            )

    # Normalize types
    df["CURRENT_ENERGY_RATING"] = df["CURRENT_ENERGY_RATING"].astype(str).str.upper().str.strip()
    df["TOTAL_FLOOR_AREA"] = pd.to_numeric(df["TOTAL_FLOOR_AREA"], errors="coerce")
    # Parse date lazily when filtering
    return df.reset_index(drop=True)

def build_units_per_building(df: pd.DataFrame) -> pd.Series:
    """Count how many records share the same BUILDING_ID."""
    if "BUILDING_ID" not in df.columns:
        return pd.Series(1, index=df.index)
    return df.groupby("BUILDING_ID")["BUILDING_ID"].transform("count").astype(int)

def apply_epc_filter(df: pd.DataFrame, chosen: list) -> pd.DataFrame:
    if not chosen:
        return df
    return df[df["CURRENT_ENERGY_RATING"].isin(chosen)]

def within_floor_area(df: pd.DataFrame, min_m2: float, max_m2: float) -> pd.DataFrame:
    return df[(df["TOTAL_FLOOR_AREA"].fillna(-1) >= min_m2) & (df["TOTAL_FLOOR_AREA"].fillna(-1) <= max_m2)]

def within_date_range(df: pd.DataFrame, start: date, end: date) -> pd.DataFrame:
    dd = df.copy()
    dd["__LD__"] = dd["LODGEMENT_DATE"].apply(coerce_date)
    return dd[(dd["__LD__"] >= start) & (dd["__LD__"] <= end)].drop(columns=["__LD__"])

def make_label_building(row: pd.Series) -> str:
    """ALWAYS return a single string (fix for the ValueError)."""
    addr  = str(row.get("__DISPLAY_ADDR__", row.get("ADDRESS", ""))).strip()
    epc   = str(row.get("CURRENT_ENERGY_RATING", "")).strip()
    units = int(row.get("units_in_building", 0)) if not pd.isna(row.get("units_in_building", 0)) else 0
    return f"{addr} – EPC {epc} • {units} units"

def google_street_view_embed(lat: float, lon: float, api_key: str, heading: int = 0, pitch: int = 0, fov: int = 90, width: int = 640, height: int = 360) -> str:
    # Static Street View
    base = "https://maps.googleapis.com/maps/api/streetview"
    params = f"size={width}x{height}&location={lat},{lon}&heading={heading}&pitch={pitch}&fov={fov}&key={api_key}"
    return f"{base}?{params}"

def maps_link(lat: float, lon: float) -> str:
    return f"https://www.google.com/maps/search/?api=1&query={lat},{lon}"

# -----------------------------
# UI
# -----------------------------

st.set_page_config(page_title="Blackpool Low EPC", layout="wide")
st.title("Blackpool Low EPC")

with st.sidebar:
    st.caption("Google API key (Street View & optional Geocoding)")
    api_key = st.text_input("Google API key", type="password", label_visibility="collapsed")

    st.markdown("---")
    flats_only = st.checkbox("Flats only", value=True)
    only_d_to_g = st.checkbox("Only EPC D–G", value=True)

    st.caption("EPC Ratings (flat-level)")
    default_epc = ["F"] if only_d_to_g else LOW_EPC
    epc_choices = st.multiselect("Pick EPC ratings", EPC_ORDER[:7], default=default_epc, label_visibility="collapsed")

    address_contains = st.text_input("Address contains")
    postcode_prefix  = st.text_input("Postcode starts with")

    st.caption("Total floor area (m²)")
    min_area, max_area = 13, 2632
    floor_area = st.slider("Total floor area (m²)", min_value=min_area, max_value=max_area, value=(min_area, max_area), label_visibility="collapsed")

    st.caption("Lodgement date range")
    default_start = date(2015, 1, 9)
    default_end = date.today()
    start_date, end_date = st.date_input("Lodgement date range", (default_start, default_end), label_visibility="collapsed")

    st.caption("Minimum units per building")
    min_units = st.number_input("Minimum units per building", min_value=1, max_value=999, value=6, step=1, label_visibility="collapsed")

    merge_buildings = st.checkbox("Merge flats into one location (one pin per building)", value=False)

st.markdown("Upload a CSV to work with your own data, or use the bundle below.")

col_u1, col_u2 = st.columns([2, 1])
with col_u1:
    upload = st.file_uploader("Upload CSV", type=["csv"])
with col_u2:
    example_btn = st.button("Load example data")

# Load data
if "df" not in st.session_state:
    st.session_state.df = pd.DataFrame()

if upload is not None:
    st.session_state.df = ensure_cols(load_csv(upload.getvalue()))
elif example_btn or st.session_state.df.empty:
    # Minimal in-memory demo dataset
    sample = pd.DataFrame({
        "__DISPLAY_ADDR__": [
            "Flat 1, 235 Promenade [FY1 6AH]",
            "Flat 2, 235 Promenade [FY1 6AH]",
            "Flat 3, 235 Promenade [FY1 6AH]",
            "Flat 10a, 235 Promenade [FY1 6AH]",
            "Flat 11, 235 Promenade [FY1 6AH]",
            "Flat 13, 235 Promenade [FY1 6AH]",
        ],
        "POSTCODE": ["FY1 6AH"] * 6,
        "CURRENT_ENERGY_RATING": ["F", "F", "F", "F", "F", "F"],
        "PROPERTY_TYPE": ["Flat"] * 6,
        "IS_FLAT": [True]*6,
        "TOTAL_FLOOR_AREA": [35, 42, 38, 40, 39, 36],
        "LODGEMENT_DATE": ["2019-03-12", "2020-05-01", "2018-10-20", "2021-06-23", "2022-07-07", "2017-12-01"],
        "LATITUDE": [53.8199]*6,
        "LONGITUDE": [-3.0550]*6,
        "BUILDING_ID": ["B-235PROM"]*6,
    })
    st.session_state.df = ensure_cols(sample)

df = st.session_state.df.copy()

# -----------------------------
# Filtering
# -----------------------------

# Flats-only
if flats_only:
    if "IS_FLAT" in df.columns:
        df = df[df["IS_FLAT"] == True]
    else:
        df = df[df["PROPERTY_TYPE"].astype(str).str.contains("flat|apartment", case=False, na=False)]

# EPC ratings
if only_d_to_g:
    df = apply_epc_filter(df, [r for r in epc_choices if r in LOW_EPC])
else:
    df = apply_epc_filter(df, epc_choices)

# Address / Postcode text filters
if address_contains.strip():
    df = df[df["__DISPLAY_ADDR__"].str.contains(address_contains, case=False, na=False)]
if postcode_prefix.strip():
    df = df[df["POSTCODE"].astype(str).str.startswith(postcode_prefix, na=False)]

# Floor area range
df = within_floor_area(df, float(floor_area[0]), float(floor_area[1]))

# Date range
start_date = start_date if isinstance(start_date, date) else start_date[0]
end_date = end_date if isinstance(end_date, date) else end_date[1]
df = within_date_range(df, start_date, end_date)

# Units per building
df["units_in_building"] = build_units_per_building(df)
df["units_in_building"] = pd.to_numeric(df["units_in_building"], errors="coerce").fillna(0).astype(int)
df = df[df["units_in_building"] >= int(min_units)]

# If merge toggle: collapse to one row per building, keeping representative lat/lon and worst EPC
if merge_buildings and not df.empty:
    agg = {
        "__DISPLAY_ADDR__": "first",
        "POSTCODE": "first",
        "LATITUDE": "median",
        "LONGITUDE": "median",
        "units_in_building": "max",
    }
    # choose worst EPC within group (G worst → A best)
    df["_epc_rank"] = df["CURRENT_ENERGY_RATING"].map({e: i for i, e in enumerate(EPC_ORDER)})
    idx = df.groupby("BUILDING_ID")["_epc_rank"].idxmax()
    worst_epc = df.loc[idx, ["BUILDING_ID", "CURRENT_ENERGY_RATING"]].set_index("BUILDING_ID")
    collapsed = df.groupby("BUILDING_ID").agg(agg)
    collapsed["CURRENT_ENERGY_RATING"] = worst_epc["CURRENT_ENERGY_RATING"]
    df_display = collapsed.reset_index(drop=False)
else:
    df_display = df.copy()

# --- IMPORTANT: build labels (always one string per row) ---
if df_display.empty:
    st.info("No matches at the current settings. Try adjusting your filters.")
    st.stop()

df_display["__LABEL__"] = df_display.apply(make_label_building, axis=1).astype(str)

# -----------------------------
# Right side: results + map
# -----------------------------

st.subheader("Selected Property / Building")
st.caption("Pick an item (drives Map & Street View):")

# left = list + selection, right = map/street view
lcol, rcol = st.columns([1.2, 1.8])

with lcol:
    # Create a stable key for radio options
    options = df_display["__LABEL__"].tolist()
    chosen_label = st.radio("", options, index=0)

    # resolve selected row
    sel = df_display[df_display["__LABEL__"] == chosen_label].iloc[0]

with rcol:
    # Map
    if pd.notna(sel.get("LATITUDE")) and pd.notna(sel.get("LONGITUDE")):
        lat = float(sel["LATITUDE"])
        lon = float(sel["LONGITUDE"])

        st.caption("Map")
        layer = pdk.Layer(
            "ScatterplotLayer",
            data=df_display.dropna(subset=["LATITUDE", "LONGITUDE"]),
            get_position="[LONGITUDE, LATITUDE]",
            get_radius=15,
            pickable=True,
        )
        view_state = pdk.ViewState(latitude=lat, longitude=lon, zoom=14)
        st.pydeck_chart(pdk.Deck(layers=[layer], initial_view_state=view_state, tooltip={"text": "{__LABEL__}"}))

        # Street View
        st.caption("Street View")
        if api_key:
            url = google_street_view_embed(lat, lon, api_key)
            st.image(url, caption=f"Street View (approx): {maps_link(lat, lon)}", use_container_width=True)
        else:
            st.info("Add a Google API key in the sidebar to load Street View preview.")
            st.link_button("Open in Google Maps", maps_link(lat, lon))
    else:
        st.warning("No coordinates for this record. Add LATITUDE/LONGITUDE columns to enable map & Street View.")

# Footer counts
st.markdown("---")
c1, c2, c3 = st.columns(3)
with c1:
    st.metric("Entries shown", len(df_display))
with c2:
    st.metric("Unique buildings", df_display["BUILDING_ID"].nunique() if "BUILDING_ID" in df_display.columns else len(df_display))
with c3:
    st.metric("Units (sum)", int(df_display["units_in_building"].sum()) if "units_in_building" in df_display.columns else len(df_display))
