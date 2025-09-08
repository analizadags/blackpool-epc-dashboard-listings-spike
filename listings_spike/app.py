from __future__ import annotations
import os, re, math
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from collections import Counter
from datetime import datetime

from utils.epc_io import load_epc_csv
from utils.map_viz import draw_map
from providers.url_builders import (
    rightmove_search_url,
    zoopla_search_url,
    generic_map_query,
    rightmove_broad_url,
    zoopla_broad_url,
)

# -------------------- Page setup & light styling --------------------
load_dotenv()
st.set_page_config(page_title="Blackpool Low-EPC Dashboard", layout="wide")

st.markdown("""
<style>
/* title spacing similar to your original */
.block-container { padding-top: 1rem; padding-bottom: 2rem; }
.small-muted { color:#6b7280; font-size:0.9rem; }
.kpi .stMetric { padding: 0.25rem 0.5rem; }
.kpi .css-184tjsw, .kpi .css-ebv4gl { font-size: 0.9rem !important; }
hr { margin: 0.5rem 0 1rem 0; }
.stTabs [data-baseweb="tab-list"] { gap: .35rem; }
.stTabs [data-baseweb="tab"] { padding: .3rem .6rem; }
</style>
""", unsafe_allow_html=True)

# -------------------- Sidebar (like the original) --------------------
st.sidebar.header("Data source")
st.sidebar.caption("Upload a CSV or use the bundled dataset. Filters apply to whichever dataset is active.")

# CSV source: load from repo root by default
DEFAULT_DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "blackpool_low_epc_with_coords.csv")

uploaded = st.sidebar.file_uploader("Upload CSV (optional)", type=["csv"])
if st.sidebar.button("Refresh data"):
    st.rerun()

st.sidebar.divider()
st.sidebar.header("Filters")

# Optional Google API key (for Street View static image); not required for links
google_api_key = st.sidebar.text_input("Google API key (Street View optional)", type="password")

display_mode = st.sidebar.radio(
    "Display mode",
    ["Individual flats", "Merged buildings (one pin per building)"],
    index=0
)

only_efg = st.sidebar.checkbox("Only EPC D–G", value=False)  # you can toggle default

# EPC multiselect (will be filled after data load)
epc_multiselect_placeholder = st.sidebar.empty()

addr_contains = st.sidebar.text_input("Address contains", value="")
pc_starts = st.sidebar.text_input("Postcodes starts with", value="")

# Optional filters if columns exist in your CSV
tfa_min, tfa_max = None, None
ldr_start, ldr_end = None, None

# Minimum units slider (per building if merged)
min_units_default = 7
min_units = st.sidebar.number_input(
    "Minimum TOTAL flats per building (uses entire CSV)",
    min_value=0, max_value=1000, value=min_units_default
)

# -------------------- Load data --------------------
def base_addr(addr: str) -> str:
    """Strip flat/unit prefix like 'Flat 3, ' so we can group buildings."""
    s = str(addr or "")
    s = re.sub(r"^(?:Flat|Apt|Apartment)\s*\w+\s*,\s*", "", s, flags=re.I)
    return s.strip()

# Load CSV (uploaded overrides default)
if uploaded:
    df = load_epc_csv(uploaded)
else:
    df = load_epc_csv(DEFAULT_DATA_PATH)

# Fill optional columns if missing
for col in ["TOTAL_FLOOR_AREA", "LODGEMENT_DATE"]:
    if col not in df.columns:
        df[col] = ""

# Build building key and infer UNITS_PER_BUILDING if missing
if "UNITS_PER_BUILDING" not in df.columns or df["UNITS_PER_BUILDING"].isna().all():
    df["__BASE_ADDR__"] = df["ADDRESS"].map(base_addr) + " | " + df["POSTCODE"].astype(str)
    counts = df["__BASE_ADDR__"].value_counts(dropna=False)
    df["UNITS_PER_BUILDING"] = df["__BASE_ADDR__"].map(counts).astype("Int64")
else:
    df["UNITS_PER_BUILDING"] = pd.to_numeric(df["UNITS_PER_BUILDING"], errors="coerce").astype("Int64")
if "__BASE_ADDR__" not in df.columns:
    df["__BASE_ADDR__"] = df["ADDRESS"].map(base_addr) + " | " + df["POSTCODE"].astype(str)

# Coerce coords
df["LAT"] = pd.to_numeric(df["LAT"], errors="coerce")
df["LON"] = pd.to_numeric(df["LON"], errors="coerce")

# Populate EPC options now that we have data
epc_options = sorted({str(x).strip() for x in df.get("EPC_CURRENT", pd.Series(dtype=str)).dropna() if str(x).strip()})
default_epcs = ["E","F","G"] if only_efg else []
epc_selected = epc_multiselect_placeholder.multiselect("EPC Ratings (flat-level)", options=epc_options, default=default_epcs)

# Optional sliders if data present
if pd.to_numeric(df["TOTAL_FLOOR_AREA"], errors="coerce").notna().any():
    tfa_series = pd.to_numeric(df["TOTAL_FLOOR_AREA"], errors="coerce").dropna()
    tfa_min_val, tfa_max_val = int(math.floor(tfa_series.min())), int(math.ceil(tfa_series.max()))
    tfa_min, tfa_max = st.sidebar.slider("Total floor area (m²)", min_value=tfa_min_val, max_value=tfa_max_val,
                                         value=(tfa_min_val, tfa_max_val))
else:
    st.sidebar.caption("")

if pd.to_datetime(df["LODGEMENT_DATE"], errors="coerce").notna().any():
    d = pd.to_datetime(df["LODGEMENT_DATE"], errors="coerce").dropna()
    if not d.empty:
        ldr_min, ldr_max = d.min().date(), d.max().date()
        ldr_start, ldr_end = st.sidebar.date_input("Lodgement date range", value=(ldr_min, ldr_max))

# -------------------- Apply filters (sidebar) --------------------
mask = pd.Series(True, index=df.index)

if epc_selected:
    mask &= df["EPC_CURRENT"].isin(epc_selected)

if addr_contains.strip():
    mask &= df["ADDRESS"].str.contains(addr_contains.strip(), case=False, na=False)

if pc_starts.strip():
    mask &= df["POSTCODE"].astype(str).str.startswith(pc_starts.strip(), na=False)

if tfa_min is not None and tfa_max is not None:
    tfa = pd.to_numeric(df["TOTAL_FLOOR_AREA"], errors="coerce")
    mask &= (tfa >= tfa_min) & (tfa <= tfa_max)

if ldr_start and ldr_end:
    ld = pd.to_datetime(df["LODGEMENT_DATE"], errors="coerce")
    mask &= (ld.dt.date >= ldr_start) & (ld.dt.date <= ldr_end)

if min_units and int(min_units) > 0:
    mask &= (df["UNITS_PER_BUILDING"].fillna(0) >= int(min_units))

df_filt = df.loc[mask].copy()

# Merge vs individual
if display_mode.startswith("Merged"):
    # one row per building (base address + postcode), pick a representative row
    grp_cols = ["__BASE_ADDR__"]
    rep = (
        df_filt.sort_values(["UNITS_PER_BUILDING", "EPC_CURRENT"], ascending=[False, True])
              .groupby(grp_cols, as_index=False)
              .first()
    )
    view = rep.copy()
else:
    view = df_filt.copy()

# -------------------- Header (title + KPIs like the original) --------------------
st.title("Blackpool Low-EPC Dashboard")

# KPIs row
total_entries = len(df_filt)
eligible_buildings = int(view["__BASE_ADDR__"].nunique()) if "__BASE_ADDR__" in view.columns else len(view)
with_coords = int(view.dropna(subset=["LAT","LON"]).shape[0])
most_common = ""
if "EPC_CURRENT" in view.columns and not view["EPC_CURRENT"].dropna().empty:
    most_common = Counter([str(x).strip() for x in view["EPC_CURRENT"] if str(x).strip()]).most_common(1)
    most_common = most_common[0][0] if most_common else ""

k1, k2, k3, k4 = st.columns([1,1,1,1], gap="large")
with k1:
    st.metric("Entries shown", f"{total_entries:,}")
with k2:
    st.metric("Eligible buildings (shown)", f"{eligible_buildings:,}")
with k3:
    st.metric("With coordinates", f"{with_coords:,}")
with k4:
    st.metric("Most common rating", most_common or "—")

st.divider()

# -------------------- Selected Property / Building --------------------
st.subheader("Selected Property / Building")

# Build a friendly label for the selector
def nice_label(row) -> str:
    addr = str(row.get("ADDRESS","")).strip()
    pc = str(row.get("POSTCODE","")).strip()
    epc = str(row.get("EPC_CURRENT","")).strip()
    return f"{addr} [{pc}] – EPC {epc or '—'}"

if display_mode.startswith("Merged"):
    # choose first row per building for selection list
    sel_df = view.copy()
else:
    sel_df = view.copy()

options = sel_df.apply(nice_label, axis=1).tolist()
selected_label = st.selectbox(
    "Pick an item (drives Map & Street View):",
    options,
    index=0 if options else None,
    placeholder="Select a property…"
)

# Resolve the selected row
if options:
    sel_idx = options.index(selected_label)
    sel_row = sel_df.iloc[sel_idx]
else:
    sel_row = None

# -------------------- Tabs (Map • Street View • Table • Diagnostics) --------------------
tabs = st.tabs(["Map", "Street View", "Table", "Diagnostics"])

# ---- Build map pins + link columns for the *current view* ----
rows = []
for _, r in view.iterrows():
    addr = str(r["ADDRESS"]).strip()
    pc   = str(r["POSTCODE"]).strip()

    rm_url = rightmove_search_url(addr, pc)
    zp_url = zoopla_search_url(addr, pc)
    gmaps  = generic_map_query(addr, pc)

    # Broader street/block fallback
    street = addr.split(",")[1].strip() if "," in addr and len(addr.split(",")) > 1 else addr
    rm_broad = rightmove_broad_url(street, pc)
    zp_broad = zoopla_broad_url(street, pc)

    label = f"{addr} [{pc}] – EPC {str(r.get('EPC_CURRENT','')).strip() or '—'}"
    popup_html = (
        f"<b>{label}</b><br>"
        f"<a href='{gmaps}' target='_blank'>Google Maps</a> | "
        f"<a href='{rm_url}' target='_blank'>Rightmove search</a> | "
        f"<a href='{zp_url}' target='_blank'>Zoopla search</a><br>"
        f"<small>Broader: <a href='{rm_broad}' target='_blank'>RM street</a> · "
        f"<a href='{zp_broad}' target='_blank'>ZP street</a></small>"
    )

    rows.append({
        "ADDRESS": addr,
        "POSTCODE": pc,
        "EPC_CURRENT": r.get("EPC_CURRENT",""),
        "LAT": r.get("LAT"),
        "LON": r.get("LON"),
        "__LABEL__": label,
        "__POPUP_HTML__": popup_html,
        "__MARKER_COLOR__": "blue",
        "RIGHTMOVE_SEARCH": rm_url,
        "ZOOPLA_SEARCH": zp_url,
        "GOOGLE_MAPS": gmaps,
        "RM_BROAD": rm_broad,
        "ZP_BROAD"_
