from __future__ import annotations
import os, re, math
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from collections import Counter

from utils.epc_io import load_epc_csv
from utils.map_viz import draw_map
from providers.url_builders import (
    rightmove_search_url,
    zoopla_search_url,
    generic_map_query,
    rightmove_broad_url,
    zoopla_broad_url,
)

# -------------------- Page setup --------------------
load_dotenv()
st.set_page_config(page_title="Blackpool Low-EPC Dashboard", layout="wide")

st.markdown("""
<style>
.block-container { padding-top: 1rem; padding-bottom: 2rem; }
.small-muted { color:#6b7280; font-size:0.9rem; }
</style>
""", unsafe_allow_html=True)

# -------------------- Sidebar --------------------
st.sidebar.header("Data source")
DEFAULT_DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "blackpool_low_epc_with_coords.csv")
uploaded = st.sidebar.file_uploader("Upload CSV (optional)", type=["csv"])
if st.sidebar.button("Refresh data"):
    st.rerun()

st.sidebar.divider()
st.sidebar.header("Filters")
google_api_key = st.sidebar.text_input("Google API key (Street View optional)", type="password")
display_mode = st.sidebar.radio("Display mode", ["Individual flats", "Merged buildings"], index=0)
only_efg = st.sidebar.checkbox("Only EPC E–G", value=False)
addr_contains = st.sidebar.text_input("Address contains", value="")
pc_starts = st.sidebar.text_input("Postcodes starts with", value="")
min_units = st.sidebar.number_input("Minimum flats per building", min_value=0, max_value=1000, value=7)

# -------------------- Load data --------------------
if uploaded:
    df = load_epc_csv(uploaded)
else:
    df = load_epc_csv(DEFAULT_DATA_PATH)

df["LAT"] = pd.to_numeric(df.get("LAT"), errors="coerce")
df["LON"] = pd.to_numeric(df.get("LON"), errors="coerce")
if "UNITS_PER_BUILDING" not in df.columns or df["UNITS_PER_BUILDING"].isna().all():
    def base_addr(s: str) -> str:
        s = str(s or "")
        return re.sub(r"^(?:Flat|Apt|Apartment)\s*\w+\s*,\s*", "", s, flags=re.I).strip()
    df["__BASE_ADDR__"] = df["ADDRESS"].map(base_addr) + " | " + df["POSTCODE"].astype(str)
    counts = df["__BASE_ADDR__"].value_counts()
    df["UNITS_PER_BUILDING"] = df["__BASE_ADDR__"].map(counts).astype("Int64")
else:
    df["UNITS_PER_BUILDING"] = pd.to_numeric(df["UNITS_PER_BUILDING"], errors="coerce").astype("Int64")

# -------------------- Apply filters --------------------
mask = (df["UNITS_PER_BUILDING"].fillna(0) >= min_units)
if only_efg:
    mask &= df["EPC_CURRENT"].isin(["E", "F", "G"])
if addr_contains:
    mask &= df["ADDRESS"].str.contains(addr_contains, case=False, na=False)
if pc_starts:
    mask &= df["POSTCODE"].astype(str).str.startswith(pc_starts)

df_filt = df.loc[mask].copy()

# Merged vs individual
if display_mode.startswith("Merged"):
    grp = df_filt.groupby("__BASE_ADDR__", as_index=False).first()
    view = grp.copy()
else:
    view = df_filt.copy()

# -------------------- Header KPIs --------------------
st.title("Blackpool Low-EPC Dashboard")
total_entries = len(df_filt)
eligible_buildings = int(view["__BASE_ADDR__"].nunique()) if "__BASE_ADDR__" in view.columns else len(view)
with_coords = int(view.dropna(subset=["LAT", "LON"]).shape[0])
most_common = Counter(view["EPC_CURRENT"].dropna()).most_common(1)
most_common = most_common[0][0] if most_common else "—"
k1, k2, k3, k4 = st.columns(4)
k1.metric("Entries shown", f"{total_entries:,}")
k2.metric("Eligible buildings", f"{eligible_buildings:,}")
k3.metric("With coordinates", f"{with_coords:,}")
k4.metric("Most common rating", most_common)

st.divider()

# -------------------- Selected Property --------------------
st.subheader("Selected Property / Building")
def nice_label(row) -> str:
    addr = str(row.get("ADDRESS", "")).strip()
    pc = str(row.get("POSTCODE", "")).strip()
    epc = str(row.get("EPC_CURRENT", "")).strip()
    return f"{addr} [{pc}] – EPC {epc or '—'}"

options = view.apply(nice_label, axis=1).tolist()
selected_label = st.selectbox("Pick an item", options, index=0 if options else None)
sel_row = view.iloc[options.index(selected_label)] if options else None

# -------------------- Build link table --------------------
rows = []
for _, r in view.iterrows():
    addr = str(r["ADDRESS"]).strip()
    pc = str(r["POSTCODE"]).strip()
    rm_url = rightmove_search_url(addr, pc)
    zp_url = zoopla_search_url(addr, pc)
    gmaps = generic_map_query(addr, pc)
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
        "EPC_CURRENT": r.get("EPC_CURRENT", ""),
        "LAT": r.get("LAT"),
        "LON": r.get("LON"),
        "__LABEL__": label,
        "__POPUP_HTML__": popup_html,
        "RIGHTMOVE_SEARCH": rm_url,
        "ZOOPLA_SEARCH": zp_url,
        "GOOGLE_MAPS": gmaps,
        "RM_BROAD": rm_broad,
        "ZP_BROAD": zp_broad,
        "UNITS_PER_BUILDING": r.get("UNITS_PER_BUILDING", ""),
    })

result_df = pd.DataFrame(rows)

# -------------------- Tabs --------------------
tabs = st.tabs(["Map", "Street View", "Table", "Diagnostics"])

with tabs[0]:
    from streamlit_folium import st_folium
    m = draw_map(result_df)
    st_folium(m, height=530, width=None)

with tabs[1]:
    if sel_row is None:
        st.info("Select a property to preview Street View.")
    else:
        addr = str(sel_row["ADDRESS"]).strip()
        pc = str(sel_row["POSTCODE"]).strip()
        lat, lon = sel_row.get("LAT"), sel_row.get("LON")
        if google_api_key and pd.notna(lat) and pd.notna(lon):
            img_url = f"https://maps.googleapis.com/maps/api/streetview?size=640x400&location={lat},{lon}&key={google_api_key}"
            st.image(img_url, caption=f"Street View near: {addr} [{pc}]")
        else:
            gmaps = generic_map_query(addr, pc)
            st.markdown(f"[Open Google Maps / Street View]({gmaps})")

with tabs[2]:
    table_cols = ["ADDRESS","POSTCODE","EPC_CURRENT","UNITS_PER_BUILDING",
                  "RIGHTMOVE_SEARCH","ZOOPLA_SEARCH","GOOGLE_MAPS","RM_BROAD","ZP_BROAD"]
    for c in table_cols:
        if c not in result_df.columns:
            result_df[c] = ""
    safe_view = result_df.reindex(columns=table_cols, fill_value="")
    st.dataframe(safe_view, use_container_width=True, height=420)
    st.download_button(
        label="Download results as CSV",
        data=safe_view.to_csv(index=False).encode("utf-8"),
        file_name="listings_overlay_results.csv",
        mime="text/csv",
        key="download_results_main"
    )

with tabs[3]:
    st.dataframe(view.head(20), use_container_width=True, height=360)
    st.code(", ".join(list(df.columns)[:80]), language="text")
