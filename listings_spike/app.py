from __future__ import annotations
import os
import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from utils.epc_io import load_epc_csv
from providers.url_builders import rightmove_search_url, zoopla_search_url, generic_map_query
from utils.map_viz import draw_map

# Load environment variables (optional)
load_dotenv()

st.set_page_config(page_title="Listings Overlay Spike", layout="wide")
st.title("Listings Overlay Spike (Rightmove / Zoopla)")
st.caption("Safe sandbox – separate from your production dashboard")

# ============ Data Input ============
col1, col2 = st.columns([2, 1])
with col1:
    uploaded = st.file_uploader(
        "Upload EPC CSV (ADDRESS, POSTCODE, LAT, LON, EPC_CURRENT, EPC_POTENTIAL, UNITS_PER_BUILDING)",
        type=["csv"]
    )
with col2:
    use_sample = st.toggle("Use sample data", value=not bool(uploaded))

if uploaded:
    df = load_epc_csv(uploaded)
elif use_sample:
    sample_path = os.path.join(os.path.dirname(__file__), "data", "sample.csv")
    df = load_epc_csv(sample_path)
else:
    st.stop()

# Filters
st.sidebar.header("Filters")
min_units = st.sidebar.number_input("Min units per building", min_value=1, max_value=500, value=7)
epc_filter = st.sidebar.multiselect(
    "EPC current label", 
    options=sorted(df["EPC_CURRENT"].dropna().unique()), 
    default=None
)

mask = (df["UNITS_PER_BUILDING"] >= min_units)
if epc_filter:
    mask &= df["EPC_CURRENT"].isin(epc_filter)

view = df.loc[mask].copy()

# Build label + links
rows = []
for _, r in view.iterrows():
    addr = str(r["ADDRESS"]).strip()
    pc = str(r["POSTCODE"]).strip()

    rm_url = rightmove_search_url(addr, pc)
    zp_url = zoopla_search_url(addr, pc)
    gmaps = generic_map_query(addr, pc)

    label = f"{addr} [{pc}]"
    popup_html = f"<b>{label}</b><br>\n" \
                 f"<a href='{gmaps}' target='_blank'>Google Maps</a> | " \
                 f"<a href='{rm_url}' target='_blank'>Rightmove search</a> | " \
                 f"<a href='{zp_url}' target='_blank'>Zoopla search</a>"

    rows.append({
        "ADDRESS": addr,
        "POSTCODE": pc,
        "LAT": r.get("LAT"),
        "LON": r.get("LON"),
        "__LABEL__": label,
        "__POPUP_HTML__": popup_html,
        "__MARKER_COLOR__": "blue",
        "RIGHTMOVE_SEARCH": rm_url,
        "ZOOPLA_SEARCH": zp_url,
        "GOOGLE_MAPS": gmaps,
    })

result_df = pd.DataFrame(rows)

# Map
st.subheader("Map")
from streamlit_folium import st_folium
m = draw_map(result_df)
st_folium(m, height=600, width=1100)

# Table
st.subheader("Results")
st.dataframe(
    result_df[["ADDRESS", "POSTCODE", "RIGHTMOVE_SEARCH", "ZOOPLA_SEARCH", "GOOGLE_MAPS"]],
    use_container_width=True
)

st.caption("Blue pins = properties to check. Click links to open Rightmove/Zoopla/Google Maps.")
