from __future__ import annotations
import os
import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from utils.epc_io import load_epc_csv
from utils.map_viz import draw_map
from providers.url_builders import (
    rightmove_search_url,
    zoopla_search_url,
    generic_map_query,
    rightmove_broad_url,
    zoopla_broad_url,
)

# Load environment variables (optional)
load_dotenv()

st.set_page_config(page_title="Listings Overlay Spike", layout="wide")
st.title("Listings Overlay Spike (Rightmove / Zoopla)")
st.caption("Safe sandbox – separate from your production dashboard")

# ============ Data Input ============
# Always load the CSV from repo root
DATA_PATH = os.path.join(
    os.path.dirname(__file__), "..", "blackpool_low_epc_with_coords.csv"
)

st.markdown("**Data source:** Built-in CSV (Blackpool EPC). "
            "You can still upload another CSV to override it.")

uploaded = st.file_uploader(
    "Optional: Upload a different EPC CSV "
    "(ADDRESS, POSTCODE, LAT, LON, EPC_CURRENT, EPC_POTENTIAL, UNITS_PER_BUILDING)",
    type=["csv"]
)

if uploaded:
    df = load_epc_csv(uploaded)
else:
    df = load_epc_csv(DATA_PATH)

# ============ Filters ============
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
st.info(
    f"Showing {len(view):,} properties (min units ≥ {min_units}"
    + (f", EPC: {', '.join(epc_filter)}" if epc_filter else "")
    + ")"
)

# ============ Build label + links ============
rows = []
for _, r in view.iterrows():
    addr = str(r["ADDRESS"]).strip()
    pc = str(r["POSTCODE"]).strip()

    # Exact searches
    rm_url = rightmove_search_url(addr, pc)
    zp_url = zoopla_search_url(addr, pc)
    gmaps = generic_map_query(addr, pc)

    # Broader street/block fallback
    street = addr.split(",")[1].strip() if "," in addr and len(addr.split(",")) > 1 else addr
    rm_broad = rightmove_broad_url(street, pc)
    zp_broad = zoopla_broad_url(street, pc)

    label = f"{addr} [{pc}]"
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
        "LAT": r.get("LAT"),
        "LON": r.get("LON"),
        "__LABEL__": label,
        "__POPUP_HTML__": popup_html,
        "__MARKER_COLOR__": "blue",
        "RIGHTMOVE_SEARCH": rm_url,
        "ZOOPLA_SEARCH": zp_url,
        "GOOGLE_MAPS": gmaps,
        "RM_BROAD": rm_broad,
        "ZP_BROAD": zp_broad,
    })

result_df = pd.DataFrame(rows)

# ============ Map ============
st.subheader("Map")
from streamlit_folium import st_folium
m = draw_map(result_df)
st_folium(m, height=600, width=1100)

# ============ Table + Download ============
st.subheader("Results")
st.dataframe(
    result_df[[
        "ADDRESS", "POSTCODE",
        "RIGHTMOVE_SEARCH", "ZOOPLA_SEARCH", "GOOGLE_MAPS",
        "RM_BROAD", "ZP_BROAD"
    ]],
    use_container_width=True
)

st.download_button(
    label="Download results as CSV",
    data=result_df.to_csv(index=False).encode("utf-8"),
    file_name="listings_overlay_results.csv",
    mime="text/csv"
)

st.caption(
    "Exact address searches open first; if nothing shows, try the ‘Broader’ street-level links."
)
