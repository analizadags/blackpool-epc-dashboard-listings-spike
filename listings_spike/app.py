from __future__ import annotations
import os, re
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from collections import Counter
import requests  # for lightweight status checks

from utils.epc_io import load_epc_csv
from utils.map_viz import draw_map
from providers.url_builders import (
    rightmove_search_url,
    zoopla_search_url,
    generic_map_query,
    rightmove_broad_url,
    zoopla_broad_url,
)

# Optional Zoopla (Piloterr) client; if file not present or no key, we fall back to "unknown"
try:
    from providers.zoopla_piloterr import PiloterrZooplaClient  # optional
except Exception:
    PiloterrZooplaClient = None

# -------------------- Page setup --------------------
load_dotenv()
st.set_page_config(page_title="Blackpool Low-EPC Dashboard", layout="wide")

st.markdown(
    """
<style>
.block-container { padding-top: 1rem; padding-bottom: 2rem; }
.small-muted { color:#6b7280; font-size:0.9rem; }
.stTabs [data-baseweb="tab-list"] { gap: .35rem; }
.stTabs [data-baseweb="tab"] { padding: .3rem .6rem; }
</style>
""",
    unsafe_allow_html=True,
)

# -------------------- Sidebar --------------------
st.sidebar.header("Data source")
DEFAULT_DATA_PATH = os.path.join(
    os.path.dirname(__file__), "..", "blackpool_low_epc_with_coords.csv"
)
uploaded = st.sidebar.file_uploader("Upload CSV (optional)", type=["csv"])
if st.sidebar.button("Refresh data"):
    st.rerun()

st.sidebar.divider()
st.sidebar.header("Filters")

google_api_key = st.sidebar.text_input(
    "Google API key (Street View optional)", type="password"
)

display_mode = st.sidebar.radio(
    "Display mode", ["Individual flats", "Merged buildings (one pin per building)"], index=0
)

# EPC quick-picks
epc_mode = st.sidebar.radio(
    "EPC filter",
    ["Any", "Only E", "Only F", "Only G", "Only E+F+G", "Custom…"],
    index=0,
)

addr_contains = st.sidebar.text_input("Address contains", value="")
pc_starts = st.sidebar.text_input("Postcodes starts with", value="")
min_units = st.sidebar.number_input(
    "Minimum TOTAL flats per building", min_value=0, max_value=1000, value=7
)

# -------------------- Load data --------------------
def base_addr(s: str) -> str:
    s = str(s or "")
    # remove flat/unit prefix like "Flat 3, " / "Apt 2, " / "Apartment 5, "
    return re.sub(r"^(?:Flat|Apt|Apartment)\s*\w+\s*,\s*", "", s, flags=re.I).strip()

if uploaded:
    df = load_epc_csv(uploaded)
else:
    df = load_epc_csv(DEFAULT_DATA_PATH)

# Ensure helpful columns exist
for col in ["TOTAL_FLOOR_AREA", "LODGEMENT_DATE"]:
    if col not in df.columns:
        df[col] = ""

# Coordinates + units
df["LAT"] = pd.to_numeric(df.get("LAT"), errors="coerce")
df["LON"] = pd.to_numeric(df.get("LON"), errors="coerce")

if "UNITS_PER_BUILDING" not in df.columns or df["UNITS_PER_BUILDING"].isna().all():
    df["__BASE_ADDR__"] = df["ADDRESS"].map(base_addr) + " | " + df["POSTCODE"].astype(str)
    counts = df["__BASE_ADDR__"].value_counts(dropna=False)
    df["UNITS_PER_BUILDING"] = df["__BASE_ADDR__"].map(counts).astype("Int64")
else:
    df["UNITS_PER_BUILDING"] = pd.to_numeric(df["UNITS_PER_BUILDING"], errors="coerce").astype("Int64")
if "__BASE_ADDR__" not in df.columns:
    df["__BASE_ADDR__"] = df["ADDRESS"].map(base_addr) + " | " + df["POSTCODE"].astype(str)

# -------------------- EPC quick-picks resolve --------------------
epc_options = sorted(
    {str(x).strip().upper() for x in df.get("EPC_CURRENT", pd.Series(dtype=str)).dropna() if str(x).strip()}
)
if not epc_options:
    epc_options = ["A", "B", "C", "D", "E", "F", "G"]

if epc_mode == "Any":
    epc_selected = []
elif epc_mode == "Only E":
    epc_selected = ["E"]
elif epc_mode == "Only F":
    epc_selected = ["F"]
elif epc_mode == "Only G":
    epc_selected = ["G"]
elif epc_mode == "Only E+F+G":
    epc_selected = ["E", "F", "G"]
else:  # Custom…
    epc_selected = st.sidebar.multiselect(
        "Choose EPC ratings", options=epc_options, default=["E", "F", "G"]
    )

# -------------------- Apply filters --------------------
mask = (df["UNITS_PER_BUILDING"].fillna(0) >= int(min_units))

if epc_selected:
    mask &= df["EPC_CURRENT"].astype(str).str.upper().isin(epc_selected)

if addr_contains.strip():
    mask &= df["ADDRESS"].str.contains(addr_contains.strip(), case=False, na=False)

if pc_starts.strip():
    mask &= df["POSTCODE"].astype(str).str.startswith(pc_starts.strip())

df_filt = df.loc[mask].copy()

# Merged vs individual
if display_mode.startswith("Merged"):
    # One row per building (choose a representative row per __BASE_ADDR__)
    view = (
        df_filt.sort_values(["UNITS_PER_BUILDING", "EPC_CURRENT"], ascending=[False, True])
        .groupby("__BASE_ADDR__", as_index=False)
        .first()
        .copy()
    )
else:
    view = df_filt.copy()

# -------------------- Header KPIs --------------------
st.title("Blackpool Low-EPC Dashboard")

total_entries = len(df_filt)
eligible_buildings = int(view["__BASE_ADDR__"].nunique()) if "__BASE_ADDR__" in view.columns else len(view)
with_coords = int(view.dropna(subset=["LAT", "LON"]).shape[0])
most_common = ""
if "EPC_CURRENT" in view.columns and not view["EPC_CURRENT"].dropna().empty:
    mc = Counter([str(x).strip().upper() for x in view["EPC_CURRENT"] if str(x).strip()])
    most_common = mc.most_common(1)[0][0] if mc else "—"
else:
    most_common = "—"

k1, k2, k3, k4 = st.columns(4)
k1.metric("Entries shown", f"{total_entries:,}")
k2.metric("Eligible buildings (shown)", f"{eligible_buildings:,}")
k3.metric("With coordinates", f"{with_coords:,}")
k4.metric("Most common rating", most_common)

st.divider()

# -------------------- Selected Property / Building --------------------
st.subheader("Selected Property / Building")

def nice_label(row) -> str:
    addr = str(row.get("ADDRESS", "")).strip()
    pc = str(row.get("POSTCODE", "")).strip()
    epc = str(row.get("EPC_CURRENT", "")).strip().upper()
    return f"{addr} [{pc}] – EPC {epc or '—'}"

options = view.apply(nice_label, axis=1).tolist()
selected_label = st.selectbox(
    "Pick an item (drives Map & Street View):",
    options,
    index=0 if options else None,
    placeholder="Select a property…",
)
sel_row = view.iloc[options.index(selected_label)] if options else None

# -------------------- Build link table for current view --------------------
rows = []
for _, r in view.iterrows():
    addr = str(r.get("ADDRESS", "")).strip()
    pc = str(r.get("POSTCODE", "")).strip()

    rm_url = rightmove_search_url(addr, pc)
    zp_url = zoopla_search_url(addr, pc)
    gmaps = generic_map_query(addr, pc)

    street = addr.split(",")[1].strip() if "," in addr and len(addr.split(",")) > 1 else addr
    rm_broad = rightmove_broad_url(street, pc)
    zp_broad = zoopla_broad_url(street, pc)

    label = f"{addr} [{pc}] – EPC {str(r.get('EPC_CURRENT','')).strip().upper() or '—'}"
    popup_html = (
        f"<b>{label}</b><br>"
        f"<a href='{gmaps}' target='_blank'>Google Maps</a> | "
        f"<a href='{rm_url}' target='_blank'>Rightmove search</a> | "
        f"<a href='{zp_url}' target='_blank'>Zoopla search</a><br>"
        f"<small>Broader: <a href='{rm_broad}' target='_blank'>RM street</a> · "
        f"<a href='{zp_broad}' target='_blank'>ZP street</a></small>"
    )

    rows.append(
        {
            "ADDRESS": addr,
            "POSTCODE": pc,
            "EPC_CURRENT": r.get("EPC_CURRENT", ""),
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
            "UNITS_PER_BUILDING": r.get("UNITS_PER_BUILDING", ""),
        }
    )

result_df = pd.DataFrame(rows)

# -------------------- Lightweight "for sale" status helpers --------------------
def _google_results_hint(url: str) -> str:
    """
    Heuristic: fetch the Google results page for our site-limited query.
    Returns: 'hit' | 'miss' | 'unknown'
    """
    try:
        r = requests.get(
            url, timeout=8,
            headers={"User-Agent": "Mozilla/5.0 (compatible; DemoApp/1.0)"}
        )
        txt = r.text.lower()
        if "did not match any documents" in txt:
            return "miss"
        if "rightmove.co.uk" in txt or "zoopla.co.uk" in txt:
            return "hit"
        return "unknown"
    except Exception:
        return "unknown"

def _zoopla_status_via_piloterr(q: str) -> str:
    """
    Uses Piloterr to check Zoopla. Needs PILOTERR_API_KEY in env/Secrets.
    Returns: 'hit' | 'miss' | 'unknown'
    """
    if PiloterrZooplaClient is None:
        return "unknown"
    try:
        client = PiloterrZooplaClient()
        items = client.search(q) or []
        return "hit" if len(items) > 0 else "miss"
    except Exception:
        return "unknown"

def _status_badge(label: str, status: str, url: str) -> str:
    icon = "✅" if status == "hit" else ("❌" if status == "miss" else "➖")
    tip  = "found" if status == "hit" else ("not found" if status == "miss" else "unknown")
    return f"**{label}:** {icon} <small>({tip})</small> · <a href='{url}' target='_blank'>open</a>"

# -------------------- Tabs --------------------
tabs = st.tabs(["Map", "Street View", "Table", "Diagnostics"])

with tabs[0]:
    from streamlit_folium import st_folium
    st.write("")  # spacer
    m = draw_map(result_df)
    st_folium(m, height=530, width=None)

with tabs[1]:
    st.write("")
    if sel_row is None:
        st.info("Select a property to preview Street View.")
    else:
        addr = str(sel_row.get("ADDRESS", "")).strip()
        pc = str(sel_row.get("POSTCODE", "")).strip()
        lat, lon = sel_row.get("LAT"), sel_row.get("LON")
        has_coords = pd.notna(lat) and pd.notna(lon)

        # exact search URLs
        rm_url = rightmove_search_url(addr, pc)
        zp_url = zoopla_search_url(addr, pc)

        # status checks
        rm_status = _google_results_hint(rm_url)
        zp_status = _zoopla_status_via_piloterr(f"{addr} {pc}")

        # Build Google Maps links (interactive, always "current")
        gmaps_search = generic_map_query(addr, pc)
        gmaps_pano = (
            f"https://www.google.com/maps/@?api=1&map_action=pano&viewpoint={lat},{lon}"
            if has_coords else gmaps_search
        )

        left, right = st.columns([2, 1], vertical_alignment="top")
        with left:
            # If we have coords+key, show static image BUT make it CLICKABLE to open interactive Street View
            if google_api_key and has_coords:
                static_img = (
                    "https://maps.googleapis.com/maps/api/streetview"
                    f"?size=800x450&location={lat},{lon}&key={google_api_key}"
                )
                # clickable image -> opens live Google Maps Street View (always up-to-date)
                st.markdown(
                    f"<a href='{gmaps_pano}' target='_blank'>"
                    f"<img src='{static_img}' style='width:100%; border-radius:8px;'/>"
                    f"</a>",
                    unsafe_allow_html=True,
                )
                st.caption(f"Street View near: {addr} [{pc}] — click image to open interactive view.")
                st.markdown(f"[Open Google Maps by address]({gmaps_search})")
            else:
                # No key or no coords -> give direct interactive link(s)
                if not has_coords:
                    st.info("This row has no LAT/LON. Opening by address search instead.")
                st.markdown(f"[Open interactive Street View / Google Maps]({gmaps_pano})")
                st.markdown(f"[Open Google Maps by address]({gmaps_search})")

        with right:
            st.markdown("### For Sale status")
            st.markdown(_status_badge("Rightmove", rm_status, rm_url), unsafe_allow_html=True)
            st.markdown(_status_badge("Zoopla",    zp_status, zp_url), unsafe_allow_html=True)
            st.caption(
                "Rightmove status uses a simple Google-results hint. "
                "Zoopla status uses Piloterr if configured; otherwise shows unknown (➖)."
            )

with tabs[2]:
    st.write("")
    table_cols = [
        "ADDRESS",
        "POSTCODE",
        "EPC_CURRENT",
        "UNITS_PER_BUILDING",
        "RIGHTMOVE_SEARCH",
        "ZOOPLA_SEARCH",
        "GOOGLE_MAPS",
        "RM_BROAD",
        "ZP_BROAD",
    ]
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
        key="download_results_main",
    )

with tabs[3]:
    st.write("")
    st.markdown("**Data snapshot**")
    st.dataframe(view.head(20), use_container_width=True, height=360)
    st.markdown("---")
    st.markdown("**Columns present**")
    st.code(", ".join(list(df.columns)[:80]), language="text")
    st.markdown("---")
    st.markdown(
        "- Buildings are grouped by stripping a flat/unit prefix (e.g., `Flat 3,`) "
        "and combining base address with postcode.\n"
        "- If `UNITS_PER_BUILDING` was missing, it is estimated by counting flats per building."
    )
