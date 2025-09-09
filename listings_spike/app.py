from __future__ import annotations
import os, re
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from collections import Counter
import requests  # HTTP for geocoding, streetview metadata, lightweight checks

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

# ✅ Google API key from Streamlit Secrets
google_api_key = st.secrets.get("GOOGLE_MAPS_API_KEY")

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

# -------------------- Helpers --------------------
def _google_results_hint(url: str) -> str:
    """Heuristic: fetch the Google results page for our site-limited query."""
    try:
        r = requests.get(url, timeout=8, headers={"User-Agent": "Mozilla/5.0 (DemoApp/1.0)"})
        txt = r.text.lower()
        if "did not match any documents" in txt:
            return "miss"
        if "rightmove.co.uk" in txt or "zoopla.co.uk" in txt:
            return "hit"
        return "unknown"
    except Exception:
        return "unknown"

def _zoopla_status_via_piloterr(q: str) -> str:
    """Uses Piloterr to check Zoopla when available."""
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

# --- New: address → place_id (Geocoding API), to match Google Maps manual search ---
def _geocode_place_id(address_q: str, key: str) -> str | None:
    if not (address_q and key):
        return None
    try:
        url = "https://maps.googleapis.com/maps/api/geocode/json"
        r = requests.get(url, params={"address": address_q, "key": key}, timeout=6)
        js = r.json() if r.ok else {}
        results = js.get("results") or []
        if results:
            return results[0].get("place_id")
    except Exception:
        pass
    return None

def _streetview_best_from_location(location: str, key: str, radius_m: int = 120) -> dict:
    """
    Use Street View METADATA to find the best, most recent OUTDOOR pano near the location.
    `location` can be "place_id:XXXX" or "lat,lon".
    Returns dict with: static_img, pano_link, date, note
    """
    if not (location and key):
        return {}
    meta_url = "https://maps.googleapis.com/maps/api/streetview/metadata"
    params = {"location": location, "source": "outdoor", "radius": radius_m, "key": key}
    try:
        resp = requests.get(meta_url, params=params, timeout=6)
        meta = resp.json() if resp.ok else {}
    except Exception:
        meta = {}
    pano_id = (meta or {}).get("pano_id") or (meta or {}).get("panoId")
    date = (meta or {}).get("date")
    if pano_id:
        static_img = (
            "https://maps.googleapis.com/maps/api/streetview"
            f"?size=800x450&pano={pano_id}&key={key}"
        )
        pano_link = f"https://www.google.com/maps/@?api=1&map_action=pano&pano={pano_id}"
        return {"static_img": static_img, "pano_link": pano_link, "date": date, "note": "pano_id"}
    # fallback uses given location
    static_img = (
        "https://maps.googleapis.com/maps/api/streetview"
        f"?size=800x450&location={location}&source=outdoor&key={key}"
    )
    if location.startswith("place_id:"):
        pano_link = f"https://www.google.com/maps/search/?api=1&query={location}"
    else:
        pano_link = f"https://www.google.com/maps/@?api=1&map_action=pano&viewpoint={location}"
    return {"static_img": static_img, "pano_link": pano_link, "date": date, "note": "viewpoint-fallback"}

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

        rm_url = rightmove_search_url(addr, pc)
        zp_url = zoopla_search_url(addr, pc)
        rm_status = _google_results_hint(rm_url)
        zp_status = _zoopla_status_via_piloterr(f"{addr} {pc}")

        gmaps_search = generic_map_query(addr, pc)

        left, right = st.columns([2, 1], vertical_alignment="top")
        with left:
            if google_api_key:
                # 1) Try place_id from the full address (better match to manual Google Maps)
                place_id = _geocode_place_id(f"{addr}, {pc}", google_api_key)
                if place_id:
                    best = _streetview_best_from_location(f"place_id:{place_id}", google_api_key, radius_m=120)
                elif has_coords:
                    best = _streetview_best_from_location(f"{lat},{lon}", google_api_key, radius_m=150)
                else:
                    best = {}

                if best:
                    st.markdown(
                        f"<a href='{best['pano_link']}' target='_blank'>"
                        f"<img src='{best['static_img']}' style='width:100%; border-radius:8px;'/>"
                        f"</a>",
                        unsafe_allow_html=True,
                    )
                    cap = f"Street View near: {addr} [{pc}]"
                    if best.get("date"):
                        cap += f" — capture: {best['date']}"
                    st.caption(cap)
                else:
                    st.warning("Couldn’t fetch Street View. Opening by address instead.")
                    st.markdown(f"[Open Google Maps / Street View]({gmaps_search})")
            else:
                # No key at runtime (e.g., local dev)
                if not has_coords:
                    st.info("This row has no LAT/LON. Opening by address search instead.")
                st.markdown(f"[Open Google Maps / Street View]({gmaps_search})")

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
        "- Street View now prefers the address **place_id** via the Geocoding API, "
        "then asks Street View Metadata with `source=outdoor` and a small radius. "
        "This matches what you see when you manually search Google Maps.\n"
        "- If place_id is unavailable, it falls back to the CSV coordinates."
    )
