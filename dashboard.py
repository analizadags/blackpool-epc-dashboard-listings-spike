import streamlit as st
import pandas as pd
import re
import folium
from folium.plugins import MarkerCluster
from streamlit_folium import st_folium
from urllib.parse import quote_plus
import requests

st.set_page_config(page_title="Blackpool Low-EPC Flats", layout="wide")

# ---------- Load data ----------
@st.cache_data
def load_data():
    df = pd.read_csv("blackpool_low_epc_with_coords.csv")
    # Normalize columns to UPPER CASE for consistency
    df.columns = df.columns.str.upper()

    # Coerce numeric types if present
    for col in ["LAT", "LON", "TOTAL_FLOOR_AREA"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Parse date (dayfirst=True for UK-style dates)
    if "LODGEMENT_DATE" in df.columns:
        df["LODGEMENT_DATE"] = pd.to_datetime(
            df["LODGEMENT_DATE"], errors="coerce", dayfirst=True
        )
    return df

df = load_data()

st.title("🏢 Blackpool Low-EPC Dashboard")
st.caption("Filters on the left. Use the tabs to switch between Map, Street View, and Table.")

# ---------- Helpers ----------
def base_address(addr: str) -> str:
    """
    Strip out 'Flat 3,' / 'Apartment 5,' prefixes to get a building-level address.
    """
    s = str(addr).strip()
    s = re.sub(r'^\s*(flat|apartment|apt)\s*\d+[a-zA-Z]?\s*,\s*', '', s, flags=re.IGNORECASE)
    return s.strip()

if {"ADDRESS", "POSTCODE"}.issubset(df.columns):
    df["BASE_ADDRESS"] = df["ADDRESS"].apply(base_address)
    df["BUILDING_ID"]  = (df["BASE_ADDRESS"].fillna("") + " | " + df["POSTCODE"].fillna("")).str.strip()
else:
    # Fallbacks to avoid crashes if CSV variant lacks these columns
    if "ADDRESS" not in df.columns:
        df["ADDRESS"] = ""
    if "POSTCODE" not in df.columns:
        df["POSTCODE"] = ""
    df["BASE_ADDRESS"] = df["ADDRESS"].apply(base_address)
    df["BUILDING_ID"]  = (df["BASE_ADDRESS"].fillna("") + " | " + df["POSTCODE"].fillna("")).str.strip()

def color_for(rating: str) -> str:
    """Map EPC to folium icon color."""
    r = str(rating).upper()
    if r == "G":
        return "red"
    if r == "F":
        return "orange"
    return "blue"  # D/E/other

def worst_epc(series: pd.Series):
    """Return worst (lowest) EPC among D/E/F/G."""
    order = {"D":1, "E":2, "F":3, "G":4}
    mapped = series.map(lambda x: order.get(str(x).upper(), None))
    x = mapped.max(skipna=True)
    rev = {v: k for k, v in order.items()}
    return rev.get(x, None)

def epc_mix(series: pd.Series) -> str:
    vc = series.astype(str).str.upper().value_counts()
    return f"D{vc.get('D',0)} E{vc.get('E',0)} F{vc.get('F',0)} G{vc.get('G',0)}"

def make_label_flat(row: pd.Series) -> str:
    addr = str(row.get("ADDRESS", ""))
    pc   = "" if pd.isna(row.get("POSTCODE")) else str(row.get("POSTCODE"))
    rtg  = str(row.get("CURRENT_ENERGY_RATING", ""))
    return f"{addr}  [{pc}]  – EPC {rtg}"

def make_label_building(row: pd.Series) -> str:
    addr  = str(row.get("ADDRESS", ""))
    pc    = "" if pd.isna(row.get("POSTCODE")) else str(row.get("POSTCODE"))
    worst = str(row.get("WORST_EPC", ""))
    n     = int(row.get("N_UNITS", 0))
    return f"{addr}  [{pc}]  – {n} units – worst EPC {worst}"

# ---------- Sidebar ----------
st.sidebar.header("Filters")

# Refresh button to reload data (and clear cache)
if st.sidebar.button("🔄 Refresh data"):
    st.cache_data.clear()
    st.rerun()

# Google Maps API key (read from secrets by default)
default_key = st.secrets.get("GOOGLE_MAPS_API_KEY", "")
api_key     = st.sidebar.text_input("Google API key (for Street View)", value=default_key, type="password")

flats_only    = st.sidebar.checkbox("Flats only", value=True)
ratings_only  = st.sidebar.checkbox("Only EPC D–G", value=True)
seven_plus    = st.sidebar.checkbox("Require ≥7 units per building", value=False)
rating_filter = st.sidebar.multiselect("EPC Ratings", ["D","E","F","G"], default=["D","E","F","G"])

has_area = ("TOTAL_FLOOR_AREA" in df.columns) and df["TOTAL_FLOOR_AREA"].notna().any()
if has_area:
    valid = df["TOTAL_FLOOR_AREA"].dropna()
    min_a, max_a = int(valid.min()), int(valid.max())
    area_range = st.sidebar.slider("Total floor area (m²)", min_a, max_a, (min_a, max_a))
else:
    st.sidebar.info("No floor area data available.")
    area_range = None

# Merge mode (auto-default on when seven_plus is checked)
merge_buildings = st.sidebar.checkbox(
    "Merge flats into one location (one pin per building)",
    value=seven_plus
)

# ---------- Apply filters at flat level ----------
df_f = df.copy()

if flats_only and "PROPERTY_TYPE" in df_f.columns:
    df_f = df_f[df_f["PROPERTY_TYPE"].astype(str).str.contains("Flat", case=False, na=False)]

if ratings_only and "CURRENT_ENERGY_RATING" in df_f.columns:
    df_f = df_f[df_f["CURRENT_ENERGY_RATING"].astype(str).str.upper().isin(rating_filter)]

if has_area:
    df_f = df_f[df_f["TOTAL_FLOOR_AREA"].between(area_range[0], area_range[1])]

# ---------- Build display dataframe ----------
if merge_buildings:
    # Ensure the columns we need exist
    for col in ["BUILDING_ID", "BASE_ADDRESS", "POSTCODE"]:
        if col not in df_f.columns:
            st.error(f"Required column '{col}' is missing for merge mode.")
            st.stop()

    # Base row per building + coords (first non-null)
    base = (
        df_f.sort_values(["BASE_ADDRESS", "POSTCODE"])
            .groupby("BUILDING_ID", as_index=False)
            .first()[["BUILDING_ID","BASE_ADDRESS","POSTCODE","LAT","LON"]]
            .rename(columns={"BASE_ADDRESS":"ADDRESS"})
    )

    # Aggregations
    counts = df_f.groupby("BUILDING_ID").size().rename("N_UNITS").reset_index()
    worst  = df_f.groupby("BUILDING_ID")["CURRENT_ENERGY_RATING"].apply(worst_epc).rename("WORST_EPC").reset_index()
    mix    = df_f.groupby("BUILDING_ID")["CURRENT_ENERGY_RATING"].apply(epc_mix).rename("RATING_MIX").reset_index()

    df_display = base.merge(counts, on="BUILDING_ID").merge(worst, on="BUILDING_ID").merge(mix, on="BUILDING_ID")

    # Apply ≥7 filter at building level if requested
    if seven_plus:
        df_display = df_display[df_display["N_UNITS"] >= 7]

    df_display["__LABEL__"] = df_display.apply(make_label_building, axis=1)

else:
    # If ≥7 filter is on but we are NOT merging, keep only flats in buildings that have ≥7 units
    if seven_plus and "BUILDING_ID" in df_f.columns:
        b_counts = df_f["BUILDING_ID"].value_counts()
        df_f = df_f[df_f["BUILDING_ID"].isin(b_counts[b_counts >= 7].index)]

    df_display = df_f.copy()
    df_display["__LABEL__"] = df_display.apply(make_label_flat, axis=1)

st.caption(f"Entries shown: **{len(df_display)}**")
if df_display.empty:
    st.warning("No properties match the filters above.")
    st.stop()

# ---------- KPI row ----------
with st.container():
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Entries shown", len(df_display))
    with c2:
        n_bldgs = (df_f["BUILDING_ID"].nunique() if "BUILDING_ID" in df_f.columns else 0)
        st.metric("Unique buildings (in data)", n_bldgs)
    with c3:
        if {"LAT","LON"}.issubset(df_display.columns):
            with_coords = df_display[["LAT","LON"]].dropna().shape[0]
        else:
            with_coords = 0
        st.metric("With coordinates", with_coords)
    with c4:
        if merge_buildings and "WORST_EPC" in df_display.columns and not df_display["WORST_EPC"].dropna().empty:
            st.metric("Worst EPC (mode)", df_display["WORST_EPC"].mode().iloc[0])
        elif (not merge_buildings) and "CURRENT_ENERGY_RATING" in df_display.columns:
            rc = df_display["CURRENT_ENERGY_RATING"].value_counts(dropna=False)
            st.metric("Most common rating", (rc.index[0] if not rc.empty else "—"))
        else:
            st.metric("Most common rating", "—")

# ---------- Stable selection ----------
if "__SEL_LABEL__" not in st.session_state:
    st.session_state["__SEL_LABEL__"] = df_display["__LABEL__"].iloc[0]
if st.session_state["__SEL_LABEL__"] not in set(df_display["__LABEL__"]):
    st.session_state["__SEL_LABEL__"] = df_display["__LABEL__"].iloc[0]

st.subheader("Selected Property / Building")
sel_label = st.selectbox(
    "Pick an item (drives Map & Street View):",
    options=list(df_display["__LABEL__"]),
    index=list(df_display["__LABEL__"]).index(st.session_state["__SEL_LABEL__"])
)
st.session_state["__SEL_LABEL__"] = sel_label
selected = df_display[df_display["__LABEL__"] == st.session_state["__SEL_LABEL__"]].iloc[0].to_dict()

# Quick details in sidebar
with st.sidebar:
    st.markdown("### Selected")
    if merge_buildings:
        st.write(
            f"**{selected.get('ADDRESS','')}**\n\n"
            f"Postcode: {selected.get('POSTCODE','')}\n\n"
            f"Units: {int(selected.get('N_UNITS',0))}\n\n"
            f"Worst EPC: {selected.get('WORST_EPC','')}\n\n"
            f"Mix: {selected.get('RATING_MIX','')}"
        )
    else:
        st.write(
            f"**{selected.get('ADDRESS','')}**\n\n"
            f"Postcode: {selected.get('POSTCODE','')}\n\n"
            f"EPC: {selected.get('CURRENT_ENERGY_RATING','')}\n\n"
            f"Building: {selected.get('BUILDING_ID','')}"
        )

# ---------- Street View helpers ----------
@st.cache_data(show_spinner=False)
def streetview_metadata(location_param: str, api_key: str):
    try:
        resp = requests.get(
            "https://maps.googleapis.com/maps/api/streetview/metadata",
            params={"location": location_param, "key": api_key, "source": "outdoor"},
            timeout=8,
        )
        return resp.json()
    except Exception as e:
        return {"status": "ERROR", "error_message": str(e)}

def find_working_streetview(lat, lon, address, postcode, api_key):
    candidates = []
    if pd.notna(lat) and pd.notna(lon):
        # Try nearby offsets around the point
        offsets = [
            (0.0, 0.0),
            (0.0003, 0.0), (-0.0003, 0.0), (0.0, 0.0003), (0.0, -0.0003),
            (0.0003, 0.0003), (0.0003, -0.0003), (-0.0003, 0.0003), (-0.0003, -0.0003),
        ]
        for dy, dx in offsets:
            candidates.append(f"{lat + dy},{lon + dx}")
    # Try address as a final fallback
    addr_str = f"{address}, {postcode}, Blackpool, UK"
    candidates.append(addr_str)

    last_meta = None
    for loc in candidates:
        meta = streetview_metadata(loc, api_key)
        last_meta = meta
        if meta.get("status") == "OK":
            loc_meta = meta.get("location", {}) or {}
            snap_lat = loc_meta.get("lat", lat if pd.notna(lat) else None)
            snap_lon = loc_meta.get("lng", lon if pd.notna(lon) else None)
            return True, snap_lat, snap_lon, meta, loc
    return False, None, None, last_meta or {}, None

# ---------- Tabs ----------
tab_map, tab_sv, tab_tbl, tab_diag = st.tabs(["🗺️ Map", "📷 Street View", "📋 Table", "🔧 Diagnostics"])

# --- Map tab ---
with tab_map:
    st.subheader("Map")

    lat_sel, lon_sel = selected.get("LAT"), selected.get("LON")
    if pd.notna(lat_sel) and pd.notna(lon_sel):
        center, zoom = [float(lat_sel), float(lon_sel)], 16
    else:
        center, zoom = [53.8175, -3.05], 12

    m = folium.Map(location=center, zoom_start=zoom)
    cluster = MarkerCluster().add_to(m)

    # highlight selected
    sel_key = (selected.get("ADDRESS",""), selected.get("POSTCODE",""))

    for _, row in df_display.dropna(subset=["LAT","LON"]).iterrows():
        if merge_buildings:
            rating = str(row.get("WORST_EPC",""))
            n_units = int(row.get("N_UNITS",0))
            mix = str(row.get("RATING_MIX",""))
        else:
            rating = str(row.get("CURRENT_ENERGY_RATING",""))
            n_units = None
            mix = ""

        # Street View link from each pin (opens in Google Maps)
        gsv_url = (
            "https://www.google.com/maps/@?api=1&map_action=pano"
            f"&viewpoint={row['LAT']},{row['LON']}"
        )

        popup_html = (
            f"<div style='font-size:14px; line-height:1.2;'>"
            f"<strong>{row.get('ADDRESS','(no address)')}</strong><br>"
            f"Postcode: {row.get('POSTCODE','')}<br>"
            + (f"Units: {n_units}<br>" if merge_buildings else "")
            + (f"Mix: {mix}<br>" if merge_buildings else f"EPC: {rating}<br>")
            + f"<br><a href='{gsv_url}' target='_blank' rel='noopener noreferrer'>🧭 Open Street View</a>"
            f"</div>"
        )

        is_sel = (row.get("ADDRESS",""), row.get("POSTCODE","")) == sel_key
        if is_sel:
            folium.CircleMarker(
                [row["LAT"], row["LON"]],
                radius=10, color="#2b8a3e", fill=True, fill_opacity=0.7
            ).add_to(m)

        folium.Marker(
            [row["LAT"], row["LON"]],
            popup=folium.Popup(popup_html, max_width=300),
            icon=folium.Icon(color=color_for(rating))
        ).add_to(cluster)

    map_key = f"map_{st.session_state['__SEL_LABEL__']}_{len(df_display)}_{int(merge_buildings)}"
    st_folium(m, width=1100, height=580, key=map_key)

# --- Street View tab ---
with tab_sv:
    st.subheader("Street View")
    address  = selected.get("ADDRESS","")
    postcode = "" if pd.isna(selected.get("POSTCODE")) else str(selected.get("POSTCODE"))
    rating   = selected.get("WORST_EPC","") if merge_buildings else selected.get("CURRENT_ENERGY_RATING","")
    lat      = selected.get("LAT")
    lon      = selected.get("LON")
    st.write(f"**{address} – EPC {rating if rating else '—'}**")

    if api_key:
        heading = st.slider("Heading (°)", 0, 360, 210, 1)
        pitch   = st.slider("Pitch (°)",  -90,  90,  10, 1)
        fov     = st.slider("FOV (°)",     30, 120,  80, 1)

        ok, snap_lat, snap_lon, meta, used_param = find_working_streetview(lat, lon, address, postcode, api_key)

        if ok and (snap_lat is not None and snap_lon is not None):
            loc_param = f"{snap_lat},{snap_lon}"
            img_url = (
                "https://maps.googleapis.com/maps/api/streetview"
                f"?size=640x400&location={loc_param}"
                f"&heading={heading}&pitch={pitch}&fov={fov}&source=outdoor&key={api_key}"
            )
            gsv_url = (
                "https://www.google.com/maps/@?api=1&map_action=pano"
                f"&viewpoint={snap_lat},{snap_lon}&heading={heading}&pitch={pitch}&fov={fov}"
            )

            # Fetch once so we can surface helpful errors (403/denied/etc.)
            try:
                r = requests.get(img_url, timeout=10)
                ctype = r.headers.get("Content-Type","")
                if r.ok and ctype.startswith("image/"):
                    st.markdown(
                        f'<a href="{gsv_url}" target="_blank" rel="noopener noreferrer">'
                        f'<img src="{img_url}" alt="Street View" style="max-width:100%;height:auto;border-radius:6px;" /></a>',
                        unsafe_allow_html=True
                    )
                else:
                    st.warning("Street View image couldn’t be fetched.")
                    st.code((r.text or str(r.status_code))[:400], language="text")
                    st.markdown(f"[Open in Google Maps Street View]({gsv_url})")
            except Exception as e:
                st.error(f"Street View fetch failed: {e}")
                st.markdown(f"[Open in Google Maps Street View]({gsv_url})")

            st.caption(address)
            date = meta.get("date")
            if date:
                st.caption(f"Street View capture date: {date}")
        else:
            status = (meta or {}).get("status", "UNKNOWN")
            msg = {
                "ZERO_RESULTS": "No Street View found near this point.",
                "REQUEST_DENIED": "Street View request denied (check API key restrictions).",
                "OVER_QUERY_LIMIT": "Daily quota exceeded for this API key.",
                "INVALID_REQUEST": "Invalid Street View request parameters.",
                "ERROR": "Network or API error while checking Street View.",
            }.get(status, f"Street View not available (status: {status}).")
            st.warning(msg)
            gsv_fallback = (
                "https://www.google.com/maps/search/?api=1&query="
                f"{quote_plus(f'{address}, {postcode}, Blackpool, UK')}&layer=c"
            )
            st.markdown(f"[Open in Google Maps Street View]({gsv_fallback})")
    else:
        gmaps_url = f"https://www.google.com/maps?q={quote_plus(address + ' ' + postcode + ' Blackpool UK')}&layer=c"
        st.info("Enter your Google API key in the sidebar to show Street View images.")
        st.markdown(f"[Open Street View for this address]({gmaps_url})")

# --- Table tab ---
with tab_tbl:
    st.subheader("Results")

    desired_merge = ["ADDRESS","POSTCODE","N_UNITS","WORST_EPC","RATING_MIX","LAT","LON"]
    desired_flat  = ["ADDRESS","BASE_ADDRESS","BUILDING_ID","POSTCODE",
                     "CURRENT_ENERGY_RATING","LODGEMENT_DATE","TOTAL_FLOOR_AREA","LAT","LON"]
    desired = desired_merge if merge_buildings else desired_flat
    available = [c for c in desired if c in df_display.columns]

    if not available:
        st.warning("Expected columns for this mode weren’t found. Showing all columns for debugging.")
        st.caption("Available columns: " + ", ".join(map(str, df_display.columns)))
        st.dataframe(df_display, use_container_width=True, height=480)
        csv_source = df_display
    else:
        st.dataframe(df_display[available], use_container_width=True, height=480)
        csv_source = df_display[available]

    csv_bytes = csv_source.to_csv(index=False).encode("utf-8")
    st.download_button("Download current view (CSV)", data=csv_bytes,
                       file_name="blackpool_epc_view.csv", mime="text/csv")

# --- Diagnostics tab ---
with tab_diag:
    st.subheader("Diagnostics")
    st.write("Raw df shape:", df.shape)
    st.write("Raw df columns:", list(df.columns))
    st.write("Display df shape:", df_display.shape)
    st.write("Display df columns:", list(df_display.columns))
    if {"LAT","LON"}.issubset(df_display.columns):
        st.write("Entries with coords:", int(df_display[["LAT","LON"]].dropna().shape[0]))
    st.write("Sample (first 10 rows of display df):")
    st.dataframe(df_display.head(10), use_container_width=True)
