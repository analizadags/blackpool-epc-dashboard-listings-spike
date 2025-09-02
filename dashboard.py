import streamlit as st
import pandas as pd
import re
import folium
from folium.plugins import MarkerCluster
from streamlit_folium import st_folium
from urllib.parse import quote_plus
import requests

st.set_page_config(page_title="Blackpool Low-EPC Flats", layout="wide")

# =========================================
# Data loading (built-in file or uploaded)
# =========================================
@st.cache_data
def _load_builtin():
    df = pd.read_csv("blackpool_low_epc_with_coords.csv")
    return df

def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = df.columns.str.upper()
    for col in ["LAT", "LON", "TOTAL_FLOOR_AREA"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    if "LODGEMENT_DATE" in df.columns:
        df["LODGEMENT_DATE"] = pd.to_datetime(df["LODGEMENT_DATE"], errors="coerce", dayfirst=True)
    return df

def _base_address(addr: str) -> str:
    s = str(addr).strip()
    s = re.sub(r'^\s*(flat|apartment|apt)\s*\d+[a-zA-Z]?\s*,\s*', '', s, flags=re.IGNORECASE)
    return s.strip()

def _ensure_building_fields(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "ADDRESS" not in df.columns:
        df["ADDRESS"] = ""
    if "POSTCODE" not in df.columns:
        df["POSTCODE"] = ""
    df["BASE_ADDRESS"] = df["ADDRESS"].apply(_base_address)
    df["BUILDING_ID"] = (df["BASE_ADDRESS"].fillna("") + " | " + df["POSTCODE"].fillna("")).str.strip()
    return df

def _apply_column_mapping(df: pd.DataFrame, mapping: dict) -> pd.DataFrame:
    """
    Map arbitrary CSV columns to the app's expected names without losing data.
    mapping keys: address, postcode, lat, lon, rating, property_type, total_floor_area, lodgement_date
    """
    df = df.copy()
    up = {c.upper(): c for c in df.columns}
    def pick(key):
        v = mapping.get(key) or ""
        return up.get(str(v).upper(), None) if v else None

    col_address    = pick("address")
    col_postcode   = pick("postcode")
    col_lat        = pick("lat")
    col_lon        = pick("lon")
    col_rating     = pick("rating")
    col_prop_type  = pick("property_type")
    col_tfa        = pick("total_floor_area")
    col_lodge_date = pick("lodgement_date")

    if col_address:    df["ADDRESS"] = df[col_address]
    if col_postcode:   df["POSTCODE"] = df[col_postcode]
    if col_lat:        df["LAT"] = pd.to_numeric(df[col_lat], errors="coerce")
    if col_lon:        df["LON"] = pd.to_numeric(df[col_lon], errors="coerce")
    if col_rating:     df["CURRENT_ENERGY_RATING"] = df[col_rating].astype(str).str.upper()
    if col_prop_type:  df["PROPERTY_TYPE"] = df[col_prop_type]
    if col_tfa:        df["TOTAL_FLOOR_AREA"] = pd.to_numeric(df[col_tfa], errors="coerce")
    if col_lodge_date: df["LODGEMENT_DATE"] = pd.to_datetime(df[col_lodge_date], errors="coerce", dayfirst=True)

    df = _ensure_building_fields(df)
    return df

def load_dataframe():
    st.sidebar.subheader("Data source")
    uploaded = st.sidebar.file_uploader("Upload CSV (optional)", type=["csv"])
    if uploaded is not None:
        raw = pd.read_csv(uploaded)
        df_up = _normalize_columns(raw)

        with st.sidebar.expander("Column mapping", expanded=True):
            cols = [""] + list(df_up.columns)

            st.caption("Map your CSV columns to fields used by the app:")
            c1, c2 = st.columns(2)
            with c1:
                addr = st.selectbox("Address", options=cols, index=(cols.index("ADDRESS") if "ADDRESS" in cols else 0), key="map_addr")
                pcd  = st.selectbox("Postcode", options=cols, index=(cols.index("POSTCODE") if "POSTCODE" in cols else 0), key="map_pcd")
                lat  = st.selectbox("Latitude", options=cols, index=(cols.index("LAT") if "LAT" in cols else 0), key="map_lat")
                lon  = st.selectbox("Longitude", options=cols, index=(cols.index("LON") if "LON" in cols else 0), key="map_lon")
            with c2:
                rat  = st.selectbox("EPC Rating", options=cols, index=(cols.index("CURRENT_ENERGY_RATING") if "CURRENT_ENERGY_RATING" in cols else 0), key="map_rat")
                pty  = st.selectbox("Property Type", options=cols, index=(cols.index("PROPERTY_TYPE") if "PROPERTY_TYPE" in cols else 0), key="map_pty")
                tfa  = st.selectbox("Total floor area", options=cols, index=(cols.index("TOTAL_FLOOR_AREA") if "TOTAL_FLOOR_AREA" in cols else 0), key="map_tfa")
                lodg = st.selectbox("Lodgement date", options=cols, index=(cols.index("LODGEMENT_DATE") if "LODGEMENT_DATE" in cols else 0), key="map_ldg")

        mapping = dict(
            address=addr or "", postcode=pcd or "", lat=lat or "", lon=lon or "",
            rating=rat or "", property_type=pty or "", total_floor_area=tfa or "", lodgement_date=lodg or "",
        )
        df = _apply_column_mapping(df_up, mapping)
    else:
        df = _normalize_columns(_load_builtin())
        df = _ensure_building_fields(df)
    return df

# load data (uploaded or bundled)
df = load_dataframe()

st.title("🏢 Blackpool Low-EPC Dashboard")
st.caption("Upload a CSV to work with your own data, or use the bundled dataset. Filters, map and Street View apply to whichever dataset is active.")

# =========================================
# Helpers
# =========================================
def color_for(rating: str) -> str:
    r = str(rating).upper()
    if r == "G": return "red"
    if r == "F": return "orange"
    return "blue"  # D/E/other

def worst_epc(series: pd.Series):
    order = {"D":1, "E":2, "F":3, "G":4}
    mapped = series.astype(str).str.upper().map(order)
    x = mapped.max(skipna=True)
    rev = {v:k for k,v in order.items()}
    return rev.get(x, None)

def epc_mix(series: pd.Series) -> str:
    vc = series.astype(str).str.upper().value_counts()
    return f"D{vc.get('D',0)} E{vc.get('E',0)} F{vc.get('F',0)} G{vc.get('G',0)}"

def make_label_flat(row: pd.Series) -> str:
    addr = str(row.get("ADDRESS",""))
    pc   = "" if pd.isna(row.get("POSTCODE")) else str(row.get("POSTCODE"))
    rtg  = str(row.get("CURRENT_ENERGY_RATING",""))
    return f"{addr}  [{pc}]  – EPC {rtg}"

def make_label_building(row: pd.Series) -> str:
    addr  = str(row.get("ADDRESS",""))
    pc    = "" if pd.isna(row.get("POSTCODE")) else str(row.get("POSTCODE"))
    worst = str(row.get("WORST_EPC",""))
    n     = int(row.get("N_UNITS",0))
    return f"{addr}  [{pc}]  – {n} units – worst EPC {worst}"

# =========================================
# Sidebar filters (with building-level label filter)
# =========================================
st.sidebar.header("Filters")

if st.sidebar.button("🔄 Refresh data"):
    st.cache_data.clear()
    st.rerun()

default_key = st.secrets.get("GOOGLE_MAPS_API_KEY", "")
api_key = st.sidebar.text_input("Google API key (Street View & optional Geocoding)", value=default_key, type="password")

flats_only    = st.sidebar.checkbox("Flats only", value=True)
ratings_only  = st.sidebar.checkbox("Only EPC D–G", value=True)
rating_filter = st.sidebar.multiselect("EPC Ratings (flat-level)", ["D","E","F","G"], default=["D","E","F","G"])

# Address / postcode filters
addr_contains = st.sidebar.text_input("Address contains", value="")
pc_prefix     = st.sidebar.text_input("Postcode starts with", value="")

# Area filter (if present)
has_area = ("TOTAL_FLOOR_AREA" in df.columns) and df["TOTAL_FLOOR_AREA"].notna().any()
if has_area:
    valid = df["TOTAL_FLOOR_AREA"].dropna()
    min_a, max_a = int(valid.min()), int(valid.max())
    area_range = st.sidebar.slider("Total floor area (m²)", min_a, max_a, (min_a, max_a))
else:
    area_range = None

# Date filter (if present)
if "LODGEMENT_DATE" in df.columns and df["LODGEMENT_DATE"].notna().any():
    min_d = pd.to_datetime(df["LODGEMENT_DATE"]).min()
    max_d = pd.to_datetime(df["LODGEMENT_DATE"]).max()
    d_from, d_to = st.sidebar.date_input("Lodgement date range", value=(min_d.date(), max_d.date()))
else:
    d_from = d_to = None

# Units threshold & merge toggle
min_units = st.sidebar.number_input("Minimum units per building", min_value=0, max_value=999, value=0, step=1)
merge_buildings = st.sidebar.checkbox("Merge flats into one location (one pin per building)", value=(min_units >= 7))

# NEW: Building-level label filter (shown only in merge mode)
building_label_filter = ["D","E","F","G"]
building_filter_mode = "Any unit has…"
if merge_buildings:
    st.sidebar.markdown("**Building label filter**")
    building_label_filter = st.sidebar.multiselect(
        "Choose building labels to show",
        ["D","E","F","G"],
        default=["D","E","F","G"],
        key="bldg_labels"
    )
    building_filter_mode = st.sidebar.radio(
        "How should buildings match?",
        ["Any unit has…", "Worst EPC is…"],
        index=0,
        key="bldg_mode"
    )

# =========================================
# Apply filters at flat level
# =========================================
df_f = df.copy()

if flats_only and "PROPERTY_TYPE" in df_f.columns:
    df_f = df_f[df_f["PROPERTY_TYPE"].astype(str).str.contains("Flat", case=False, na=False)]

if ratings_only and "CURRENT_ENERGY_RATING" in df_f.columns:
    df_f = df_f[df_f["CURRENT_ENERGY_RATING"].astype(str).str.upper().isin(rating_filter)]

if addr_contains.strip():
    df_f = df_f[df_f["ADDRESS"].astype(str).str.contains(addr_contains, case=False, na=False)]

if pc_prefix.strip():
    df_f = df_f[df_f["POSTCODE"].astype(str).str.startswith(pc_prefix, na=False)]

if area_range and "TOTAL_FLOOR_AREA" in df_f.columns:
    df_f = df_f[df_f["TOTAL_FLOOR_AREA"].between(area_range[0], area_range[1])]

if d_from and d_to and "LODGEMENT_DATE" in df_f.columns:
    mask = df_f["LODGEMENT_DATE"].between(pd.to_datetime(d_from), pd.to_datetime(d_to))
    df_f = df_f[mask]

# =========================================
# Build display dataframe (merge or flat)
# =========================================
if merge_buildings:
    base = (
        df_f.sort_values(["BASE_ADDRESS","POSTCODE"])
            .groupby("BUILDING_ID", as_index=False)
            .first()[["BUILDING_ID","BASE_ADDRESS","POSTCODE","LAT","LON"]]
            .rename(columns={"BASE_ADDRESS":"ADDRESS"})
    )
    counts = df_f.groupby("BUILDING_ID").size().rename("N_UNITS").reset_index()
    worst  = df_f.groupby("BUILDING_ID")["CURRENT_ENERGY_RATING"].apply(worst_epc).rename("WORST_EPC").reset_index()
    mix    = df_f.groupby("BUILDING_ID")["CURRENT_ENERGY_RATING"].apply(epc_mix).rename("RATING_MIX").reset_index()

    df_display = base.merge(counts, on="BUILDING_ID").merge(worst, on="BUILDING_ID").merge(mix, on="BUILDING_ID")

    # Apply ≥ min units
    if min_units > 0:
        df_display = df_display[df_display["N_UNITS"] >= int(min_units)]

    # NEW: Building label filter (operates on the CURRENTLY filtered flat-level set df_f)
    # Tip: This composes with your flat-level "EPC Ratings" selection.
    if set(building_label_filter) != {"D","E","F","G"}:
        selected_set = set([s.upper() for s in building_label_filter])
        if building_filter_mode.startswith("Worst"):
            df_display = df_display[df_display["WORST_EPC"].isin(selected_set)]
        else:
            has_sel = (
                df_f.groupby("BUILDING_ID")["CURRENT_ENERGY_RATING"]
                    .apply(lambda s: s.astype(str).str.upper().isin(selected_set).any())
            )
            df_display = df_display[df_display["BUILDING_ID"].isin(has_sel[has_sel].index)]

    df_display["__LABEL__"] = df_display.apply(make_label_building, axis=1)

else:
    # If min_units is set but not merging, keep flats only from buildings meeting threshold
    if min_units > 0 and "BUILDING_ID" in df_f.columns:
        b_counts = df_f["BUILDING_ID"].value_counts()
        df_f = df_f[df_f["BUILDING_ID"].isin(b_counts[b_counts >= int(min_units)].index)]
    df_display = df_f.copy()
    df_display["__LABEL__"] = df_display.apply(make_label_flat, axis=1)

st.caption(f"Entries shown: **{len(df_display)}**")
if df_display.empty:
    st.warning("No properties match the filters above.")
    st.stop()

# =========================================
# KPI row
# =========================================
with st.container():
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Entries shown", len(df_display))
    with c2:
        n_bldgs = (df_f["BUILDING_ID"].nunique() if "BUILDING_ID" in df_f.columns else 0)
        st.metric("Unique buildings (in data)", n_bldgs)
    with c3:
        with_coords = df_display[["LAT","LON"]].dropna().shape[0] if {"LAT","LON"}.issubset(df_display.columns) else 0
        st.metric("With coordinates", with_coords)
    with c4:
        if merge_buildings and "WORST_EPC" in df_display.columns and not df_display["WORST_EPC"].dropna().empty:
            st.metric("Worst EPC (mode)", df_display["WORST_EPC"].mode().iloc[0])
        elif (not merge_buildings) and "CURRENT_ENERGY_RATING" in df_display.columns:
            rc = df_display["CURRENT_ENERGY_RATING"].value_counts(dropna=False)
            st.metric("Most common rating", (rc.index[0] if not rc.empty else "—"))
        else:
            st.metric("Most common rating", "—")

# =========================================
# Stable selection
# =========================================
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

# =========================================
# Optional geocoding for selected (place_id link)
# =========================================
@st.cache_data(show_spinner=False)
def geocode_place_id(addr_str: str, api_key: str):
    try:
        r = requests.get(
            "https://maps.googleapis.com/maps/api/geocode/json",
            params={"address": addr_str, "key": api_key},
            timeout=8,
        )
        js = r.json()
        if js.get("status") == "OK" and js.get("results"):
            res = js["results"][0]
            return res.get("place_id"), res.get("geometry", {}).get("location", {})
    except Exception:
        pass
    return None, {}

addr_str_sel = f"{selected.get('ADDRESS','')}, {selected.get('POSTCODE','')}, Blackpool, UK"
place_id_sel, geom_sel = geocode_place_id(addr_str_sel, api_key) if api_key else (None, {})
addr_q_sel = quote_plus(addr_str_sel)
if place_id_sel:
    sel_gmaps_link = f"https://www.google.com/maps/search/?api=1&query={addr_q_sel}&query_place_id={place_id_sel}&layer=c"
else:
    sel_gmaps_link = f"https://www.google.com/maps/search/?api=1&query={addr_q_sel}&layer=c"

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
    st.markdown(f"[Open in Google Maps (address-accurate)]({sel_gmaps_link})")

# =========================================
# Street View helpers (freshest pano)
# =========================================
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

def _parse_sv_date(s):
    try:
        if not s: return (0,0)
        parts = str(s).split("-")
        y = int(parts[0]) if parts[0].isdigit() else 0
        m = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else 0
        return (y, m)
    except Exception:
        return (0,0)

def find_freshest_streetview(lat, lon, address, postcode, api_key):
    candidates = []
    if pd.notna(lat) and pd.notna(lon):
        steps = [0.0, 0.00010, 0.00020, 0.00030]  # ~0–33 m
        dirs = [(0,0),(1,0),(-1,0),(0,1),(0,-1),(1,1),(1,-1),(-1,1),(-1,-1)]
        for d in steps:
            for sy, sx in dirs:
                y = lat + sy*d
                x = lon + sx*d
                candidates.append(f"{y},{x}")
    candidates.append(f"{address}, {postcode}, Blackpool, UK")

    best_key = None
    best_meta = None
    best_used = None

    for loc in dict.fromkeys(candidates):
        meta = streetview_metadata(loc, api_key)
        if meta.get("status") == "OK":
            dkey = _parse_sv_date(meta.get("date"))
            if (best_key is None) or (dkey > best_key):
                best_key, best_meta, best_used = dkey, meta, loc

    if best_meta:
        loc_meta = best_meta.get("location", {}) or {}
        snap_lat = loc_meta.get("lat", lat if pd.notna(lat) else None)
        snap_lon = loc_meta.get("lng", lon if pd.notna(lon) else None)
        pano_id  = best_meta.get("pano_id")
        return True, snap_lat, snap_lon, best_meta, best_used, pano_id

    return False, None, None, {}, None, None

# =========================================
# Tabs
# =========================================
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

        # Address-accurate Street View link (like manual search)
        addr = str(row.get("ADDRESS",""))
        pc   = str(row.get("POSTCODE","")) if pd.notna(row.get("POSTCODE")) else ""
        addr_q = quote_plus(f"{addr}, {pc}, Blackpool, UK")
        gsv_url = f"https://www.google.com/maps/search/?api=1&query={addr_q}&layer=c"
        if not addr.strip():
            gsv_url = (
                "https://www.google.com/maps/@?api=1&map_action=pano"
                f"&viewpoint={row['LAT']},{row['LON']}"
            )

        popup_html = (
            f"<div style='font-size:14px; line-height:1.2;'>"
            f"<strong>{addr or '(no address)'} </strong><br>"
            f"Postcode: {pc}<br>"
            + (f"Units: {n_units}<br>" if merge_buildings else "")
            + (f"Mix: {mix}<br>" if merge_buildings else f"EPC: {rating}<br>")
            + f"<br><a href='{gsv_url}' target='_blank' rel='noopener noreferrer'>🧭 Open Street View</a>"
            f"</div>"
        )

        is_sel = (row.get("ADDRESS",""), row.get("POSTCODE","")) == sel_key
        if is_sel:
            folium.CircleMarker([row["LAT"], row["LON"]], radius=10, color="#2b8a3e", fill=True, fill_opacity=0.7).add_to(m)

        folium.Marker(
            [row["LAT"], row["LON"]],
            popup=folium.Popup(popup_html, max_width=300),
            icon=folium.Icon(color=color_for(rating))
        ).add_to(cluster)

    map_key = f"map_{st.session_state['__SEL_LABEL__']}_{len(df_display)}_{int(merge_buildings)}"
    st_folium(m, width=1100, height=580, key=map_key)

# --- Street View tab (freshest pano) ---
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

        ok, snap_lat, snap_lon, meta, used_param, pano_id = find_freshest_streetview(
            lat, lon, address, postcode, api_key
        )

        if ok:
            if pano_id:
                img_url = (
                    "https://maps.googleapis.com/maps/api/streetview"
                    f"?size=640x400&pano={pano_id}"
                    f"&heading={heading}&pitch={pitch}&fov={fov}&source=outdoor&key={api_key}"
                )
            else:
                loc_param = f"{snap_lat},{snap_lon}"
                img_url = (
                    "https://maps.googleapis.com/maps/api/streetview"
                    f"?size=640x400&location={loc_param}"
                    f"&heading={heading}&pitch={pitch}&fov={fov}&source=outdoor&key={api_key}"
                )
            gsv_url = (
                "https://www.google.com/maps/@?api=1&map_action=pano"
                f"&viewpoint={snap_lat},{snap_lon}"
            )

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
            if meta.get("date"):
                st.caption(f"Street View capture date: {meta.get('date')} (freshest nearby)")

            with st.expander("Street View Debug"):
                st.write("Chosen candidate:", used_param)
                st.write("Pano ID:", pano_id)
                st.write("Metadata status:", meta.get("status"))
                st.code(img_url, language="text")
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

# --- Table tab (robust) ---
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

# --- Diagnostics tab (robust) ---
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
