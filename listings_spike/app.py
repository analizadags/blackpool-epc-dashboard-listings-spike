from __future__ import annotations
import os, re, math
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from collections import Counter
import requests

from utils.epc_io import load_epc_csv
from utils.map_viz import draw_map
from providers.url_builders import (
    rightmove_search_url,
    zoopla_search_url,
    generic_map_query,
    rightmove_broad_url,
    zoopla_broad_url,
)

try:
    from providers.zoopla_piloterr import PiloterrZooplaClient
except Exception:
    PiloterrZooplaClient = None

# -------------------- App setup --------------------
load_dotenv()
st.set_page_config(page_title="Blackpool Low-EPC Dashboard", layout="wide")
st.markdown("""
<style>
.block-container { padding-top: 1rem; padding-bottom: 2rem; }
.small-muted { color:#6b7280; font-size:0.9rem; }
.stTabs [data-baseweb="tab-list"] { gap: .35rem; }
.stTabs [data-baseweb="tab"] { padding: .3rem .6rem; }
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

google_api_key = st.secrets.get("GOOGLE_MAPS_API_KEY")

display_mode = st.sidebar.radio("Display mode",
    ["Individual flats", "Merged buildings (one pin per building)"], index=0)

epc_mode = st.sidebar.radio("EPC filter",
    ["Any", "Only E", "Only F", "Only G", "Only E+F+G", "Custom…"], index=0)

addr_contains = st.sidebar.text_input("Address contains", value="")
pc_starts    = st.sidebar.text_input("Postcodes starts with", value="")
min_units    = st.sidebar.number_input("Minimum TOTAL flats per building", 0, 1000, 7)

# -------------------- Load data --------------------
def base_addr(s: str) -> str:
    s = str(s or "")
    return re.sub(r"^(?:Flat|Apt|Apartment)\s*\w+\s*,\s*", "", s, flags=re.I).strip()

df = load_epc_csv(uploaded) if uploaded else load_epc_csv(DEFAULT_DATA_PATH)
for c in ["TOTAL_FLOOR_AREA", "LODGEMENT_DATE"]:
    if c not in df: df[c] = ""

df["LAT"] = pd.to_numeric(df.get("LAT"), errors="coerce")
df["LON"] = pd.to_numeric(df.get("LON"), errors="coerce")

if "UNITS_PER_BUILDING" not in df or df["UNITS_PER_BUILDING"].isna().all():
    df["__BASE_ADDR__"] = df["ADDRESS"].map(base_addr) + " | " + df["POSTCODE"].astype(str)
    counts = df["__BASE_ADDR__"].value_counts(dropna=False)
    df["UNITS_PER_BUILDING"] = df["__BASE_ADDR__"].map(counts).astype("Int64")
else:
    df["UNITS_PER_BUILDING"] = pd.to_numeric(df["UNITS_PER_BUILDING"], errors="coerce").astype("Int64")
if "__BASE_ADDR__" not in df:
    df["__BASE_ADDR__"] = df["ADDRESS"].map(base_addr) + " | " + df["POSTCODE"].astype(str)

# -------------------- EPC picks --------------------
epc_options = sorted({str(x).strip().upper() for x in df.get("EPC_CURRENT", pd.Series(dtype=str)).dropna() if str(x).strip()})
if not epc_options: epc_options = ["A","B","C","D","E","F","G"]

if epc_mode == "Any":            epc_selected = []
elif epc_mode == "Only E":       epc_selected = ["E"]
elif epc_mode == "Only F":       epc_selected = ["F"]
elif epc_mode == "Only G":       epc_selected = ["G"]
elif epc_mode == "Only E+F+G":   epc_selected = ["E","F","G"]
else:
    epc_selected = st.sidebar.multiselect("Choose EPC ratings", epc_options, default=["E","F","G"])

# -------------------- Filter --------------------
mask = (df["UNITS_PER_BUILDING"].fillna(0) >= int(min_units))
if epc_selected:           mask &= df["EPC_CURRENT"].astype(str).str.upper().isin(epc_selected)
if addr_contains.strip():  mask &= df["ADDRESS"].str.contains(addr_contains.strip(), case=False, na=False)
if pc_starts.strip():      mask &= df["POSTCODE"].astype(str).str.startswith(pc_starts.strip())
df_filt = df.loc[mask].copy()

view = (df_filt.sort_values(["UNITS_PER_BUILDING","EPC_CURRENT"], ascending=[False,True])
        .groupby("__BASE_ADDR__", as_index=False).first()
        if display_mode.startswith("Merged") else df_filt.copy())

# -------------------- KPIs --------------------
st.title("Blackpool Low-EPC Dashboard")
k1,k2,k3,k4 = st.columns(4)
k1.metric("Entries shown", f"{len(df_filt):,}")
k2.metric("Eligible buildings (shown)", f"{(view['__BASE_ADDR__'].nunique() if '__BASE_ADDR__' in view else len(view)):,}")
k3.metric("With coordinates", f"{view.dropna(subset=['LAT','LON']).shape[0]:,}")
most = "—"
if "EPC_CURRENT" in view and not view["EPC_CURRENT"].dropna().empty:
    most = Counter([str(x).strip().upper() for x in view["EPC_CURRENT"] if str(x).strip()]).most_common(1)[0][0]
k4.metric("Most common rating", most)
st.divider()

# -------------------- Selection --------------------
st.subheader("Selected Property / Building")
def nice_label(row)->str:
    return f"{row.get('ADDRESS','').strip()} [{str(row.get('POSTCODE','')).strip()}] – EPC {str(row.get('EPC_CURRENT','')).strip().upper() or '—'}"
options = view.apply(nice_label, axis=1).tolist()
selected_label = st.selectbox("Pick an item (drives Map & Street View):", options, index=0 if options else None)
sel_row = view.iloc[options.index(selected_label)] if options else None

# -------------------- Links table --------------------
rows=[]
for _,r in view.iterrows():
    addr=str(r.get("ADDRESS","")).strip(); pc=str(r.get("POSTCODE","")).strip()
    rm=rightmove_search_url(addr,pc); zp=zoopla_search_url(addr,pc); g=generic_map_query(addr,pc)
    street = addr.split(",")[1].strip() if "," in addr and len(addr.split(","))>1 else addr
    rm_b=rightmove_broad_url(street,pc); zp_b=zoopla_broad_url(street,pc)
    label=f"{addr} [{pc}] – EPC {str(r.get('EPC_CURRENT','')).strip().upper() or '—'}"
    html=(f"<b>{label}</b><br><a href='{g}' target='_blank'>Google Maps</a> | "
          f"<a href='{rm}' target='_blank'>Rightmove search</a> | "
          f"<a href='{zp}' target='_blank'>Zoopla search</a><br>"
          f"<small>Broader: <a href='{rm_b}' target='_blank'>RM street</a> · "
          f"<a href='{zp_b}' target='_blank'>ZP street</a></small>")
    rows.append({
        "ADDRESS":addr,"POSTCODE":pc,"EPC_CURRENT":r.get("EPC_CURRENT",""),
        "LAT":r.get("LAT"),"LON":r.get("LON"),
        "__LABEL__":label,"__POPUP_HTML__":html,"__MARKER_COLOR__":"blue",
        "RIGHTMOVE_SEARCH":rm,"ZOOPLA_SEARCH":zp,"GOOGLE_MAPS":g,"RM_BROAD":rm_b,"ZP_BROAD":zp_b,
        "UNITS_PER_BUILDING":r.get("UNITS_PER_BUILDING",""),
    })
result_df=pd.DataFrame(rows)

# -------------------- For-sale helpers --------------------
def _google_results_hint(url:str)->str:
    try:
        r=requests.get(url,timeout=8,headers={"User-Agent":"Mozilla/5.0 (DemoApp/1.0)"})
        t=r.text.lower()
        if "did not match any documents" in t: return "miss"
        if "rightmove.co.uk" in t or "zoopla.co.uk" in t: return "hit"
        return "unknown"
    except Exception:
        return "unknown"

def _zoopla_status_via_piloterr(q:str)->str:
    if PiloterrZooplaClient is None: return "unknown"
    try:
        return "hit" if (PiloterrZooplaClient().search(q) or []) else "miss"
    except Exception:
        return "unknown"

def _status_badge(label:str,status:str,url:str)->str:
    icon = "✅" if status=="hit" else ("❌" if status=="miss" else "➖")
    tip  = "found" if status=="hit" else ("not found" if status=="miss" else "unknown")
    return f"**{label}:** {icon} <small>({tip})</small> · <a href='{url}' target='_blank'>open</a>"

# -------------------- Geocode / Roads / Street View --------------------
def _geocode(address_q:str,key:str):
    """Return (place_id, lat, lon) for the address using Geocoding API."""
    if not (address_q and key): return (None,None,None)
    try:
        r=requests.get("https://maps.googleapis.com/maps/api/geocode/json",
                       params={"address":address_q,"key":key}, timeout=6)
        js=r.json() if r.ok else {}
        res=(js.get("results") or [])
        if not res: return (None,None,None)
        pid=res[0].get("place_id")
        loc=(res[0].get("geometry") or {}).get("location") or {}
        return (pid, loc.get("lat"), loc.get("lng"))
    except Exception:
        return (None,None,None)

def _snap_to_road(lat, lon, key):
    """Snap to the nearest road centerline using Roads API (falls back on error)."""
    if not (pd.notna(lat) and pd.notna(lon) and key): return (None, None)
    try:
        r=requests.get("https://roads.googleapis.com/v1/nearestRoads",
                       params={"points": f"{lat},{lon}", "key": key}, timeout=6)
        js=r.json() if r.ok else {}
        pts=(js.get("snappedPoints") or [])
        if not pts: return (None, None)
        loc=pts[0].get("location") or {}
        return (loc.get("latitude"), loc.get("longitude"))
    except Exception:
        return (None, None)

def _sv_meta(location:str,key:str,radius:int)->dict:
    """Street View Metadata call; location can be 'place_id:...' or 'lat,lon'."""
    try:
        r=requests.get("https://maps.googleapis.com/maps/api/streetview/metadata",
                       params={"location":location,"source":"outdoor","radius":radius,"key":key},
                       timeout=6)
        js=r.json() if r.ok else {}
    except Exception:
        js={}
    pano = js.get("pano_id") or js.get("panoId")
    if not pano: return {}
    date = js.get("date")
    loc  = (js.get("location") or {})
    lat  = loc.get("lat"); lon = loc.get("lng")
    static = f"https://maps.googleapis.com/maps/api/streetview?size=800x450&pano={pano}&key={key}"
    link   = f"https://www.google.com/maps/@?api=1&map_action=pano&pano={pano}"
    return {"pano":pano,"date":date,"static":static,"link":link,"lat":lat,"lon":lon}

def _haversine(lat1,lon1,lat2,lon2):
    R=6371000.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2-lat1); dl = math.radians(lon2-lon1)
    a = math.sin(dphi/2)**2 + math.cos(p1)*math.cos(p2)*math.sin(dl/2)**2
    return 2*R*math.asin(math.sqrt(a))

def _offset(lat, lon, bearing_deg, dist_m):
    R=6378137.0
    br = math.radians(bearing_deg)
    lat1 = math.radians(lat); lon1 = math.radians(lon)
    lat2 = math.asin(math.sin(lat1)*math.cos(dist_m/R) + math.cos(lat1)*math.sin(dist_m/R)*math.cos(br))
    lon2 = lon1 + math.atan2(math.sin(br)*math.sin(dist_m/R)*math.cos(lat1),
                             math.cos(dist_m/R)-math.sin(lat1)*math.sin(lat2))
    return (math.degrees(lat2), math.degrees(lon2))

def _best_streetview(addr:str, pc:str, csv_lat, csv_lon, key:str)->dict:
    """
    Global strategy (applies to all addresses):
      1) Geocode to (place_id, geo_latlon)
      2) Snap geo_latlon to road via Roads API → search Street View tightly at snapped point
         (radii 25 → 45), plus small micro-ring (12m/22m) around snapped point
      3) If none, search around raw geocoded lat/lon (micro + 35m)
      4) If none, search place_id (50 → 120m)
      5) If none, fallback to CSV lat/lon (50 → 120 → 200m)
    Returns {'static','link','date'} or {}.
    """
    pid, glat, glon = _geocode(f"{addr}, {pc}", key)

    # (2) Snap to road (frontage on carriageway)
    if glat is not None and glon is not None:
        s_lat, s_lon = _snap_to_road(glat, glon, key)
        if s_lat is not None and s_lon is not None:
            # try very tight at snapped point
            for r_ in (25, 45):
                hit=_sv_meta(f"{s_lat},{s_lon}", key, r_)
                if hit: return {"static":hit["static"], "link":hit["link"], "date":hit.get("date"), "note":f"snap {r_}m"}
            # micro-ring around snapped point
            best=None; best_d=1e12
            samples=[f"{s_lat},{s_lon}"]
            for ring in (12, 22):
                for b in range(0,360,45):
                    sl, so = _offset(s_lat, s_lon, b, ring)
                    samples.append(f"{sl},{so}")
            for loc in samples:
                hit=_sv_meta(loc, key, 35)
                if hit and (hit.get("lat") is not None and hit.get("lon") is not None):
                    d=_haversine(s_lat, s_lon, hit["lat"], hit["lon"])
                    if d<best_d:
                        best, best_d = hit, d
            if best:
                return {"static":best["static"], "link":best["link"], "date":best.get("date"), "note":f"snap micro d≈{int(best_d)}m"}

    # (3) Micro-search around geocoded (building) point
    if glat is not None and glon is not None:
        samples=[f"{glat},{glon}"]
        for ring in (12, 22):
            for b in range(0,360,45):
                sl, so = _offset(glat, glon, b, ring)
                samples.append(f"{sl},{so}")
        best=None; best_d=1e12
        for loc in samples:
            hit=_sv_meta(loc, key, 35)
            if hit and (hit.get("lat") is not None and hit.get("lon") is not None):
                d=_haversine(glat, glon, hit["lat"], hit["lon"])
                if d<best_d:
                    best, best_d = hit, d
        if best:
            return {"static":best["static"], "link":best["link"], "date":best.get("date"), "note":f"geo micro d≈{int(best_d)}m"}

    # (4) place_id wider
    if pid:
        for r_ in (50, 120):
            hit=_sv_meta(f"place_id:{pid}", key, r_)
            if hit: return {"static":hit["static"], "link":hit["link"], "date":hit.get("date"), "note":f"pid {r_}m"}

    # (5) CSV fallback
    if pd.notna(csv_lat) and pd.notna(csv_lon):
        for r_ in (50, 120, 200):
            hit=_sv_meta(f"{csv_lat},{csv_lon}", key, r_)
            if hit: return {"static":hit["static"], "link":hit["link"], "date":hit.get("date"), "note":f"csv {r_}m"}

    return {}

# -------------------- Tabs --------------------
tabs = st.tabs(["Map","Street View","Table","Diagnostics"])

with tabs[0]:
    from streamlit_folium import st_folium
    st.write("")
    m = draw_map(result_df)
    st_folium(m, height=530, width=None)

with tabs[1]:
    st.write("")
    if sel_row is None:
        st.info("Select a property to preview Street View.")
    else:
        addr=str(sel_row.get("ADDRESS","")).strip()
        pc=str(sel_row.get("POSTCODE","")).strip()
        lat,lon=sel_row.get("LAT"),sel_row.get("LON")

        rm_url=rightmove_search_url(addr,pc)
        zp_url=zoopla_search_url(addr,pc)
        rm_status=_google_results_hint(rm_url)
        zp_status=_zoopla_status_via_piloterr(f"{addr} {pc}")
        gmaps_search=generic_map_query(addr,pc)

        left,right=st.columns([2,1], vertical_alignment="top")
        with left:
            if google_api_key:
                best=_best_streetview(addr, pc, lat, lon, google_api_key)
                if best:
                    st.markdown(
                        f"<a href='{best['link']}' target='_blank'>"
                        f"<img src='{best['static']}' style='width:100%; border-radius:8px;'/>"
                        f"</a>", unsafe_allow_html=True
                    )
                    cap=f"Street View near: {addr} [{pc}]"
                    if best.get("date"): cap+=f" — capture: {best['date']}"
                    st.caption(cap)
                else:
                    st.warning("Couldn’t fetch Street View. Opening by address instead.")
                    st.markdown(f"[Open Google Maps / Street View]({gmaps_search})")
            else:
                st.markdown(f"[Open Google Maps / Street View]({gmaps_search})")

        with right:
            st.markdown("### For Sale status")
            st.markdown(_status_badge("Rightmove", rm_status, rm_url), unsafe_allow_html=True)
            st.markdown(_status_badge("Zoopla",    zp_status, zp_url), unsafe_allow_html=True)
            st.caption("Rightmove uses a simple Google-results hint. Zoopla uses Piloterr if configured.")

with tabs[2]:
    st.write("")
    cols=["ADDRESS","POSTCODE","EPC_CURRENT","UNITS_PER_BUILDING",
          "RIGHTMOVE_SEARCH","ZOOPLA_SEARCH","GOOGLE_MAPS","RM_BROAD","ZP_BROAD"]
    for c in cols:
        if c not in result_df.columns: result_df[c]=""
    safe_view=result_df.reindex(columns=cols, fill_value="")
    st.dataframe(safe_view, use_container_width=True, height=420)
    st.download_button(
        "Download results as CSV",
        safe_view.to_csv(index=False).encode("utf-8"),
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
        "- Uses Geocoding + Roads (Nearest Roads) to anchor the pin on the carriageway, "
        "then Street View Metadata with tight radii. If none found, widens in steps and falls back gracefully.\n"
        "- Works for **all addresses**; no per-address hacks."
    )
