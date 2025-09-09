from __future__ import annotations
import os, re
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
DEFAULT_DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "blackpool_low_epc_with_coords.csv")
uploaded = st.sidebar.file_uploader("Upload CSV (optional)", type=["csv"])
if st.sidebar.button("Refresh data"):
    st.rerun()

st.sidebar.divider()
st.sidebar.header("Filters")

google_api_key = st.secrets.get("GOOGLE_MAPS_API_KEY")

display_mode = st.sidebar.radio(
    "Display mode", ["Individual flats", "Merged buildings (one pin per building)"], index=0
)
epc_mode = st.sidebar.radio(
    "EPC filter", ["Any", "Only E", "Only F", "Only G", "Only E+F+G", "Custom…"], index=0
)
addr_contains = st.sidebar.text_input("Address contains", value="")
pc_starts = st.sidebar.text_input("Postcodes starts with", value="")
min_units = st.sidebar.number_input("Minimum TOTAL flats per building", 0, 1000, 7)

# -------------------- Load data --------------------
def base_addr(s: str) -> str:
    s = str(s or "")
    return re.sub(r"^(?:Flat|Apt|Apartment)\s*\w+\s*,\s*", "", s, flags=re.I).strip()

df = load_epc_csv(uploaded) if uploaded else load_epc_csv(DEFAULT_DATA_PATH)

for col in ["TOTAL_FLOOR_AREA", "LODGEMENT_DATE"]:
    if col not in df.columns:
        df[col] = ""

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

# -------------------- EPC quick-picks --------------------
epc_options = sorted({str(x).strip().upper() for x in df.get("EPC_CURRENT", pd.Series(dtype=str)).dropna() if str(x).strip()})
if not epc_options:
    epc_options = ["A","B","C","D","E","F","G"]

if epc_mode == "Any":
    epc_selected = []
elif epc_mode == "Only E":
    epc_selected = ["E"]
elif epc_mode == "Only F":
    epc_selected = ["F"]
elif epc_mode == "Only G":
    epc_selected = ["G"]
elif epc_mode == "Only E+F+G":
    epc_selected = ["E","F","G"]
else:
    epc_selected = st.sidebar.multiselect("Choose EPC ratings", options=epc_options, default=["E","F","G"])

# -------------------- Filters --------------------
mask = (df["UNITS_PER_BUILDING"].fillna(0) >= int(min_units))
if epc_selected:
    mask &= df["EPC_CURRENT"].astype(str).str.upper().isin(epc_selected)
if addr_contains.strip():
    mask &= df["ADDRESS"].str.contains(addr_contains.strip(), case=False, na=False)
if pc_starts.strip():
    mask &= df["POSTCODE"].astype(str).str.startswith(pc_starts.strip())
df_filt = df.loc[mask].copy()

view = (
    df_filt.sort_values(["UNITS_PER_BUILDING","EPC_CURRENT"], ascending=[False,True])
    .groupby("__BASE_ADDR__", as_index=False).first()
    if display_mode.startswith("Merged") else df_filt.copy()
)

# -------------------- KPIs --------------------
st.title("Blackpool Low-EPC Dashboard")
k1,k2,k3,k4 = st.columns(4)
k1.metric("Entries shown", f"{len(df_filt):,}")
k2.metric("Eligible buildings (shown)", f"{(view['__BASE_ADDR__'].nunique() if '__BASE_ADDR__' in view else len(view)):,}")
k3.metric("With coordinates", f"{view.dropna(subset=['LAT','LON']).shape[0]:,}")
most_common = "—"
if "EPC_CURRENT" in view and not view["EPC_CURRENT"].dropna().empty:
    from collections import Counter
    most_common = Counter([str(x).strip().upper() for x in view["EPC_CURRENT"] if str(x).strip()]).most_common(1)[0][0]
k4.metric("Most common rating", most_common)
st.divider()

# -------------------- Selected property --------------------
st.subheader("Selected Property / Building")
def nice_label(row)->str:
    return f"{row.get('ADDRESS','').strip()} [{str(row.get('POSTCODE','')).strip()}] – EPC {str(row.get('EPC_CURRENT','')).strip().upper() or '—'}"
options = view.apply(nice_label, axis=1).tolist()
selected_label = st.selectbox("Pick an item (drives Map & Street View):", options, index=0 if options else None)
sel_row = view.iloc[options.index(selected_label)] if options else None

# -------------------- Build link table --------------------
rows=[]
for _,r in view.iterrows():
    addr=str(r.get("ADDRESS","")).strip(); pc=str(r.get("POSTCODE","")).strip()
    rm_url=rightmove_search_url(addr,pc); zp_url=zoopla_search_url(addr,pc); gmaps=generic_map_query(addr,pc)
    street = addr.split(",")[1].strip() if "," in addr and len(addr.split(","))>1 else addr
    rm_broad=rightmove_broad_url(street,pc); zp_broad=zoopla_broad_url(street,pc)
    label=f"{addr} [{pc}] – EPC {str(r.get('EPC_CURRENT','')).strip().upper() or '—'}"
    popup_html=(
        f"<b>{label}</b><br>"
        f"<a href='{gmaps}' target='_blank'>Google Maps</a> | "
        f"<a href='{rm_url}' target='_blank'>Rightmove search</a> | "
        f"<a href='{zp_url}' target='_blank'>Zoopla search</a><br>"
        f"<small>Broader: <a href='{rm_broad}' target='_blank'>RM street</a> · "
        f"<a href='{zp_broad}' target='_blank'>ZP street</a></small>"
    )
    rows.append({
        "ADDRESS":addr,"POSTCODE":pc,"EPC_CURRENT":r.get("EPC_CURRENT",""),
        "LAT":r.get("LAT"),"LON":r.get("LON"),
        "__LABEL__":label,"__POPUP_HTML__":popup_html,"__MARKER_COLOR__":"blue",
        "RIGHTMOVE_SEARCH":rm_url,"ZOOPLA_SEARCH":zp_url,"GOOGLE_MAPS":gmaps,
        "RM_BROAD":rm_broad,"ZP_BROAD":zp_broad,"UNITS_PER_BUILDING":r.get("UNITS_PER_BUILDING",""),
    })
result_df=pd.DataFrame(rows)

# -------------------- “For sale” helpers --------------------
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
        items = (PiloterrZooplaClient().search(q) or [])
        return "hit" if items else "miss"
    except Exception:
        return "unknown"

def _status_badge(label:str,status:str,url:str)->str:
    icon = "✅" if status=="hit" else ("❌" if status=="miss" else "➖")
    tip  = "found" if status=="hit" else ("not found" if status=="miss" else "unknown")
    return f"**{label}:** {icon} <small>({tip})</small> · <a href='{url}' target='_blank'>open</a>"

# -------------------- NEW: Geocode + Street View (tighter matching) --------------------
def _geocode(address_q:str,key:str):
    """Return (place_id, lat, lon) for the address using Geocoding API."""
    if not (address_q and key): return (None,None,None)
    try:
        r=requests.get(
            "https://maps.googleapis.com/maps/api/geocode/json",
            params={"address":address_q,"key":key}, timeout=6
        )
        js=r.json() if r.ok else {}
        res=(js.get("results") or [])
        if not res: return (None,None,None)
        pid=res[0].get("place_id")
        loc=(res[0].get("geometry") or {}).get("location") or {}
        return (pid, loc.get("lat"), loc.get("lng"))
    except Exception:
        return (None,None,None)

def _sv_meta(location:str,key:str,radius:int)->dict:
    """Street View Metadata call; location can be 'place_id:...' or 'lat,lon'."""
    try:
        r=requests.get(
            "https://maps.googleapis.com/maps/api/streetview/metadata",
            params={"location":location,"source":"outdoor","radius":radius,"key":key},
            timeout=6
        )
        js=r.json() if r.ok else {}
    except Exception:
        js={}
    pano=js.get("pano_id") or js.get("panoId")
    if not pano: return {}
    date=js.get("date")
    static_url=(
        "https://maps.googleapis.com/maps/api/streetview"
        f"?size=800x450&pano={pano}&key={key}"
    )
    link=f"https://www.google.com/maps/@?api=1&map_action=pano&pano={pano}"
    return {"pano":pano,"date":date,"static":static_url,"link":link}

def _best_streetview(addr:str, pc:str, lat, lon, key:str)->dict:
    """
    Order:
      A) Geocode address → (place_id, geo_latlon)
         A1) try geo_latlon radii 30→60→100 (very tight around frontage)
         A2) try place_id radii 50→120
      B) If still nothing and we have CSV lat/lon → radii 50→120→200
    Returns {'static','link','date'} or {}.
    """
    pid, glat, glon = _geocode(f"{addr}, {pc}", key)
    # A1: geocoded lat/lon (frontage point)
    if glat is not None and glon is not None:
        for r_ in (30, 60, 100):
            hit=_sv_meta(f"{glat},{glon}", key, r_)
            if hit: return {"static":hit["static"], "link":hit["link"], "date":hit.get("date"), "where":f"geo {r_}m"}
    # A2: place_id (sometimes pinned differently)
    if pid:
        for r_ in (50, 120):
            hit=_sv_meta(f"place_id:{pid}", key, r_)
            if hit: return {"static":hit["static"], "link":hit["link"], "date":hit.get("date"), "where":f"pid {r_}m", "pid":pid}
    # B) CSV lat/lon fallback
    if pd.notna(lat) and pd.notna(lon):
        for r_ in (50, 120, 200):
            hit=_sv_meta(f"{lat},{lon}", key, r_)
            if hit: return {"static":hit["static"], "link":hit["link"], "date":hit.get("date"), "where":f"csv {r_}m"}
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
    cols=["ADDRESS","POSTCODE","EPC_CURRENT","UNITS_PER_BUILDING","RIGHTMOVE_SEARCH","ZOOPLA_SEARCH","GOOGLE_MAPS","RM_BROAD","ZP_BROAD"]
    for c in cols:
        if c not in result_df.columns: result_df[c]=""
    safe_view=result_df.reindex(columns=cols, fill_value="")
    st.dataframe(safe_view, use_container_width=True, height=420)
    st.download_button(
        "Download results as CSV", safe_view.to_csv(index=False).encode("utf-8"),
        file_name="listings_overlay_results.csv", mime="text/csv", key="download_results_main"
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
        "- Street View now geocodes the address first, then searches OUTDOOR panos with a very tight radius (30→60→100 m) around the geocoded point.\n"
        "- If needed, it widens to 50→120 m on place_id, then falls back to CSV LAT/LON.\n"
        "- The preview and link both use the chosen pano id; capture date is shown."
    )
