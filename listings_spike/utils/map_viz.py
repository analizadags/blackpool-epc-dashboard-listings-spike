from __future__ import annotations
import folium
from folium.plugins import MarkerCluster
import pandas as pd

def draw_map(df, lat_col="LAT", lon_col="LON", label_col="__LABEL__"):
    if df is None or len(df) == 0:
        return folium.Map(location=[54.0, -2.5], zoom_start=6)

    # Coerce and drop rows without coordinates
    work = df.copy()
    for col in (lat_col, lon_col):
        if col not in work.columns:
            work[col] = None
    work[lat_col] = pd.to_numeric(work[lat_col], errors="coerce")
    work[lon_col] = pd.to_numeric(work[lon_col], errors="coerce")
    work = work.dropna(subset=[lat_col, lon_col])

    if work.empty:
        return folium.Map(location=[54.0, -2.5], zoom_start=6)

    center = [work[lat_col].mean(), work[lon_col].mean()]
    m = folium.Map(location=center, zoom_start=12, tiles="cartodbpositron")

    cluster = MarkerCluster().add_to(m)
    for _, row in work.iterrows():
        label = row.get(label_col, "")
        color = row.get("__MARKER_COLOR__", "blue")
        popup_html = row.get("__POPUP_HTML__", label)
        try:
            folium.Marker(
                [row[lat_col], row[lon_col]],
                tooltip=label,
                popup=folium.Popup(popup_html, max_width=350),
                icon=folium.Icon(color=color, icon="home"),
            ).add_to(cluster)
        except Exception:
            # Skip odd rows silently
            continue
    return m
