from __future__ import annotations
import folium
from folium.plugins import MarkerCluster

def draw_map(df, lat_col="LAT", lon_col="LON", label_col="__LABEL__"):
    if df.empty:
        return folium.Map(location=[54.0, -2.5], zoom_start=6)

    center = [df[lat_col].mean(), df[lon_col].mean()]
    m = folium.Map(location=center, zoom_start=12, tiles="cartodbpositron")

    cluster = MarkerCluster().add_to(m)
    for _, row in df.iterrows():
        label = row.get(label_col, "")
        color = row.get("__MARKER_COLOR__", "blue")
        popup_html = row.get("__POPUP_HTML__", label)
        folium.Marker(
            [row[lat_col], row[lon_col]],
            tooltip=label,
            popup=folium.Popup(popup_html, max_width=350),
            icon=folium.Icon(color=color, icon="home"),
        ).add_to(cluster)
    return m

