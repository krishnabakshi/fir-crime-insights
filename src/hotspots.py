"""Hotspot detection and the interactive folium map.

* K-Means (k from the elbow method) splits the city into patrol ZONES: every
  incident belongs to one zone, which suits beat planning.
* DBSCAN finds dense HOTSPOTS (a few hundred metres across) and ignores
  scattered incidents as noise, which suits picket / PCR placement. DBSCAN
  needs no k, only a radius (eps) and a minimum number of incidents.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN, KMeans
from sklearn.metrics import silhouette_score

from src.contract import RANDOM_STATE, STATIONS


def to_km(lat: pd.Series, lon: pd.Series, ref_lat: float) -> np.ndarray:
    """Project lat/lon to kilometres so K-Means distances are real distances.

    StandardScaler is NOT used here: scaling latitude and longitude separately
    would stretch the map and distort which incidents are close together.
    """
    return np.column_stack([lat.to_numpy() * 110.57,
                            lon.to_numpy() * 111.32 * np.cos(np.radians(ref_lat))])


def elbow_curve(X: np.ndarray, k_range=range(2, 16)) -> pd.DataFrame:
    rows = []
    for k in k_range:
        km = KMeans(n_clusters=k, n_init=10, random_state=RANDOM_STATE).fit(X)
        sample = np.random.default_rng(RANDOM_STATE).choice(len(X), size=min(1500, len(X)), replace=False)
        rows.append({"k": k, "inertia": km.inertia_,
                     "silhouette": silhouette_score(X[sample], km.labels_[sample])})
    return pd.DataFrame(rows)


def knee_point(curve: pd.DataFrame) -> int:
    """Elbow = point farthest from the straight line joining the first and last
    points of the (normalised) inertia curve."""
    x = (curve["k"] - curve["k"].min()) / (curve["k"].max() - curve["k"].min())
    y = (curve["inertia"] - curve["inertia"].min()) / (curve["inertia"].max() - curve["inertia"].min())
    dist = np.abs(y - (1 - x))  # line from (0,1) to (1,0)
    return int(curve.loc[dist.idxmax(), "k"])


def find_zones(df: pd.DataFrame, district: str = "Indore", k: int | None = None) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """K-Means patrol zones for one district.

    Returns (incidents with zone id, zone summary, elbow curve)."""
    d = df[df["district"] == district].dropna(subset=["latitude", "longitude"]).copy()
    X = to_km(d["latitude"], d["longitude"], d["latitude"].mean())
    curve = elbow_curve(X)
    k = k or knee_point(curve)
    km = KMeans(n_clusters=k, n_init=10, random_state=RANDOM_STATE).fit(X)
    d["zone"] = km.labels_
    return d, _summarise(d, X, "zone"), curve


def find_dense_hotspots(df: pd.DataFrame, district: str = "Indore", eps_km: float = 0.25,
                        min_samples: int = 25) -> tuple[pd.DataFrame, pd.DataFrame]:
    """DBSCAN hotspots. Returns (incidents with hotspot id, -1 = noise; summary)."""
    d = df[df["district"] == district].dropna(subset=["latitude", "longitude"]).copy()
    X = to_km(d["latitude"], d["longitude"], d["latitude"].mean())
    d["hotspot"] = DBSCAN(eps=eps_km, min_samples=min_samples).fit_predict(X)
    inside = d["hotspot"].to_numpy() >= 0
    summ = _summarise(d[inside], X[inside], "hotspot")
    summ["is_hotspot"] = True
    return d, summ


def _summarise(d: pd.DataFrame, X: np.ndarray, key: str) -> pd.DataFrame:
    summ = (d.groupby(key)
              .agg(firs=("fir_no", "count"), lat=("latitude", "mean"), lon=("longitude", "mean"),
                   top_station=("police_station", lambda s: s.value_counts().index[0]),
                   top_crime=("crime_type", lambda s: s.value_counts().index[0]),
                   top_time=("time_of_day", lambda s: s.value_counts().index[0] if s.notna().any() else "n/a"),
                   last_90d=("date_occurred", lambda s: int((s >= s.max() - pd.Timedelta(days=90)).sum())))
              .reset_index())
    # radius covering 80% of the cluster's incidents (km)
    radii = []
    for c in summ[key]:
        pts = X[d[key].to_numpy() == c]
        radii.append(float(np.percentile(np.linalg.norm(pts - pts.mean(axis=0), axis=1), 80)))
    summ["radius_km"] = np.round(radii, 2)
    summ["share_%"] = (summ["firs"] / summ["firs"].sum() * 100).round(1)
    summ["density_per_km2"] = (summ["firs"] / (np.pi * summ["radius_km"].clip(lower=0.2) ** 2)).round(1)
    return summ.sort_values("density_per_km2", ascending=False).reset_index(drop=True)


def station_concentration(df: pd.DataFrame) -> pd.DataFrame:
    t = df.groupby(["police_station", "district"]).size().rename("firs").reset_index()
    t["share_%"] = (t["firs"] / t["firs"].sum() * 100).round(1)
    return t.sort_values("firs", ascending=False).reset_index(drop=True)


def build_map(incidents: pd.DataFrame, hotspots: pd.DataFrame, zones: pd.DataFrame | None = None,
              forecast: pd.DataFrame | None = None, out_html: str | None = None):
    """Folium map: heat layer, DBSCAN hotspots, K-Means zones, stations + forecast."""
    import folium
    from folium.plugins import HeatMap

    centre = [incidents["latitude"].mean(), incidents["longitude"].mean()]
    fmap = folium.Map(location=centre, zoom_start=12, tiles="OpenStreetMap")
    HeatMap(incidents[["latitude", "longitude"]].values.tolist(), radius=11, blur=14,
            name="Incident heat map").add_to(fmap)

    if zones is not None:
        zl = folium.FeatureGroup(name="K-Means patrol zones", show=False)
        for _, r in zones.iterrows():
            folium.Circle([r["lat"], r["lon"]], radius=float(r["radius_km"]) * 1000, color="#2980b9",
                          fill=False, weight=2, dash_array="6",
                          popup=f"Zone {r['zone']}: {r['firs']} FIRs, mostly {r['top_crime']}").add_to(zl)
        zl.add_to(fmap)

    hs = folium.FeatureGroup(name="DBSCAN hotspots")
    for _, r in hotspots.iterrows():
        folium.Circle(
            location=[r["lat"], r["lon"]], radius=max(float(r["radius_km"]), 0.15) * 1000, color="#c0392b",
            fill=True, fill_opacity=0.35,
            popup=folium.Popup(
                f"<b>Hotspot {r['hotspot']}</b><br>FIRs: {r['firs']} | last 90 days: {r['last_90d']}<br>"
                f"Top crime: {r['top_crime']}<br>Peak time: {r['top_time']}<br>Station: {r['top_station']}",
                max_width=260),
        ).add_to(hs)
    hs.add_to(fmap)

    st = folium.FeatureGroup(name="Police stations + 4-week forecast")
    fc = forecast.set_index("police_station") if forecast is not None else None
    for name, meta in STATIONS.items():
        if meta["district"] != incidents["district"].iloc[0]:
            continue
        txt = f"<b>{name} PS</b>"
        if fc is not None and name in fc.index:
            r = fc.loc[name]
            txt += (f"<br>Forecast next 4 weeks: {r['forecast_next_weeks']} FIRs (rank {r['risk_rank']})"
                    f"<br>Peak: {r['peak_time_of_day']}, {r['peak_weekday']}")
        folium.Marker([meta["lat"], meta["lon"]], popup=folium.Popup(txt, max_width=260),
                      icon=folium.Icon(color="blue", icon="info-sign")).add_to(st)
    st.add_to(fmap)
    folium.LayerControl().add_to(fmap)
    if out_html:
        fmap.save(out_html)
    return fmap
