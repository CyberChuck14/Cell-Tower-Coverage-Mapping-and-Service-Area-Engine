"""Folium layers for towers, coverage circles, quality grid, Voronoi, Delaunay."""

from __future__ import annotations

import base64
import io
from dataclasses import dataclass
from typing import Iterable

import folium
import matplotlib

matplotlib.use("Agg")
import numpy as np
import pandas as pd
from folium.plugins import MarkerCluster
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.colors import Normalize
from matplotlib.figure import Figure
from scipy.ndimage import gaussian_filter
from scipy.spatial import Voronoi
from shapely.geometry import LineString, MultiPoint, Point, Polygon, box
from shapely.ops import polygonize, unary_union

from src.config import (
    CHENNAI_BOUNDS,
    CHENNAI_CENTER_LAT,
    CHENNAI_CENTER_LON,
    CoverageConfig,
)
from src.distance import to_local_xy, to_wgs84
from src.signal_model import assumed_radius, estimate_coverage
from src.spatial_grid import ZoneMappedColumnBlocks
from src.triangulation import DCEL, delaunay_segments_wgs

CATEGORY_COLORS = {"green": "#22c55e", "yellow": "#eab308", "red": "#ef4444"}


@dataclass
class CoverageField:
    """Regular lat/lon score lattice used for filled-contour overlays."""

    scores: np.ndarray
    south: float
    west: float
    north: float
    east: float


def base_map(center: tuple[float, float] | None = None, zoom: int = 12) -> folium.Map:
    """OpenStreetMap only – no API key required."""
    lat, lon = center or (CHENNAI_CENTER_LAT, CHENNAI_CENTER_LON)
    fmap = folium.Map(
        location=[lat, lon],
        zoom_start=zoom,
        tiles="OpenStreetMap",
        control_scale=True,
        prefer_canvas=True,
    )
    return fmap


def add_legend(fmap: folium.Map) -> None:
    html = """
    <div style="position:fixed;bottom:24px;left:24px;z-index:9999;
                background:rgba(15,18,24,0.88);color:#e5e7eb;padding:10px 12px;
                border-radius:8px;font-size:12px;border:1px solid #334155;
                font-family:Segoe UI,sans-serif;max-width:280px;">
      <b>Estimated proximity coverage</b><br/>
      <div style="height:12px;width:220px;margin:6px 0 2px;
                  background:linear-gradient(to right,#00007f,#0000ff,#00ffff,#00ff00,#ffff00,#ff0000,#7f0000);
                  border-radius:2px;"></div>
      <div style="display:flex;justify-content:space-between;width:220px;font-size:11px;color:#cbd5e1;">
        <span>0 weak</span><span>50</span><span>100 strong</span>
      </div>
      White isolines label modelled score.<br/>
      Grey circles are assumed tower range.<br/>
      <span style="color:#60a5fa">â—</span> Selected location<br/>
      Not measured dBm / RSSI.
    </div>
    """
    fmap.get_root().html.add_child(folium.Element(html))


def add_tower_markers(fmap: folium.Map, towers: pd.DataFrame, cluster: bool = True) -> None:
    if towers.empty:
        return
    layer = MarkerCluster(name="Cell / tower records") if cluster else folium.FeatureGroup(name="Cell / tower records")
    if not cluster:
        layer.add_to(fmap)
    target = layer
    for row in towers.itertuples(index=False):
        measured = "Measured signal fields: none in this record."
        if bool(row.has_measured_signal):
            measured = "Record includes a verified measurement field."
        origin = "SYNTHETIC DEMO â€” not a real cell site." if bool(row.is_synthetic) else f"Source: {row.source}"
        html = (
            f"<b>ID {row.tower_id}</b> ({row.source_id})<br/>"
            f"{row.lat:.5f}, {row.lon:.5f}<br/>"
            f"Radio: {row.radio}<br/>"
            f"Operator: {row.operator}<br/>"
            f"{origin}<br/>{measured}"
        )
        color = "#f97316" if bool(row.is_synthetic) else "#38bdf8"
        folium.CircleMarker(
            location=[float(row.lat), float(row.lon)],
            radius=4,
            color=color,
            fill=True,
            fill_opacity=0.9,
            popup=folium.Popup(html, max_width=280),
            tooltip=f"{row.radio} Â· {row.operator}",
        ).add_to(target)
    if cluster:
        layer.add_to(fmap)


def add_coverage_circles(
    fmap: folium.Map,
    towers: pd.DataFrame,
    cfg: CoverageConfig,
    nearest_id: int | None = None,
) -> None:
    if towers.empty:
        return
    group = folium.FeatureGroup(name="Assumed coverage circles", show=True)
    for row in towers.itertuples(index=False):
        radius = assumed_radius(str(row.radio), cfg, getattr(row, "source_range_m", None))
        selected = nearest_id is not None and int(row.tower_id) == int(nearest_id)
        folium.Circle(
            location=[float(row.lat), float(row.lon)],
            radius=radius,
            color="#fbbf24" if selected else "#64748b",
            weight=2 if selected else 1,
            fill=True,
            fill_opacity=0.08 if not selected else 0.16,
            fill_color="#f59e0b" if selected else "#94a3b8",
        ).add_to(group)
    group.add_to(fmap)


def add_user_marker(fmap: folium.Map, lat: float, lon: float) -> None:
    folium.Marker(
        location=[lat, lon],
        tooltip="Selected location",
        popup=f"{lat:.5f}, {lon:.5f}",
        icon=folium.Icon(color="blue", icon="info-sign"),
    ).add_to(fmap)


def _field_shape(cfg: CoverageConfig, west: float, south: float, east: float, north: float) -> tuple[int, int]:
    max_cells = max(16, int(cfg.max_grid_cells))
    res = max(80.0, float(cfg.grid_resolution_m))
    sw = to_local_xy(west, south)
    ne = to_local_xy(east, north)
    width_m = max(1.0, ne[0] - sw[0])
    height_m = max(1.0, ne[1] - sw[1])
    nx = max(8, int(round(width_m / res)))
    ny = max(8, int(round(height_m / res)))
    if nx * ny > max_cells:
        scale = (max_cells / (nx * ny)) ** 0.5
        nx = max(8, int(nx * scale))
        ny = max(8, int(ny * scale))
    return nx, ny


def coverage_score_field(
    towers: pd.DataFrame,
    index: ZoneMappedColumnBlocks,
    cfg: CoverageConfig,
    allowed_ids: set[int] | None,
    *,
    lat: float | None = None,
    lon: float | None = None,
    half_extent_m: float | None = None,
    bounds: dict[str, float] | None = None,
) -> CoverageField:
    """Sample modelled scores on a regular lon/lat lattice for contour mapping."""
    if half_extent_m is not None and lat is not None and lon is not None:
        qx, qy = to_local_xy(lon, lat)
        west_lon, south_lat = to_wgs84(qx - half_extent_m, qy - half_extent_m)
        east_lon, north_lat = to_wgs84(qx + half_extent_m, qy + half_extent_m)
        west, south, east, north = west_lon, south_lat, east_lon, north_lat
    else:
        b = bounds or CHENNAI_BOUNDS
        south, west, north, east = b["south"], b["west"], b["north"], b["east"]

    nx, ny = _field_shape(cfg, west, south, east, north)
    lons = np.linspace(west, east, nx)
    lats = np.linspace(south, north, ny)
    scores = np.zeros((ny, nx), dtype=float)
    if towers is not None and not towers.empty:
        for j, clat in enumerate(lats):
            for i, clon in enumerate(lons):
                scores[j, i] = estimate_coverage(float(clat), float(clon), towers, index, cfg, allowed_ids).score
    return CoverageField(scores=scores, south=south, west=west, north=north, east=east)


def coverage_grid_polygons(
    lat: float,
    lon: float,
    towers: pd.DataFrame,
    index: ZoneMappedColumnBlocks,
    cfg: CoverageConfig,
    allowed_ids: set[int] | None,
    half_extent_m: float = 2500.0,
) -> list[tuple[list[list[float]], str, float]]:
    """Return (latlon_ring, category, score) for a local grid around the point."""
    field = coverage_score_field(
        towers,
        index,
        cfg,
        allowed_ids,
        lat=lat,
        lon=lon,
        half_extent_m=half_extent_m,
    )
    ny, nx = field.scores.shape
    dlon = (field.east - field.west) / max(nx, 1)
    dlat = (field.north - field.south) / max(ny, 1)
    cells: list[tuple[list[list[float]], str, float]] = []
    for j in range(ny):
        for i in range(nx):
            score = float(field.scores[j, i])
            result_cat = "green" if score >= cfg.green_threshold else "yellow" if score >= cfg.yellow_threshold else "red"
            lon0 = field.west + i * dlon
            lat0 = field.south + j * dlat
            ring = [
                [lat0, lon0],
                [lat0, lon0 + dlon],
                [lat0 + dlat, lon0 + dlon],
                [lat0 + dlat, lon0],
                [lat0, lon0],
            ]
            cells.append((ring, result_cat, score))
            if len(cells) >= cfg.max_grid_cells:
                return cells
    return cells


def _signal_colormap():
    """Red -> Yellow -> Green colormap: red = weak (outer areas), green = strong."""
    from matplotlib.colors import LinearSegmentedColormap
    return LinearSegmentedColormap.from_list(
        "signal",
        [
            (0.00, "#dc2626"),
            (0.35, "#facc15"),
            (0.65, "#22c55e"),
            (1.00, "#16a34a"),
        ],
    )


def _contour_png(field: CoverageField) -> bytes:
    """Smooth filled-contour gradient: red = weak signal, green = strong signal."""
    raw = np.clip(np.nan_to_num(field.scores, nan=0.0), 0.0, 100.0)
    sigma = 1.5 if min(raw.shape) >= 12 else 0.8
    scores = gaussian_filter(raw, sigma=sigma, mode="nearest")
    ny, nx = scores.shape
    aspect = max(0.45, min(2.2, (field.east - field.west) / max(field.north - field.south, 1e-9)))
    fig = Figure(figsize=(6.4 * aspect, 6.4), dpi=140, facecolor="none")
    FigureCanvasAgg(fig)
    ax = fig.add_axes([0.0, 0.0, 1.0, 1.0], facecolor="none")
    ax.set_axis_off()
    xs = np.linspace(field.west, field.east, nx)
    ys = np.linspace(field.south, field.north, ny)
    levels = np.linspace(0.0, 100.0, 40)
    ax.contourf(
        xs,
        ys,
        scores,
        levels=levels,
        cmap=_signal_colormap(),
        norm=Normalize(vmin=0.0, vmax=100.0),
        extend="neither",
        antialiased=True,
    )
    ax.set_xlim(field.west, field.east)
    ax.set_ylim(field.south, field.north)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", transparent=True, dpi=140)
    return buf.getvalue()


def add_coverage_gradient(fmap: folium.Map, field: CoverageField) -> None:
    if field.scores.size == 0:
        return
    png = _contour_png(field)
    uri = "data:image/png;base64," + base64.b64encode(png).decode("ascii")
    folium.raster_layers.ImageOverlay(
        image=uri,
        bounds=[[field.south, field.west], [field.north, field.east]],
        opacity=0.62,
        name="Coverage quality gradient",
        overlay=True,
        control=True,
        show=True,
        pixelated=False,
    ).add_to(fmap)


def add_coverage_grid(fmap: folium.Map, cells: list[tuple[list[list[float]], str, float]]) -> None:
    """Backward-compatible discrete cells; prefer add_coverage_gradient."""
    group = folium.FeatureGroup(name="Coverage quality grid", show=True)
    for ring, category, score in cells:
        folium.Polygon(
            locations=ring,
            color=CATEGORY_COLORS[category],
            weight=0,
            fill=True,
            fill_opacity=0.28,
            popup=f"Modelled score {score:.1f} ({category})",
        ).add_to(group)
    group.add_to(fmap)


def add_delaunay(fmap: folium.Map, towers: pd.DataFrame, dcel: DCEL) -> None:
    segs = delaunay_segments_wgs(towers, dcel)
    group = folium.FeatureGroup(name="Delaunay neighbours", show=False)
    for (a, b) in segs:
        folium.PolyLine([a, b], color="#a78bfa", weight=1.2, opacity=0.7).add_to(group)
    group.add_to(fmap)


def voronoi_polygons_wgs(
    towers: pd.DataFrame,
    clip_bounds: dict[str, float] | None = None,
) -> list[tuple[list[list[float]], int]]:
    """Nearest-tower partition clipped to the map region. Not a radio footprint."""
    if towers is None or len(towers) < 3:
        return []
    b = clip_bounds or CHENNAI_BOUNDS
    sw = to_local_xy(b["west"], b["south"])
    ne = to_local_xy(b["east"], b["north"])
    clip = box(sw[0], sw[1], ne[0], ne[1])
    points = towers[["x", "y"]].to_numpy(dtype=float)
    ids = towers["tower_id"].to_numpy(dtype=int)
    try:
        vor = Voronoi(points)
    except Exception:
        return []

    # Finite regions via ridge reconstruction (standard SciPy Voronoi finite-region trick).
    center = points.mean(axis=0)
    ptp = points.max(axis=0) - points.min(axis=0)
    radius = max(float(ptp.max()) * 3.0, 5000.0)
    ridge_lines = []
    for (p1, p2), (v1, v2) in zip(vor.ridge_points, vor.ridge_vertices):
        if v1 >= 0 and v2 >= 0:
            ridge_lines.append(LineString([vor.vertices[v1], vor.vertices[v2]]))
        else:
            v = v1 if v1 >= 0 else v2
            if v < 0:
                continue
            t = points[p1] - points[p2]
            t = t / max((t**2).sum() ** 0.5, 1e-9)
            n = [t[1], -t[0]]
            midpoint = (points[p1] + points[p2]) / 2
            direction = 1 if ((midpoint - center) * n).sum() >= 0 else -1
            far = vor.vertices[v] + direction * radius * n
            ridge_lines.append(LineString([vor.vertices[v], far]))
    if not ridge_lines:
        return []
    regions = list(polygonize(unary_union(ridge_lines)))
    mp = MultiPoint([Point(x, y) for x, y in points])
    out = []
    for poly in regions:
        if not isinstance(poly, Polygon) or poly.is_empty:
            continue
        clipped = poly.intersection(clip)
        if clipped.is_empty:
            continue
        polys = [clipped] if isinstance(clipped, Polygon) else list(getattr(clipped, "geoms", []))
        for part in polys:
            if not isinstance(part, Polygon) or part.is_empty:
                continue
            c = part.representative_point()
            nearest_i = min(range(len(points)), key=lambda i: (points[i][0] - c.x) ** 2 + (points[i][1] - c.y) ** 2)
            ring = []
            for x, y in list(part.exterior.coords):
                lon, lat = to_wgs84(float(x), float(y))
                ring.append([lat, lon])
            out.append((ring, int(ids[nearest_i])))
    _ = mp  # retained to document that sites are the Voronoi generators
    return out


def add_voronoi(fmap: folium.Map, polygons: Iterable[tuple[list[list[float]], int]], towers: pd.DataFrame) -> None:
    group = folium.FeatureGroup(name="Nearest-tower Voronoi (geometric)", show=False)
    by_id = {int(r.tower_id): r for r in towers.itertuples(index=False)}
    palette = ["#0ea5e9", "#14b8a6", "#8b5cf6", "#f43f5e", "#84cc16", "#f97316"]
    for ring, tid in polygons:
        rec = by_id.get(tid)
        color = palette[tid % len(palette)]
        name = f"Nearest recorded ID {tid}" if rec is None else f"{rec.radio} Â· {rec.operator}"
        folium.Polygon(
            locations=ring,
            color=color,
            weight=1,
            fill=True,
            fill_opacity=0.12,
            popup=f"Geometric nearest-tower cell for {name}. Not a radio footprint.",
        ).add_to(group)
    group.add_to(fmap)
