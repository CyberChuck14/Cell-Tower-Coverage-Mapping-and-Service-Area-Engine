"""Geographic and projected distance helpers.

Haversine is used for WGS84 point-to-point distances.
Local metric work (blocks, Delaunay, Voronoi, grids) uses UTM zone 44N.
"""

from __future__ import annotations

import math
from functools import lru_cache

import numpy as np
from pyproj import Transformer

from src.config import GEOGRAPHIC_CRS, LOCAL_CRS

EARTH_RADIUS_M = 6_371_000.0


def validate_lat_lon(lat: float, lon: float) -> None:
    if lat is None or lon is None or not math.isfinite(lat) or not math.isfinite(lon):
        raise ValueError("Coordinates must be finite numbers.")
    if not -90.0 <= lat <= 90.0:
        raise ValueError(f"Latitude out of range: {lat}")
    if not -180.0 <= lon <= 180.0:
        raise ValueError(f"Longitude out of range: {lon}")


def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in metres using the haversine formula."""
    validate_lat_lon(lat1, lon1)
    validate_lat_lon(lat2, lon2)
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dl / 2) ** 2
    return 2 * EARTH_RADIUS_M * math.asin(min(1.0, math.sqrt(a)))


def haversine_many_m(
    lat: float,
    lon: float,
    lats: np.ndarray,
    lons: np.ndarray,
) -> np.ndarray:
    """Vectorised haversine from one query point to arrays of towers."""
    validate_lat_lon(lat, lon)
    phi1 = np.radians(lat)
    phi2 = np.radians(lats.astype(float))
    dphi = np.radians(lats.astype(float) - lat)
    dl = np.radians(lons.astype(float) - lon)
    a = np.sin(dphi / 2) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dl / 2) ** 2
    return 2 * EARTH_RADIUS_M * np.arcsin(np.minimum(1.0, np.sqrt(a)))


@lru_cache(maxsize=4)
def _to_local_transformer() -> Transformer:
    return Transformer.from_crs(GEOGRAPHIC_CRS, LOCAL_CRS, always_xy=True)


@lru_cache(maxsize=4)
def _to_wgs_transformer() -> Transformer:
    return Transformer.from_crs(LOCAL_CRS, GEOGRAPHIC_CRS, always_xy=True)


def to_local_xy(lon: float, lat: float) -> tuple[float, float]:
    """Convert WGS84 (lon, lat) to local metres (x easting, y northing)."""
    validate_lat_lon(lat, lon)
    x, y = _to_local_transformer().transform(lon, lat)
    return float(x), float(y)


def to_local_xy_arrays(lons: np.ndarray, lats: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x, y = _to_local_transformer().transform(lons.astype(float), lats.astype(float))
    return np.asarray(x, dtype=float), np.asarray(y, dtype=float)


def to_wgs84(x: float, y: float) -> tuple[float, float]:
    """Return (lon, lat) from local metres."""
    lon, lat = _to_wgs_transformer().transform(x, y)
    return float(lon), float(lat)


def euclidean_m(x1: float, y1: float, x2: float, y2: float) -> float:
    return math.hypot(x2 - x1, y2 - y1)


def aabb_min_distance_m(
    qx: float,
    qy: float,
    min_x: float,
    min_y: float,
    max_x: float,
    max_y: float,
) -> float:
    """Minimum Euclidean distance from a point to an axis-aligned rectangle.

    Zero if the point is inside the rectangle. Used as a lower bound so that
    a spatial block can be skipped when this value already exceeds the best
    tower distance found so far.
    """
    dx = 0.0 if min_x <= qx <= max_x else min(abs(qx - min_x), abs(qx - max_x))
    if qx < min_x:
        dx = min_x - qx
    elif qx > max_x:
        dx = qx - max_x
    else:
        dx = 0.0
    if qy < min_y:
        dy = min_y - qy
    elif qy > max_y:
        dy = qy - max_y
    else:
        dy = 0.0
    return math.hypot(dx, dy)
