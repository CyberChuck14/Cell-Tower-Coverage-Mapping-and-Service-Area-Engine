from __future__ import annotations

import math

import pytest

from src.distance import aabb_min_distance_m, euclidean_m, haversine_m, to_local_xy, validate_lat_lon


def test_identical_points() -> None:
    assert haversine_m(13.0827, 80.2707, 13.0827, 80.2707) == 0.0
    x, y = to_local_xy(80.2707, 13.0827)
    assert euclidean_m(x, y, x, y) == 0.0


def test_known_chennai_distance_order() -> None:
    # Marina Beach area vs roughly Anna Nagar — several kilometres.
    d = haversine_m(13.0500, 80.2824, 13.0850, 80.2100)
    assert 6_000 < d < 12_000


def test_invalid_coordinates() -> None:
    with pytest.raises(ValueError):
        validate_lat_lon(95, 80)
    with pytest.raises(ValueError):
        haversine_m(13, 80, math.nan, 80)


def test_projected_local_is_metric_scale() -> None:
    x1, y1 = to_local_xy(80.2707, 13.0827)
    x2, y2 = to_local_xy(80.2807, 13.0827)
    d_xy = euclidean_m(x1, y1, x2, y2)
    d_h = haversine_m(13.0827, 80.2707, 13.0827, 80.2807)
    assert abs(d_xy - d_h) / d_h < 0.02


def test_aabb_inside_is_zero() -> None:
    assert aabb_min_distance_m(5, 5, 0, 0, 10, 10) == 0.0
    assert aabb_min_distance_m(15, 5, 0, 0, 10, 10) == 5.0
