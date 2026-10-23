from __future__ import annotations

import pandas as pd

from src.data_loader import normalise_frame
from src.distance import to_local_xy
from src.spatial_grid import ZoneMappedColumnBlocks


def _towers(rows: list[dict]) -> pd.DataFrame:
    df = normalise_frame(pd.DataFrame(rows))
    return df


def _pt(lat: float, lon: float, source_id: str) -> dict:
    return {
        "lat": lat,
        "lon": lon,
        "radio": "LTE",
        "operator": "Demo",
        "source": "synthetic",
        "is_synthetic": True,
        "source_id": source_id,
    }


def test_nearest_in_same_block() -> None:
    df = _towers(
        [
            _pt(13.0827, 80.2707, "a"),
            _pt(13.0830, 80.2710, "b"),
        ]
    )
    idx = ZoneMappedColumnBlocks(df, block_size_m=5000)
    qlat, qlon = 13.08271, 80.27071
    got = idx.nearest(qlat, qlon, k=1)
    brute = idx.brute_force_nearest(qlat, qlon, k=1)
    assert got[0][0] == brute[0][0]


def test_nearest_in_neighbouring_block() -> None:
    # Force small blocks so two nearby points land in different cells.
    df = _towers(
        [
            _pt(13.0827, 80.2707, "west"),
            _pt(13.0827, 80.2900, "east"),
        ]
    )
    idx = ZoneMappedColumnBlocks(df, block_size_m=200)
    qlat, qlon = 13.0827, 80.2880  # closer to east
    got = idx.nearest(qlat, qlon, k=1)
    brute = idx.brute_force_nearest(qlat, qlon, k=1)
    assert got[0][0] == brute[0][0]
    east_id = int(df.loc[df["source_id"] == "east", "tower_id"].iloc[0])
    assert got[0][0] == east_id
    # Confirm they sit in different blocks when possible.
    wx, wy = to_local_xy(80.2707, 13.0827)
    ex, ey = to_local_xy(80.2900, 13.0827)
    assert idx._locate_indices(wx, wy) != idx._locate_indices(ex, ey)


def test_nearest_in_distant_block() -> None:
    df = _towers(
        [
            _pt(12.90, 80.12, "sw"),
            _pt(13.20, 80.32, "ne"),
        ]
    )
    idx = ZoneMappedColumnBlocks(df, block_size_m=800)
    qlat, qlon = 13.19, 80.31
    got = idx.nearest(qlat, qlon, k=1)
    brute = idx.brute_force_nearest(qlat, qlon, k=1)
    assert got[0][0] == brute[0][0]


def test_index_matches_brute_force_synthetic() -> None:
    from src.data_loader import generate_synthetic_towers

    df = generate_synthetic_towers(n=80, seed=7)
    idx = ZoneMappedColumnBlocks(df, block_size_m=1200)
    queries = [(13.0827, 80.2707), (13.05, 80.22), (13.15, 80.30), (12.95, 80.18)]
    for lat, lon in queries:
        a = idx.nearest(lat, lon, k=3)
        b = idx.brute_force_nearest(lat, lon, k=3)
        assert [t[0] for t in a] == [t[0] for t in b]


def test_range_query_boundary() -> None:
    df = _towers(
        [
            _pt(13.0827, 80.2707, "centre"),
            _pt(13.10, 80.30, "far"),
        ]
    )
    idx = ZoneMappedColumnBlocks(df, block_size_m=1000)
    cid = int(df.loc[df["source_id"] == "centre", "tower_id"].iloc[0])
    inside = idx.range_query(13.0827, 80.2707, 50)
    assert inside and inside[0][0] == cid
    none = idx.range_query(13.0827, 80.2707, 0)
    # distance 0 includes the tower at the query point
    assert any(t[0] == cid for t in none)
    empty = idx.range_query(13.20, 80.10, 10)
    assert empty == []


def test_empty_index() -> None:
    idx = ZoneMappedColumnBlocks(pd.DataFrame(), block_size_m=1000)
    assert idx.nearest(13.08, 80.27) == []
    assert idx.range_query(13.08, 80.27, 500) == []
