from __future__ import annotations

import pandas as pd

from src.data_loader import generate_synthetic_towers, normalise_frame
from src.triangulation import (
    adjacency_graph,
    assert_valid,
    build_dcel_from_towers,
    validate_dcel,
)


def _pts(*coords: tuple[float, float]) -> pd.DataFrame:
    rows = []
    for i, (lat, lon) in enumerate(coords):
        rows.append(
            {
                "lat": lat,
                "lon": lon,
                "radio": "LTE",
                "operator": "D",
                "source": "synthetic",
                "is_synthetic": True,
                "source_id": str(i),
            }
        )
    return normalise_frame(pd.DataFrame(rows))


def test_degenerate_fewer_than_three() -> None:
    dcel = build_dcel_from_towers(_pts((13.08, 80.27), (13.09, 80.28)))
    assert dcel.faces == []
    assert len(dcel.vertices) == 2


def test_empty_and_duplicates() -> None:
    assert build_dcel_from_towers(pd.DataFrame()).faces == []
    dcel = build_dcel_from_towers(_pts((13.08, 80.27), (13.08, 80.27), (13.08, 80.27)))
    assert dcel.faces == []


def test_twin_and_next_on_triangle() -> None:
    df = _pts((13.08, 80.27), (13.09, 80.28), (13.07, 80.29), (13.10, 80.26))
    dcel = build_dcel_from_towers(df)
    assert dcel.faces
    assert_valid(dcel)
    twins = [e for e in dcel.half_edges if e.twin is not None]
    assert twins
    for e in twins:
        assert e.twin.twin is e
    for f in dcel.faces:
        edges = dcel.face_edges(f)
        assert len(edges) == 3
        assert edges[0].next.next.next is edges[0]


def test_boundary_half_edges() -> None:
    df = _pts((13.08, 80.27), (13.09, 80.28), (13.07, 80.29))
    dcel = build_dcel_from_towers(df)
    assert_valid(dcel)
    boundary = dcel.boundary_edges()
    assert boundary
    assert all(e.twin is None for e in boundary)


def test_synthetic_cloud_valid() -> None:
    df = generate_synthetic_towers(n=30, seed=3)
    dcel = build_dcel_from_towers(df)
    errs = validate_dcel(dcel)
    assert errs == []
    graph = adjacency_graph(dcel)
    assert graph
    sample = next(iter(graph))
    assert dcel.adjacent_tower_ids(sample) == graph[sample]
