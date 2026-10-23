from __future__ import annotations

from src.config import CoverageConfig
from src.data_loader import generate_synthetic_towers
from src.map_layers import _contour_png, coverage_score_field
from src.spatial_grid import ZoneMappedColumnBlocks


def test_score_field_shape_and_range() -> None:
    df = generate_synthetic_towers(n=30, seed=3)
    idx = ZoneMappedColumnBlocks(df)
    cfg = CoverageConfig(grid_resolution_m=800, max_grid_cells=100)
    allowed = set(int(i) for i in df["tower_id"])
    field = coverage_score_field(df, idx, cfg, allowed)
    assert field.scores.ndim == 2
    assert field.south < field.north
    assert field.west < field.east
    assert field.scores.min() >= 0.0
    assert field.scores.max() <= 100.0
    png = _contour_png(field)
    assert png[:8] == b"\x89PNG\r\n\x1a\n"
