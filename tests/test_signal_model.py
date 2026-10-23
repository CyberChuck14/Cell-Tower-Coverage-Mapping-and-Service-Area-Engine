from __future__ import annotations

import pandas as pd

from src.config import CoverageConfig
from src.data_loader import generate_synthetic_towers, normalise_frame
from src.signal_model import category_from_score, estimate_coverage
from src.spatial_grid import ZoneMappedColumnBlocks


def test_thresholds() -> None:
    cfg = CoverageConfig()
    assert category_from_score(80, cfg) == "green"
    assert category_from_score(40, cfg) == "yellow"
    assert category_from_score(0, cfg) == "red"


def test_no_tower_inside_radius() -> None:
    raw = pd.DataFrame(
        [
            {
                "lat": 13.20,
                "lon": 80.32,
                "radio": "LTE",
                "operator": "D",
                "source": "synthetic",
                "is_synthetic": True,
                "source_id": "far",
            }
        ]
    )
    df = normalise_frame(raw)
    idx = ZoneMappedColumnBlocks(df, block_size_m=1000)
    cfg = CoverageConfig(default_radius_m=200)
    cfg.radius_by_radio_m = {k: 200 for k in cfg.radius_by_radio_m}
    result = estimate_coverage(12.90, 80.12, df, idx, cfg)
    assert result.score == 0
    assert result.category == "red"
    assert result.contributing == 0


def test_missing_area_flagged() -> None:
    df = generate_synthetic_towers(n=20, seed=2)
    idx = ZoneMappedColumnBlocks(df)
    result = estimate_coverage(8.09, 77.54, df, idx, CoverageConfig())
    assert result.outside_dataset or (result.nearest_distance_m or 0) > 50_000


def test_empty_dataset() -> None:
    idx = ZoneMappedColumnBlocks(pd.DataFrame())
    result = estimate_coverage(13.08, 80.27, pd.DataFrame(), idx)
    assert result.outside_dataset
    assert result.nearest_id is None


def test_score_decreases_with_distance() -> None:
    df = generate_synthetic_towers(n=40, seed=4)
    idx = ZoneMappedColumnBlocks(df)
    cfg = CoverageConfig()
    near = estimate_coverage(13.0827, 80.2707, df, idx, cfg)
    far = estimate_coverage(12.85, 80.09, df, idx, cfg)
    assert near.score >= far.score
