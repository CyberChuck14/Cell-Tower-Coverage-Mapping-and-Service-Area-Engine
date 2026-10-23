"""Distance-based coverage score. Not a calibrated radio-propagation model."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from src.config import CoverageConfig
from src.spatial_grid import ZoneMappedColumnBlocks


@dataclass
class CoverageResult:
    score: float
    category: str
    nearest_id: int | None
    nearest_distance_m: float | None
    contributing: int
    model_name: str
    assumptions: str
    outside_dataset: bool = False


def category_from_score(score: float, cfg: CoverageConfig) -> str:
    if score >= cfg.green_threshold:
        return "green"
    if score >= cfg.yellow_threshold:
        return "yellow"
    return "red"


def assumed_radius(radio: str, cfg: CoverageConfig, source_range_m: float | None = None) -> float:
    if cfg.use_source_range and source_range_m is not None:
        try:
            value = float(source_range_m)
            if value > 0:
                return value
        except (TypeError, ValueError):
            pass
    return cfg.radius_by_radio_m.get(str(radio).upper(), cfg.default_radius_m)


def _row_by_id(towers: pd.DataFrame, tower_id: int):
    hits = towers.loc[towers["tower_id"] == tower_id]
    if hits.empty:
        return None
    return hits.iloc[0]


def estimate_coverage(
    lat: float,
    lon: float,
    towers: pd.DataFrame,
    index: ZoneMappedColumnBlocks,
    cfg: CoverageConfig | None = None,
    allowed_ids: set[int] | None = None,
) -> CoverageResult:
    """Proximity model: score = 100 * max(0, 1 - distance / assumed_radius).

    Combined by taking the maximum score among candidates inside their
    assumed radius. This is an illustrative proximity score, not dBm,
    RSSI, or a measured percentage of signal strength.
    """
    cfg = cfg or CoverageConfig()
    assumptions = (
        "Modelled coverage uses distance to recorded cell locations and an "
        "assumed service radius by radio technology. It is not a measured "
        "RSRP/RSSI value and does not prove a real-world dead zone."
    )
    if towers is None or towers.empty:
        return CoverageResult(
            score=0.0,
            category="red",
            nearest_id=None,
            nearest_distance_m=None,
            contributing=0,
            model_name="distance-proximity",
            assumptions=assumptions,
            outside_dataset=True,
        )

    max_r = max(cfg.radius_by_radio_m.values()) if cfg.radius_by_radio_m else cfg.default_radius_m
    candidates = index.range_query(lat, lon, max_r, allowed_ids=allowed_ids)
    nearest = index.nearest(lat, lon, allowed_ids=allowed_ids, k=1)
    nearest_id = nearest[0][0] if nearest else None
    nearest_d = nearest[0][1] if nearest else None

    best = 0.0
    contributing = 0
    for tid, dist in candidates:
        rec = _row_by_id(towers, tid)
        if rec is None:
            continue
        radius = assumed_radius(str(rec.radio), cfg, rec.get("source_range_m") if hasattr(rec, "get") else rec["source_range_m"])
        if dist > radius:
            continue
        score = 100.0 * max(0.0, 1.0 - dist / radius)
        contributing += 1
        if score > best:
            best = score

    outside = nearest_d is not None and nearest_d > max_r * 2
    return CoverageResult(
        score=best,
        category=category_from_score(best, cfg),
        nearest_id=nearest_id,
        nearest_distance_m=nearest_d,
        contributing=contributing,
        model_name="distance-proximity",
        assumptions=assumptions,
        outside_dataset=outside or nearest_id is None,
    )
