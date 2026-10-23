"""Measure spatial search, bitmap filters, and coverage-grid generation."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from time import perf_counter

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.bitmap_filters import benchmark_filters
from src.config import CoverageConfig
from src.data_loader import generate_synthetic_towers, load_towers
from src.map_layers import coverage_grid_polygons
from src.spatial_grid import ZoneMappedColumnBlocks


def timed(fn, repeats: int = 5) -> float:
    t0 = perf_counter()
    for _ in range(repeats):
        fn()
    return (perf_counter() - t0) / repeats


def main() -> None:
    df, meta = load_towers()
    if df.empty:
        df = generate_synthetic_towers(n=200, seed=1)
        meta = {"source": "synthetic-inline", "record_count": len(df)}

    idx = ZoneMappedColumnBlocks(df, block_size_m=1500)
    queries = [(13.0827, 80.2707), (13.04, 80.22), (13.12, 80.29)]

    def brute():
        for lat, lon in queries:
            idx.brute_force_nearest(lat, lon, k=1)

    def indexed():
        for lat, lon in queries:
            idx.nearest(lat, lon, k=1)

    equal = True
    for lat, lon in queries:
        a = idx.nearest(lat, lon, k=5)
        b = idx.brute_force_nearest(lat, lon, k=5)
        if [t[0] for t in a] != [t[0] for t in b]:
            equal = False

    t_brute = timed(brute, repeats=20)
    t_idx = timed(indexed, repeats=20)

    radios = ["LTE"] if "LTE" in set(df["radio"]) else list(df["radio"].unique()[:1])
    ops = list(df["operator"].unique()[:2])
    filt = benchmark_filters(df, radios=list(radios), operators=ops, repeats=30)

    cfg_coarse = CoverageConfig(grid_resolution_m=500, max_grid_cells=400)
    cfg_fine = CoverageConfig(grid_resolution_m=250, max_grid_cells=900)
    allowed = set(int(i) for i in df["tower_id"])

    def grid(cfg):
        coverage_grid_polygons(13.0827, 80.2707, df, idx, cfg, allowed, half_extent_m=2000)

    t_coarse = timed(lambda: grid(cfg_coarse), repeats=3)
    t_fine = timed(lambda: grid(cfg_fine), repeats=3)

    report = {
        "dataset_source": meta.get("source"),
        "n": int(len(df)),
        "occupied_blocks": len(idx.blocks),
        "nearest_results_equal": equal,
        "brute_force_seconds_per_pass": t_brute,
        "zone_index_seconds_per_pass": t_idx,
        "filter_benchmark": filt,
        "coverage_grid_coarse_s": t_coarse,
        "coverage_grid_fine_s": t_fine,
        "notes": "Wall-clock on this machine; not a claim of asymptotic O(log n).",
    }
    out = ROOT / "benchmarks"
    out.mkdir(exist_ok=True)
    path = out / "results.json"
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
