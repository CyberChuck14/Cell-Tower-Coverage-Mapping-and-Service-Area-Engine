from __future__ import annotations

from src.bitmap_filters import benchmark_filters, build_bitmaps, python_set_filter
from src.data_loader import generate_synthetic_towers


def test_intersection_union_difference() -> None:
    df = generate_synthetic_towers(n=60, seed=5)
    bm = build_bitmaps(df)
    radios = bm.radios()[:2]
    ops = bm.operators()[:2]
    filtered = bm.filter(radios=radios, operators=ops)
    baseline = python_set_filter(df, radios=radios, operators=ops)
    assert set(filtered) == baseline
    union = bm.by_radio[radios[0]] | bm.by_radio[radios[1]] if len(radios) > 1 else bm.by_radio[radios[0]]
    assert set(union) >= set(bm.by_radio[radios[0]])
    diff = bm.all_ids - bm.synthetic
    assert set(diff).isdisjoint(set(bm.synthetic))


def test_empty_filters() -> None:
    df = generate_synthetic_towers(n=10, seed=0)
    bm = build_bitmaps(df)
    assert set(bm.filter(radios=["NOT-A-RADIO"])) == set()
    empty = build_bitmaps(df.iloc[0:0])
    assert len(empty.all_ids) == 0


def test_benchmark_equality() -> None:
    df = generate_synthetic_towers(n=80, seed=9)
    radios = ["LTE"]
    ops = sorted(df["operator"].unique())[:2]
    result = benchmark_filters(df, radios=radios, operators=list(ops), repeats=8)
    assert result["results_equal"] is True
    assert result["backend"] == "pyroaring.BitMap"
