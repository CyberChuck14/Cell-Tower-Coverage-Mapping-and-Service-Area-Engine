"""Roaring Bitmap filters over stable tower integer IDs.

PyRoaring stores compressed integer sets. We keep one bitmap per
categorical value (operator, radio, source) plus an ``all_ids`` universe.

Intersections of filter bitmaps yield candidate IDs that the spatial
index then searches. Ordinary Python sets are used only as a baseline
in benchmarks — they are not claimed to be Roaring Bitmaps.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from time import perf_counter

import pandas as pd
from pyroaring import BitMap


@dataclass
class TowerBitmaps:
    all_ids: BitMap = field(default_factory=BitMap)
    by_radio: dict[str, BitMap] = field(default_factory=dict)
    by_operator: dict[str, BitMap] = field(default_factory=dict)
    by_source: dict[str, BitMap] = field(default_factory=dict)
    synthetic: BitMap = field(default_factory=BitMap)
    real_location: BitMap = field(default_factory=BitMap)
    backend: str = "pyroaring.BitMap"

    def radios(self) -> list[str]:
        return sorted(self.by_radio)

    def operators(self) -> list[str]:
        return sorted(self.by_operator)

    def filter(
        self,
        radios: list[str] | None = None,
        operators: list[str] | None = None,
        sources: list[str] | None = None,
        exclude_synthetic: bool = False,
    ) -> BitMap:
        result = BitMap(self.all_ids)
        if radios:
            acc = BitMap()
            for r in radios:
                if r in self.by_radio:
                    acc |= self.by_radio[r]
            result &= acc
        if operators:
            acc = BitMap()
            for op in operators:
                if op in self.by_operator:
                    acc |= self.by_operator[op]
            result &= acc
        if sources:
            acc = BitMap()
            for s in sources:
                if s in self.by_source:
                    acc |= self.by_source[s]
            result &= acc
        if exclude_synthetic:
            result -= self.synthetic
        return result

    def as_set(self, bitmap: BitMap) -> set[int]:
        return set(bitmap)

    def intersect_with_candidates(self, bitmap: BitMap, candidate_ids: list[int] | set[int]) -> BitMap:
        return bitmap & BitMap(candidate_ids)


def build_bitmaps(towers: pd.DataFrame) -> TowerBitmaps:
    bm = TowerBitmaps()
    if towers is None or towers.empty:
        return bm
    for row in towers.itertuples(index=False):
        tid = int(row.tower_id)
        bm.all_ids.add(tid)
        radio = str(row.radio)
        operator = str(row.operator)
        source = str(row.source)
        bm.by_radio.setdefault(radio, BitMap()).add(tid)
        bm.by_operator.setdefault(operator, BitMap()).add(tid)
        bm.by_source.setdefault(source, BitMap()).add(tid)
        if bool(row.is_synthetic):
            bm.synthetic.add(tid)
        else:
            bm.real_location.add(tid)
    return bm


def python_set_filter(
    towers: pd.DataFrame,
    radios: list[str] | None = None,
    operators: list[str] | None = None,
) -> set[int]:
    ids = set(int(t) for t in towers["tower_id"].tolist()) if not towers.empty else set()
    if radios:
        ids &= set(int(t) for t in towers.loc[towers["radio"].isin(radios), "tower_id"])
    if operators:
        ids &= set(int(t) for t in towers.loc[towers["operator"].isin(operators), "tower_id"])
    return ids


def benchmark_filters(
    towers: pd.DataFrame,
    radios: list[str],
    operators: list[str],
    repeats: int = 50,
) -> dict[str, float | int | bool]:
    bitmaps = build_bitmaps(towers)
    t0 = perf_counter()
    last_bm = None
    for _ in range(repeats):
        last_bm = bitmaps.filter(radios=radios, operators=operators)
    t_bm = (perf_counter() - t0) / repeats

    t0 = perf_counter()
    last_set = None
    for _ in range(repeats):
        last_set = python_set_filter(towers, radios=radios, operators=operators)
    t_set = (perf_counter() - t0) / repeats

    equal = set(last_bm or []) == (last_set or set())
    return {
        "n": int(len(towers)),
        "repeats": repeats,
        "bitmap_seconds": t_bm,
        "python_set_seconds": t_set,
        "results_equal": equal,
        "backend": bitmaps.backend,
    }
