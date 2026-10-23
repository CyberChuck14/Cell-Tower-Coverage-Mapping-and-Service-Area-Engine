"""Zone-Mapped Column Blocks spatial index.

The plane is partitioned into vertical columns of width ``block_size_m``
(in the local UTM metre CRS). Each column is further split into square
blocks. Every occupied block stores:

* bounding rectangle (min/max X and Y)
* the tower IDs that fall inside it

Nearest-neighbour search uses a best-first expansion over neighbouring
blocks. A block is pruned when the minimum distance from the query point
to the block's AABB is already worse than the best tower found so far.
That guarantees the nearest tower is not missed merely because it sits
in an adjacent block.

Complexity (honest):
* Construction: O(n + B) time and O(n + B) space, where B is the number
  of occupied blocks (B <= n).
* Query: O(n) worst case if every block must be inspected (uniform dense
  data, or a query whose nearest neighbour is far and pruning fails).
* Average case on clustered local data: far fewer than n distance
  evaluations because AABB lower bounds discard empty/far columns.
  This is **not** O(log n) in the worst case.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from heapq import heappop, heappush
from typing import Iterable

import numpy as np
import pandas as pd

from src.config import DEFAULT_BLOCK_SIZE_M
from src.distance import aabb_min_distance_m, euclidean_m, to_local_xy


@dataclass
class SpatialBlock:
    block_id: int
    col: int
    row: int
    min_x: float
    max_x: float
    min_y: float
    max_y: float
    tower_ids: list[int] = field(default_factory=list)

    def min_distance(self, qx: float, qy: float) -> float:
        return aabb_min_distance_m(qx, qy, self.min_x, self.min_y, self.max_x, self.max_y)


class ZoneMappedColumnBlocks:
    def __init__(
        self,
        towers: pd.DataFrame,
        block_size_m: float = DEFAULT_BLOCK_SIZE_M,
    ) -> None:
        self.block_size_m = float(block_size_m)
        self.towers = towers.reset_index(drop=True)
        self._by_id = {int(r.tower_id): r for r in self.towers.itertuples(index=False)}
        self.blocks: list[SpatialBlock] = []
        self._index: dict[tuple[int, int], SpatialBlock] = {}
        self.origin_x = 0.0
        self.origin_y = 0.0
        self.n_cols = 0
        self.n_rows = 0
        if not self.towers.empty:
            self._build()

    def _build(self) -> None:
        xs = self.towers["x"].to_numpy(dtype=float)
        ys = self.towers["y"].to_numpy(dtype=float)
        self.origin_x = float(xs.min())
        self.origin_y = float(ys.min())
        span_x = max(self.block_size_m, float(xs.max() - self.origin_x))
        span_y = max(self.block_size_m, float(ys.max() - self.origin_y))
        self.n_cols = int(math.floor(span_x / self.block_size_m)) + 1
        self.n_rows = int(math.floor(span_y / self.block_size_m)) + 1

        buckets: dict[tuple[int, int], list[int]] = {}
        for row in self.towers.itertuples(index=False):
            col, r = self._locate_indices(float(row.x), float(row.y))
            buckets.setdefault((col, r), []).append(int(row.tower_id))

        blocks: list[SpatialBlock] = []
        for bid, ((col, r), ids) in enumerate(sorted(buckets.items())):
            min_x = self.origin_x + col * self.block_size_m
            min_y = self.origin_y + r * self.block_size_m
            block = SpatialBlock(
                block_id=bid,
                col=col,
                row=r,
                min_x=min_x,
                max_x=min_x + self.block_size_m,
                min_y=min_y,
                max_y=min_y + self.block_size_m,
                tower_ids=ids,
            )
            blocks.append(block)
            self._index[(col, r)] = block
        self.blocks = blocks

    def _locate_indices(self, x: float, y: float) -> tuple[int, int]:
        if not math.isfinite(x) or not math.isfinite(y) or self.n_cols == 0:
            return 0, 0
        col = int(math.floor((x - self.origin_x) / self.block_size_m))
        row = int(math.floor((y - self.origin_y) / self.block_size_m))
        col = min(max(col, 0), max(self.n_cols - 1, 0))
        row = min(max(row, 0), max(self.n_rows - 1, 0))
        return col, row

    def block_at(self, x: float, y: float) -> SpatialBlock | None:
        return self._index.get(self._locate_indices(x, y))

    def neighbouring_keys(self, col: int, row: int, radius: int) -> Iterable[tuple[int, int]]:
        for dc in range(-radius, radius + 1):
            for dr in range(-radius, radius + 1):
                yield col + dc, row + dr

    def _tower_xy(self, tower_id: int) -> tuple[float, float]:
        rec = self._by_id[tower_id]
        return float(rec.x), float(rec.y)

    def nearest(
        self,
        lat: float,
        lon: float,
        allowed_ids: set[int] | None = None,
        k: int = 1,
    ) -> list[tuple[int, float]]:
        """Return up to k nearest (tower_id, distance_m) pairs.

        Expanding-block / best-first search. Empty index returns [].
        """
        if self.towers.empty:
            return []
        qx, qy = to_local_xy(lon, lat)
        start_col, start_row = self._locate_indices(qx, qy)

        heap: list[tuple[float, int, int]] = []
        heappush(heap, (0.0, start_col, start_row))
        visited: set[tuple[int, int]] = set()
        best: list[tuple[float, int]] = []  # (-distance for easy? use sorted list)
        # Keep a max-heap of the current k best as (-dist, id) inverted: store (dist, id) and sort.
        found: list[tuple[float, int]] = []

        def cutoff() -> float:
            if k <= 0:
                return 0.0
            if len(found) < k:
                return math.inf
            found.sort()
            return found[k - 1][0]

        while heap:
            dmin, col, row = heappop(heap)
            key = (col, row)
            if key in visited:
                continue
            visited.add(key)
            if dmin > cutoff():
                continue
            block = self._index.get(key)
            if block is not None:
                for tid in block.tower_ids:
                    if allowed_ids is not None and tid not in allowed_ids:
                        continue
                    tx, ty = self._tower_xy(tid)
                    dist = euclidean_m(qx, qy, tx, ty)
                    found.append((dist, tid))
                found.sort()
                found[:] = found[:k]
            # Expand to the 8-neighbourhood (and further rings via the heap).
            for nc, nr in self.neighbouring_keys(col, row, 1):
                if (nc, nr) in visited:
                    continue
                if nc < 0 or nr < 0 or nc >= self.n_cols or nr >= self.n_rows:
                    continue
                nblock = self._index.get((nc, nr))
                if nblock is None:
                    # Empty block: still need a lower bound using the geometric cell.
                    min_x = self.origin_x + nc * self.block_size_m
                    min_y = self.origin_y + nr * self.block_size_m
                    nd = aabb_min_distance_m(
                        qx,
                        qy,
                        min_x,
                        min_y,
                        min_x + self.block_size_m,
                        min_y + self.block_size_m,
                    )
                else:
                    nd = nblock.min_distance(qx, qy)
                if nd <= cutoff():
                    heappush(heap, (nd, nc, nr))

        found.sort()
        return [(tid, dist) for dist, tid in found[:k]]

    def range_query(
        self,
        lat: float,
        lon: float,
        radius_m: float,
        allowed_ids: set[int] | None = None,
    ) -> list[tuple[int, float]]:
        if self.towers.empty or radius_m < 0:
            return []
        qx, qy = to_local_xy(lon, lat)
        results: list[tuple[int, float]] = []
        for block in self.blocks:
            if block.min_distance(qx, qy) > radius_m:
                continue
            for tid in block.tower_ids:
                if allowed_ids is not None and tid not in allowed_ids:
                    continue
                tx, ty = self._tower_xy(tid)
                dist = euclidean_m(qx, qy, tx, ty)
                if dist <= radius_m:
                    results.append((tid, dist))
        results.sort(key=lambda t: t[1])
        return results

    def brute_force_nearest(
        self,
        lat: float,
        lon: float,
        allowed_ids: set[int] | None = None,
        k: int = 1,
    ) -> list[tuple[int, float]]:
        if self.towers.empty:
            return []
        qx, qy = to_local_xy(lon, lat)
        scored: list[tuple[float, int]] = []
        for row in self.towers.itertuples(index=False):
            tid = int(row.tower_id)
            if allowed_ids is not None and tid not in allowed_ids:
                continue
            dist = euclidean_m(qx, qy, float(row.x), float(row.y))
            scored.append((dist, tid))
        scored.sort()
        return [(tid, dist) for dist, tid in scored[:k]]


def rebuild_index(towers: pd.DataFrame, block_size_m: float = DEFAULT_BLOCK_SIZE_M) -> ZoneMappedColumnBlocks:
    return ZoneMappedColumnBlocks(towers, block_size_m=block_size_m)
