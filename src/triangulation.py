"""Delaunay triangulation with an explicit Half-Edge / DCEL representation.

SciPy computes the Delaunay triangulation (Bowyer-Watson / Qhull internally).
This module then builds our own Doubly Connected Edge List:

* Vertex — a site (tower) in projected metres
* HalfEdge — a directed edge with origin, twin, next, and incident face
* Face — a triangle referenced by one outer half-edge

The DCEL is the data structure this project implements and tests.
SciPy is only the triangle generator.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterator

import numpy as np
import pandas as pd
from scipy.spatial import Delaunay, QhullError


@dataclass
class Vertex:
    vid: int
    x: float
    y: float
    tower_id: int | None
    incident_edge: "HalfEdge | None" = None


@dataclass
class Face:
    fid: int
    outer: "HalfEdge | None" = None
    tower_ids: tuple[int | None, ...] = ()


@dataclass
class HalfEdge:
    hid: int
    origin: Vertex
    twin: "HalfEdge | None" = None
    next: "HalfEdge | None" = None
    face: Face | None = None

    @property
    def destination(self) -> Vertex | None:
        if self.next is None:
            return None
        return self.next.origin


@dataclass
class DCEL:
    vertices: list[Vertex] = field(default_factory=list)
    half_edges: list[HalfEdge] = field(default_factory=list)
    faces: list[Face] = field(default_factory=list)
    triangulation_backend: str = "scipy.spatial.Delaunay"
    notes: str = (
        "Triangle geometry comes from SciPy; half-edge linking is implemented here."
    )

    def face_edges(self, face: Face) -> list[HalfEdge]:
        if face.outer is None:
            return []
        edges = []
        e = face.outer
        start = e.hid
        for _ in range(8):
            edges.append(e)
            if e.next is None:
                break
            e = e.next
            if e.hid == start:
                break
        return edges

    def neighbouring_faces(self, face: Face) -> list[Face]:
        neighbours = []
        for e in self.face_edges(face):
            if e.twin is not None and e.twin.face is not None:
                neighbours.append(e.twin.face)
        return neighbours

    def adjacent_tower_ids(self, tower_id: int) -> set[int]:
        adj: set[int] = set()
        for v in self.vertices:
            if v.tower_id != tower_id:
                continue
            start = v.incident_edge
            if start is None:
                continue
            e = start
            seen = set()
            while e is not None and e.hid not in seen:
                seen.add(e.hid)
                dest = e.destination
                if dest is not None and dest.tower_id is not None:
                    adj.add(dest.tower_id)
                if e.twin is not None and e.twin.next is not None:
                    e = e.twin.next
                else:
                    break
                if e is start:
                    break
        adj.discard(tower_id)
        return adj

    def boundary_edges(self) -> list[HalfEdge]:
        return [e for e in self.half_edges if e.twin is None]


class DCELValidationError(Exception):
    pass


def validate_dcel(dcel: DCEL) -> list[str]:
    """Return a list of invariant violations (empty means valid)."""
    errors: list[str] = []
    for e in dcel.half_edges:
        if e.twin is not None:
            if e.twin.twin is not e:
                errors.append(f"twin.twin mismatch on half-edge {e.hid}")
            if e.twin.origin is not e.destination:
                errors.append(f"twin origin is not destination on {e.hid}")
        if e.next is None:
            errors.append(f"missing next on half-edge {e.hid}")
            continue
        if e.face is not None:
            cycle = dcel.face_edges(e.face)
            if len(cycle) != 3:
                errors.append(f"face {e.face.fid} is not a triangle ({len(cycle)} edges)")
            if cycle and cycle[0].next and cycle[0].next.next and cycle[0].next.next.next is not cycle[0]:
                errors.append(f"next cycle broken on face {e.face.fid}")
            for ce in cycle:
                if ce.face is not e.face:
                    errors.append(f"face pointer mismatch on {ce.hid}")
    for f in dcel.faces:
        if f.outer is None:
            errors.append(f"face {f.fid} has no outer edge")
    return errors


def assert_valid(dcel: DCEL) -> None:
    errs = validate_dcel(dcel)
    if errs:
        raise DCELValidationError("; ".join(errs[:8]))


def build_dcel_from_towers(towers: pd.DataFrame) -> DCEL:
    """Build a DCEL from unique projected tower coordinates."""
    dcel = DCEL()
    if towers is None or towers.empty:
        return dcel

    pts = []
    tower_ids = []
    seen: set[tuple[float, float]] = set()
    for row in towers.itertuples(index=False):
        key = (round(float(row.x), 3), round(float(row.y), 3))
        if key in seen:
            continue
        seen.add(key)
        pts.append((float(row.x), float(row.y)))
        tower_ids.append(int(row.tower_id))

    if len(pts) < 3:
        for i, ((x, y), tid) in enumerate(zip(pts, tower_ids)):
            dcel.vertices.append(Vertex(vid=i, x=x, y=y, tower_id=tid))
        return dcel

    points = np.asarray(pts, dtype=float)
    try:
        tri = Delaunay(points)
    except (QhullError, ValueError):
        for i, ((x, y), tid) in enumerate(zip(pts, tower_ids)):
            dcel.vertices.append(Vertex(vid=i, x=x, y=y, tower_id=tid))
        return dcel

    vertices = [
        Vertex(vid=i, x=float(points[i, 0]), y=float(points[i, 1]), tower_id=tower_ids[i])
        for i in range(len(points))
    ]
    dcel.vertices = vertices

    edge_map: dict[tuple[int, int], HalfEdge] = {}
    hid = 0
    for fid, simplex in enumerate(tri.simplices):
        a, b, c = (int(simplex[0]), int(simplex[1]), int(simplex[2]))
        # Orient CCW
        x1, y1 = points[a]
        x2, y2 = points[b]
        x3, y3 = points[c]
        if (x2 - x1) * (y3 - y1) - (y2 - y1) * (x3 - x1) < 0:
            b, c = c, b
        verts = (a, b, c)
        face = Face(fid=fid, tower_ids=(vertices[a].tower_id, vertices[b].tower_id, vertices[c].tower_id))
        edges = []
        for i in range(3):
            o = verts[i]
            he = HalfEdge(hid=hid, origin=vertices[o], face=face)
            hid += 1
            edges.append(he)
            dcel.half_edges.append(he)
            if vertices[o].incident_edge is None:
                vertices[o].incident_edge = he
        for i in range(3):
            edges[i].next = edges[(i + 1) % 3]
        face.outer = edges[0]
        dcel.faces.append(face)
        for i in range(3):
            o = verts[i]
            d = verts[(i + 1) % 3]
            edge_map[(o, d)] = edges[i]

    for (o, d), he in edge_map.items():
        twin = edge_map.get((d, o))
        if twin is not None:
            he.twin = twin

    return dcel


def delaunay_segments_wgs(towers: pd.DataFrame, dcel: DCEL) -> list[tuple[tuple[float, float], tuple[float, float]]]:
    """Unique undirected edges as ((lat, lon), (lat, lon)) for Folium."""
    from src.distance import to_wgs84

    seen: set[tuple[int, int]] = set()
    segs = []
    by_tid = {int(r.tower_id): r for r in towers.itertuples(index=False)}
    for e in dcel.half_edges:
        if e.destination is None:
            continue
        a = e.origin.tower_id
        b = e.destination.tower_id
        if a is None or b is None:
            continue
        key = (min(a, b), max(a, b))
        if key in seen:
            continue
        seen.add(key)
        ra, rb = by_tid.get(a), by_tid.get(b)
        if ra is None or rb is None:
            oa = to_wgs84(e.origin.x, e.origin.y)
            ob = to_wgs84(e.destination.x, e.destination.y)
            segs.append(((oa[1], oa[0]), (ob[1], ob[0])))
        else:
            segs.append(((float(ra.lat), float(ra.lon)), (float(rb.lat), float(rb.lon))))
    return segs


def adjacency_graph(dcel: DCEL) -> dict[int, set[int]]:
    """Simple adjacency list extracted from the DCEL for comparison."""
    graph: dict[int, set[int]] = {}
    for v in dcel.vertices:
        if v.tower_id is None:
            continue
        graph.setdefault(v.tower_id, set())
        graph[v.tower_id] |= dcel.adjacent_tower_ids(v.tower_id)
    return graph
