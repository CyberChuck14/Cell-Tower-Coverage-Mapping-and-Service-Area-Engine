"""Load, validate, and normalise cell-site records."""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.config import (
    CHENNAI_BOUNDS,
    DEMO_DIR,
    PROCESSED_DIR,
    RAW_DIR,
    TOWER_COLUMNS,
)
from src.distance import to_local_xy_arrays, validate_lat_lon

SYNTHETIC_PATH = DEMO_DIR / "synthetic_chennai_towers.csv"
PROCESSED_PATH = PROCESSED_DIR / "towers.csv"
METADATA_PATH = PROCESSED_DIR / "dataset_meta.json"

INDIA_OPERATORS = {
    (404, 10): "Airtel",
    (404, 45): "Airtel",
    (404, 49): "Airtel",
    (404, 94): "Airtel",
    (405, 51): "Airtel",
    (405, 52): "Airtel",
    (404, 20): "Vodafone Idea",
    (404, 5): "Vodafone Idea",
    (404, 11): "Vodafone Idea",
    (404, 84): "Vodafone Idea",
    (404, 86): "Vodafone Idea",
    (405, 67): "Jio",
    (405, 840): "Jio",
    (405, 854): "Jio",
    (405, 855): "Jio",
    (405, 856): "Jio",
    (404, 71): "BSNL",
    (404, 72): "BSNL",
    (404, 80): "BSNL",
}


def _in_bounds(lat: float, lon: float, bounds: dict[str, float] | None = None) -> bool:
    b = bounds or CHENNAI_BOUNDS
    return b["south"] <= lat <= b["north"] and b["west"] <= lon <= b["east"]


def operator_from_mcc_mnc(mcc: Any, mnc: Any) -> str:
    try:
        key = (int(mcc), int(mnc))
    except (TypeError, ValueError):
        return "Unknown"
    return INDIA_OPERATORS.get(key, f"MCC {key[0]} / MNC {key[1]}")


def empty_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=TOWER_COLUMNS)


def validate_record(row: dict[str, Any]) -> dict[str, Any] | None:
    try:
        lat = float(row["lat"])
        lon = float(row["lon"])
        validate_lat_lon(lat, lon)
    except (KeyError, TypeError, ValueError):
        return None
    if not _in_bounds(lat, lon):
        return None
    radio = str(row.get("radio") or "UNKNOWN").strip().upper() or "UNKNOWN"
    if radio in {"GSM", "UMTS", "LTE", "NR", "CDMA", "UNKNOWN"}:
        pass
    source = str(row.get("source") or "unknown")
    is_synthetic = bool(row.get("is_synthetic", source == "synthetic"))
    out = {
        "source_id": str(row.get("source_id", "")),
        "lat": lat,
        "lon": lon,
        "radio": radio,
        "operator": str(row.get("operator") or "Unknown"),
        "mcc": row.get("mcc"),
        "mnc": row.get("mnc"),
        "source": source,
        "is_synthetic": is_synthetic,
        "has_measured_signal": bool(row.get("has_measured_signal", False)),
        "source_range_m": row.get("source_range_m"),
        "samples": row.get("samples"),
        "updated_unix": row.get("updated_unix"),
    }
    return out


def normalise_frame(df: pd.DataFrame, bounds: dict[str, float] | None = None) -> pd.DataFrame:
    if df is None or df.empty:
        return empty_frame()
    records: list[dict[str, Any]] = []
    seen: set[tuple[Any, ...]] = set()
    for row in df.to_dict(orient="records"):
        rec = validate_record(row)
        if rec is None:
            continue
        key = (
            round(rec["lat"], 6),
            round(rec["lon"], 6),
            rec["radio"],
            rec["operator"],
            rec["source_id"],
        )
        if key in seen:
            continue
        seen.add(key)
        records.append(rec)
    out = pd.DataFrame(records)
    if out.empty:
        return empty_frame()
    out = out.reset_index(drop=True)
    out.insert(0, "tower_id", np.arange(len(out), dtype=np.int32))
    xs, ys = to_local_xy_arrays(out["lon"].to_numpy(), out["lat"].to_numpy())
    out["x"] = xs
    out["y"] = ys
    return out


def generate_synthetic_towers(n: int = 96, seed: int = 42) -> pd.DataFrame:
    """Plausible Chennai-area points labelled as synthetic. Not real infrastructure."""
    rng = np.random.default_rng(seed)
    b = CHENNAI_BOUNDS
    radios = np.array(["GSM", "UMTS", "LTE", "NR"])
    operators = np.array(["Demo-Airtel", "Demo-Jio", "Demo-Vi", "Demo-BSNL"])
    # Cluster around a few neighbourhoods rather than a uniform scatter.
    centres = np.array(
        [
            [13.0827, 80.2707],  # Chennai Central
            [13.06, 80.25],  # T. Nagar / south-central
            [13.04, 80.17],  # west
            [13.11, 80.29],  # north
            [12.99, 80.22],  # south
            [13.09, 80.22],  # Anna Nagar-ish
        ]
    )
    rows = []
    for i in range(n):
        cy, cx = centres[i % len(centres)]
        lat = float(np.clip(cy + rng.normal(0, 0.018), b["south"] + 0.01, b["north"] - 0.01))
        lon = float(np.clip(cx + rng.normal(0, 0.018), b["west"] + 0.01, b["east"] - 0.01))
        radio = str(rng.choice(radios, p=[0.15, 0.2, 0.5, 0.15]))
        operator = str(rng.choice(operators))
        rows.append(
            {
                "source_id": f"SYN-{i:04d}",
                "lat": lat,
                "lon": lon,
                "radio": radio,
                "operator": operator,
                "mcc": 404,
                "mnc": 0,
                "source": "synthetic",
                "is_synthetic": True,
                "has_measured_signal": False,
                "source_range_m": math.nan,
                "samples": 0,
                "updated_unix": None,
            }
        )
    return normalise_frame(pd.DataFrame(rows))


def write_synthetic_demo(path: Path | None = None, n: int = 96) -> Path:
    path = path or SYNTHETIC_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    df = generate_synthetic_towers(n=n)
    keep = [c for c in TOWER_COLUMNS if c in df.columns]
    df[keep].to_csv(path, index=False)
    return path


def parse_opencellid_csv(path: Path, bounds: dict[str, float] | None = None) -> pd.DataFrame:
    """Map OpenCelliD export columns to the internal schema.

    OpenCelliD `range` is documented as estimated cell range in metres.
    `averageSignal` is deprecated and always 0, so it is not treated as RSSI.
    """
    usecols = ["radio", "mcc", "net", "area", "cell", "lon", "lat", "range", "samples", "updated"]
    chunks = []
    b = bounds or CHENNAI_BOUNDS
    for chunk in pd.read_csv(path, usecols=lambda c: c in usecols, chunksize=200_000):
        chunk = chunk.rename(columns={"net": "mnc", "lat": "lat", "lon": "lon"})
        mask = (
            chunk["lat"].between(b["south"], b["north"])
            & chunk["lon"].between(b["west"], b["east"])
            & chunk["mcc"].isin([404, 405])
        )
        sub = chunk.loc[mask].copy()
        if sub.empty:
            continue
        sub["source_id"] = (
            sub["radio"].astype(str)
            + "-"
            + sub["mcc"].astype(str)
            + "-"
            + sub["mnc"].astype(str)
            + "-"
            + sub["area"].astype(str)
            + "-"
            + sub["cell"].astype(str)
        )
        sub["operator"] = [
            operator_from_mcc_mnc(mcc, mnc) for mcc, mnc in zip(sub["mcc"], sub["mnc"])
        ]
        sub["source"] = "opencellid"
        sub["is_synthetic"] = False
        sub["has_measured_signal"] = False
        sub["source_range_m"] = sub.get("range")
        sub["updated_unix"] = sub.get("updated")
        chunks.append(sub)
    if not chunks:
        return empty_frame()
    raw = pd.concat(chunks, ignore_index=True)
    return normalise_frame(raw)


def parse_osm_overpass(payload: dict[str, Any]) -> pd.DataFrame:
    rows = []
    for el in payload.get("elements", []):
        if el.get("type") != "node":
            continue
        lat = el.get("lat")
        lon = el.get("lon")
        tags = el.get("tags") or {}
        raw_radio = str(
            tags.get("technology")
            or tags.get("communication:mobile")
            or tags.get("communication:mobile_phone")
            or "UNKNOWN"
        ).upper()
        if raw_radio in {"YES", "TRUE", "1"}:
            raw_radio = "UNKNOWN"
        radio = raw_radio
        operator = tags.get("operator") or tags.get("owner") or "Unknown (OSM)"
        rows.append(
            {
                "source_id": f"OSM-{el.get('id')}",
                "lat": lat,
                "lon": lon,
                "radio": str(radio).upper(),
                "operator": operator,
                "mcc": None,
                "mnc": None,
                "source": "osm",
                "is_synthetic": False,
                "has_measured_signal": False,
                "source_range_m": math.nan,
                "samples": None,
                "updated_unix": None,
            }
        )
    return normalise_frame(pd.DataFrame(rows)) if rows else empty_frame()


def save_processed(df: pd.DataFrame, meta: dict[str, Any], path: Path = PROCESSED_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cols = [c for c in TOWER_COLUMNS if c in df.columns]
    extra = [c for c in ("x", "y") if c in df.columns]
    df[cols + extra].to_csv(path, index=False)
    METADATA_PATH.write_text(json.dumps(meta, indent=2), encoding="utf-8")


def load_metadata() -> dict[str, Any]:
    if METADATA_PATH.exists():
        return json.loads(METADATA_PATH.read_text(encoding="utf-8"))
    return {}


def load_towers(prefer_processed: bool = True) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Load the best available dataset. Never invents coordinates as real."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    DEMO_DIR.mkdir(parents=True, exist_ok=True)

    if prefer_processed and PROCESSED_PATH.exists():
        df = pd.read_csv(PROCESSED_PATH)
        df = normalise_frame(df)
        meta = load_metadata() or {
            "source": "processed",
            "licence": "see data/README.md",
            "record_count": int(len(df)),
        }
        meta["record_count"] = int(len(df))
        return df, meta

    if SYNTHETIC_PATH.exists():
        df = normalise_frame(pd.read_csv(SYNTHETIC_PATH))
        meta = {
            "source": "synthetic",
            "licence": "Demonstration data generated by this project. Not real cell sites.",
            "attribution": "Synthetic DEMO — do not cite as telecom infrastructure.",
            "record_count": int(len(df)),
            "coverage": "Chennai demonstration bounding box",
            "is_synthetic": True,
        }
        return df, meta

    df = generate_synthetic_towers()
    write_synthetic_demo()
    meta = {
        "source": "synthetic",
        "licence": "Demonstration data generated by this project. Not real cell sites.",
        "attribution": "Synthetic DEMO — do not cite as telecom infrastructure.",
        "record_count": int(len(df)),
        "coverage": "Chennai demonstration bounding box",
        "is_synthetic": True,
    }
    return df, meta


def opencellid_token() -> str | None:
    token = os.environ.get("OPENCELLID_API_KEY") or os.environ.get("OPENCELLID_TOKEN")
    if token:
        return token.strip() or None
    return None
