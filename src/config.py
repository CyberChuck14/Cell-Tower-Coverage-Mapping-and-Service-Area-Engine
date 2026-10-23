"""Central configuration for the Chennai coverage engine."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
DEMO_DIR = DATA_DIR / "demo"
GEOCODER_CACHE = ROOT / ".cache" / "geopy"

# Chennai demonstration area (WGS84). Slightly larger than the municipal core.
CHENNAI_CENTER_LAT = 13.0827
CHENNAI_CENTER_LON = 80.2707
CHENNAI_BOUNDS = {
    "south": 12.82,
    "west": 80.08,
    "north": 13.28,
    "east": 80.35,
}

# UTM zone 44N is appropriate for ~80°E.
LOCAL_CRS = "EPSG:32644"
GEOGRAPHIC_CRS = "EPSG:4326"

# Assumed service radii (metres). These are modelling assumptions, not operator RF plans.
DEFAULT_RADIUS_M = 1000.0
RADIUS_BY_RADIO_M: dict[str, float] = {
    "GSM": 1500.0,
    "CDMA": 1400.0,
    "UMTS": 1200.0,
    "LTE": 900.0,
    "NR": 600.0,
    "UNKNOWN": DEFAULT_RADIUS_M,
}

GREEN_THRESHOLD = 60.0
YELLOW_THRESHOLD = 25.0

# Zone-mapped column blocks: columns along X, rows along Y in projected metres.
DEFAULT_BLOCK_SIZE_M = 1500.0

MAX_GRID_CELLS = 900
DEFAULT_GRID_RESOLUTION_M = 400.0

NOMINATIM_USER_AGENT = "cell-tower-coverage-dsa-case-study"
NOMINATIM_MIN_DELAY_SECONDS = 1.1

INDIA_MCC = {404, 405}

TOWER_COLUMNS = [
    "tower_id",
    "source_id",
    "lat",
    "lon",
    "radio",
    "operator",
    "mcc",
    "mnc",
    "source",
    "is_synthetic",
    "has_measured_signal",
    "source_range_m",
    "samples",
    "updated_unix",
]


@dataclass
class CoverageConfig:
    default_radius_m: float = DEFAULT_RADIUS_M
    radius_by_radio_m: dict[str, float] = field(default_factory=lambda: dict(RADIUS_BY_RADIO_M))
    green_threshold: float = GREEN_THRESHOLD
    yellow_threshold: float = YELLOW_THRESHOLD
    use_source_range: bool = False
    block_size_m: float = DEFAULT_BLOCK_SIZE_M
    grid_resolution_m: float = DEFAULT_GRID_RESOLUTION_M
    max_grid_cells: int = MAX_GRID_CELLS
