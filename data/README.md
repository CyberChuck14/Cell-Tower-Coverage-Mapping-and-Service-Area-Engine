# Datasets

## Layout

- `raw/` — untouched downloads (OpenCelliD country extracts, OSM Overpass JSON). Not committed.
- `processed/` — cleaned, schema-normalised CSV used by the application.
- `demo/synthetic_chennai_towers.csv` — **synthetic demonstration records only**. Committed so the app runs without credentials.

## Sources (attempted in this order)

1. **OpenCelliD** (preferred authentic cell-location database)
   - Docs: https://docs.opencellid.org/docs/downloads/overview
   - Format: https://docs.opencellid.org/docs/downloads/database-format
   - Licence: CC BY-SA 4.0 — attribution required: https://docs.opencellid.org/docs/attribution
   - Access: API token via `OPENCELLID_API_KEY`
   - India MCC values typically 404 and 405
   - `range` is documented as an *estimated* cell range in metres from measurements; this app does **not** treat it as a calibrated radio footprint unless you explicitly enable that experimental option.
   - `averageSignal` is documented as deprecated and always `0`.

2. **OpenStreetMap Overpass** (licensed map features, not a complete cell inventory)
   - Licence: ODbL — https://www.openstreetmap.org/copyright
   - Features tagged as communication towers / masts in the Chennai bounding box.
   - These are genuine OSM locations, **not** a full operator cell-site database.

3. **Synthetic DEMO** (always available)
   - Plausible coordinates inside the Chennai demonstration rectangle.
   - Operator and radio values are illustrative labels, **not** real infrastructure.

## Internal schema

| Column | Meaning |
| --- | --- |
| `tower_id` | Stable integer index used by bitmaps and the spatial index |
| `source_id` | Original identifier from the source |
| `lat`, `lon` | WGS84 degrees |
| `radio` | Radio technology string |
| `operator` | Operator / network label |
| `mcc`, `mnc` | Mobile country/network codes when known |
| `source` | `opencellid` / `osm` / `synthetic` |
| `is_synthetic` | Boolean |
| `has_measured_signal` | Always false unless a verified measurement column is present |
| `source_range_m` | Source-provided range if documented; unused by default model |
| `samples` | Measurement count when provided by OpenCelliD |
| `updated_unix` | Last-updated timestamp when provided |

Run `python scripts/prepare_dataset.py` after cloning to refresh processed data.
