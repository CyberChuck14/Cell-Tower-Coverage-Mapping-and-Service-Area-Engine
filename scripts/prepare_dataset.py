"""Acquire and normalise tower records for the Chennai demonstration area.

Order:
1. OpenCelliD country/bulk extract if OPENCELLID_API_KEY is set.
2. OpenStreetMap Overpass communication towers (ODbL).
3. Synthetic DEMO dataset (always written so the app can run).
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import requests
from dotenv import load_dotenv

from src.config import CHENNAI_BOUNDS, PROCESSED_DIR, RAW_DIR
from src.data_loader import (
    empty_frame,
    generate_synthetic_towers,
    load_towers,
    opencellid_token,
    parse_opencellid_csv,
    parse_osm_overpass,
    save_processed,
    write_synthetic_demo,
)

OVERPASS_URL = "https://overpass-api.de/api/interpreter"
OPENCELLID_COUNTRY = "https://opencellid.org/ocid/downloads?token={token}&type=full&file=cell_towers.csv.gz"


def try_opencellid() -> tuple[object, dict]:
    token = opencellid_token()
    meta = {"attempted": True, "source": "opencellid", "ok": False}
    if not token:
        meta["error"] = "OPENCELLID_API_KEY not set; skipped authentic OpenCelliD download."
        return empty_frame(), meta
    dest = RAW_DIR / "cell_towers.csv.gz"
    url = OPENCELLID_COUNTRY.format(token=token)
    try:
        with requests.get(url, stream=True, timeout=60) as resp:
            if resp.status_code != 200:
                meta["error"] = f"OpenCelliD HTTP {resp.status_code} (token not logged)."
                return empty_frame(), meta
            RAW_DIR.mkdir(parents=True, exist_ok=True)
            with dest.open("wb") as fh:
                for chunk in resp.iter_content(1 << 20):
                    if chunk:
                        fh.write(chunk)
        df = parse_opencellid_csv(dest)
        meta.update(
            {
                "ok": not df.empty,
                "record_count": int(len(df)),
                "licence": "OpenCelliD CC BY-SA 4.0",
                "attribution": "Cell location data from OpenCelliD (unwiredlabs) — https://opencellid.org/",
                "path": str(dest),
            }
        )
        if df.empty:
            meta["error"] = "Download succeeded but no rows fell inside the Chennai bounding box / India MCC filter."
        return df, meta
    except Exception as exc:  # noqa: BLE001
        meta["error"] = f"OpenCelliD download failed: {type(exc).__name__}"
        return empty_frame(), meta


def try_osm() -> tuple[object, dict]:
    b = CHENNAI_BOUNDS
    query = f"""
    [out:json][timeout:45];
    (
      node["man_made"="mast"]({b['south']},{b['west']},{b['north']},{b['east']});
      node["man_made"="tower"]["tower:type"="communication"]({b['south']},{b['west']},{b['north']},{b['east']});
      node["man_made"="communications_tower"]({b['south']},{b['west']},{b['north']},{b['east']});
      node["communication:mobile_phone"]({b['south']},{b['west']},{b['north']},{b['east']});
    );
    out body;
    """
    meta = {"attempted": True, "source": "osm", "ok": False, "licence": "ODbL", "attribution": "© OpenStreetMap contributors"}
    try:
        headers = {
            "User-Agent": "cell-tower-coverage-dsa-case-study/1.0 (academic; contact: student)",
            "Accept": "application/json",
        }
        resp = requests.post(OVERPASS_URL, data={"data": query}, headers=headers, timeout=50)
        if resp.status_code != 200:
            meta["error"] = f"Overpass HTTP {resp.status_code}"
            return empty_frame(), meta
        payload = resp.json()
        RAW_DIR.mkdir(parents=True, exist_ok=True)
        (RAW_DIR / "osm_overpass.json").write_text(json.dumps(payload), encoding="utf-8")
        df = parse_osm_overpass(payload)
        meta["ok"] = not df.empty
        meta["record_count"] = int(len(df))
        if df.empty:
            meta["error"] = "Overpass returned no matching nodes in the bounding box."
        return df, meta
    except Exception as exc:  # noqa: BLE001
        meta["error"] = f"Overpass failed: {type(exc).__name__}: {exc}"
        return empty_frame(), meta


def main() -> None:
    load_dotenv(ROOT / ".env")
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    attempts = []
    chosen = empty_frame()
    chosen_meta: dict = {}

    ocid_df, ocid_meta = try_opencellid()
    attempts.append(ocid_meta)
    if ocid_meta.get("ok"):
        chosen, chosen_meta = ocid_df, ocid_meta

    osm_df, osm_meta = try_osm()
    attempts.append(osm_meta)
    if chosen.empty and osm_meta.get("ok"):
        chosen, chosen_meta = osm_df, osm_meta

    synthetic_path = write_synthetic_demo()
    syn = generate_synthetic_towers()
    attempts.append(
        {
            "source": "synthetic",
            "ok": True,
            "record_count": int(len(syn)),
            "path": str(synthetic_path),
            "licence": "Project-generated DEMO. Not real infrastructure.",
        }
    )
    if chosen.empty:
        chosen = syn
        chosen_meta = dict(attempts[-1])
        chosen_meta["is_synthetic"] = True
        chosen_meta["warning"] = (
            "Authentic OpenCelliD data was not loaded. The application is using "
            "labelled synthetic demonstration points inside the Chennai box."
        )

    chosen_meta = dict(chosen_meta)
    chosen_meta["attempts"] = attempts
    chosen_meta["coverage"] = CHENNAI_BOUNDS
    chosen_meta["record_count"] = int(len(chosen))
    save_processed(chosen, chosen_meta)
    print(json.dumps({k: v for k, v in chosen_meta.items() if k != "attempts"}, indent=2))
    print("--- attempts ---")
    print(json.dumps(attempts, indent=2, default=str))
    df, meta = load_towers()
    print(f"Loaded {len(df)} rows via load_towers(); source={meta.get('source')}")


if __name__ == "__main__":
    os.chdir(ROOT)
    main()
