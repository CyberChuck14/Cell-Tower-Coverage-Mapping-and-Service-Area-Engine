"""Nominatim geocoding with on-disk cache and rate limiting."""

from __future__ import annotations

import json
import time
from pathlib import Path

from geopy.exc import GeocoderServiceError, GeocoderTimedOut, GeocoderUnavailable
from geopy.geocoders import Nominatim

from src.config import GEOCODER_CACHE, NOMINATIM_MIN_DELAY_SECONDS, NOMINATIM_USER_AGENT

_last_request = 0.0


def _cache_path() -> Path:
    GEOCODER_CACHE.mkdir(parents=True, exist_ok=True)
    return GEOCODER_CACHE / "nominatim.json"


def _load_cache() -> dict:
    path = _cache_path()
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return {}
    return {}


def _save_cache(cache: dict) -> None:
    _cache_path().write_text(json.dumps(cache), encoding="utf-8")


def geocode_chennai(query: str, user_agent: str | None = None) -> tuple[float, float, str] | tuple[None, None, str]:
    """Return (lat, lon, display_name) or (None, None, error)."""
    q = (query or "").strip()
    if not q:
        return None, None, "Enter an address or place name."
    key = q.lower()
    cache = _load_cache()
    if key in cache:
        hit = cache[key]
        if hit and "lat" in hit:
            return float(hit["lat"]), float(hit["lon"]), str(hit.get("display_name") or q)
        return None, None, str(hit.get("error") or "Not found (cached).")

    global _last_request
    wait = NOMINATIM_MIN_DELAY_SECONDS - (time.time() - _last_request)
    if wait > 0:
        time.sleep(wait)

    geolocator = Nominatim(user_agent=user_agent or NOMINATIM_USER_AGENT, timeout=8)
    try:
        _last_request = time.time()
        location = geolocator.geocode(f"{q}, Chennai, Tamil Nadu, India")
    except (GeocoderTimedOut, GeocoderUnavailable, GeocoderServiceError) as exc:
        return None, None, f"Geocoding service error: {exc}. Use map click or manual coordinates."
    except Exception as exc:  # noqa: BLE001 — surface any geopy/network failure to the UI
        return None, None, f"Geocoding failed: {exc}. Use map click or manual coordinates."

    if location is None:
        cache[key] = {"error": "No result for that query in Chennai."}
        _save_cache(cache)
        return None, None, "No result for that query in Chennai."

    cache[key] = {
        "lat": location.latitude,
        "lon": location.longitude,
        "display_name": location.address,
    }
    _save_cache(cache)
    return float(location.latitude), float(location.longitude), str(location.address)
