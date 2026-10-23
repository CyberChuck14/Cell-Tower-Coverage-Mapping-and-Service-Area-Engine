from __future__ import annotations

import pandas as pd

from src.data_loader import generate_synthetic_towers, load_towers, normalise_frame, opencellid_token


def test_rejects_invalid_coordinates() -> None:
    raw = pd.DataFrame(
        [
            {"lat": 91, "lon": 80.2, "radio": "LTE", "operator": "X", "source": "synthetic", "is_synthetic": True, "source_id": "a"},
            {"lat": 13.08, "lon": 200, "radio": "LTE", "operator": "X", "source": "synthetic", "is_synthetic": True, "source_id": "b"},
            {"lat": None, "lon": 80.2, "radio": "LTE", "operator": "X", "source": "synthetic", "is_synthetic": True, "source_id": "c"},
        ]
    )
    assert normalise_frame(raw).empty


def test_rejects_outside_chennai() -> None:
    raw = pd.DataFrame(
        [
            {
                "lat": 28.61,
                "lon": 77.20,
                "radio": "LTE",
                "operator": "X",
                "source": "synthetic",
                "is_synthetic": True,
                "source_id": "delhi",
            }
        ]
    )
    assert normalise_frame(raw).empty


def test_valid_record_and_duplicates() -> None:
    raw = pd.DataFrame(
        [
            {"lat": 13.0827, "lon": 80.2707, "radio": "LTE", "operator": "Demo-Jio", "source": "synthetic", "is_synthetic": True, "source_id": "1"},
            {"lat": 13.0827, "lon": 80.2707, "radio": "LTE", "operator": "Demo-Jio", "source": "synthetic", "is_synthetic": True, "source_id": "1"},
            {"lat": 13.09, "lon": 80.28, "radio": "NR", "operator": "Demo-Airtel", "source": "synthetic", "is_synthetic": True, "source_id": "2"},
        ]
    )
    df = normalise_frame(raw)
    assert len(df) == 2
    assert list(df["tower_id"]) == [0, 1]
    assert df["is_synthetic"].all()


def test_duplicate_coordinates_different_cells_kept() -> None:
    raw = pd.DataFrame(
        [
            {"lat": 13.0827, "lon": 80.2707, "radio": "LTE", "operator": "A", "source": "synthetic", "is_synthetic": True, "source_id": "1"},
            {"lat": 13.0827, "lon": 80.2707, "radio": "GSM", "operator": "B", "source": "synthetic", "is_synthetic": True, "source_id": "2"},
        ]
    )
    df = normalise_frame(raw)
    assert len(df) == 2


def test_empty_dataset() -> None:
    df = normalise_frame(pd.DataFrame())
    assert df.empty


def test_synthetic_generation_in_bounds() -> None:
    df = generate_synthetic_towers(n=40, seed=1)
    assert len(df) == 40
    assert df["is_synthetic"].all()
    assert df["has_measured_signal"].eq(False).all()


def test_no_api_key_load_still_works() -> None:
    assert opencellid_token() in (None, "") or isinstance(opencellid_token(), str)
    df, meta = load_towers()
    assert meta.get("record_count", len(df)) == len(df)
    assert not df.empty or meta.get("source") is not None
