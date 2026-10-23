"""Chennai Cell Tower Signal Coverage – per-carrier demo (DSA Case Study).

Data basis: TRAI Independent Drive Test, Chennai, July 2026.
  - Airtel: 0.075% poor-signal samples  (densest, strongest network)
  - Jio:    0.285% poor-signal samples  (excellent; scattered weak pockets)
  - BSNL:   1.900% poor-signal samples  (notable gaps on western & northern corridors)
Tower positions are modelled from the TRAI July 2026 route-area evidence.
A SIM can ONLY use its own carrier's towers. Carrier signals are never mixed.
"""

from __future__ import annotations

import io
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import folium
import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from src.config import CHENNAI_BOUNDS, CHENNAI_CENTER_LAT, CHENNAI_CENTER_LON, CoverageConfig
from src.map_layers import CoverageField, add_coverage_gradient, base_map
from src.signal_model import assumed_radius
from src.spatial_grid import ZoneMappedColumnBlocks
from src.distance import to_local_xy_arrays

# ── Fixed Chennai demo locations ─────────────────────────────────────────────
DEMO_LOCATIONS: dict[str, tuple[float, float]] = {
    "Anna Nagar":        (13.0900, 80.2100),   # TRAI P1 priority corridor
    "Poes Garden":       (13.0349, 80.2506),
    "T. Nagar":          (13.0358, 80.2340),
    "Nungambakkam":      (13.0569, 80.2425),
    "Egmore":            (13.0780, 80.2707),   # TRAI named corridor
    "Kilpauk":           (13.0808, 80.2360),   # TRAI named corridor
    "Mylapore":          (13.0368, 80.2676),
    "Adyar":             (13.0063, 80.2574),
    "Saidapet":          (13.0200, 80.2230),   # TRAI P1 corridor
    "Guindy":            (13.0070, 80.2209),   # TRAI P1 corridor
    "Velachery":         (12.9815, 80.2180),
    "Perambur":          (13.1166, 80.2457),   # TRAI May-2025 area
    "Sholinganallur":    (12.9010, 80.2279),
    "Tambaram":          (12.9249, 80.1000),   # outer — western dead zone
    "Ambattur":          (13.1143, 80.1548),   # outer north-west
    "Koyambedu":         (13.0700, 80.1950),   # TRAI P1 corridor
    "Pammal":            (12.9700, 80.1400),   # TRAI P1 western corridor
    "Chromepet":         (12.9516, 80.1426),
}

# ── Per-carrier CoverageConfig ────────────────────────────────────────────────
# Tower radii are tuned so that:
#   Airtel → mostly green city-core, rare yellow/red (0.075% poor in TRAI)
#   Jio    → largely green, scattered yellow pockets (0.285% poor in TRAI)
#   BSNL   → significant yellow/red on western & outer corridors (1.9% poor in TRAI)

CARRIER_CFG: dict[str, CoverageConfig] = {}

# Airtel — densest private network, 26 towers, widest radii
_airtel = CoverageConfig(default_radius_m=2200.0, green_threshold=55.0, yellow_threshold=20.0)
_airtel.radius_by_radio_m = {"LTE": 2200.0, "GSM": 3000.0, "UMTS": 2500.0, "NR": 1500.0, "CDMA": 2000.0, "UNKNOWN": 2000.0}
CARRIER_CFG["Airtel"] = _airtel

# Jio — strong LTE network, 25 towers, slightly tighter radii than Airtel
_jio = CoverageConfig(default_radius_m=2000.0, green_threshold=55.0, yellow_threshold=20.0)
_jio.radius_by_radio_m = {"LTE": 2000.0, "GSM": 2800.0, "UMTS": 2200.0, "NR": 1400.0, "CDMA": 1800.0, "UNKNOWN": 1800.0}
CARRIER_CFG["Jio"] = _jio

# BSNL — older GSM network, 15 towers, smaller effective radii (weaker signal)
_bsnl = CoverageConfig(default_radius_m=1600.0, green_threshold=55.0, yellow_threshold=20.0)
_bsnl.radius_by_radio_m = {"GSM": 1800.0, "LTE": 1400.0, "UMTS": 1600.0, "NR": 1000.0, "CDMA": 1400.0, "UNKNOWN": 1400.0}
CARRIER_CFG["BSNL"] = _bsnl

# ── TRAI evidence banner per carrier ─────────────────────────────────────────
TRAI_BADGE = {
    "Airtel": ("✅ Airtel", "#22c55e", "TRAI Jul 2026: 0.075% poor samples · Strongest in Chennai"),
    "Jio":    ("📶 Jio",    "#3b82f6", "TRAI Jul 2026: 0.285% poor samples · Strong with scattered gaps"),
    "BSNL":   ("⚠️ BSNL",   "#f59e0b", "TRAI Jul 2026: 1.90% poor samples · Notable gaps (P1 corridors)"),
}

OPERATOR_DOT_COLOR = {"Jio": "#3b82f6", "Airtel": "#ef4444", "BSNL": "#22c55e"}

FIXED_TOWER_PATH = ROOT / "data" / "demo" / "chennai_towers_fixed.csv"

# ── Streamlit config ──────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Chennai Signal Coverage",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded",
)
st.markdown("""
<style>
  .block-container{padding-top:1rem;padding-bottom:.4rem;max-width:100%}
  header{background:transparent}
  [data-testid="stSidebar"]{background:#0f172a;color:#e2e8f0}
  .stMetric{background:#1e293b;border-radius:8px;padding:.5rem;margin-bottom:.4rem}
</style>
""", unsafe_allow_html=True)


# ── Data loading ──────────────────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def _load_towers() -> pd.DataFrame:
    df = pd.read_csv(FIXED_TOWER_PATH)
    xs, ys = to_local_xy_arrays(df["lon"].to_numpy(), df["lat"].to_numpy())
    df["x"] = xs
    df["y"] = ys
    df["tower_id"]            = df["tower_id"].astype(int)
    df["is_synthetic"]        = df["is_synthetic"].astype(bool)
    df["has_measured_signal"] = df["has_measured_signal"].astype(bool)
    df["source_range_m"]      = pd.to_numeric(df["source_range_m"], errors="coerce")
    return df


@st.cache_resource(show_spinner=False)
def _build_index(payload: bytes) -> ZoneMappedColumnBlocks:
    df = pd.read_json(io.BytesIO(payload))
    return ZoneMappedColumnBlocks(df, block_size_m=1500.0)


def _payload(df: pd.DataFrame) -> bytes:
    cols = [c for c in df.columns if c in {
        "tower_id", "source_id", "lat", "lon", "x", "y",
        "radio", "operator", "source", "is_synthetic",
        "has_measured_signal", "source_range_m",
    }]
    return df[cols].to_json(orient="records").encode("utf-8")


# ── Signal scoring ────────────────────────────────────────────────────────────

def composite_score(
    lat: float, lon: float,
    towers: pd.DataFrame,
    index: ZoneMappedColumnBlocks,
    cfg: CoverageConfig,
    allowed: set[int],
) -> tuple[float, int, float | None]:
    """Sum individual tower contributions (capped at 100).

    Each tower within assumed_radius adds: 100 * (1 – dist/radius).
    Summing means areas with multiple towers get high scores; sparse
    areas get low scores. This is a proximity model, NOT measured dBm.
    """
    max_r = max(cfg.radius_by_radio_m.values())
    candidates = index.range_query(lat, lon, max_r, allowed_ids=allowed)
    nearest = index.nearest(lat, lon, allowed_ids=allowed, k=1)
    nearest_d = nearest[0][1] if nearest else None

    total, contributing = 0.0, 0
    for tid, dist in candidates:
        row = towers.loc[towers["tower_id"] == tid]
        if row.empty:
            continue
        r = assumed_radius(str(row.iloc[0]["radio"]), cfg)
        if dist > r:
            continue
        total += 100.0 * (1.0 - dist / r)
        contributing += 1

    return min(100.0, total), contributing, nearest_d


def _score_color(score: float) -> str:
    if score >= 55:
        return "#22c55e"
    if score >= 20:
        return "#f59e0b"
    return "#ef4444"


def _score_label(score: float) -> str:
    if score >= 55:
        return "Good Signal"
    if score >= 20:
        return "Moderate / Fair"
    return "Weak / Dead Zone"


# ── Gradient field (per-carrier, cached by operator name) ────────────────────

@st.cache_data(show_spinner=False)
def _build_gradient_field(operator: str, tower_payload: bytes) -> CoverageField:
    """Build a 60×60 lat/lon lattice of composite scores for one carrier.

    Cached per operator — rebuilds only when towers change.
    """
    df = pd.read_json(io.BytesIO(tower_payload))
    # Recompute projected coords
    xs, ys = to_local_xy_arrays(df["lon"].to_numpy(), df["lat"].to_numpy())
    df["x"] = xs
    df["y"] = ys
    df["tower_id"] = df["tower_id"].astype(int)
    df["source_range_m"] = pd.to_numeric(df.get("source_range_m", pd.Series(dtype=float)), errors="coerce")

    op_towers = df[df["operator"] == operator].copy()
    allowed = set(int(i) for i in op_towers["tower_id"])
    index = ZoneMappedColumnBlocks(df, block_size_m=1500.0)
    cfg = CARRIER_CFG[operator]

    b = CHENNAI_BOUNDS
    nx, ny = 60, 60
    lons = np.linspace(b["west"], b["east"], nx)
    lats = np.linspace(b["south"], b["north"], ny)
    scores = np.zeros((ny, nx), dtype=float)
    for j, clat in enumerate(lats):
        for i, clon in enumerate(lons):
            s, _, _ = composite_score(float(clat), float(clon), op_towers, index, cfg, allowed)
            scores[j, i] = s

    return CoverageField(
        scores=scores,
        south=b["south"], west=b["west"],
        north=b["north"], east=b["east"],
    )


# ── Map builder ───────────────────────────────────────────────────────────────

def build_map(
    operator: str,
    op_towers: pd.DataFrame,
    field: CoverageField,
    sel: dict | None,
) -> folium.Map:
    center = (sel["lat"], sel["lon"]) if sel else (CHENNAI_CENTER_LAT, CHENNAI_CENTER_LON)
    fmap = base_map(center=center, zoom=13 if sel else 12)

    # Coverage gradient
    add_coverage_gradient(fmap, field)

    # Tower markers — only this carrier's towers
    dot_color = OPERATOR_DOT_COLOR[operator]
    grp = folium.FeatureGroup(name=f"{operator} cell towers", show=True)
    for row in op_towers.itertuples(index=False):
        folium.CircleMarker(
            location=[float(row.lat), float(row.lon)],
            radius=7,
            color="white",
            weight=1.5,
            fill=True,
            fill_color=dot_color,
            fill_opacity=0.9,
            tooltip=f"{operator} tower ({row.radio})",
            popup=folium.Popup(
                f"<b>{operator}</b> · {row.radio}<br/>"
                f"ID: {row.tower_id}<br/>"
                f"{row.lat:.4f}, {row.lon:.4f}<br/>"
                f"<i>Source: TRAI Jul 2026 modelled</i>",
                max_width=200,
            ),
        ).add_to(grp)
    grp.add_to(fmap)

    # Selected location pin
    if sel:
        folium.Marker(
            location=[sel["lat"], sel["lon"]],
            tooltip=sel["label"],
            popup=folium.Popup(
                f"<b>{sel['label']}</b><br/>{sel['lat']:.4f}, {sel['lon']:.4f}",
                max_width=180,
            ),
            icon=folium.Icon(color="red", icon="info-sign"),
        ).add_to(fmap)

    # Legend
    _, badge_color, _ = TRAI_BADGE[operator]
    legend_html = f"""
    <div style="position:fixed;bottom:20px;left:20px;z-index:9999;
                background:rgba(15,23,42,0.93);color:#f1f5f9;
                padding:12px 16px;border-radius:10px;font-size:13px;
                border:1px solid #334155;font-family:Segoe UI,sans-serif;
                min-width:210px;">
      <b style="font-size:14px;">{operator} Signal Strength</b>
      <div style="height:13px;width:190px;margin:7px 0 3px;
                  background:linear-gradient(to right,#dc2626,#f59e0b,#22c55e);
                  border-radius:4px;"></div>
      <div style="display:flex;justify-content:space-between;width:190px;
                  font-size:11px;color:#94a3b8;margin-bottom:8px;">
        <span>Weak / dead zone</span><span>Strong</span>
      </div>
      <hr style="border-color:#334155;margin:6px 0;"/>
      <span style="color:{badge_color};font-size:15px;">●</span>
      &nbsp;{operator} cell tower<br/>
      <span style="font-size:14px;">📍</span>&nbsp;Selected location<br/>
      <hr style="border-color:#334155;margin:6px 0;"/>
      <span style="font-size:10px;color:#64748b;">
        Proximity model · not measured dBm<br/>
        TRAI IDT Chennai Jul 2026 basis
      </span>
    </div>
    """
    fmap.get_root().html.add_child(folium.Element(legend_html))
    folium.LayerControl(collapsed=True).add_to(fmap)
    return fmap


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    towers_all = _load_towers()
    full_payload = _payload(towers_all)

    # ── Sidebar ───────────────────────────────────────────────────────────────
    with st.sidebar:
        st.title("📡 Chennai Cell Signal")
        st.caption("DSA Case Study · TRAI Jul 2026 basis")
        st.markdown("---")

        st.subheader("📱 Select Carrier")
        operator = st.radio(
            "Your SIM network:",
            options=["Jio", "Airtel", "BSNL"],
            index=0,
            help="Signal shown is ONLY from the selected carrier's towers. "
                 "A Jio SIM cannot use Airtel or BSNL towers.",
        )

        st.markdown("---")
        st.subheader("📍 Select Location")
        loc_choice = st.selectbox(
            "Choose an area in Chennai:",
            options=["— None —"] + list(DEMO_LOCATIONS.keys()),
        )

        # TRAI data badge
        st.markdown("---")
        badge_label, badge_color, badge_note = TRAI_BADGE[operator]
        st.markdown(
            f"<div style='background:{badge_color}22;border:1px solid {badge_color};"
            f"border-radius:8px;padding:8px 10px;font-size:12px;color:#e2e8f0;'>"
            f"<b>{badge_label}</b><br/>{badge_note}</div>",
            unsafe_allow_html=True,
        )

        # DSA info
        op_towers = towers_all[towers_all["operator"] == operator]
        op_index = _build_index(full_payload)

        st.markdown("---")
        st.subheader("DSA Structures")
        st.caption(
            f"• Zone-Mapped Column Blocks: **{len(op_index.blocks)}** cells\n\n"
            f"• {operator} towers indexed: **{len(op_towers)}**\n\n"
            f"• Signal lattice: **60 × 60** grid with Gaussian smoothing"
        )

    # ── Resolve location ──────────────────────────────────────────────────────
    sel: dict | None = None
    if loc_choice != "— None —" and loc_choice in DEMO_LOCATIONS:
        lat, lon = DEMO_LOCATIONS[loc_choice]
        sel = {"lat": lat, "lon": lon, "label": loc_choice}

    # ── Layout ────────────────────────────────────────────────────────────────
    info_col, map_col = st.columns([1, 3.2], gap="small")

    # Build per-operator filtered index
    cfg = CARRIER_CFG[operator]
    allowed: set[int] = set(int(i) for i in op_towers["tower_id"])

    with info_col:
        st.markdown(f"### {operator} Signal Report")

        if sel is None:
            st.info("Select a location from the sidebar to see signal strength.")
        else:
            score, contributing, nearest_d = composite_score(
                sel["lat"], sel["lon"], op_towers, op_index, cfg, allowed
            )
            color  = _score_color(score)
            label  = _score_label(score)

            # Score card
            st.markdown(
                f"""
                <div style="background:{color}18;border:2px solid {color};
                            border-radius:12px;padding:1rem;text-align:center;
                            margin-bottom:.75rem;">
                  <div style="font-size:2.8rem;font-weight:700;color:{color};">{score:.0f}</div>
                  <div style="font-size:1rem;color:{color};font-weight:600;">{label}</div>
                  <div style="font-size:.8rem;color:#94a3b8;margin-top:4px;">
                    {sel['label']} · {operator}
                  </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            c1, c2 = st.columns(2)
            c1.metric("Score", f"{score:.0f} / 100")
            if nearest_d is not None:
                c2.metric("Nearest tower", f"{nearest_d:.0f} m")
            st.caption(f"{operator} towers contributing: **{contributing}**")

            # Nearest towers
            near3 = op_index.nearest(sel["lat"], sel["lon"], allowed_ids=allowed, k=3)
            if near3:
                rows = []
                for tid, dist in near3:
                    rr = op_towers.loc[op_towers["tower_id"] == tid]
                    if not rr.empty:
                        r = rr.iloc[0]
                        rows.append({"Type": r["radio"], "Dist (m)": int(dist)})
                if rows:
                    st.markdown(f"**Nearest {operator} towers:**")
                    try:
                        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
                    except TypeError:
                        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

            # Status message
            if score >= 55:
                st.success(f"✅ {operator} signal is strong here.")
            elif score >= 20:
                st.warning(f"⚠️ {operator} signal is moderate – partial coverage.")
            else:
                st.error(
                    f"🔴 Weak / no {operator} signal here.\n\n"
                    f"Nearest {operator} tower: {int(nearest_d) if nearest_d else '?'} m away."
                )

        st.markdown("---")
        st.caption(
            "Score = proximity model (0–100). Not measured dBm/RSSI.  \n"
            "Tower positions modelled from TRAI IDT Chennai Jul 2026 route evidence."
        )

    with map_col:
        with st.spinner(f"Building {operator} signal map…"):
            field = _build_gradient_field(operator, full_payload)
            fmap = build_map(operator, op_towers, field, sel)
        map_html = fmap.get_root().render()
        components.html(map_html, height=720)


if __name__ == "__main__":
    main()
