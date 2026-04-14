from __future__ import annotations

import re
from datetime import datetime, timedelta
from typing import Any, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

import dash
from dash import Input, Output, State, dcc, html, ctx, dash_table, ALL
from dash.exceptions import PreventUpdate
import dash_bootstrap_components as dbc
import dash_mantine_components as dmc


PLANNING_ENERGY_MULTIPLIER = 250.0  # kWh per MW for a 15-min block (0.25h * 1000)

# Battery application: default is per 15-min block; keep as a constant so we can later switch to windowed mode.
DEFAULT_BATTERY_WINDOW_BLOCKS = 1


def _simplify_band_label(label: str) -> str:
    """Keep the % range part; strip extra explanatory text like '(scaled)'."""
    s = str(label or "").strip()
    if not s:
        return ""
    # strip trailing parentheses info
    s = s.split("(", 1)[0].strip()
    # normalize dash variants (en-dash/em-dash/minus/etc) so "10–15" == "10-15"
    s = re.sub(r"[\u2010\u2011\u2012\u2013\u2014\u2212]", "-", s)
    # normalize spacing around hyphen
    s = re.sub(r"\s*-\s*", "-", s)
    # normalize whitespace
    s = re.sub(r"\s+", " ", s)
    # normalize numeric formatting so "0.0" -> "0", "10.50" -> "10.5"
    # (helps match bands coming from config vs generated/parsed labels)
    s = re.sub(r"(\d+)\.(0+)(?=\D|$)", r"\1", s)
    s = re.sub(r"(\d+\.\d*?)0+(?=\D|$)", r"\1", s)
    s = re.sub(r"\.(?=\D|$)", "", s)  # drop trailing dot if any
    return s


def _band_label_from_energy_mw_col(col: str) -> tuple[str, str]:
    """Parse Analysis band MW column -> (stable_label, display_label).

    Examples:
      - "UI Energy 0-10% (MW)"  -> ("UI 0-10%", "UI 0-10")
      - "OI Energy 10-15% (MW)" -> ("OI 10-15%", "OI 10-15")
      - "UI Energy >15% (MW)"   -> ("UI >15%", "UI >15")
    """
    s = str(col or "").strip()
    m_gt = re.match(r"^(UI|OI)\s+Energy\s+>(\d+(?:\.\d+)?)%\s+\(MW\)\s*$", s, flags=re.IGNORECASE)
    if m_gt:
        d = m_gt.group(1).upper()
        lo = m_gt.group(2)
        return (f"{d} >{lo}%", f"{d} >{lo}")
    m_rng = re.match(r"^(UI|OI)\s+Energy\s+(\d+(?:\.\d+)?)\s*-\s*(\d+(?:\.\d+)?)%\s+\(MW\)\s*$", s, flags=re.IGNORECASE)
    if m_rng:
        d = m_rng.group(1).upper()
        lo = m_rng.group(2)
        hi = m_rng.group(3)
        return (f"{d} {lo}-{hi}%", f"{d} {lo}-{hi}")
    cleaned = s.replace("(MW)", "").strip()
    return (cleaned, cleaned)


def _long_band_df_from_bandwise_df(bw_df: pd.DataFrame) -> pd.DataFrame:
    """Convert authoritative bandwise (wide) df -> long per-(block, direction, band).

    Input expectation (DO NOT CHANGE upstream DSM logic):
      - One row per timestamp/block (and per plant if multiple plants selected)
      - Band energy columns: "UI Energy ... (MW)" / "OI Energy ... (MW)"
      - Band DSM columns:    "UI DSM ... (₹)"   / "OI DSM ... (₹)"

    Output columns (for battery post-processing):
      - block keys: region, plant_name, date, time_block, from_time, to_time, Custom Setting
      - direction: UI / OI
      - band_label: stable label like "UI 0-10%"
      - band_display: display label like "UI 0-10"
      - energy_kwh_before
      - band_dsm_before
      - band_penalty_rate (₹/kWh), computed as band_dsm_before / energy_kwh_before (safe; 0 if energy=0)
    """
    if bw_df is None or bw_df.empty:
        return pd.DataFrame()

    bw = bw_df.copy()

    # Mandatory filter (global rule): Schedule=0 must be dropped before any battery/stats
    if "Scheduled_MW" in bw.columns:
        bw["Scheduled_MW"] = pd.to_numeric(bw["Scheduled_MW"], errors="coerce").fillna(0.0)
        bw = bw[bw["Scheduled_MW"] > 0].copy()

    band_mw_cols = [c for c in bw.columns if isinstance(c, str) and c.endswith("(MW)") and ("UI Energy" in c or "OI Energy" in c)]
    if not band_mw_cols:
        return pd.DataFrame()

    rows = []
    keep_cols = [c for c in ("region", "plant_name", "date", "time_block", "from_time", "to_time", "Custom Setting") if c in bw.columns]
    base = bw[keep_cols].copy() if keep_cols else pd.DataFrame(index=bw.index)

    for mw_col in band_mw_cols:
        stable_label, display_label = _band_label_from_energy_mw_col(mw_col)
        stable_label = _simplify_band_label(stable_label)
        direction = "UI" if stable_label.upper().startswith("UI") else ("OI" if stable_label.upper().startswith("OI") else "")

        energy_kwh = pd.to_numeric(bw[mw_col], errors="coerce").fillna(0.0) * PLANNING_ENERGY_MULTIPLIER

        dsm_col = mw_col.replace("Energy", "DSM").replace("(MW)", "(₹)")
        band_dsm = pd.to_numeric(bw[dsm_col], errors="coerce").fillna(0.0) if dsm_col in bw.columns else pd.Series(0.0, index=bw.index)

        # Effective penalty rate per kWh (no formula invention; derived from existing DSM output)
        with np.errstate(divide="ignore", invalid="ignore"):
            rate = np.where(energy_kwh.to_numpy() > 0, (band_dsm.to_numpy() / energy_kwh.to_numpy()), 0.0)

        tmp = base.copy()
        tmp["direction"] = direction
        tmp["band_label"] = stable_label
        tmp["band_display"] = display_label
        tmp["energy_kwh_before"] = energy_kwh.to_numpy()
        tmp["band_dsm_before"] = band_dsm.to_numpy()
        tmp["band_penalty_rate"] = rate
        rows.append(tmp)

    out = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    return out


def apply_battery_blockwise(
    df_long: pd.DataFrame,
    battery_kwh: float,
    window_blocks: int = DEFAULT_BATTERY_WINDOW_BLOCKS,
) -> pd.DataFrame:
    """Apply battery per block (or per window of blocks) WITHOUT changing DSM/band logic.

    Business rules:
      - Battery applies independently to UI and OI
      - By default battery_kwh applies per 15-min block (window_blocks=1)
      - Within each (block/window, direction), reduce highest penalty bands first
      - DSM after is recomputed ONLY as: energy_after_kwh * band_penalty_rate
      - No reslicing, no recompute of deviation %, no band logic changes
    """
    if df_long is None or df_long.empty:
        return pd.DataFrame()
    try:
        cap = float(battery_kwh or 0.0)
    except Exception:
        cap = 0.0
    if cap <= 0:
        out = df_long.copy()
        out["battery_applied_kwh"] = 0.0
        out["energy_kwh_after"] = pd.to_numeric(out.get("energy_kwh_before"), errors="coerce").fillna(0.0)
        out["band_dsm_after"] = pd.to_numeric(out.get("band_dsm_before"), errors="coerce").fillna(0.0)
        out["window_id"] = 0
        return out

    out = df_long.copy()
    out["energy_kwh_before"] = pd.to_numeric(out.get("energy_kwh_before"), errors="coerce").fillna(0.0)
    out["band_penalty_rate"] = pd.to_numeric(out.get("band_penalty_rate"), errors="coerce").fillna(0.0)
    out["band_dsm_before"] = pd.to_numeric(out.get("band_dsm_before"), errors="coerce").fillna(0.0)

    # Build grouping key for per-block processing (per plant + timestamp)
    key_cols = [c for c in ("region", "plant_name", "date", "time_block", "Custom Setting") if c in out.columns]
    if not key_cols:
        key_cols = ["_row"]  # fallback: treat as one big group
        out["_row"] = 0

    # Assign window_id: keeps future N-block mode pluggable
    window_blocks = int(window_blocks or 1)
    window_blocks = max(1, window_blocks)
    out["_order"] = np.arange(len(out))

    def _assign_window_ids(g: pd.DataFrame) -> pd.DataFrame:
        # sort by (date, time_block) if present; else stable order
        sort_cols = [c for c in ("date", "time_block") if c in g.columns]
        gg = g.sort_values(sort_cols + ["_order"] if sort_cols else ["_order"]).copy()
        gg["window_id"] = (np.arange(len(gg)) // window_blocks).astype(int)
        return gg

    out = out.groupby(key_cols, group_keys=False).apply(_assign_window_ids)

    # Allocate per (key, window_id, direction)
    out["battery_applied_kwh"] = 0.0
    out["energy_kwh_after"] = out["energy_kwh_before"].copy()

    group_cols = key_cols + ["window_id", "direction"]
    for _, gidx in out.groupby(group_cols).groups.items():
        idx = list(gidx)
        # Sort within the block/window by penalty severity (desc), tie-breaker by energy (desc)
        sub = out.loc[idx, ["band_penalty_rate", "energy_kwh_before"]].copy()
        order = sub.sort_values(["band_penalty_rate", "energy_kwh_before"], ascending=[False, False]).index.tolist()

        remaining = cap
        for i in order:
            if remaining <= 0:
                break
            e = float(out.at[i, "energy_kwh_after"])
            if e <= 0:
                continue
            red = min(remaining, e)
            out.at[i, "battery_applied_kwh"] = float(red)
            out.at[i, "energy_kwh_after"] = float(e - red)
            remaining -= red

    # DSM recomputation ONLY using remaining band energy (no new DSM logic)
    out["band_dsm_after"] = out["energy_kwh_after"] * out["band_penalty_rate"]

    # Cleanup helper columns
    out = out.drop(columns=["_order", "_row"], errors="ignore")
    return out


def _slice_pct(abs_err: float, lower: float, upper: float) -> float:
    """Compute the % slice that falls within [lower, upper) for a given abs_err.
    This is the single source of truth for DSM band slicing (matches dsm_dashboard.py).
    """
    return max(0.0, min(abs_err, upper) - lower)


def _kwh_from_slice(slice_pct_val: float, basis_mw: float) -> float:
    """Convert a band slice (%) to energy (kWh) for a 15-min block.
    15-min = 0.25 h; MW → kW × 1000; energy = P(kW) × h
    Matches dsm_dashboard.py and Sheet-3 Excel formula.
    """
    return (slice_pct_val / 100.0) * basis_mw * 0.25 * 1000.0


def _build_band_lookup(bands_rows: list[dict]) -> dict[str, tuple[float, float]]:
    """Build a lookup: simplified_band_label → (lower_pct, upper_pct)."""
    lookup = {}
    if not bands_rows:
        return lookup
    for b in bands_rows:
        label = str(b.get("label", "")).strip()
        if not label:
            continue
        simplified = _simplify_band_label(label)
        if not simplified:
            continue
        lower = float(b.get("lower_pct", 0.0))
        upper = float(b.get("upper_pct", 1000.0))
        lookup[simplified] = (lower, upper)
    return lookup


def _get_band_penalty_rate(band_row: dict) -> tuple[str, float]:
    """Extract penalty rate info from a band configuration row.
    
    Returns (rate_type, rate_value) where:
    - rate_type: 'none', 'ppa_fraction', 'ppa_multiple', 'flat_per_kwh', etc.
    - rate_value: the numeric rate (0 for free zones)
    """
    rate_type = str(band_row.get("rate_type", "")).strip().lower()
    try:
        rate_value = float(band_row.get("rate_value", 0.0))
    except (ValueError, TypeError):
        rate_value = 0.0
    
    # Treat empty/unknown rate_type or 0 rate_value as "no penalty"
    if not rate_type or rate_value == 0:
        return ("none", 0.0)
    
    return (rate_type, rate_value)


def _compute_band_dsm_loss(energy_kwh: float, ppa: float, rate_type: str, rate_value: float) -> float:
    """Compute DSM loss for a band based on its penalty settings.
    
    Different rate_types:
    - 'none': No penalty (free zone) → 0
    - 'ppa_fraction': Penalty = energy_kwh × (rate_value × PPA / 1000)
    - 'ppa_multiple': Penalty = energy_kwh × (rate_value × PPA)
    - 'flat_per_kwh': Penalty = energy_kwh × rate_value
    - 'scaled_excess': Penalty = energy_kwh × rate_value (scaled by deviation)
    """
    if energy_kwh <= 0 or rate_type == "none" or rate_value == 0:
        return 0.0
    
    if rate_type in ("ppa_fraction", "ppa_frac"):
        # rate_value is fraction of PPA (e.g., 0.1 = 10% of PPA per kWh)
        return energy_kwh * rate_value * ppa / 1000.0
    elif rate_type in ("ppa_multiple", "ppa_mult"):
        # rate_value is multiplier of PPA
        return energy_kwh * rate_value * ppa
    elif rate_type in ("flat_per_kwh", "flat"):
        # rate_value is ₹ per kWh
        return energy_kwh * rate_value
    elif rate_type in ("scaled_excess", "scaled"):
        # rate_value is penalty rate for excess
        return energy_kwh * rate_value * ppa / 1000.0
    else:
        # Default: treat as flat rate
        return energy_kwh * rate_value


def _energy_df_from_detail(detail_df: pd.DataFrame, bands_rows: list[dict] = None) -> pd.DataFrame:
    """Build the exact per-block energy dataset used by the plot + download.

    ✅ CORRECT DSM LOGIC (slice-based, matches dsm_dashboard.py and Sheet-3):
      For each block where Schedule > 0, compute energy for ALL bands in that direction:
        - If abs_err=18% in UI direction:
          - UI 0-10%: slice = 10% → energy = (10/100) * basis_MW * 250
          - UI 10-15%: slice = 5% → energy = (5/100) * basis_MW * 250
          - UI >15%: slice = 3% → energy = (3/100) * basis_MW * 250
        - If abs_err=5% in UI direction:
          - UI 0-10%: slice = 5% → energy = (5/100) * basis_MW * 250
          - UI 10-15%: slice = 0% → energy = 0 (INCLUDED in stats!)
          - UI >15%: slice = 0% → energy = 0 (INCLUDED in stats!)

      This creates rows for ALL bands per block (including energy = 0).
    
    ✅ STATISTICS: Include 0 energy blocks in Median/Average/Max calculations
      Example: If 5 blocks have energies [1, 0, 2, 3, 4] for a band, median = 2 (not 2.5)
    
    ✅ DSM LOSS: Computed per band based on penalty settings (free zones = 0)
    """
    if detail_df is None or detail_df.empty:
        return pd.DataFrame()

    df = detail_df.copy()
    
    # Check required columns
    for col in ("abs_err", "basis_MW"):
        if col not in df.columns:
            print(f"DEBUG - Battery: Missing required column: {col}")
            return pd.DataFrame()
    
    # Need direction column
    if "direction" not in df.columns:
        print("DEBUG - Battery: Missing 'direction' column")
        return pd.DataFrame()

    if "dsm_skipped_zero_inputs" in df.columns:
        df = df[df["dsm_skipped_zero_inputs"] != True]  # noqa: E712

    # ✅ CRITICAL FILTER: Only include blocks where Scheduled_MW > 0
    initial_count = len(df)
    for sched_col in ("Scheduled_MW", "scheduled_mw", "schedule_mw"):
        if sched_col in df.columns:
            df[sched_col] = pd.to_numeric(df[sched_col], errors="coerce")
            df = df[df[sched_col].fillna(0) > 0].copy()
            print(f"DEBUG - Battery: Filtered {initial_count} → {len(df)} blocks (Scheduled_MW > 0)")
            break

    df["abs_err"] = pd.to_numeric(df["abs_err"], errors="coerce").fillna(0)
    df["basis_MW"] = pd.to_numeric(df["basis_MW"], errors="coerce").fillna(0)
    
    # Get PPA for DSM loss calculation
    ppa_col = None
    for pc in ("PPA", "ppa"):
        if pc in df.columns:
            ppa_col = pc
            df[pc] = pd.to_numeric(df[pc], errors="coerce").fillna(0)
            break
    
    if df.empty or not bands_rows:
        return pd.DataFrame()

    # ✅ Build band info with penalty rates
    ui_bands = []
    oi_bands = []
    for b in bands_rows:
        direction = str(b.get("direction", "")).strip().upper()
        lower = float(b.get("lower_pct", 0.0))
        upper = float(b.get("upper_pct", 1000.0))
        label = str(b.get("label", "")).strip()
        if not label:
            label = f"{direction} {lower}-{upper}%" if upper < 1000 else f"{direction} >{lower}%"
        simplified = _simplify_band_label(label)
        
        rate_type, rate_value = _get_band_penalty_rate(b)
        
        band_info = {
            "direction": direction,
            "lower": lower,
            "upper": upper,
            "label": simplified,
            "rate_type": rate_type,
            "rate_value": rate_value,
        }
        if direction == "UI":
            ui_bands.append(band_info)
        elif direction == "OI":
            oi_bands.append(band_info)
    
    # Sort bands by lower bound (ascending)
    ui_bands = sorted(ui_bands, key=lambda x: x["lower"])
    oi_bands = sorted(oi_bands, key=lambda x: x["lower"])
    
    print(f"DEBUG - Battery: UI bands: {[(b['label'], b['rate_type'], b['rate_value']) for b in ui_bands]}")
    print(f"DEBUG - Battery: OI bands: {[(b['label'], b['rate_type'], b['rate_value']) for b in oi_bands]}")
    
    # ✅ Build expanded rows: one row per (block, band) for ALL bands in that direction
    # INCLUDES energy = 0 rows (for correct median/average/max calculation)
    expanded_rows = []
    
    for idx, row in df.iterrows():
        abs_err = float(row["abs_err"])
        basis_mw = float(row["basis_MW"])
        ppa = float(row[ppa_col]) if ppa_col else 0.0
        row_direction = str(row.get("direction", "")).strip().upper()
        
        # Select bands based on direction
        bands_to_check = ui_bands if row_direction == "UI" else oi_bands if row_direction == "OI" else []
        
        for band_info in bands_to_check:
            lower = band_info["lower"]
            upper = band_info["upper"]
            band_label = band_info["label"]
            rate_type = band_info["rate_type"]
            rate_value = band_info["rate_value"]
            
            # Compute slice for this band (can be 0)
            slice_pct_val = _slice_pct(abs_err, lower, upper)
            
            # Convert to kWh: (slice% / 100) * basis_MW * 0.25h * 1000
            energy_kwh = _kwh_from_slice(slice_pct_val, basis_mw)
            band_value_mw = energy_kwh / PLANNING_ENERGY_MULTIPLIER
            
            # Compute DSM loss for this band (0 for free zones)
            band_dsm_loss = _compute_band_dsm_loss(energy_kwh, ppa, rate_type, rate_value)
            
            # Create row for this band (EVEN if energy = 0)
            new_row = row.to_dict()
            new_row["band"] = band_label
            new_row["band_label"] = band_label
            new_row["slice_pct"] = slice_pct_val
            new_row["energy_kwh"] = energy_kwh
            new_row["band_value_mw"] = band_value_mw
            new_row["band_dsm_loss"] = band_dsm_loss
            new_row["band_rate_type"] = rate_type
            new_row["band_rate_value"] = rate_value
            expanded_rows.append(new_row)
    
    if not expanded_rows:
        print("DEBUG - Battery: No expanded rows created")
        return pd.DataFrame()
    
    result_df = pd.DataFrame(expanded_rows)
    
    # Replace inf with nan, but do NOT filter out energy = 0 rows
    result_df = result_df.replace([np.inf, -np.inf], np.nan)
    result_df["energy_kwh"] = result_df["energy_kwh"].fillna(0)
    result_df["band_dsm_loss"] = result_df["band_dsm_loss"].fillna(0)

    print(f"DEBUG - Battery: Final energy_df has {len(result_df)} rows (including energy=0)")
    
    # Show sample of energy values per band for debugging
    sample_stats = result_df.groupby("band")["energy_kwh"].agg(["count", "min", "median", "max"])
    print(f"DEBUG - Battery: Sample stats per band:\n{sample_stats}")

    # Keep a clean, insight-friendly set of columns
    keep = []
    for c in [
        "region",
        "plant_name",
        "date",
        "time_block",
        "from_time",
        "to_time",
        "direction",
        "band_label",
        "band",
        "abs_err",
        "slice_pct",
        "basis_MW",
        "band_value_mw",
        "energy_kwh",
        "band_dsm_loss",
        "band_rate_type",
        "band_rate_value",
        "penalty",
        "Custom Setting",
    ]:
        if c in result_df.columns:
            keep.append(c)
    if keep:
        return result_df[keep].copy()
    return result_df.copy()


def _band_sort_key(b: str) -> Tuple[int, float, str]:
    """UI bands first then OI; within that ascending % (for consistent Y ordering)."""
    s = str(b or "").upper()
    dir_key = 0 if s.startswith("UI") else (1 if s.startswith("OI") else 2)
    m = re.search(r"(\d+(\.\d+)?)", s)
    lo = float(m.group(1)) if m else 9999.0
    return (dir_key, lo, s)


def _band_label_from_row(r: dict) -> str:
    if not isinstance(r, dict):
        return ""
    lbl = r.get("label")
    if lbl:
        return str(lbl)
    direction = str(r.get("direction") or "").strip().upper()
    lo = r.get("lower_pct", "")
    hi = r.get("upper_pct", "")
    return f"{direction} {lo}-{hi}".strip()


def _penalty_severity_from_band_row(r: dict) -> float:
    """Ranking score used for battery allocation priority (higher = more severe).

    We do not recompute DSM; we just prioritize bands with higher penalty rate.
    """
    if not isinstance(r, dict):
        return 0.0
    rt = str(r.get("rate_type") or "").strip().lower()
    try:
        rv = float(r.get("rate_value") or 0.0)
    except Exception:
        rv = 0.0

    # Order buckets: scaled_excess > ppa_multiple > ppa_fraction > flat_per_kwh > unknown
    if rt in ("scaled_excess", "scaled"):
        base = 5.0
    elif rt in ("ppa_multiple", "ppa_mult"):
        base = 4.0
    elif rt in ("ppa_fraction", "ppa_frac"):
        base = 3.0
    elif rt in ("flat_per_kwh", "flat"):
        base = 1.0
    else:
        base = 0.0
    return base * 1000.0 + rv  # ensure bucket dominates, then rate_value breaks ties


def _build_download_df(detail_df: pd.DataFrame, bands_rows: list[dict]) -> pd.DataFrame:
    """Build download DataFrame with all band energies as separate columns.
    
    Structure:
      Region, Plant Name, Date, Block, From Time, To Time,
      Scheduled Power(MW), AvC(MW), Injected Power, PPA,
      Error%, Absolute Error, Direction,
      Deviation (MW),
      UI Energy 0-10% (kWh), UI Energy 10-15% (kWh), UI Energy >15% (kWh),
      OI Energy 0-10% (kWh), OI Energy 10-15% (kWh), OI Energy >15% (kWh),
      Total DSM, Revenue Loss
    """
    # CORE PRINCIPLE: Download must consume authoritative Analysis output, not recompute bands.
    if detail_df is None or detail_df.empty or not bands_rows:
        return pd.DataFrame()

    df = detail_df.copy()

    # If caller passed raw detail, convert it to authoritative bandwise df.
    # If caller already passed bandwise df, this is a no-op (get_bandwise_analysis_df will preserve base cols + add band cols).
    import dsm_dashboard as core  # noqa: WPS433
    bw = core.get_bandwise_analysis_df(df, bands_rows)
    if bw is None or bw.empty:
        return pd.DataFrame()

    # Mandatory global filter: Scheduled > 0
    if "Scheduled_MW" in bw.columns:
        bw["Scheduled_MW"] = pd.to_numeric(bw["Scheduled_MW"], errors="coerce").fillna(0.0)
        bw = bw[bw["Scheduled_MW"] > 0].copy()

    # Create kWh columns from MW columns (×250) WITHOUT overwriting MW
    band_mw_cols = [c for c in bw.columns if isinstance(c, str) and c.endswith("(MW)") and ("UI Energy" in c or "OI Energy" in c)]
    for mw_col in band_mw_cols:
        kwh_col = mw_col.replace("(MW)", "(kWh)")
        bw[kwh_col] = pd.to_numeric(bw[mw_col], errors="coerce").fillna(0.0) * PLANNING_ENERGY_MULTIPLIER

    # Prepare output columns in the requested order
    output = pd.DataFrame()
    
    # Base columns
    for col, out_name in [
        ("region", "Region"),
        ("plant_name", "Plant Name"),
        ("date", "Date"),
        ("time_block", "Block"),
        ("from_time", "From Time"),
        ("to_time", "To Time"),
        ("Scheduled_MW", "Scheduled Power(MW)"),
        ("AvC_MW", "AvC(MW)"),
        ("Actual_MW", "Injected Power"),
        ("PPA", "PPA"),
        ("error_pct", "Error%"),
        ("abs_err", "Absolute Error"),
        ("direction", "Direction"),
    ]:
        if col in bw.columns:
            output[out_name] = bw[col]
    
    # Deviation (MW)
    if "abs_err" in bw.columns and "basis_MW" in bw.columns:
        output["Deviation (MW)"] = (pd.to_numeric(bw["abs_err"], errors="coerce").fillna(0.0) / 100.0) * pd.to_numeric(bw["basis_MW"], errors="coerce").fillna(0.0)
    
    # Band-specific energy columns (already computed above)
    for col in bw.columns:
        if isinstance(col, str) and ("UI Energy" in col or "OI Energy" in col) and col.endswith("(kWh)"):
            output[col] = bw[col]
    
    # DSM columns (if present in detail_df)
    # Total DSM comes from authoritative per-band DSM columns
    if "Total DSM (₹)" in bw.columns:
        output["Total DSM"] = bw["Total DSM (₹)"]
    elif "penalty" in bw.columns:
        output["Total DSM"] = bw["penalty"]

    if "revenue_loss" in bw.columns:
        output["Revenue Loss"] = bw["revenue_loss"]
    
    return output


def _compute_band_stats(energy_df: pd.DataFrame, bands_rows: list[dict] | None = None) -> pd.DataFrame:
    """Band-wise summary stats used in Section A (Median/Average/Max) and DSM Loss (₹).

    CORE PRINCIPLE:
    - This function expects **wide Analysis output** that already contains:
      - band-wise energy columns in MW: "UI Energy ... (MW)" / "OI Energy ... (MW)"
      - band-wise DSM columns in ₹:  "UI DSM ... (₹)" / "OI DSM ... (₹)"
    - It must NOT reslice or recompute DSM.

    Rules:
    - Caller must apply Schedule>0 filter BEFORE calling (global filter).
    - Convert MW->kWh by multiplying by 250 (create new values, do not mutate MW logic).
    - Median/Average/Max computed on kWh values after excluding zeros (user rule).
    - DSM loss is sum of the provided per-band DSM column (free-zone bands will naturally be 0).
    """
    if energy_df is None or energy_df.empty:
        return pd.DataFrame(columns=["Band", "Median Energy (kWh)", "Average Energy (kWh)", "Max Energy (kWh)", "DSM Loss (₹)"])

    df = energy_df.copy()

    band_mw_cols = [c for c in df.columns if isinstance(c, str) and c.endswith("(MW)") and ("UI Energy" in c or "OI Energy" in c)]
    if not band_mw_cols:
        return pd.DataFrame(columns=["Band", "Median Energy (kWh)", "Average Energy (kWh)", "Max Energy (kWh)", "DSM Loss (₹)"])

    rows = []
    for mw_col in band_mw_cols:
        band_label, _band_disp = _band_label_from_energy_mw_col(mw_col)
        band_name = _simplify_band_label(band_label)

        kwh_all = pd.to_numeric(df[mw_col], errors="coerce").fillna(0.0) * PLANNING_ENERGY_MULTIPLIER
        # USER RULE: exclude zeros from stats computation
        kwh = kwh_all[kwh_all > 0]

        # DSM col is expected to match naming from core.get_bandwise_analysis_df
        dsm_col = mw_col.replace("Energy", "DSM").replace("(MW)", "(₹)")
        if dsm_col in df.columns:
            dsm_vals = pd.to_numeric(df[dsm_col], errors="coerce").fillna(0.0)
            dsm_sum = float(dsm_vals.sum())
        else:
            dsm_sum = 0.0

        rows.append(
            {
                "Band": band_name,
                "Median Energy (kWh)": float(kwh.median()) if len(kwh) else 0.0,
                "Average Energy (kWh)": float(kwh.mean()) if len(kwh) else 0.0,
                "Max Energy (kWh)": float(kwh.max()) if len(kwh) else 0.0,
                "DSM Loss (₹)": float(dsm_sum),
            }
        )

    stats_df = pd.DataFrame(rows)
    # Stable order: UI bands then OI, by lower bound
    stats_df = stats_df.sort_values("Band", key=lambda s: s.map(lambda x: _band_sort_key(str(x)))).reset_index(drop=True)

    # Ensure all configured bands appear (even if 0 rows in this run)
    if bands_rows:
        all_bands = []
        for r in bands_rows:
            b = _simplify_band_label(_band_label_from_row(r))
            if b:
                all_bands.append(b)
        all_bands = sorted(set(all_bands), key=_band_sort_key)
        if all_bands:
            full = pd.DataFrame({"Band": all_bands})
            stats_df = full.merge(stats_df, on="Band", how="left")
            stats_df = stats_df.fillna(
                {
                    "Median Energy (kWh)": 0.0,
                    "Average Energy (kWh)": 0.0,
                    "Max Energy (kWh)": 0.0,
                    "DSM Loss (₹)": 0.0,
                }
            )

    stats_df = stats_df.sort_values("Band", key=lambda s: s.map(lambda x: _band_sort_key(str(x)))).reset_index(drop=True)
    return stats_df


def build_energy_scatter_plot(detail_df: pd.DataFrame, bands_rows: list[dict] = None) -> go.Figure:
    """Scatter plot (Battery Investment Section A).

    CORE PRINCIPLE:
    - Do NOT reslice/recompute DSM/bands here.
    - Consume Analysis-tab bandwise MW outputs and only convert MW -> kWh (×250), then plot.

    Plot spec (per user):
    - Y axis: band names (e.g. UI 0-10, UI 10-15, OI 0-10 ...)
    - X axis: energy values (kWh)
    - Plot only points where energy > 0
    """
    if detail_df is None or detail_df.empty:
        fig = px.scatter(pd.DataFrame({"energy_kwh": [], "band": []}), x="energy_kwh", y="band", color="band")
        fig.update_layout(template="plotly_white", title="Battery Investment – No data")
        return fig

    bw = detail_df.copy()
    # If we were passed raw detail (no band MW cols), derive authoritative bandwise df once.
    has_band_cols = any(
        isinstance(c, str) and c.endswith("(MW)") and ("UI Energy" in c or "OI Energy" in c)
        for c in bw.columns
    )
    if not has_band_cols:
        import dsm_dashboard as core  # noqa: WPS433
        bw = core.get_bandwise_analysis_df(bw, bands_rows or [])
    if bw is None or bw.empty:
        fig = px.scatter(pd.DataFrame({"energy_kwh": [], "band": []}), x="energy_kwh", y="band", color="band")
        fig.update_layout(template="plotly_white", title="Battery Investment – No data")
        return fig

    # Mandatory global filter: Scheduled > 0, dropped upfront
    sched_col = "Scheduled_MW" if "Scheduled_MW" in bw.columns else None
    if sched_col:
        bw[sched_col] = pd.to_numeric(bw[sched_col], errors="coerce").fillna(0.0)
        bw = bw[bw[sched_col] > 0].copy()

    # Identify band MW columns from Analysis output
    band_mw_cols = [c for c in bw.columns if isinstance(c, str) and c.endswith("(MW)") and ("UI Energy" in c or "OI Energy" in c)]
    if not band_mw_cols:
        fig = px.scatter(pd.DataFrame({"energy_kwh": [], "band": []}), x="energy_kwh", y="band", color="band")
        fig.update_layout(template="plotly_white", title="Battery Investment – No band columns")
        return fig

    # Build long df: one row per (block, band) where energy > 0
    plot_rows = []
    for mw_col in band_mw_cols:
        band_label, band_disp = _band_label_from_energy_mw_col(mw_col)
        # MW -> kWh
        kwh = pd.to_numeric(bw[mw_col], errors="coerce").fillna(0.0) * PLANNING_ENERGY_MULTIPLIER
        # IMPORTANT: keep bw's index so we can align hover columns after filtering
        tmp = pd.DataFrame(
            {"energy_kwh": kwh, "band": band_disp, "band_label": _simplify_band_label(band_label)},
            index=bw.index,
        )
        mask = tmp["energy_kwh"] > 0
        tmp = tmp.loc[mask].copy()
        if tmp.empty:
            continue
        # Hover info (if present)
        for c in ("plant_name", "date", "time_block", "from_time", "to_time", "Custom Setting"):
            if c in bw.columns:
                tmp[c] = bw.loc[mask, c].values
        plot_rows.append(tmp)

    plot_df = pd.concat(plot_rows, ignore_index=True) if plot_rows else pd.DataFrame({"energy_kwh": [], "band": [], "band_label": []})
    band_order = sorted(plot_df["band"].dropna().astype(str).unique().tolist(), key=_band_sort_key)

    fig = px.scatter(
        plot_df,
        x="energy_kwh",
        y="band",
        color="band",
        render_mode="webgl",
        hover_data={
            "plant_name": True if "plant_name" in plot_df.columns else False,
            "date": True if "date" in plot_df.columns else False,
            "time_block": True if "time_block" in plot_df.columns else False,
            "Custom Setting": True if "Custom Setting" in plot_df.columns else False,
            "energy_kwh": ":.0f",
        },
        category_orders={"band": band_order} if band_order else None,
        labels={"energy_kwh": "Energy (kWh) = Energy (MW) × 250", "band": "Band"},
        title="Battery Investment – Energy distribution by Band (Schedule>0 only, Energy>0 only)",
    )
    fig.update_traces(marker=dict(size=6, opacity=0.65))
    fig.update_layout(template="plotly_white", height=650, margin=dict(l=70, r=30, t=70, b=60))
    return fig


def battery_investment_layout():
    """Main content block (hidden by default; shown via sidebar nav)."""
    return html.Div(
        [
            # Stores (avoid recomputation + enable download/investment)
            dcc.Store(id="battery-detail-store"),  # Original detail data for download
            dcc.Store(id="battery-energy-store"),  # Filtered energy data for scatter
            dcc.Store(id="battery-band-stats-store"),
            dcc.Store(id="battery-meta-store"),
            dcc.Store(id="battery-investment-store"),
            dcc.Download(id="battery-download-energy"),
            dcc.Download(id="battery-download-investment"),

            # Filters (mirrors Analysis UX)
            html.Div(
                [
                    html.Div(
                        [
                            html.Span("🔋", style={"fontSize": "1.5rem", "marginRight": "10px"}),
                            html.Span(
                                "Battery Investment – Planning Tool",
                                style={"fontSize": "1.1rem", "fontWeight": "600", "color": "#333"},
                            ),
                        ],
                        style={"marginBottom": "1.5rem"},
                    ),
                    dbc.Row(
                        [
                            dbc.Col(
                                [
                                    html.Label(
                                        [html.Span("🌐 ", style={"marginRight": "6px"}), "Region"],
                                        style={"fontWeight": "600", "marginBottom": "10px", "color": "#333", "fontSize": "0.95rem"},
                                    ),
                                    dmc.MultiSelect(
                                        id="battery-region-dd",
                                        value=[],
                                        data=[],
                                        searchable=True,
                                        clearable=True,
                                        placeholder="Select Region(s)...",
                                        comboboxProps={"keepOpened": True},
                                        maxDropdownHeight=300,
                                    ),
                                ],
                                md=4,
                            ),
                            dbc.Col(
                                [
                                    html.Label(
                                        [html.Span("🏭 ", style={"marginRight": "6px"}), "State"],
                                        style={"fontWeight": "600", "marginBottom": "10px", "color": "#333", "fontSize": "0.95rem"},
                                    ),
                                    dmc.MultiSelect(
                                        id="battery-state-dd",
                                        value=[],
                                        data=[],
                                        searchable=True,
                                        clearable=True,
                                        placeholder="Select State(s)...",
                                        comboboxProps={"keepOpened": True},
                                        maxDropdownHeight=300,
                                    ),
                                ],
                                md=2,
                            ),
                            dbc.Col(
                                [
                                    html.Label(
                                        [html.Span("🏭 ", style={"marginRight": "6px"}), "Resource"],
                                        style={"fontWeight": "600", "marginBottom": "10px", "color": "#333", "fontSize": "0.95rem"},
                                    ),
                                    dmc.MultiSelect(
                                        id="battery-resource-type-dd",
                                        value=[],
                                        data=[],
                                        searchable=True,
                                        clearable=True,
                                        placeholder="Select Resource(s)...",
                                        comboboxProps={"keepOpened": True},
                                        maxDropdownHeight=300,
                                    ),
                                ],
                                md=2,
                            ),
                            dbc.Col(
                                [
                                    html.Label(
                                        [html.Span("📅 ", style={"marginRight": "6px"}), "Date Range"],
                                        style={"fontWeight": "600", "marginBottom": "10px", "color": "#333", "fontSize": "0.95rem"},
                                    ),
                                    dcc.DatePickerRange(
                                        id="battery-date-range",
                                        start_date=(datetime.now() - timedelta(days=6)).date(),
                                        end_date=datetime.now().date(),
                                        display_format="DD-MMM-YYYY",
                                        minimum_nights=0,
                                        style={"fontSize": "0.95rem"},
                                    ),
                                ],
                                md=4,
                            ),
                        ],
                        className="mb-4",
                    ),
                    dbc.Row(
                        [
                            dbc.Col(
                                [
                                    html.Label(
                                        [html.Span("🏢 ", style={"marginRight": "6px"}), "QCA (Agency)"],
                                        style={"fontWeight": "600", "marginBottom": "10px", "color": "#333", "fontSize": "0.95rem"},
                                    ),
                                    dmc.MultiSelect(
                                        id="battery-qca-dd",
                                        value=[],
                                        data=[],
                                        searchable=True,
                                        clearable=True,
                                        placeholder="Select QCA(s)...",
                                        comboboxProps={"keepOpened": True},
                                        maxDropdownHeight=300,
                                    ),
                                ],
                                md=4,
                            ),
                            dbc.Col(
                                [
                                    html.Label(
                                        [html.Span("🔌 ", style={"marginRight": "6px"}), "Pooling Station"],
                                        style={"fontWeight": "600", "marginBottom": "10px", "color": "#333", "fontSize": "0.95rem"},
                                    ),
                                    dmc.MultiSelect(
                                        id="battery-pooling-dd",
                                        value=[],
                                        data=[],
                                        searchable=True,
                                        clearable=True,
                                        placeholder="Select Pooling Station(s)...",
                                        comboboxProps={"keepOpened": True},
                                        maxDropdownHeight=300,
                                    ),
                                ],
                                md=4,
                            ),
                            dbc.Col(
                                [
                                    html.Label(
                                        [html.Span("🏭 ", style={"marginRight": "6px"}), "Plant"],
                                        style={"fontWeight": "600", "marginBottom": "10px", "color": "#333", "fontSize": "0.95rem"},
                                    ),
                                    dmc.MultiSelect(
                                        id="battery-plant-dd",
                                        value=[],
                                        data=[],
                                        searchable=True,
                                        clearable=True,
                                        placeholder="Select Plant(s) or Select All...",
                                        comboboxProps={"keepOpened": True},
                                        maxDropdownHeight=400,
                                    ),
                                ],
                                md=4,
                            ),
                        ],
                        className="mb-4",
                    ),
                    # Application Mode + Custom Setting(s) (mirrors Analysis)
                    dbc.Row(
                        [
                            dbc.Col(
                                [
                                    html.Label(
                                        "Application Mode",
                                        style={"fontWeight": 600, "marginBottom": "10px", "color": "#333", "fontSize": "0.95rem"},
                                    ),
                                    dcc.RadioItems(
                                        id="battery-custom-setting-mode",
                                        options=[
                                            {"label": "Apply SAME custom setting to all plants", "value": "GLOBAL"},
                                            {"label": "Apply DIFFERENT custom settings per plant", "value": "PLANT_WISE"},
                                        ],
                                        value="GLOBAL",
                                        inline=True,
                                        style={"fontSize": "0.9rem"},
                                    ),
                                ],
                                md=12,
                            )
                        ],
                        className="mb-2",
                    ),
                    # Global preset selection (GLOBAL mode)
                    dbc.Row(
                        [
                            dbc.Col(
                                [
                                    html.Label(
                                        "Custom Setting(s)",
                                        style={"fontWeight": 600, "marginBottom": "10px", "color": "#333", "fontSize": "0.95rem"},
                                    ),
                                    dmc.MultiSelect(
                                        id="battery-preset-select",
                                        data=[],
                                        placeholder="(Optional) Choose one or more saved presets",
                                        value=[],
                                        searchable=True,
                                        clearable=True,
                                        comboboxProps={"keepOpened": True},
                                        maxDropdownHeight=300,
                                    ),
                                ],
                                md=4,
                            ),
                        ],
                        className="mb-2",
                        id="battery-global-setting-row",
                    ),
                    # Plant-wise mapping (PLANT_WISE mode)
                    dbc.Row(
                        [
                            dbc.Col(
                                [
                                    dcc.Store(
                                        id="battery-setting-assignments-store",
                                        data={
                                            "next_id": 1,
                                            "rows": [{"row": 0, "preset": None, "plants": [], "apply_remaining": True}],
                                        },
                                    ),
                                    html.Div(id="battery-plant-setting-mapping-container", style={"display": "none"}),
                                ],
                                md=12,
                            )
                        ],
                        className="mb-2",
                    ),
                    # Exclude Plants Section (only when SELECT_ALL is active)
                    html.Div(
                        [
                            dbc.Row(
                                [
                                    dbc.Col(
                                        [
                                            html.Div(
                                                [
                                                    html.Label(
                                                        [
                                                            html.Span("❌ ", style={"marginRight": "6px", "color": "#dc3545"}),
                                                            "Exclude Plants from Analysis (Optional)",
                                                            html.Small(
                                                                " - Only available when 'Select All' is active",
                                                                style={
                                                                    "marginLeft": "8px",
                                                                    "color": "#666",
                                                                    "fontWeight": "normal",
                                                                    "fontSize": "0.85rem",
                                                                },
                                                            ),
                                                        ],
                                                        style={
                                                            "fontWeight": "600",
                                                            "marginBottom": "10px",
                                                            "color": "#333",
                                                            "fontSize": "0.95rem",
                                                        },
                                                    ),
                                                    dmc.MultiSelect(
                                                        id="battery-exclude-plant-dd",
                                                        value=[],
                                                        data=[],
                                                        searchable=True,
                                                        clearable=True,
                                                        disabled=True,
                                                        placeholder="Select plants to exclude...",
                                                        comboboxProps={"keepOpened": True},
                                                        maxDropdownHeight=300,
                                                    ),
                                                ],
                                                style={
                                                    "padding": "1rem",
                                                    "backgroundColor": "#fff3cd",
                                                    "border": "1px solid #ffc107",
                                                    "borderRadius": "8px",
                                                    "marginBottom": "1rem",
                                                },
                                            )
                                        ],
                                        md=12,
                                    )
                                ],
                                className="mb-3",
                            )
                        ],
                        id="battery-exclude-section",
                        style={"display": "none"},
                    ),
                    dbc.Row(
                        [
                            dbc.Col(
                                [
                                    dbc.Button(
                                        [html.Span("📊 ", style={"marginRight": "8px"}), "Run Analysis"],
                                        id="battery-run",
                                        size="lg",
                                        className="w-100",
                                        disabled=True,
                                        style={
                                            "backgroundColor": "#ff6b35",
                                            "border": "none",
                                            "fontWeight": "600",
                                            "padding": "14px",
                                            "boxShadow": "0 4px 6px rgba(255, 107, 53, 0.3)",
                                        },
                                    )
                                ],
                                md=6,
                                className="offset-md-3",
                            )
                        ]
                    ),
                ],
                style={
                    "padding": "2rem",
                    "backgroundColor": "#fff",
                    "borderRadius": "12px",
                    "marginBottom": "2rem",
                    "boxShadow": "0 2px 8px rgba(0,0,0,0.05)",
                },
            ),
            # SECTION A: DSM Energy Understanding
            html.Div(
                [
                    html.H5("SECTION A: DSM Energy Understanding", style={"marginTop": "0.5rem", "marginBottom": "1rem", "color": "#333"}),
                    html.Div(id="battery-error", className="mb-2"),
                    dcc.Loading(type="circle", children=dcc.Graph(id="battery-scatter", figure=build_energy_scatter_plot(pd.DataFrame()))),
                    dbc.Row(
                        [
                            dbc.Col(
                                [
                                    html.H6("Band Statistics", style={"marginTop": "0.5rem"}),
                                    dash_table.DataTable(
                                        id="battery-band-stats-table",
                                        columns=[
                                            {"name": "Band", "id": "Band"},
                                            {"name": "Median Energy (kWh)", "id": "Median Energy (kWh)", "type": "numeric", "format": {"specifier": ",.0f"}},
                                            {"name": "Average Energy (kWh)", "id": "Average Energy (kWh)", "type": "numeric", "format": {"specifier": ",.0f"}},
                                            {"name": "Max Energy (kWh)", "id": "Max Energy (kWh)", "type": "numeric", "format": {"specifier": ",.0f"}},
                                            {"name": "DSM Loss (₹)", "id": "DSM Loss (₹)", "type": "numeric", "format": {"specifier": ",.0f"}},
                                        ],
                                        data=[],
                                        sort_action="native",
                                        style_table={"overflowX": "auto"},
                                        style_cell={"padding": "8px", "textAlign": "left"},
                                        style_header={"fontWeight": "600", "backgroundColor": "#f8f9fa"},
                                        page_size=12,
                                    ),
                                    dbc.Button(
                                        "Download Energy Data (CSV)",
                                        id="battery-download-energy-btn",
                                        color="success",
                                        className="mt-2",
                                        outline=True,
                                    ),
                                ],
                                md=12,
                            )
                        ]
                    ),
                ],
                style={"padding": "0 0 1rem 0"},
            ),
            # SECTION B: Battery Investment Decision
            html.Div(
                [
                    html.H5("SECTION B: Battery Investment Decision", style={"marginTop": "1.25rem", "marginBottom": "1rem", "color": "#333"}),
                    dbc.Card(
                        [
                            dbc.CardBody(
                                [
                                    dbc.Row(
                                        [
                                            dbc.Col(
                                                [
                                                    html.Label("Battery Capacity (kWh)", style={"fontWeight": 600}),
                                                    dbc.Input(id="battery-capacity-kwh", type="number", min=0, step=1, value=0),
                                                ],
                                                md=4,
                                            ),
                                            dbc.Col(
                                                [
                                                    html.Label("Battery Cost per kWh (₹/kWh)", style={"fontWeight": 600}),
                                                    dbc.Input(id="battery-cost-per-kwh", type="number", min=0, step=0.01, value=0),
                                                ],
                                                md=4,
                                            ),
                                            dbc.Col(
                                                [
                                                    html.Label(" ", style={"display": "block"}),
                                                    dbc.Button("Run Investment", id="battery-run-investment", color="primary", className="w-100"),
                                                ],
                                                md=4,
                                            ),
                                        ],
                                        className="mb-2",
                                    ),
                                    html.Div(id="battery-investment-summary", className="mt-2"),
                                    dbc.Button(
                                        "Download Investment Output (Excel)",
                                        id="battery-download-investment-btn",
                                        color="secondary",
                                        outline=True,
                                        className="mt-2",
                                    ),
                                    dcc.Loading(type="circle", children=dcc.Graph(id="battery-scatter-after")),
                                ]
                            )
                        ],
                        className="shadow-sm",
                    ),
                ]
            ),
        ],
        id="battery-investment-content",
        style={"display": "none"},
    )


def register_battery_callbacks(app):
    """Register all callbacks for Battery Investment tab."""

    # Import the main module lazily to avoid circular import at import time.
    import dsm_dashboard as core  # noqa: WPS433

    @app.callback(
        Output("battery-preset-select", "data"),
        Input("presets-store", "data"),
    )
    def _battery_load_preset_options(presets):
        presets = presets or []
        return [{"label": p.get("name", "Unnamed"), "value": p.get("name", "Unnamed")} for p in presets]

    @app.callback(
        Output("battery-global-setting-row", "style"),
        Output("battery-plant-setting-mapping-container", "style"),
        Output("battery-preset-select", "disabled"),
        Input("battery-custom-setting-mode", "value"),
    )
    def _battery_toggle_mapping_ui(mode):
        if mode == "PLANT_WISE":
            return {"display": "none"}, {"display": "block"}, True
        return {"display": "block"}, {"display": "none"}, False

    @app.callback(
        Output("battery-custom-setting-mode", "value"),
        Output("battery-setting-assignments-store", "data", allow_duplicate=True),
        Input("battery-preset-select", "value"),
        Input("battery-plant-dd", "value"),
        State("battery-custom-setting-mode", "value"),
        prevent_initial_call="initial_duplicate",
    )
    def _battery_auto_switch_to_assignment_ui(selected_presets, plants_value, mode):
        plant_list = core._normalize_multi(plants_value)
        plant_list = [p for p in plant_list if p and p != "SELECT_ALL"]
        has_multi_plants = len(plant_list) > 1

        if not has_multi_plants or not selected_presets:
            raise PreventUpdate

        presets_list = list(selected_presets) if isinstance(selected_presets, list) else [selected_presets]
        presets_list = [p for p in presets_list if p]
        if not presets_list:
            raise PreventUpdate

        rows = []
        if len(presets_list) == 1:
            rows = [{"row": 0, "preset": presets_list[0], "plants": [], "apply_remaining": True}]
        else:
            rows = [{"row": i, "preset": p, "plants": [], "apply_remaining": False} for i, p in enumerate(presets_list)]

        store = {"next_id": len(rows), "rows": rows}
        return "PLANT_WISE", store

    @app.callback(
        Output("battery-setting-assignments-store", "data"),
        Input("battery-btn-add-setting-row", "n_clicks"),
        Input("battery-btn-reset-setting-rows", "n_clicks"),
        Input({"type": "battery-setting-row-remove", "row": ALL}, "n_clicks"),
        Input({"type": "battery-setting-row-preset", "row": ALL}, "value"),
        Input({"type": "battery-setting-row-scope", "row": ALL}, "value"),
        Input({"type": "battery-setting-row-plants", "row": ALL}, "value"),
        State({"type": "battery-setting-row-preset", "row": ALL}, "id"),
        State({"type": "battery-setting-row-scope", "row": ALL}, "id"),
        State({"type": "battery-setting-row-plants", "row": ALL}, "id"),
        State({"type": "battery-setting-row-remove", "row": ALL}, "id"),
        State("battery-setting-assignments-store", "data"),
        prevent_initial_call=True,
    )
    def _battery_update_setting_assignments(
        n_add,
        n_reset,
        n_remove_clicks,
        preset_values,
        scope_values,
        plants_values,
        preset_ids,
        scope_ids,
        plants_ids,
        remove_ids,
        store,
    ):
        store = store or {"next_id": 1, "rows": [{"row": 0, "preset": None, "plants": [], "apply_remaining": True}]}
        store.setdefault("next_id", 1)
        store.setdefault("rows", [{"row": 0, "preset": None, "plants": [], "apply_remaining": True}])

        trig = ctx.triggered_id

        if trig == "battery-btn-reset-setting-rows":
            return {"next_id": 1, "rows": [{"row": 0, "preset": None, "plants": [], "apply_remaining": True}]}

        if trig == "battery-btn-add-setting-row":
            rid = int(store.get("next_id") or 1)
            store["next_id"] = rid + 1
            rows = list(store.get("rows") or [])
            rows.append({"row": rid, "preset": None, "plants": [], "apply_remaining": False})
            store["rows"] = rows
            return store

        if isinstance(trig, dict) and trig.get("type") == "battery-setting-row-remove":
            rid = int(trig.get("row", -1))
            rows = [r for r in (store.get("rows") or []) if int(r.get("row", -2)) != rid]
            if not rows:
                rows = [{"row": 0, "preset": None, "plants": [], "apply_remaining": True}]
                store["next_id"] = 1
            store["rows"] = rows
            return store

        preset_map: dict[int, str | None] = {}
        scope_map: dict[int, str | None] = {}
        plants_map: dict[int, list[str]] = {}

        for pid, val in zip(preset_ids or [], preset_values or []):
            rid = int((pid or {}).get("row", -1))
            preset_map[rid] = val

        for sid, val in zip(scope_ids or [], scope_values or []):
            rid = int((sid or {}).get("row", -1))
            scope_map[rid] = val

        for plid, val in zip(plants_ids or [], plants_values or []):
            rid = int((plid or {}).get("row", -1))
            if val is None:
                plants_map[rid] = []
            elif isinstance(val, list):
                plants_map[rid] = [str(x) for x in val if x]
            else:
                plants_map[rid] = [str(val)]

        updated_rows = []
        for r in (store.get("rows") or []):
            rid = int(r.get("row", 0))
            preset = preset_map.get(rid, r.get("preset"))
            scope = scope_map.get(rid, "REMAINING" if r.get("apply_remaining") else "SELECTED")
            apply_remaining = (scope == "REMAINING")
            plants_sel = [] if apply_remaining else plants_map.get(rid, r.get("plants") or [])
            updated_rows.append({"row": rid, "preset": preset, "plants": plants_sel, "apply_remaining": apply_remaining})

        store["rows"] = updated_rows
        return store

    @app.callback(
        Output("battery-plant-setting-mapping-container", "children"),
        Input("battery-custom-setting-mode", "value"),
        Input("battery-setting-assignments-store", "data"),
        Input("battery-plant-dd", "value"),
        Input("presets-store", "data"),
        State("battery-region-dd", "value"),
        State("battery-state-dd", "value"),
        State("battery-resource-type-dd", "value"),
        State("battery-qca-dd", "value"),
        State("battery-pooling-dd", "value"),
        State("battery-exclude-plant-dd", "value"),
    )
    def _battery_build_mapping_table(mode, assignments_store, plants, presets_store, regions, states, resources, qcas, pools, excluded_plants):
        if mode != "PLANT_WISE":
            return dash.no_update
        if not plants:
            return html.Div("Select plants first to assign custom settings.", className="text-muted")

        plant_list = core._normalize_multi(plants)
        is_select_all = "SELECT_ALL" in plant_list
        if is_select_all:
            dfm = core._filter_master(regions, states, resources, qcas, pools)
            all_plants = sorted({p for p in dfm.get("plant_name", pd.Series(dtype=str)).dropna().astype(str).tolist() if p.strip()})
            excluded_list = excluded_plants if isinstance(excluded_plants, list) else ([excluded_plants] if excluded_plants else [])
            excluded_list = [str(x) for x in excluded_list if x]
            resolved_plants = [p for p in all_plants if p not in excluded_list]
        else:
            resolved_plants = [str(p) for p in plant_list if p and str(p) != "SELECT_ALL"]

        if not resolved_plants:
            return html.Div("No plants selected.", className="text-muted")

        presets = presets_store or []
        preset_options = [{"label": p.get("name", "Unnamed"), "value": p.get("name", "Unnamed")} for p in presets]

        store = assignments_store or {}
        rows_state = list(store.get("rows") or [])
        if not rows_state:
            rows_state = [{"row": 0, "preset": None, "plants": [], "apply_remaining": True}]

        assigned: set[str] = set()
        ui_rows = []
        for r in rows_state:
            rid = int(r.get("row", 0))
            preset_val = r.get("preset")
            apply_remaining = bool(r.get("apply_remaining", False))

            remaining = [p for p in resolved_plants if p not in assigned]
            plants_val = [p for p in (r.get("plants") or []) if p in remaining]
            scope_value = "REMAINING" if apply_remaining else "SELECTED"

            preview_selected = remaining if apply_remaining else plants_val
            for p in preview_selected:
                assigned.add(p)

            ui_rows.append(
                dbc.Card(
                    [
                        dbc.CardBody(
                            [
                                dbc.Row(
                                    [
                                        dbc.Col(
                                            [
                                                html.Label("Custom Setting", style={"fontWeight": 600}),
                                                dcc.Dropdown(
                                                    id={"type": "battery-setting-row-preset", "row": rid},
                                                    options=preset_options,
                                                    value=preset_val,
                                                    placeholder="Select custom setting",
                                                    clearable=True,
                                                ),
                                            ],
                                            md=5,
                                        ),
                                        dbc.Col(
                                            [
                                                html.Label("Apply To", style={"fontWeight": 600}),
                                                dcc.RadioItems(
                                                    id={"type": "battery-setting-row-scope", "row": rid},
                                                    options=[
                                                        {"label": f"All remaining plants ({len(remaining)})", "value": "REMAINING"},
                                                        {"label": "Choose specific plants", "value": "SELECTED"},
                                                    ],
                                                    value=scope_value,
                                                    inline=True,
                                                ),
                                            ],
                                            md=5,
                                        ),
                                        dbc.Col(
                                            [
                                                html.Label(" ", style={"display": "block"}),
                                                dbc.Button(
                                                    "Remove",
                                                    id={"type": "battery-setting-row-remove", "row": rid},
                                                    outline=True,
                                                    color="secondary",
                                                ),
                                            ],
                                            md=2,
                                            className="text-end",
                                        ),
                                    ],
                                    className="mb-2",
                                ),
                                dmc.MultiSelect(
                                    id={"type": "battery-setting-row-plants", "row": rid},
                                    data=[{"label": p, "value": p} for p in remaining],
                                    value=([] if apply_remaining else plants_val),
                                    searchable=True,
                                    clearable=True,
                                    placeholder="Select plants for this custom setting...",
                                    disabled=apply_remaining,
                                    comboboxProps={"keepOpened": True},
                                    maxDropdownHeight=260,
                                ),
                                html.Div(
                                    f"Will apply to: {', '.join(preview_selected) if preview_selected else '(none)'}",
                                    className="text-muted",
                                    style={"marginTop": "6px", "fontSize": "0.9rem"},
                                ),
                            ]
                        )
                    ],
                    className="mb-2",
                )
            )

        remaining_final = [p for p in resolved_plants if p not in assigned]
        controls = dbc.Row(
            [
                dbc.Col(html.Strong("Assign Custom Settings to Plants"), md=6),
                dbc.Col(
                    html.Div(
                        [
                            dbc.Button("Add another custom setting", id="battery-btn-add-setting-row", color="primary", size="sm", className="me-2"),
                            dbc.Button("Reset", id="battery-btn-reset-setting-rows", color="secondary", outline=True, size="sm"),
                        ],
                        className="text-end",
                    ),
                    md=6,
                ),
            ],
            className="align-items-center mb-2",
        )

        status = (
            [dbc.Alert(f"Unassigned plants: {', '.join(remaining_final)}", color="warning", className="mt-2")]
            if remaining_final
            else [dbc.Alert("All selected plants are assigned to a custom setting.", color="success", className="mt-2")]
        )

        return html.Div([controls] + ui_rows + status)

    @app.callback(
        Output("battery-region-dd", "data"),
        Input("nav-store", "data"),
        prevent_initial_call=False,
    )
    def _battery_load_regions(nav_data):
        try:
            df = core._load_plant_master_df()
            regions = sorted({str(r).upper() for r in df.get("region", pd.Series(dtype=str)).dropna().tolist() if str(r).strip()})
            if not regions:
                regions = core.get_regions_from_duckdb()
            return [{"label": r, "value": r} for r in regions]
        except Exception as e:
            print(f"Error loading battery regions: {e}")
            return [{"label": r, "value": r} for r in ["NRPC", "SRPC", "WRPC"]]

    @app.callback(
        Output("battery-state-dd", "data"),
        Output("battery-state-dd", "value"),
        Input("battery-region-dd", "value"),
        State("battery-state-dd", "value"),
        prevent_initial_call=False,
    )
    def _battery_states(regions, current_state_value):
        if not regions:
            return [], []
        df = core._filter_master(regions, None, None, None, None)
        states = sorted({s for s in df.get("state", pd.Series(dtype=str)).dropna().astype(str).tolist() if s.strip()})
        opts = [{"label": "✓ All", "value": core.ALL_SENTINEL}] + [{"label": s, "value": s} for s in states]
        cur = core._strip_all_sentinel(core._normalize_multi(current_state_value))
        if not cur:
            return opts, []
        valid = [v for v in cur if v in states]
        return opts, (valid if valid else [])

    @app.callback(
        Output("battery-resource-type-dd", "data"),
        Output("battery-resource-type-dd", "value"),
        Input("battery-region-dd", "value"),
        Input("battery-state-dd", "value"),
        State("battery-resource-type-dd", "value"),
        prevent_initial_call=False,
    )
    def _battery_resources(regions, states, current_resource_value):
        if not regions:
            return [], []
        df = core._filter_master(regions, states, None, None, None)
        res = sorted({r for r in df.get("resource", pd.Series(dtype=str)).dropna().astype(str).tolist() if r.strip()})
        opts = [{"label": "✓ All", "value": core.ALL_SENTINEL}] + [{"label": r, "value": r} for r in res]
        cur = core._strip_all_sentinel(core._normalize_multi(current_resource_value))
        if not cur:
            return opts, []
        valid = [v for v in cur if v in res]
        return opts, (valid if valid else [])

    @app.callback(
        Output("battery-qca-dd", "data"),
        Output("battery-qca-dd", "value"),
        Input("battery-region-dd", "value"),
        Input("battery-state-dd", "value"),
        Input("battery-resource-type-dd", "value"),
        State("battery-qca-dd", "value"),
        prevent_initial_call=False,
    )
    def _battery_qca(regions, states, resources, current_qca_value):
        if not regions:
            return [], []
        df = core._filter_master(regions, states, resources, None, None)
        qcas = sorted({q for q in df.get("qca", pd.Series(dtype=str)).dropna().astype(str).tolist() if q.strip()})
        opts = [{"label": "✓ All", "value": core.ALL_SENTINEL}] + [{"label": q, "value": q} for q in qcas]
        cur = core._strip_all_sentinel(core._normalize_multi(current_qca_value))
        if not cur:
            return opts, []
        valid = [v for v in cur if v in qcas]
        return opts, (valid if valid else [])

    @app.callback(
        Output("battery-pooling-dd", "data"),
        Output("battery-pooling-dd", "value"),
        Input("battery-region-dd", "value"),
        Input("battery-state-dd", "value"),
        Input("battery-resource-type-dd", "value"),
        Input("battery-qca-dd", "value"),
        State("battery-pooling-dd", "value"),
        prevent_initial_call=False,
    )
    def _battery_pools(regions, states, resources, qcas, current_pool_value):
        if not regions:
            return [], []
        df = core._filter_master(regions, states, resources, qcas, None)
        pools = sorted({p for p in df.get("pooling_station", pd.Series(dtype=str)).dropna().astype(str).tolist() if p.strip()})
        opts = [{"label": "✓ All", "value": core.ALL_SENTINEL}] + [{"label": p, "value": p} for p in pools]
        cur = core._strip_all_sentinel(core._normalize_multi(current_pool_value))
        if not cur:
            return opts, []
        valid = [v for v in cur if v in pools]
        return opts, (valid if valid else [])

    @app.callback(
        Output("battery-plant-dd", "data"),
        Output("battery-plant-dd", "value"),
        Input("battery-region-dd", "value"),
        Input("battery-state-dd", "value"),
        Input("battery-resource-type-dd", "value"),
        Input("battery-qca-dd", "value"),
        Input("battery-pooling-dd", "value"),
        State("battery-plant-dd", "value"),
        prevent_initial_call=False,
    )
    def _battery_plants(regions, states, resources, qcas, pools, current_plant_value):
        if not regions:
            return [], []
        df = core._filter_master(regions, states, resources, qcas, pools)
        filtered_plants = sorted({p for p in df.get("plant_name", pd.Series(dtype=str)).dropna().astype(str).tolist() if p.strip()})
        if not filtered_plants:
            return [], []

        def truncate_label(name: str, max_length: int = 50) -> str:
            return name if len(name) <= max_length else name[: max_length - 3] + "..."

        cur = core._normalize_multi(current_plant_value)
        selected_plants = set(cur) if cur else set()
        available_plants = [p for p in filtered_plants if p not in selected_plants]
        options = [{"label": "✓ Select All", "value": "SELECT_ALL"}] + [{"label": truncate_label(p), "value": p} for p in available_plants]

        if cur:
            valid = [p for p in cur if p == "SELECT_ALL" or p in filtered_plants]
            return options, valid
        return options, []

    @app.callback(
        Output("battery-exclude-section", "style"),
        Output("battery-exclude-plant-dd", "data"),
        Output("battery-exclude-plant-dd", "value"),
        Input("battery-plant-dd", "value"),
        Input("battery-region-dd", "value"),
        Input("battery-state-dd", "value"),
        Input("battery-resource-type-dd", "value"),
        Input("battery-qca-dd", "value"),
        Input("battery-pooling-dd", "value"),
        State("battery-exclude-plant-dd", "value"),
        prevent_initial_call=False,
    )
    def _battery_exclude_section(plant_value, regions, states, resources, qcas, pools, current_exclude_value):
        is_select_all = False
        if isinstance(plant_value, list):
            is_select_all = "SELECT_ALL" in plant_value
        elif plant_value == "SELECT_ALL":
            is_select_all = True

        if not is_select_all:
            return {"display": "none"}, [], []

        if not regions:
            return {"display": "block"}, [], (current_exclude_value if current_exclude_value else [])

        df = core._filter_master(regions, states, resources, qcas, pools)
        filtered_plants = sorted({p for p in df.get("plant_name", pd.Series(dtype=str)).dropna().astype(str).tolist() if p.strip()})
        opts = [{"label": p, "value": p} for p in filtered_plants]

        cur = core._normalize_multi(current_exclude_value)
        cur = [p for p in cur if p in filtered_plants]
        return {"display": "block"}, opts, cur

    @app.callback(
        Output("battery-exclude-plant-dd", "disabled"),
        Input("battery-exclude-section", "style"),
        prevent_initial_call=False,
    )
    def _battery_exclude_disabled(style):
        if style and style.get("display") == "block":
            return False
        return True

    @app.callback(
        Output("battery-run", "disabled"),
        Input("battery-region-dd", "value"),
        Input("battery-plant-dd", "value"),
        Input("battery-date-range", "start_date"),
        Input("battery-date-range", "end_date"),
        prevent_initial_call=False,
    )
    def _battery_toggle_run(regions, plants, start_date, end_date):
        if not regions or not plants or not start_date or not end_date:
            return True
        plant_list = core._normalize_multi(plants)
        plant_list = [p for p in plant_list if p]
        if not plant_list:
            return True
        return False

    @app.callback(
        Output("battery-scatter", "figure"),
        Output("battery-error", "children"),
        Output("battery-band-stats-table", "data"),
        Output("battery-detail-store", "data"),
        Output("battery-energy-store", "data"),
        Output("battery-band-stats-store", "data"),
        Output("battery-meta-store", "data"),
        Input("battery-run", "n_clicks"),
        State("battery-region-dd", "value"),
        State("battery-state-dd", "value"),
        State("battery-resource-type-dd", "value"),
        State("battery-qca-dd", "value"),
        State("battery-pooling-dd", "value"),
        State("battery-plant-dd", "value"),
        State("battery-exclude-plant-dd", "value"),
        State("battery-date-range", "start_date"),
        State("battery-date-range", "end_date"),
        State("battery-custom-setting-mode", "value"),
        State("battery-preset-select", "value"),
        State("battery-setting-assignments-store", "data"),
        State("presets-store", "data"),
        # Reuse existing DSM config controls (do not duplicate DSM logic)
        State("err-mode", "value"),
        State("x-pct", "value"),
        State("bands-table", "data"),
        prevent_initial_call=True,
    )
    def _battery_run(
        n,
        regions,
        states,
        resources,
        qcas,
        pools,
        plants,
        excluded_plants,
        start_date,
        end_date,
        mode,
        selected_preset_names,
        assignments_store,
        presets_store,
        err_mode,
        x_pct,
        bands_rows,
    ):
        if not n:
            raise PreventUpdate

        # Resolve SELECT_ALL using master (same as Analysis UX)
        plant_list = core._normalize_multi(plants)
        is_select_all = "SELECT_ALL" in plant_list
        if is_select_all:
            dfm = core._filter_master(regions, states, resources, qcas, pools)
            all_plants = sorted({p for p in dfm.get("plant_name", pd.Series(dtype=str)).dropna().astype(str).tolist() if p.strip()})
            excluded_list = excluded_plants if isinstance(excluded_plants, list) else ([excluded_plants] if excluded_plants else [])
            excluded_list = [str(x) for x in excluded_list if x]
            resolved_plants = [p for p in all_plants if p not in excluded_list]
        else:
            resolved_plants = [str(p) for p in plant_list if p and str(p) != "SELECT_ALL"]

        if not regions or not resolved_plants:
            empty = build_energy_scatter_plot(pd.DataFrame())
            return empty, dbc.Alert("Select at least one region and plant.", color="warning"), [], {}, {}, {}

        # Decide how to apply settings (GLOBAL vs PLANT_WISE) – reuse presets-store only, do not change DSM logic.
        application_mode = mode if mode else "GLOBAL"
        presets_store = presets_store or []
        name_to_settings = {p["name"]: p.get("settings", {}) for p in (presets_store or []) if isinstance(p, dict) and "name" in p}

        def run_once(with_settings: dict, target_plants: list[str]):
            final_err_mode = with_settings.get("err_mode", err_mode)
            final_x_pct = with_settings.get("x_pct", x_pct)
            saved_bands = with_settings.get("bands", None)
            final_bands = saved_bands if (saved_bands and len(saved_bands) > 0) else (bands_rows or core.DEFAULT_BANDS.copy())
            unpaid_threshold = float(with_settings.get("unpaid_oi_threshold", 15.0))
            res = core._compute_pipeline(regions, target_plants, start_date, end_date, final_err_mode, final_x_pct, final_bands, unpaid_threshold, excluded_plants)
            return res, final_err_mode, float(final_x_pct or 0), final_bands

        multi_detail = []
        used_bands_rows = None

        try:
            if application_mode == "PLANT_WISE":
                store = assignments_store or {}
                rows_state = list(store.get("rows") or [])
                if not rows_state:
                    empty = build_energy_scatter_plot(pd.DataFrame())
                    return empty, dbc.Alert("Please assign custom settings to plants (PLANT_WISE mode).", color="warning"), [], {}, {}, {}

                assigned: set[str] = set()
                for r in rows_state:
                    setting_name = r.get("preset")
                    if not setting_name:
                        empty = build_energy_scatter_plot(pd.DataFrame())
                        return empty, dbc.Alert("Please select a Custom Setting for each assignment row.", color="warning"), [], {}, {}, {}
                    st = name_to_settings.get(setting_name)
                    if not st:
                        empty = build_energy_scatter_plot(pd.DataFrame())
                        return empty, dbc.Alert(f"Preset '{setting_name}' not found. Please reselect it.", color="warning"), [], {}, {}, {}

                    apply_remaining = bool(r.get("apply_remaining", False))
                    remaining = [p for p in resolved_plants if p not in assigned]
                    if apply_remaining:
                        target_plants = remaining
                    else:
                        raw_plants = r.get("plants") or []
                        target_plants = [p for p in raw_plants if p in remaining]

                    if not target_plants:
                        empty = build_energy_scatter_plot(pd.DataFrame())
                        return empty, dbc.Alert(f"No plants selected for Custom Setting '{setting_name}'.", color="warning"), [], {}, {}, {}

                    for p in target_plants:
                        assigned.add(p)

                    res, fm, fx, fb = run_once(st, target_plants)
                    if res.get("error"):
                        empty = build_energy_scatter_plot(pd.DataFrame())
                        return empty, dbc.Alert(str(res.get("error")), color="danger"), [], {}, {}, {}

                    df_detail = res.get("df", pd.DataFrame())
                    if isinstance(df_detail, pd.DataFrame) and not df_detail.empty:
                        df_detail = df_detail.copy()
                        df_detail["Custom Setting"] = setting_name
                        multi_detail.append(df_detail)
                        used_bands_rows = fb

                missing = [p for p in resolved_plants if p not in assigned]
                if missing:
                    empty = build_energy_scatter_plot(pd.DataFrame())
                    return empty, dbc.Alert(f"Unassigned plants: {', '.join(missing)}", color="warning"), [], {}, {}, {}

            elif selected_preset_names:
                preset_list = list(selected_preset_names) if isinstance(selected_preset_names, list) else [selected_preset_names]
                preset_list = [p for p in preset_list if p]
                for nm in preset_list:
                    st = name_to_settings.get(nm, {})
                    res, fm, fx, fb = run_once(st, resolved_plants)
                    if res.get("error"):
                        empty = build_energy_scatter_plot(pd.DataFrame())
                        return empty, dbc.Alert(str(res.get("error")), color="danger"), [], {}, {}, {}
                    df_detail = res.get("df", pd.DataFrame())
                    if isinstance(df_detail, pd.DataFrame) and not df_detail.empty:
                        df_detail = df_detail.copy()
                        df_detail["Custom Setting"] = nm
                        multi_detail.append(df_detail)
                        used_bands_rows = fb
            else:
                base_settings = {"err_mode": err_mode, "x_pct": x_pct, "bands": bands_rows}
                res, fm, fx, fb = run_once(base_settings, resolved_plants)
                if res.get("error"):
                    empty = build_energy_scatter_plot(pd.DataFrame())
                    return empty, dbc.Alert(str(res.get("error")), color="danger"), [], {}, {}, {}
                df_detail = res.get("df", pd.DataFrame())
                if isinstance(df_detail, pd.DataFrame) and not df_detail.empty:
                    df_detail = df_detail.copy()
                    df_detail["Custom Setting"] = "Current"
                    multi_detail.append(df_detail)
                    used_bands_rows = fb
        except Exception as e:
            empty = build_energy_scatter_plot(pd.DataFrame())
            return empty, dbc.Alert(f"Failed to run analysis: {e}", color="danger"), [], {}, {}, {}

        if not multi_detail:
            empty = build_energy_scatter_plot(pd.DataFrame())
            return empty, dbc.Alert("No data found for the selection.", color="warning"), [], {}, {}, {}

        combined_detail = pd.concat(multi_detail, ignore_index=True) if len(multi_detail) > 1 else multi_detail[0]
        final_bands_rows = used_bands_rows or (bands_rows or core.DEFAULT_BANDS.copy())

        # ============================================================
        # CORE PRINCIPLE: Consume Analysis output, do not recompute DSM/bands
        # ============================================================
        bw_df = core.get_bandwise_analysis_df(combined_detail, final_bands_rows, err_mode=str(err_mode or "default"), x_pct=float(x_pct or 50.0))
        if bw_df is None or bw_df.empty:
            empty = build_energy_scatter_plot(pd.DataFrame())
            return empty, dbc.Alert("No bandwise analysis output available.", color="warning"), [], {}, {}, {}

        # Mandatory global filter: Scheduled > 0 (drop upfront)
        if "Scheduled_MW" in bw_df.columns:
            bw_df["Scheduled_MW"] = pd.to_numeric(bw_df["Scheduled_MW"], errors="coerce").fillna(0.0)
            bw_df = bw_df[bw_df["Scheduled_MW"] > 0].copy()

        fig = build_energy_scatter_plot(bw_df, final_bands_rows)
        band_stats = _compute_band_stats(bw_df, final_bands_rows)

        # Cache/store datasets for download/investment (avoid browser OOM)
        detail_pack = core._pack_df_for_store(bw_df)  # authoritative per-block bandwise df
        energy_pack = core._pack_df_for_store(bw_df)  # energy_store now holds authoritative df too
        stats_pack = core._pack_df_for_store(band_stats)
        meta = {"bands_rows": final_bands_rows}

        note = dbc.Alert(
            f"Energy conversion is for planning only: Band Energy (kWh) = Band Energy (MW) × {int(PLANNING_ENERGY_MULTIPLIER)}. "
            f"Schedule=0 rows are dropped before plotting/statistics.",
            color="info",
            className="mt-2",
        )
        return fig, note, band_stats.to_dict("records"), detail_pack, energy_pack, stats_pack, meta

    @app.callback(
        Output("battery-download-energy", "data"),
        Input("battery-download-energy-btn", "n_clicks"),
        State("battery-detail-store", "data"),
        State("battery-meta-store", "data"),
        prevent_initial_call=True,
    )
    def _battery_download_energy(n, detail_store, meta_store):
        if not n:
            raise PreventUpdate
        detail_df = core._detail_df_from_store(detail_store or {})
        if detail_df.empty:
            raise PreventUpdate
        
        meta = meta_store or {}
        bands_rows = meta.get("bands_rows") or core.DEFAULT_BANDS.copy()
        
        # Build full download DataFrame with all band energy columns
        download_df = _build_download_df(detail_df, bands_rows)
        if download_df.empty:
            raise PreventUpdate
        
        return dcc.send_data_frame(download_df.to_csv, "battery_energy_data.csv", index=False)

    @app.callback(
        Output("battery-investment-summary", "children"),
        Output("battery-scatter-after", "figure"),
        Output("battery-investment-store", "data"),
        Input("battery-run-investment", "n_clicks"),
        State("battery-energy-store", "data"),
        State("battery-band-stats-store", "data"),
        State("battery-meta-store", "data"),
        State("battery-capacity-kwh", "value"),
        State("battery-cost-per-kwh", "value"),
        prevent_initial_call=True,
    )
    def _battery_run_investment(n, energy_store, stats_store, meta_store, cap_kwh, cost_per_kwh):
        if not n:
            raise PreventUpdate
        energy_df = core._detail_df_from_store(energy_store or {})
        stats_df = core._detail_df_from_store(stats_store or {})
        if energy_df.empty or stats_df.empty:
            return dbc.Alert("Run Section A (Run Analysis) first.", color="warning"), build_energy_scatter_plot(pd.DataFrame()), {}

        try:
            cap = float(cap_kwh or 0)
            cpk = float(cost_per_kwh or 0)
        except Exception:
            cap = 0.0
            cpk = 0.0

        # Battery cost rule (fixed):
        # Battery Cost (₹) = Battery Capacity (kWh) × Battery Cost per kWh (₹/kWh)
        cost = cap * cpk

        # ============================================================
        # NEW LOGIC (POST-PROCESS ONLY): block-wise battery application
        # ============================================================
        # Build long df from authoritative bandwise output (no reslicing; no DSM formula change)
        df_long = _long_band_df_from_bandwise_df(energy_df)
        if df_long.empty:
            return dbc.Alert("No bandwise energy rows available for investment.", color="warning"), build_energy_scatter_plot(pd.DataFrame()), {}

        df_applied = apply_battery_blockwise(df_long, battery_kwh=cap, window_blocks=DEFAULT_BATTERY_WINDOW_BLOCKS)

        original_dsm = float(pd.to_numeric(df_applied.get("band_dsm_before"), errors="coerce").fillna(0.0).sum())
        new_dsm = float(pd.to_numeric(df_applied.get("band_dsm_after"), errors="coerce").fillna(0.0).sum())
        red_pct = 0.0 if original_dsm <= 0 else ((original_dsm - new_dsm) / original_dsm) * 100.0

        verdict = "✅ Investment Positive" if (cost + new_dsm) <= original_dsm else "❌ Bad Investment"

        summary_rows = [
            {"Metric": "Original DSM Loss (₹)", "Value": round(original_dsm, 2)},
            {"Metric": "New DSM Loss (₹)", "Value": round(new_dsm, 2)},
            {"Metric": "DSM Reduction (%)", "Value": round(red_pct, 2)},
            {"Metric": "Battery Cost per kWh (₹/kWh)", "Value": round(cpk, 2)},
            {"Metric": "Total Battery Cost (₹)", "Value": round(cost, 2)},
            {"Metric": "Remark", "Value": verdict},
        ]
        summary_table = dash_table.DataTable(
            columns=[{"name": "Metric", "id": "Metric"}, {"name": "Value", "id": "Value"}],
            data=summary_rows,
            style_table={"overflowX": "auto"},
            style_cell={"padding": "8px", "textAlign": "left"},
            style_header={"fontWeight": "600", "backgroundColor": "#f8f9fa"},
        )

        # After-battery scatter: X=energy_after, Y=band names, show only >0
        plot_df = df_applied.copy()
        plot_df["energy_kwh_after"] = pd.to_numeric(plot_df.get("energy_kwh_after"), errors="coerce").fillna(0.0)
        plot_df = plot_df[plot_df["energy_kwh_after"] > 0].copy()
        plot_df["band"] = plot_df.get("band_display", plot_df.get("band_label", "")).astype(str)
        band_order = sorted(plot_df["band"].dropna().astype(str).unique().tolist(), key=_band_sort_key)

        fig_after = px.scatter(
            plot_df,
            x="energy_kwh_after",
            y="band",
            color="band",
            render_mode="webgl",
            hover_data={
                "plant_name": True if "plant_name" in plot_df.columns else False,
                "date": True if "date" in plot_df.columns else False,
                "time_block": True if "time_block" in plot_df.columns else False,
                "direction": True if "direction" in plot_df.columns else False,
                "battery_applied_kwh": ":.0f",
                "band_penalty_rate": ":.4f",
                "band_dsm_before": ":.2f",
                "band_dsm_after": ":.2f",
            },
            category_orders={"band": band_order} if band_order else None,
            labels={"energy_kwh_after": "Energy after Battery (kWh)", "band": "Band"},
            title="After Battery: block-wise band energy reduced from highest-penalty bands first",
        )
        fig_after.update_traces(marker=dict(size=6, opacity=0.65))
        fig_after.update_layout(template="plotly_white", height=650, margin=dict(l=70, r=30, t=70, b=60))

        inv_pack = core._pack_df_for_store(df_applied)
        return summary_table, fig_after, inv_pack

    @app.callback(
        Output("battery-download-investment", "data"),
        Input("battery-download-investment-btn", "n_clicks"),
        State("battery-investment-store", "data"),
        prevent_initial_call=True,
    )
    def _battery_download_investment(n, inv_store):
        if not n:
            raise PreventUpdate
        df = core._detail_df_from_store(inv_store or {})
        if df.empty:
            raise PreventUpdate

        out = pd.DataFrame()
        for col, name in [
            ("region", "Region"),
            ("plant_name", "Plant Name"),
            ("date", "Date"),
            ("time_block", "Block"),
            ("direction", "Direction"),
            ("band_display", "Band"),
        ]:
            if col in df.columns:
                out[name] = df[col]
        out["Energy Before (kWh)"] = pd.to_numeric(df.get("energy_kwh_before"), errors="coerce").fillna(0.0)
        out["Battery Applied (kWh)"] = pd.to_numeric(df.get("battery_applied_kwh"), errors="coerce").fillna(0.0)
        out["Energy After (kWh)"] = pd.to_numeric(df.get("energy_kwh_after"), errors="coerce").fillna(0.0)
        out["DSM Before (₹)"] = pd.to_numeric(df.get("band_dsm_before"), errors="coerce").fillna(0.0)
        out["DSM After (₹)"] = pd.to_numeric(df.get("band_dsm_after"), errors="coerce").fillna(0.0)

        # Excel download (derived from same df, no recomputation)
        return dcc.send_data_frame(out.to_excel, "battery_investment_output.xlsx", index=False, sheet_name="Investment")



