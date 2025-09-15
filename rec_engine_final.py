#!/usr/bin/env python3
"""
rec_engine.py — Interactive content-based recommender for Square Yards

Run:
  python rec_engine.py
…then follow the on-screen prompts for:
  - CSV path
  - Lead fields (ProjectId, Lat/Lon, Grade, Price, Size, BHK, Status, HasFocus)
  - top_k

Keeps all prior logic:
  - Cleans data
  - Weighted similarity (price/area/bhk/grade/status/distance)
  - Status priority: New Launch > Under Construction > Partially RTM > RTM
  - Two lists: Unbiased and Unbiased+Biased (HasFocus boost)
  - Dynamic candidate filtering: expand price/area first, then distance in steps
  - Soft distance penalty so far items don’t dominate
  - Prints lists sorted by score, grouped by ProjectId while preserving global order
"""

import os
import sys
import pandas as pd
import numpy as np
from math import radians, sin, cos, sqrt, atan2
from typing import Dict, Tuple

# =========================
# Config / Tunables
# =========================
WEIGHTS = {
    'price' : 0.30,
    'area'  : 0.30,
    'bhk'   : 0.10,
    'grade' : 0.10,
    'status': 0.10,
    'dist'  : 0.10
}

STATUS_PRIORITY_BOOST = {
    'new launch'                 : 1.20,
    'under construction'         : 1.10,
    'partially ready to move'    : 1.05,
    'ready to move'              : 1.00
}

FOCUS_BOOST = 0.15                        # +15% when HasFocus == 1
GRADE_MAP  = {'Silver': 1, 'Gold': 2, 'Platinum': 3}

# Filtering behavior
POOL_MULT = 2                              # candidate pool target = max(top_k*POOL_MULT, 20)

DIST_INIT_KM        = 5.0                  # start radius
DIST_STEP_KM        = 2.5                  # step when we must expand distance
HARD_MAX_DIST_KM    = 30.0                 # absolute ceiling for radius

PRICE_BAND_INIT_FRAC = 0.10                # ±10% of lead price initially
AREA_BAND_INIT_FRAC  = 0.10                # ±10% of lead area initially
PRICE_BAND_MAX_FRAC  = 1.00                # expand up to ±100% (0–2× window)
AREA_BAND_MAX_FRAC   = 1.00

# Fixed soft distance scale (penalty stays strong even when radius widens)
DIST_SOFT_SCALE_KM   = 8.0                 # exp(-d/scale): ~0.37 at 8km, ~0.14 at 16km


# =========================
# Utilities
# =========================
def norm_status(s: str) -> str:
    return (s or 'ready to move').strip().lower() if isinstance(s, str) else 'ready to move'

def haversine(lat1, lon1, lat2, lon2) -> float:
    R = 6371.0  # km
    phi1, phi2 = radians(lat1), radians(lat2)
    dphi, dlambda = radians(lat2 - lat1), radians(lon2 - lon1)
    a = sin(dphi/2)**2 + cos(phi1)*cos(phi2)*sin(dlambda/2)**2
    return 2 * R * atan2(sqrt(a), sqrt(1 - a))

def sim_price(p1, p2):   return max(0.0, 1 - abs(p1 - p2) / max(p1, 1e-9))
def sim_area(a1, a2):    return max(0.0, 1 - abs(a1 - a2) / max(a1, 1e-9))
def sim_bhk(b1, b2):     return max(0.0, 1 - abs(b1 - b2) / max(b1, 1e-9))
def sim_grade(g1, g2):   return max(0.0, 1 - abs(g1 - g2) / 2.0)  # grades 1..3 → denom=2
def sim_status_equal(s1, s2): return 1.0 if norm_status(s1) == norm_status(s2) else 0.0
def status_priority_boost(status_norm: str) -> float:
    return STATUS_PRIORITY_BOOST.get(status_norm, 1.00)

def sim_dist_soft(d_km: float) -> float:
    from math import exp
    return max(0.0, min(1.0, exp(-d_km / DIST_SOFT_SCALE_KM)))


# =========================
# Data loading & cleaning
# =========================
def load_and_clean(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    # Harmonize project-name column if needed
    if 'Project Name' not in df.columns and 'ProjectName' in df.columns:
        df['Project Name'] = df['ProjectName']

    # Normalize placeholder nulls
    df.replace(['Inactive', 'NULL', 'Blanks', 'Blank', '(Blank)', '(Blanks)', ''], np.nan, inplace=True)

    required = [
        'ProjectId', 'Project Name', 'LowCost', 'ProjectMinSize', 'UnitBHKOptions',
        'DeveloperGradeCat', 'ProjectStatusDesc', 'Latitude', 'Longitude'
    ]
    df.dropna(subset=required, inplace=True)

    # Filter invalid numerics
    df = df[
        (df['LowCost'] > 0) &
        (df['ProjectMinSize'] > 0) &
        (df['UnitBHKOptions'] > 0)
    ].reset_index(drop=True)

    # Developer grade mapping
    df = df[df['DeveloperGradeCat'].isin(GRADE_MAP)].copy()
    df['DevGradeOrd'] = df['DeveloperGradeCat'].map(GRADE_MAP).astype(int)

    # Ensure HasFocus exists & numeric
    if 'HasFocus' not in df.columns:
        df['HasFocus'] = 0
    df['HasFocus'] = df['HasFocus'].fillna(0).astype(int)

    # Normalize status text
    df['ProjectStatusNorm'] = df['ProjectStatusDesc'].apply(norm_status)

    return df


# =========================
# Core recommender (expand price/area first; then distance in steps)
# =========================
def recommend_base(lead: Dict, df_projects: pd.DataFrame, top_k: int = 10) -> pd.DataFrame:
    """
    Builds a candidate pool by expanding price/area bands up to caps,
    then increasing distance incrementally (up to HARD_MAX_DIST_KM).
    Uses fixed soft distance penalty for scoring.
    """
    # Exclude the same project
    cand = df_projects[df_projects.ProjectId != lead['ProjectId']].copy()

    # Precompute distances
    cand['GeoDistKm'] = cand.apply(
        lambda r: haversine(lead['Latitude'], lead['Longitude'], r['Latitude'], r['Longitude']),
        axis=1
    )

    pool_size   = max(top_k * POOL_MULT, 20)
    dist_thresh = DIST_INIT_KM
    price_band  = PRICE_BAND_INIT_FRAC * lead['LowCost']
    area_band   = AREA_BAND_INIT_FRAC  * lead['ProjectMinSize']

    max_price_band = PRICE_BAND_MAX_FRAC * lead['LowCost']
    max_area_band  = AREA_BAND_MAX_FRAC  * lead['ProjectMinSize']

    while True:
        filt = cand[
            (cand.GeoDistKm <= dist_thresh) &
            (abs(cand.LowCost - lead['LowCost']) <= price_band) &
            (abs(cand.ProjectMinSize - lead['ProjectMinSize']) <= area_band)
        ]
        if len(filt) >= pool_size:
            break

        expanded = False
        # Expand price/area first (up to caps)
        if price_band < max_price_band or area_band < max_area_band:
            price_band = min(price_band * 2, max_price_band)
            area_band  = min(area_band  * 2, max_area_band)
            expanded = True

        # Then expand distance in small steps, capped by HARD_MAX_DIST_KM
        if len(filt) < pool_size and dist_thresh < HARD_MAX_DIST_KM:
            dist_thresh = min(dist_thresh + DIST_STEP_KM, HARD_MAX_DIST_KM)
            expanded = True

        if not expanded:
            break

    if len(filt) == 0:
        filt = cand[cand.GeoDistKm <= dist_thresh]

    # Compute similarity components
    base_scores = []
    status_boosts = []
    for _, row in filt.iterrows():
        score  = WEIGHTS['price']  * sim_price(lead['LowCost'],       row['LowCost'])
        score += WEIGHTS['area']   * sim_area(lead['ProjectMinSize'], row['ProjectMinSize'])
        score += WEIGHTS['bhk']    * sim_bhk(lead['UnitBHKOptions'],  row['UnitBHKOptions'])
        score += WEIGHTS['grade']  * sim_grade(lead['DevGradeOrd'],    row['DevGradeOrd'])
        score += WEIGHTS['status'] * sim_status_equal(lead['ProjectStatusDesc'], row['ProjectStatusDesc'])
        score += WEIGHTS['dist']   * sim_dist_soft(row['GeoDistKm'])
        base_scores.append(score)
        status_boosts.append(status_priority_boost(row['ProjectStatusNorm']))

    filt = filt.assign(
        SimilarityScore     = base_scores,
        StatusPriorityBoost = status_boosts,
        FinalDistThreshUsed = dist_thresh,
        FinalPriceBandAbs   = price_band,
        FinalAreaBandAbs    = area_band,
        PoolSizeTarget      = pool_size
    )
    return filt


def recommend_lists(lead: Dict, df_projects: pd.DataFrame, top_k: int = 10) -> Tuple[pd.DataFrame, pd.DataFrame]:
    cand = recommend_base(lead, df_projects, top_k=max(top_k*POOL_MULT, 20))

    # Unbiased score = base similarity × status priority
    cand['UnbiasedScore'] = cand['SimilarityScore'] * cand['StatusPriorityBoost']
    unbiased_sorted = cand.sort_values('UnbiasedScore', ascending=False) \
                          .head(top_k).reset_index(drop=True)

    # Biased score = Unbiased × (1 + FOCUS_BOOST × HasFocus)
    cand['FocusMultiplier'] = 1.0 + (FOCUS_BOOST * cand['HasFocus'].astype(int))
    cand['BiasedScore'] = cand['UnbiasedScore'] * cand['FocusMultiplier']
    biased_sorted = cand.sort_values('BiasedScore', ascending=False) \
                        .head(top_k).reset_index(drop=True)

    return unbiased_sorted, biased_sorted


# =========================
# Pretty printing: grouped by ProjectId while preserving global rank order
# =========================
def print_grouped_in_rank_order(df_sorted: pd.DataFrame, score_col: str, title: str):
    print(f"\n================ {title} ================\n")
    order, seen = [], set()
    for pid in df_sorted['ProjectId']:
        if pid not in seen:
            seen.add(pid); order.append(pid)
    for pid in order:
        block = df_sorted[df_sorted['ProjectId'] == pid]
        pname = block['Project Name'].iloc[0]
        top_score = block[score_col].iloc[0]
        print(f"=== Project {pid}: {pname} — top {score_col}={top_score:.4f} ===")
        print(block.to_string(index=False))
        print()


# =========================
# Interactive prompts
# =========================
def prompt_path(default: str) -> str:
    while True:
        val = input(f"Path to city CSV [{default}]: ").strip()
        path = val or default
        if os.path.exists(path):
            return path
        print(f"  ✖ File not found: {path}. Try again.")

def prompt_float(msg: str, default: float | None = None) -> float:
    while True:
        s = input(f"{msg}" + (f" [{default}]" if default is not None else "") + ": ").strip()
        if s == "" and default is not None:
            return float(default)
        try:
            return float(s)
        except ValueError:
            print("  ✖ Please enter a number.")

def prompt_int(msg: str, default: int | None = None) -> int:
    while True:
        s = input(f"{msg}" + (f" [{default}]" if default is not None else "") + ": ").strip()
        if s == "" and default is not None:
            return int(default)
        try:
            return int(s)
        except ValueError:
            print("  ✖ Please enter an integer.")

def prompt_choice(msg: str, options: list[str], default: str | None = None) -> str:
    opts_str = "/".join(options)
    while True:
        s = input(f"{msg} ({opts_str})" + (f" [{default}]" if default else "") + ": ").strip()
        if s == "" and default:
            return default
        if s in options:
            return s
        # allow case-insensitive and title-case matches
        for o in options:
            if s.lower() == o.lower():
                return o
        print(f"  ✖ Choose one of: {opts_str}")

def build_lead_interactively(df: pd.DataFrame) -> Dict:
    print("\nProvide the resale lead details (press Enter to auto-pick first Ready to Move row):")
    auto = input("Auto-pick from CSV? (y/N): ").strip().lower()
    if auto in ("y", "yes"):
        rtm = df[df['ProjectStatusNorm'] == 'ready to move']
        row = rtm.iloc[0] if not rtm.empty else df.iloc[0]
        lead = {
            'ProjectId'         : int(row['ProjectId']),
            'Latitude'          : float(row['Latitude']),
            'Longitude'         : float(row['Longitude']),
            'DevGradeOrd'       : int(row['DevGradeOrd']),
            'LowCost'           : float(row['LowCost']),
            'ProjectMinSize'    : float(row['ProjectMinSize']),
            'UnitBHKOptions'    : float(row['UnitBHKOptions']),
            'ProjectStatusDesc' : str(row['ProjectStatusDesc']),
            'HasFocus'          : 0
        }
        print("→ Auto-picked lead:", lead)
        return lead

    # Manual entry
    project_id = prompt_int("Lead ProjectId")
    lat        = prompt_float("Latitude")
    lon        = prompt_float("Longitude")
    grade      = prompt_choice("Developer Grade", list(GRADE_MAP.keys()))
    lowcost    = prompt_float("LowCost (price)")
    size       = prompt_float("Size (ProjectMinSize)")
    bhk        = prompt_float("BHK (UnitBHKOptions)")
    status     = prompt_choice("ProjectStatusDesc",
                    ["New Launch","Under Construction","Partially Ready to Move","Ready to Move"],
                    default="Ready to Move")
    focus      = prompt_choice("HasFocus", ["0","1"], default="0")

    return {
        'ProjectId'         : project_id,
        'Latitude'          : lat,
        'Longitude'         : lon,
        'DevGradeOrd'       : GRADE_MAP[grade],
        'LowCost'           : lowcost,
        'ProjectMinSize'    : size,
        'UnitBHKOptions'    : bhk,
        'ProjectStatusDesc' : status,
        'HasFocus'          : int(focus)
    }


# =========================
# Main
# =========================
if __name__ == '__main__':
    print("\n=== Square Yards Recommendation Engine (Interactive) ===\n")
    default_path = "data/Gurgaon_Properties_Rec_RAG.csv"
    csv_path = prompt_path(default_path)

    df = load_and_clean(csv_path)
    lead = build_lead_interactively(df)
    top_k = prompt_int("How many recommendations (top_k)?", default=10)

    unbiased_sorted, biased_sorted = recommend_lists(lead, df, top_k=top_k)

    # Print grouped results preserving global score order
    print_grouped_in_rank_order(unbiased_sorted, 'UnbiasedScore', 'UNBIASED RECOMMENDATIONS')
    print_grouped_in_rank_order(biased_sorted,   'BiasedScore',   'UNBIASED + BIASED (HasFocus) RECOMMENDATIONS')

    print("\nDone.\n")