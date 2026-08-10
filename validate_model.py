#!/usr/bin/env python3
"""Acceptance checks for a corrector .joblib the moment it lands.

    venv/bin/python validate_model.py corrector_grid_2deg_v01.joblib

Checks, in order:
 1. bundle structure + provenance metadata
 2. feature census: grid nodes vs driver cols vs station cols; driver
    names must be a subset of the hindcast builder's 93 columns
 3. grid-node alignment: every dBNg node must land EXACTLY on the live
    1-deg grid (guards the cell-centered 30.5/32.5-style silent offset —
    predict_aual's 1.01-deg tolerance would NOT catch it)
 4. live prediction on the newest realtime grid file (drivers filled with
    0 / stations NaN — plumbing + timing only, NOT a skill check)
"""
import glob
import sys
import time

import joblib
import numpy as np

from predict_aual import _GCOL, Corrector

LIVE_GLOB = ("/data/tuija/cdimarco/LAUREN/realtime-geodgp/"
             "deltaB_*_real4.out")

# the 93 driver columns, rebuilt from geodgp_common's recipe (9 base +
# 72 lagmd + 12 Dst_diff) without importing its heavy deps
SW_COLS = ['Bx, nT (GSE, GSM)', 'By, nT (GSM)', 'Bz, nT (GSM)',
           'Vx Velocity, km/s, GSE', 'Proton Density, n/cc', 'Temperature, K']
DRIVER93 = (SW_COLS + ['Dipole Tilt Angle', 'f10.7_index', 'Dst']
            + [f'{c}_lagmd_{lag * 5}' for lag in range(12) for c in SW_COLS]
            + [f'Dst_diff_{j}' for j in range(1, 13)])


def main(path):
    b = joblib.load(path)
    print("== 1. bundle")
    print("  keys:", sorted(b.keys()))
    for k, v in b.items():
        if k not in ("models", "fcols"):
            print(f"  {k}: {v}")
    assert set(b["models"]) >= {"AU", "AL"}, "missing AU/AL models"

    print("== 2. feature census")
    fcols = b["fcols"]
    grid = [c for c in fcols if _GCOL.match(c)]
    other = [c for c in fcols if not _GCOL.match(c)]
    driver = [c for c in other if c in DRIVER93]
    station = [c for c in other if c not in DRIVER93]
    print(f"  {len(fcols)} features = {len(grid)} grid nodes + "
          f"{len(driver)} drivers + {len(station)} station cols")
    missing93 = [c for c in DRIVER93 if c not in other]
    if missing93:
        print(f"  drivers NOT used by model ({len(missing93)}):",
              missing93[:5], "...")
    if station:
        print("  station cols (NaN-tolerant):", station[:6],
              "..." if len(station) > 6 else "")
    unknown = [c for c in station if not any(
        s in c for s in ('ABK', 'DIK', 'CCS', 'TIK', 'CWE', 'BRW', 'CMO',
                         'YKC', 'FCC', 'PBQ', 'NAQ', 'LRV'))]
    if unknown:
        print(f"  WARNING: {len(unknown)} non-driver cols match no AE "
              f"station: {unknown[:5]}")

    print("== 3. grid-node alignment vs live 1-deg grid")
    c = Corrector(path)
    lats = sorted({la for _, la, _ in c.grid_cols})
    lons = sorted({lo for _, _, lo in c.grid_cols})
    off = [(la, lo) for _, la, lo in c.grid_cols
           if la != round(la) or lo != round(lo)]
    print(f"  lats {lats[0]}..{lats[-1]} ({len(lats)} distinct), "
          f"lons {lons[0]}..{lons[-1]} ({len(lons)} distinct)")
    assert not off, f"NON-INTEGER nodes (silent nearest-node offset!): {off[:5]}"
    assert lats[0] >= 30.0, f"node below 30N: {lats[0]}"
    print("  all nodes integer-degree and >= 30N: exact match on live grid")

    print("== 4. live prediction (plumbing only)")
    latest = sorted(glob.glob(LIVE_GLOB))[-1]
    # drivers zeroed, everything else NaN — mirrors the realtime caller,
    # which supplies drivers and NaNs any station/stray cols (e.g. the
    # 0-split 'Datetime' index column that leaked into v01's fcols)
    feats = {col: 0.0 for col in driver}
    feats.update({col: np.nan for col in station})
    t0 = time.perf_counter()
    au, al = c.predict(latest, feats)
    dt = (time.perf_counter() - t0) * 1e3
    print(f"  {latest.rsplit('/', 1)[-1]} -> AU={au:.1f} AL={al:.1f} "
          f"({dt:.1f} ms)")
    assert np.isfinite(au) and np.isfinite(al)
    print("ALL CHECKS PASSED")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1
         else "corrector_grid_2deg_v01.joblib")
