#!/usr/bin/env python3
"""Realtime station features for the corrector — replicates the Pleiades
training builder (stations_{year}.parquet, relayed 2026-08-10) with the
realtime-necessary substitutions, each marked [RT] below:

Per station (NLAGS = 6), per valid time t:
  dh    = H - quiet-day baseline. Baseline: minute-of-day mean over the 5
          quietest days (min daily std), 61-min circularly smoothed —
          training used the calendar month's days; [RT] we use the trailing
          30 full UTC days (the causal analog; baselines vary slowly).
          Skipped (all-NaN station) when window coverage < 20%, as in
          training's month gate.
  rm    = dh.rolling('5min', min_periods=3).median()
  age   = minutes from the station's last valid sample to t, >= 0.
          Training simulated this from the very same feed's latency log;
          [RT] it is the live feed's actual staleness.
  dh{k} = rm at (t - round(age) - 5k min), k = 0..5; NaN before data start.
  mlt   = aacgmv2: station mlon fixed at July-1-of-current-year, hourly
          convert_mlt anchors + minute/60 advance (training's exact
          recipe, quirks included), as _mltsin/_mltcos.

H store: state/groundmag/h_store.parquet — 1-min x station float matrix,
trailing STORE_DAYS days. Upserted every minute by fetch_groundmag_rt
(update_h_store) and bootstrappable with --backfill N (one N-day GIN
request per station). Baselines are cached per UTC day.

The station set is driven by the model's own fcols (KIR was excluded in
training for its sign-inverted realtime channel, so it never appears).
"""
import json
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
STATE = HERE / "state" / "groundmag"
STORE = STATE / "h_store.parquet"
BASE_CACHE = STATE / "baselines.parquet"

NLAGS = 6
STORE_DAYS = 45
BASE_WINDOW_DAYS = 30
QUIET_DAYS = 5
MIN_VALID_FRAC = 0.2


def update_h_store(series):
    """Merge {code: {'YYYY-MM-DDTHH:MM': H}} into the rolling store.
    New values win over stored ones (GIN backfills/revises); trailing
    STORE_DAYS kept. Called by fetch_groundmag_rt every run."""
    new = {c: pd.Series({pd.Timestamp(k): v for k, v in s.items()},
                        dtype="float64")
           for c, s in series.items() if s}
    if not new:
        return
    new_df = pd.DataFrame(new).sort_index()
    if STORE.exists():
        old = pd.read_parquet(STORE)
        df = new_df.combine_first(old)
    else:
        df = new_df
    cutoff = pd.Timestamp(datetime.utcnow()) - pd.Timedelta(days=STORE_DAYS)
    df = df.loc[df.index >= cutoff].astype("float32")
    STATE.mkdir(parents=True, exist_ok=True)
    tmp = STORE.with_suffix(".tmp")
    df.to_parquet(tmp)
    tmp.replace(STORE)


def _quiet_baseline(h):
    """1440-array minute-of-day baseline from a station's trailing-window
    series — the training builder's per-month method, verbatim."""
    if h.notna().mean() < MIN_VALID_FRAC:
        return None
    std = h.groupby(h.index.date).std()
    quiet = std.dropna().nsmallest(QUIET_DAYS).index
    if len(quiet) == 0:
        return None
    qd = h[np.isin(h.index.date, quiet)]
    mod = qd.index.hour * 60 + qd.index.minute
    curve = qd.groupby(mod).mean().reindex(range(1440))
    ext = pd.concat([curve, curve, curve], ignore_index=True)
    sm = ext.rolling(61, center=True, min_periods=10).mean()[1440:2880]
    sm.index = range(1440)
    return sm


def _baselines(today, stations):
    """{station: 1440-Series} for today's UTC date, cached in one parquet
    (a 'date' attr row via column name); recomputed once per day."""
    if BASE_CACHE.exists():
        try:
            cached = pd.read_parquet(BASE_CACHE)
            if cached.attrs.get("date") == str(today) or \
                    cached.columns.name == str(today):
                return {c: cached[c] for c in cached.columns
                        if c in stations and cached[c].notna().any()}
        except Exception:
            pass
    store = pd.read_parquet(STORE)
    end = pd.Timestamp(today)                      # midnight UTC today
    start = end - pd.Timedelta(days=BASE_WINDOW_DAYS)
    win = store.loc[(store.index >= start) & (store.index < end)]
    out = {}
    cols = {}
    for sta in store.columns:
        b = _quiet_baseline(win[sta]) if sta in win else None
        cols[sta] = b if b is not None else pd.Series(np.nan, index=range(1440))
        if b is not None:
            out[sta] = cols[sta]
    doc = pd.DataFrame(cols)
    doc.columns.name = str(today)
    tmp = BASE_CACHE.with_suffix(".tmp")
    doc.to_parquet(tmp)
    tmp.replace(BASE_CACHE)
    return {c: v for c, v in out.items() if c in stations}


_MLON = {}


def _station_mlon(sta, coords, year):
    import aacgmv2
    key = (sta, year)
    if key not in _MLON:
        lat, lon = coords[sta]
        _, mlon, _ = aacgmv2.convert_latlon(
            lat, lon, 0.0, datetime(year, 7, 1), method_code="G2A")
        _MLON[key] = mlon
    return _MLON[key]


def _mlt_lookup(hours, mlon):
    import aacgmv2
    return {h: aacgmv2.convert_mlt(mlon, h.to_pydatetime())[0] for h in hours}


def build(valid_times, stations, coords):
    """DataFrame indexed by valid_times with {STA}_{dh0..5,age,mltsin,mltcos}
    for every requested station (all-NaN columns where a station has no
    usable data — the model treats NaN as 'station dark', as in training)."""
    times = pd.DatetimeIndex(sorted(valid_times))
    if not STORE.exists():
        raise FileNotFoundError("h_store.parquet missing — fetcher not run")
    store = pd.read_parquet(STORE)
    # dh/rm only need history back to (oldest target - age - 30 min); 5 days
    # is generous and keeps this light
    tail = store.loc[store.index >= times.min() - pd.Timedelta(days=5)]
    grid = pd.date_range(tail.index.min(), max(tail.index.max(), times.max()),
                         freq="1min")
    today = datetime.utcnow().date()
    baselines = _baselines(today, set(stations))
    hours = pd.date_range(times.min().floor("h"), times.max().ceil("h"),
                          freq="1h")
    year = times.max().year

    out = {}
    for sta in stations:
        cols = {f"{sta}_{s}": np.full(len(times), np.nan, dtype=np.float32)
                for s in [f"dh{k}" for k in range(NLAGS)]
                + ["age", "mltsin", "mltcos"]}
        if sta in coords:
            mlon = _station_mlon(sta, coords, year)
            mlt_h = _mlt_lookup(hours, mlon)
            mlt = np.array([(mlt_h[t.floor("h")] + t.minute / 60.0) % 24
                            for t in times])
            ang = 2 * np.pi * mlt / 24.0
            cols[f"{sta}_mltsin"] = np.sin(ang).astype(np.float32)
            cols[f"{sta}_mltcos"] = np.cos(ang).astype(np.float32)
        if sta in tail.columns and sta in baselines:
            h = tail[sta].reindex(grid)
            mod = grid.hour * 60 + grid.minute
            dh = h - baselines[sta].reindex(mod).to_numpy()
            rm = dh.rolling("5min", min_periods=3).median().to_numpy()
            valid = h.notna()
            if valid.any():
                last_valid = h.index[valid][-1]
                pos = {ts: i for i, ts in enumerate(grid)}
                for j, t in enumerate(times):
                    age = max(0.0, (t - last_valid).total_seconds() / 60.0)
                    cols[f"{sta}_age"][j] = age
                    anchor = t - pd.Timedelta(minutes=round(age))
                    for k in range(NLAGS):
                        ts = anchor - pd.Timedelta(minutes=5 * k)
                        i = pos.get(ts)
                        if i is not None:
                            cols[f"{sta}_dh{k}"][j] = rm[i]
        out.update(cols)
    return pd.DataFrame(out, index=times)


def model_stations(fcols):
    """Station codes the model was trained with, from its own column list."""
    return sorted({c.split("_")[0] for c in fcols if c.endswith("_dh0")})


def backfill(days=31):
    """One-time store bootstrap: one N-day GIN request per station."""
    from fetch_groundmag_rt import STATIONS, fetch_station
    import concurrent.futures
    start = (datetime.utcnow() - timedelta(days=days)).strftime("%Y-%m-%d")
    series = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as ex:
        futs = {ex.submit(fetch_station, c, start, days + 1): c
                for c in STATIONS}
        for fut in concurrent.futures.as_completed(futs):
            code = futs[fut]
            series[code] = fut.result()
            print(f"{code}: {len(series[code])} minutes", flush=True)
    update_h_store(series)
    df = pd.read_parquet(STORE)
    print(f"store: {df.shape[0]} minutes x {df.shape[1]} stations, "
          f"{df.index.min()} .. {df.index.max()}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--backfill", type=int, metavar="DAYS")
    a = ap.parse_args()
    if a.backfill:
        backfill(a.backfill)
    else:
        from fetch_groundmag_rt import STATIONS
        from predict_aual import Corrector
        c = Corrector("models/corrector_grid_2deg_v01.joblib")
        stas = model_stations([col for _, col in c.other_cols])
        now = pd.Timestamp(datetime.utcnow()).floor("min")
        df = build([now - pd.Timedelta(minutes=2)], stas, STATIONS)
        row = df.iloc[0]
        live = [s for s in stas if not np.isnan(row.get(f"{s}_dh0", np.nan))]
        print(f"{len(stas)} model stations, {len(live)} with live dh0: {live}")
        print(row[[f"{live[0]}_{x}" for x in
                   ("dh0", "dh1", "age", "mltsin", "mltcos")]].to_string()
              if live else "(no live stations yet — backfill/fetch first)")
