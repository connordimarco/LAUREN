#!/usr/bin/env python3
"""Realtime Kyoto Dst for the AU/AL corrector feature row.

Fetches dst_realtime/YYYYMM/dstYYMM.for.request (WDC format, same layout
the hindcast's load_dst_wdc parses), caches per month, refetches the
current month at most every 20 min. Returns an hourly pandas Series
stamped HH:30 (the GeoDGP/pyspedas convention).

Usage:
    from fetch_dst_rt import dst_hourly
    s = dst_hourly(hours_back=14)   # covers the 12 Dst lags + slack
"""
import time
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

CACHE = Path(__file__).resolve().parent / "cache" / "dst"
BASE = "https://wdc.kugi.kyoto-u.ac.jp/dst_realtime/{ym}/dst{ym2}.for.request"
UA = {"User-Agent":
      "GITM-RT research (University of Michigan; cdimarco@umich.edu)"}
CURRENT_MONTH_TTL_S = 20 * 60


def _fetch_month(year, month):
    ym = f"{year:04d}{month:02d}"
    dest = CACHE / f"dst{ym}.for.request"
    fresh_enough = (dest.exists() and
                    time.time() - dest.stat().st_mtime < CURRENT_MONTH_TTL_S)
    now = datetime.now(timezone.utc)
    is_past_month = (year, month) < (now.year, now.month)
    if dest.exists() and (is_past_month or fresh_enough):
        return dest
    url = BASE.format(ym=ym, ym2=ym[2:])
    req = urllib.request.Request(url, headers=UA)
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            body = r.read()
        if len(body) < 100:
            raise ValueError(f"short body {len(body)}")
        CACHE.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_suffix(".tmp")
        tmp.write_bytes(body)
        tmp.replace(dest)
    except Exception:
        if not dest.exists():       # stale cache beats no data
            raise
    return dest


def _parse_wdc(path):
    """WDC dst .for.request -> hourly series stamped HH:30 (NaN for 9999).

    Matches geodgp_ae/geodgp_common.load_dst_wdc: 24 4-char fields at
    cols 20:116, value = base*100 + field, fill 9999 -> NaN.
    """
    times, vals = [], []
    for line in path.read_text().splitlines():
        if not line.startswith("DST"):
            continue
        yy, mm, dd = int(line[3:5]), int(line[5:7]), int(line[8:10])
        century = int(line[14:16]) if line[14:16].strip() else 20
        base = int(line[16:20])
        for h in range(24):
            v = int(line[20 + 4 * h:24 + 4 * h])
            times.append(datetime(century * 100 + yy, mm, dd, h, 30))
            vals.append(np.nan if v == 9999 else base * 100 + v)
    return pd.Series(vals, index=pd.DatetimeIndex(times), name="Dst").sort_index()


def dst_hourly(hours_back=14):
    """Hourly Dst covering [now - hours_back, now], HH:30 stamps, no NaN
    at the tail (trailing NaN hours are dropped — realtime Dst lags the
    clock by up to ~1-2 h; callers ffill per GeoDGP convention)."""
    now = datetime.now(timezone.utc).replace(tzinfo=None)
    t0 = now - timedelta(hours=hours_back)
    months = {(t0.year, t0.month), (now.year, now.month)}
    parts = [_parse_wdc(_fetch_month(y, m)) for y, m in sorted(months)]
    s = pd.concat(parts).sort_index()
    s = s[~s.index.duplicated(keep="last")]
    s = s.loc[s.index >= t0].dropna()
    if s.empty:
        raise RuntimeError("no realtime Dst in window")
    return s


if __name__ == "__main__":
    s = dst_hourly()
    print(s.tail(6).to_string())
    age = (datetime.now(timezone.utc).replace(tzinfo=None)
           - s.index[-1]).total_seconds() / 60
    print(f"latest Dst is {age:.0f} min old")
