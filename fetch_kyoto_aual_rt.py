#!/usr/bin/env python3
"""Daily fetch of the REAL (observed) AU/AL from WDC Kyoto's realtime
quicklook AE service, for live validation of LAUREN's predictions.

Kyoto posts day files at
  https://wdc.kugi.kyoto-u.ac.jp/ae_realtime/data_dir/YYYY/MM/DD/{au,al}YYMMDD
in WDC hourly format (60 one-minute values + hourly mean per line),
typically ~2 weeks behind real time. This script keeps a rolling store
(state/kyoto/aual_store.parquet, KEEP_DAYS) and writes two website
products into realtime-aual/ — named last_*.json ON PURPOSE so the MIDL
relay's `last_*.json` glob ships them to herot data/aual/ unchanged:

  last_kyoto.json      observed AU/AL, last 35 d, 15-min means
                       (matches the 35d prediction view's resampling;
                       the 24h/7d views never overlap Kyoto's lag)
  last_validation.json daily r + RMSE of LAUREN vs observed over the
                       predictions store (36 d), plus a 35 d summary

Run once per day from cron (dev/kyoto_fetch.sh). Steady state is ~6
requests/day (re-fetch the newest 3 posted days for quicklook revisions
+ probe forward until two consecutive missing days). Quiet failure: any
network error just waits for tomorrow's run — no watchdog by design.

Data courtesy of WDC for Geomagnetism, Kyoto (quicklook AE; provisional).
"""
import json
import sys
import time
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
STORE = ROOT / "state" / "kyoto" / "aual_store.parquet"
OUTDIR = ROOT / "realtime-aual"
PRED = ROOT / "state" / "predictions.csv"
LOGS = ROOT / "logs"

BASE = "https://wdc.kugi.kyoto-u.ac.jp/ae_realtime/data_dir"
KEEP_DAYS = 60          # rolling store bound (Kyoto is the archive)
LOOKBACK_DAYS = 45      # bootstrap window / oldest day ever fetched
REFETCH_DAYS = 3        # newest posted days re-fetched for revisions
MISS_STOP = 2           # stop probing forward after N consecutive 404 days
VIEW_DAYS = 35          # matches the site's longest prediction view
MIN_DAY_ROWS = 360      # need >=6 h of overlap for a daily statistic


def fetch_day(day):
    """{'AU': series, 'AL': series} of 1-min values for one UT day, or
    None if either file is missing (day not posted yet)."""
    out = {}
    for idx in ("au", "al"):
        url = f"{BASE}/{day:%Y/%m/%d}/{idx}{day:%y%m%d}"
        try:
            with urllib.request.urlopen(url, timeout=30) as r:
                text = r.read().decode("ascii", "replace")
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return None
            raise
        rows = []
        for line in text.splitlines():
            toks = line.split()
            if len(toks) < 63 or toks[2] != "QUICKLK":
                continue
            hour = int(toks[1][7:9])
            base = pd.Timestamp(day) + pd.Timedelta(hours=hour)
            for k, v in enumerate(toks[3:63]):     # 60 min; [63] is hourly mean
                v = float(v)
                rows.append((base + pd.Timedelta(minutes=k),
                             np.nan if abs(v) >= 9000 else v))
        if not rows:
            return None
        out[idx.upper()] = pd.Series(dict(rows))
        time.sleep(0.5)                            # be polite to the WDC
    return out


def update_store(now):
    store = pd.read_parquet(STORE) if STORE.exists() else \
        pd.DataFrame(columns=["AU", "AL"], index=pd.DatetimeIndex([], name="t"))
    have_days = set(store.index.normalize().unique())
    newest = max(have_days) if have_days else None
    start = (now - timedelta(days=LOOKBACK_DAYS)).date()
    if newest is not None:
        start = max(start, (newest - pd.Timedelta(days=REFETCH_DAYS - 1)).date())
    fetched, missing, day = 0, 0, start
    while day <= now.date() and missing < MISS_STOP:
        got = fetch_day(day)
        if got is None:
            missing += 1
        else:
            missing = 0
            new = pd.DataFrame({"AU": got["AU"], "AL": got["AL"]})
            new.index.name = "t"
            store = pd.concat([store[~store.index.normalize()
                                     .isin([pd.Timestamp(day)])], new])
            fetched += 1
        day += timedelta(days=1)
    store = store.sort_index()
    store = store[store.index >= pd.Timestamp(now.date())
                  - pd.Timedelta(days=KEEP_DAYS)]
    STORE.parent.mkdir(parents=True, exist_ok=True)
    tmp = STORE.with_suffix(".tmp")
    store.to_parquet(tmp)
    tmp.replace(STORE)
    return store, fetched


def write_kyoto_view(store, now):
    keep = store[store.index >= now - pd.Timedelta(days=VIEW_DAYS)]
    keep = keep.resample("15min").mean().dropna(how="all")
    doc = {"fetched_utc": now.strftime("%Y-%m-%dT%H:%M:%SZ"),
           "source": "WDC for Geomagnetism, Kyoto — realtime (quicklook) AE",
           "t": [t.strftime("%Y-%m-%dT%H:%M:%SZ") for t in keep.index],
           "au": [None if np.isnan(v) else round(v, 1) for v in keep["AU"]],
           "al": [None if np.isnan(v) else round(v, 1) for v in keep["AL"]]}
    tmp = OUTDIR / "last_kyoto.json.tmp"
    tmp.write_text(json.dumps(doc))
    tmp.replace(OUTDIR / "last_kyoto.json")
    return len(keep)


def write_kyoto_full(store, now):
    """Full 1-min observed series, 35 d, epoch-ms timestamps (the live
    page's file; last_kyoto.json stays 15-min for cached older JS).
    Reindexed to a regular minute grid so unposted stretches become
    explicit nulls (drawn as gaps, not bridged)."""
    keep = store[store.index >= now - pd.Timedelta(days=VIEW_DAYS)]
    if len(keep):
        keep = keep.reindex(pd.date_range(keep.index.min(), keep.index.max(),
                                          freq="min"))
    doc = {"fetched_utc": now.strftime("%Y-%m-%dT%H:%M:%SZ"),
           "source": "WDC for Geomagnetism, Kyoto — realtime (quicklook) AE",
           "tms": [int(t.value // 10**6) for t in keep.index],
           "au": [None if np.isnan(v) else round(v, 1) for v in keep["AU"]],
           "al": [None if np.isnan(v) else round(v, 1) for v in keep["AL"]]}
    tmp = OUTDIR / "last_kyoto_full.json.tmp"
    tmp.write_text(json.dumps(doc))
    tmp.replace(OUTDIR / "last_kyoto_full.json")
    return len(keep)


def write_validation_view(store, now):
    pred = pd.read_csv(PRED, index_col=0, parse_dates=True)
    pred = pred[~pred.index.duplicated(keep="last")].sort_index()
    m = pred[["AU", "AL"]].join(store, how="inner", rsuffix="_obs").dropna()
    m = m[m.index >= now - pd.Timedelta(days=VIEW_DAYS)]
    days = []
    for day, d in m.groupby(m.index.normalize()):
        if len(d) < MIN_DAY_ROWS:
            continue
        row = {"d": day.strftime("%Y-%m-%d"), "n": int(len(d))}
        for q in ("AU", "AL"):
            err = d[q] - d[f"{q}_obs"]
            row[f"r_{q.lower()}"] = round(float(d[q].corr(d[f"{q}_obs"])), 3)
            row[f"rmse_{q.lower()}"] = round(float(np.sqrt((err**2).mean())), 1)
        days.append(row)
    summary = {"n": int(len(m))}
    if len(m) >= MIN_DAY_ROWS:
        for q in ("AU", "AL"):
            err = m[q] - m[f"{q}_obs"]
            summary[f"r_{q.lower()}"] = round(float(m[q].corr(m[f"{q}_obs"])), 3)
            summary[f"rmse_{q.lower()}"] = round(float(np.sqrt((err**2).mean())), 1)
    doc = {"computed_utc": now.strftime("%Y-%m-%dT%H:%M:%SZ"),
           "note": ("LAUREN prediction vs Kyoto realtime (quicklook) AE, "
                    "1-min pairs; quicklook posts ~2 weeks behind real time"),
           "days": days, "summary": summary}
    tmp = OUTDIR / "last_validation.json.tmp"
    tmp.write_text(json.dumps(doc))
    tmp.replace(OUTDIR / "last_validation.json")
    return len(days), summary


def main():
    now = datetime.now(timezone.utc).replace(tzinfo=None)
    store, fetched = update_store(now)
    npts = write_kyoto_view(store, now)
    write_kyoto_full(store, now)
    ndays, summary = write_validation_view(store, now)
    line = (f"{now.strftime('%Y-%m-%dT%H:%M:%S')} OK fetched_days={fetched} "
            f"store_rows={len(store)} view_pts={npts} stat_days={ndays} "
            f"r_al={summary.get('r_al')} rmse_al={summary.get('rmse_al')}\n")
    LOGS.mkdir(parents=True, exist_ok=True)
    with open(LOGS / f"kyoto_{now.strftime('%Y%m')}.log", "a") as fh:
        fh.write(line)
    sys.stdout.write(line)
    return 0


if __name__ == "__main__":
    sys.exit(main())
