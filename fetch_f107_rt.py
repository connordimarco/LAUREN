#!/usr/bin/env python3
"""Realtime F10.7 (+ 81-day history/average) for the corrector and GITM-RT.

Two sources, two roles:
- SWPC observed 10.7 cm flux (json/f107_cm_flux.json, ~42 days, intraday):
  the freshest current value. Cache cache/f107.json, TTL 3 h.
- CelesTrak SW-Last5Years.csv (daily, years deep, updated daily): history
  for the 81-day averages GITM needs. Only type==OBS rows are used (the
  file continues into predicted monthlies). Cache cache/f107_sw5y.csv,
  TTL 12 h. A stale cache always beats no data.

Product realtime-f107/f107.json (written only on change, so the SWORD
rsync ships it only when something moved):
  f107          latest observed flux (SWPC, intraday)
  time_tag      its observation time
  f107a_last81  trailing 81-day mean of the merged daily series, incl.
                the current day (n_days says how many days actually
                contributed — normally 81)
  daily         merged {YYYY-MM-DD: flux} series (~90 days): CelesTrak
                OBS overlaid with SWPC's last-obs-of-day for days
                CelesTrak has not published yet (i.e. today)
GITM-RT reads this instead of running its own SWPC fetch cron; the daily
series is deep enough to recompute centered/trailing averages remotely.

f107_latest() (the corrector's hook) returns the current flux and
refreshes the product as a side effect; the hindcast convention (OMNI2
daily observed value, ffilled) is unchanged.
"""
import csv
import json
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
CACHE = HERE / "cache" / "f107.json"
CACHE_HIST = HERE / "cache" / "f107_sw5y.csv"
PRODUCT = HERE / "realtime-f107" / "f107.json"
TTL_S = 3 * 3600
TTL_HIST_S = 12 * 3600
DAILY_DEPTH = 90
UA = {"User-Agent":
      "GITM-RT research (University of Michigan; cdimarco@umich.edu)"}
# timeseries first (explicit time_tags), single-value summary as fallback
URLS = ("https://services.swpc.noaa.gov/json/f107_cm_flux.json",
        "https://services.swpc.noaa.gov/products/summary/10cm-flux.json")
URL_HIST = "https://celestrak.org/SpaceData/SW-Last5Years.csv"


def _get(url, timeout=30):
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def _swpc():
    """{'flux', 'time_tag', 'daily': {date: last-obs-of-day flux}} via cache."""
    if CACHE.exists() and time.time() - CACHE.stat().st_mtime < TTL_S:
        return json.loads(CACHE.read_text())
    payload = None
    for url in URLS:
        try:
            rows = sorted(json.loads(_get(url, 20)),
                          key=lambda d: d["time_tag"])
            daily = {}
            for r in rows:                      # sorted: last obs of day wins
                daily[r["time_tag"][:10]] = float(r["flux"])
            payload = {"flux": float(rows[-1]["flux"]),
                       "time_tag": rows[-1]["time_tag"], "daily": daily}
            break
        except Exception:
            continue
    if payload is None:
        if CACHE.exists():                      # stale beats none
            return json.loads(CACHE.read_text())
        raise RuntimeError("no F10.7 from SWPC and no cache")
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    tmp = CACHE.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload))
    tmp.replace(CACHE)
    return payload


def _celestrak_daily():
    """{date: observed flux} from CelesTrak (type==OBS rows only), via cache.
    Returns {} if CelesTrak is unreachable and no cache exists — the product
    then degrades to SWPC's ~42 days and says so via n_days."""
    if not (CACHE_HIST.exists()
            and time.time() - CACHE_HIST.stat().st_mtime < TTL_HIST_S):
        try:
            body = _get(URL_HIST, 60)
            if len(body) < 10000:
                raise ValueError(f"short body {len(body)}")
            CACHE_HIST.parent.mkdir(parents=True, exist_ok=True)
            tmp = CACHE_HIST.with_suffix(".tmp")
            tmp.write_bytes(body)
            tmp.replace(CACHE_HIST)
        except Exception:
            if not CACHE_HIST.exists():
                return {}
    out = {}
    with open(CACHE_HIST, newline="") as fh:
        for row in csv.DictReader(fh):
            if row.get("F10.7_DATA_TYPE") == "OBS" and row.get("F10.7_OBS"):
                out[row["DATE"]] = float(row["F10.7_OBS"])
    return out


def _publish(payload):
    """Write the product only on change, so rsync ships it only then."""
    text = json.dumps(payload)
    if PRODUCT.exists() and PRODUCT.read_text() == text:
        return
    PRODUCT.parent.mkdir(parents=True, exist_ok=True)
    tmp = PRODUCT.with_suffix(".tmp")
    tmp.write_text(text)
    tmp.replace(PRODUCT)


def refresh_product():
    """Rebuild realtime-f107/f107.json from both sources. Returns payload."""
    swpc = _swpc()
    daily = _celestrak_daily()
    daily.update(swpc["daily"])         # SWPC wins where both have a day
    days = sorted(daily)[-DAILY_DEPTH:]
    last81 = days[-81:]
    payload = {
        "f107": swpc["flux"],
        "time_tag": swpc["time_tag"],
        "f107a_last81": round(sum(daily[d] for d in last81) / len(last81), 1),
        "n_days": len(last81),
        "daily": {d: daily[d] for d in days},
        "sources": {"current": "SWPC f107_cm_flux",
                    "history": "CelesTrak SW-Last5Years (OBS)"},
    }
    _publish(payload)
    return payload


def f107_latest():
    """Latest observed F10.7 (sfu) as float; refreshes the product."""
    return refresh_product()["f107"]


if __name__ == "__main__":
    p = refresh_product()
    print(f"F10.7 = {p['f107']:.1f} sfu ({p['time_tag']})  "
          f"F10.7a(last81) = {p['f107a_last81']:.1f} over {p['n_days']} days; "
          f"daily series {min(p['daily'])} .. {max(p['daily'])}")
