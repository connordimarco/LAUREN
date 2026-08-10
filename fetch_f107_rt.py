#!/usr/bin/env python3
"""Realtime F10.7 for the corrector feature row.

SWPC observed 10.7 cm flux (services.swpc.noaa.gov). The hindcast uses
OMNI2's daily F10.7 ffilled to 1-min; the realtime equivalent is the latest
observed value held constant over the feature frame — exactly equivalent
for the extracted rows, since f10.7_index is never lagged.

Caches to cache/f107.json (TTL 3 h); a stale cache beats no data. The
current value is also mirrored to realtime-f107/f107.json (written only on
change, so rsync pushes it to SWORD only when SWPC posts a new flux) —
GITM-RT reads that instead of running its own SWPC fetch cron.
"""
import json
import time
import urllib.request
from pathlib import Path

CACHE = Path(__file__).resolve().parent / "cache" / "f107.json"
PRODUCT = Path(__file__).resolve().parent / "realtime-f107" / "f107.json"
TTL_S = 3 * 3600
UA = {"User-Agent":
      "GITM-RT research (University of Michigan; connordimarco@gmail.com)"}
# timeseries first (explicit time_tags), single-value summary as fallback
URLS = ("https://services.swpc.noaa.gov/json/f107_cm_flux.json",
        "https://services.swpc.noaa.gov/products/summary/10cm-flux.json")


def _fetch():
    for url in URLS:
        try:
            req = urllib.request.Request(url, headers=UA)
            with urllib.request.urlopen(req, timeout=20) as r:
                obj = json.load(r)
            rows = sorted(obj, key=lambda d: d["time_tag"])
            return float(rows[-1]["flux"]), rows[-1]["time_tag"]
        except Exception:
            continue
    return None, None


def _publish(payload):
    """Mirror the current value into realtime-f107/ (only on change, so the
    SWORD rsync ships it only when SWPC posts a new flux)."""
    text = json.dumps(payload)
    if PRODUCT.exists() and PRODUCT.read_text() == text:
        return
    PRODUCT.parent.mkdir(parents=True, exist_ok=True)
    tmp = PRODUCT.with_suffix(".tmp")
    tmp.write_text(text)
    tmp.replace(PRODUCT)


def f107_latest():
    """Latest observed F10.7 (sfu) as float."""
    if CACHE.exists() and time.time() - CACHE.stat().st_mtime < TTL_S:
        payload = json.loads(CACHE.read_text())
        _publish(payload)
        return payload["flux"]
    flux, tag = _fetch()
    if flux is None:
        if CACHE.exists():                      # stale beats none
            payload = json.loads(CACHE.read_text())
            _publish(payload)
            return payload["flux"]
        raise RuntimeError("no F10.7 from SWPC and no cache")
    payload = {"flux": flux, "time_tag": tag}
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    tmp = CACHE.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload))
    tmp.replace(CACHE)
    _publish(payload)
    return flux


if __name__ == "__main__":
    print(f"F10.7 = {f107_latest():.1f} sfu")
