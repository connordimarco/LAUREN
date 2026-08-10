#!/usr/bin/env python3
"""Realtime ground-magnetometer fetcher for LAUREN.

Adapted from the SWORD ae_proto health monitor (station_fetch_package):
same INTERMAGNET GIN (BGS) source, same station roster, same health PDF —
but where the original kept only per-station *latency*, this keeps the
actual 1-min **H-component** series the corrector's station features are
built from (AU/AL are defined from H-component deviations).

Dumps (atomically replaced each run) in realtime-groundmag/:
  groundmag_h.json     per-station 1-min H series on a common 26 h grid:
                       {generated_utc, window_start, cadence_min,
                        stations: {CODE: {lat, lon, h[], last_valid,
                                          age_min}}}   (null = missing)
  station_health.pdf   north-polar staleness map + latency table

State: state/groundmag/history.jsonl (7-day latency window, self-pruning).
Log: one line per run to logs/groundmag_YYYYMM.log (pruned by the tick's
6-month log retention, same pattern as tick_/push_ logs).

H per station: reported 'H' column directly, else hypot(X, Y). Component
letters are parsed from the IAGA-2002 data header row. Values with
|v| >= 88887 (IAGA missing/not-recorded sentinels) become null.
"""
import concurrent.futures
import json
import math
import os
import sys
import urllib.parse
import urllib.request
from datetime import datetime, timedelta
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT_DIR = HERE / "realtime-groundmag"
STATE = HERE / "state" / "groundmag"
HISTORY = STATE / "history.jsonl"
JSON_OUT = OUT_DIR / "groundmag_h.json"
PDF_OUT = OUT_DIR / "station_health.pdf"
COASTLINES = HERE / "dev" / "coastlines.json"
LOGS = HERE / "logs"

GIN_URL = "https://imag-data.bgs.ac.uk/GIN_V1/GINServices"
REQ_TIMEOUT_S = 25
WORKERS = 8
AVG_WINDOW_DAYS = 7
WINDOW_H = 26                  # series depth kept in the dump
MISSING = 88887.0

# code: (lat, lon_east). First 12 are the official AE ring; the rest are
# additional high-latitude INTERMAGNET observatories. Russian AE stations
# (DIK/CCS/TIK/PBK) have no public realtime feed — kept for the health map;
# Canada/UK GIN feeds carry deliberate embargoes and run stale by policy.
STATIONS = {
    "ABK": (68.36, 18.82), "DIK": (73.55, 80.57), "CCS": (77.72, 104.28),
    "TIK": (71.58, 129.00), "PBK": (70.09, 170.93), "BRW": (71.32, -156.62),
    "CMO": (64.87, -147.86), "YKC": (62.48, -114.48), "FCC": (58.76, -94.09),
    "SNK": (56.54, -79.22), "NAQ": (61.16, -45.44), "LRV": (64.18, -21.70),
    "THL": (77.47, -69.23), "GDH": (69.25, -53.53), "DED": (70.36, -148.79),
    "SHU": (55.35, -160.46), "SIT": (57.06, -135.33), "CBB": (69.12, -105.03),
    "RES": (74.69, -94.90), "BLC": (64.32, -96.01), "IQA": (63.75, -68.52),
    "MEA": (54.62, -113.35), "KIR": (67.84, 20.42), "SOD": (67.37, 26.63),
    "NUR": (60.51, 24.66), "HRN": (77.00, 15.55), "LYC": (64.61, 18.75),
    "DOB": (62.07, 9.11), "YAK": (61.96, 129.66), "SPG": (60.54, 29.72),
    "LER": (60.14, -1.18), "MGD": (60.05, 150.73), "UPS": (59.90, 17.35),
    "BOX": (58.07, 38.23), "ARS": (56.43, 58.57), "BFE": (55.62, 11.67),
    "ESK": (55.31, -3.21), "NVS": (54.85, 83.23), "HLP": (54.60, 18.81),
    "WNG": (53.73, 9.05), "PET": (52.97, 158.25), "IRT": (52.17, 104.45),
    "NGK": (52.07, 12.68), "VAL": (51.93, -10.25), "BEL": (51.84, 20.79),
    "HAD": (51.00, -4.48), "KIV": (50.72, 30.30), "MAB": (50.30, 5.68),
    "DOU": (50.10, 4.60),
}

BINS = [(10, "#1a9850", "≤ 10 min"), (30, "#91cf60", "≤ 30 min"),
        (120, "#fee08b", "≤ 2 h"), (1440, "#fc8d59", "≤ 24 h"),
        (float("inf"), "#d73027", "> 24 h")]
NODATA_COLOR = "#999999"


def staleness_color(latency_min):
    if latency_min is None:
        return NODATA_COLOR
    for edge, color, _ in BINS:
        if latency_min <= edge:
            return color
    return NODATA_COLOR


def fetch_station(code, start_date, duration_days=2):
    """{minute_iso: H_nT} for the window (empty dict on any failure).
    Default 2 days so the UTC-midnight rollover never looks like a dead
    station; station_features_rt --backfill passes more."""
    params = urllib.parse.urlencode({
        "Request": "GetData", "format": "Iaga2002", "testObsys": "0",
        "observatoryIagaCode": code, "samplesPerDay": "1440",
        "publicationState": "adj-or-rep", "dataStartDate": start_date,
        "dataDuration": str(duration_days)})
    try:
        with urllib.request.urlopen(GIN_URL + "?" + params,
                                    timeout=REQ_TIMEOUT_S) as r:
            body = r.read().decode("utf-8", "replace")
    except Exception:
        return {}
    comps = None
    out = {}
    for line in body.splitlines():
        parts = line.split()
        if not parts:
            continue
        if parts[0] == "DATE":                  # data header: DATE TIME DOY ABKX ...
            comps = [p[len(code):] or p[-1] for p in parts[3:] if p != "|"]
            continue
        if len(parts) < 7 or not parts[0][:2].isdigit() or comps is None:
            continue
        try:
            vals = [float(v) for v in parts[3:3 + len(comps)]]
        except ValueError:
            continue
        by = {c: (None if abs(v) >= MISSING else v)
              for c, v in zip(comps, vals)}
        if by.get("H") is not None:
            h = by["H"]
        elif by.get("X") is not None and by.get("Y") is not None:
            h = math.hypot(by["X"], by["Y"])
        else:
            continue
        out[parts[0] + "T" + parts[1][:5]] = h   # minute key YYYY-MM-DDTHH:MM
    return out


def load_prune_history(now):
    rows = []
    if HISTORY.exists():
        for line in HISTORY.read_text().splitlines():
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    cutoff = (now - timedelta(days=AVG_WINDOW_DAYS)).strftime("%Y-%m-%dT%H:%M:%S")
    kept = [r for r in rows if r.get("t", "") >= cutoff]
    if len(kept) < len(rows):
        tmp = HISTORY.with_suffix(".tmp")
        tmp.write_text("".join(json.dumps(r) + "\n" for r in kept))
        tmp.replace(HISTORY)
    return kept


def fmt_latency(m):
    if m is None:
        return "—"
    if m < 90:
        return f"{m:.0f} m"
    if m < 48 * 60:
        return f"{m / 60:.1f} h"
    return f"{m / 1440:.1f} d"


def render_pdf(latencies, averages, now, n_runs):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.patches import Patch

    fig = plt.figure(figsize=(8.5, 11))
    ax = fig.add_axes([0.10, 0.41, 0.80, 0.47], projection="polar")
    ax.set_theta_zero_location("S")
    ax.set_theta_direction(1)
    ax.set_rlim(0, 40)
    ax.set_rgrids([10, 20, 30, 40], labels=["80°", "70°", "60°", "50°"],
                  fontsize=7, color="#666666")
    ax.set_thetagrids([0, 45, 90, 135, 180, 225, 270, 315],
                      labels=["0°", "45°E", "90°E", "135°E", "180°",
                              "135°W", "90°W", "45°W"], fontsize=7)
    ax.grid(True, linewidth=0.4, alpha=0.5)
    try:
        cl = json.loads(COASTLINES.read_text())
        lon = np.array([v if v is not None else np.nan for v in cl["lon"]], float)
        lat = np.array([v if v is not None else np.nan for v in cl["lat"]], float)
        lat = np.where(lat >= 50.0, lat, np.nan)
        ax.plot(np.radians(lon), 90.0 - lat, lw=0.4, color="#aaaaaa", zorder=1)
    except Exception:
        pass
    for code, (lat, lon) in STATIONS.items():
        th = math.radians(lon % 360.0)
        r = 90.0 - lat
        ax.plot([th], [r], "o", ms=8, color=staleness_color(latencies.get(code)),
                mec="#333333", mew=0.5, zorder=3)
        ax.annotate(code, (th, r), textcoords="offset points", xytext=(4, 4),
                    fontsize=5.5, zorder=4)
    items = [Patch(facecolor=c, edgecolor="#333333", label=lbl)
             for _, c, lbl in BINS]
    items.append(Patch(facecolor=NODATA_COLOR, edgecolor="#333333", label="no data"))
    ax.legend(handles=items, loc="lower left", bbox_to_anchor=(-0.12, -0.08),
              fontsize=7, title="data age", title_fontsize=7, framealpha=0.9)
    fig.suptitle("High-latitude magnetometer feed health — INTERMAGNET GIN "
                 "(reported)", fontsize=12, y=0.965)
    fig.text(0.5, 0.942,
             f"generated {now:%Y-%m-%d %H:%M} UT  •  run #{n_runs}  •  "
             f"avg over last {AVG_WINDOW_DAYS} d  •  AE-ring stations: "
             "ABK DIK CCS TIK PBK BRW CMO YKC FCC SNK NAQ LRV",
             ha="center", fontsize=7.5, color="#444444")
    order = sorted(STATIONS, key=lambda c: (latencies.get(c) is None,
                                            latencies.get(c) or 0.0))
    ncol = 3
    per = (len(order) + ncol - 1) // ncol
    for i in range(ncol):
        chunk = order[i * per:(i + 1) * per]
        if not chunk:
            continue
        cells = [[c, fmt_latency(latencies.get(c)), fmt_latency(averages.get(c))]
                 for c in chunk]
        tax = fig.add_axes([0.06 + 0.32 * i, 0.04, 0.29, 0.33])
        tax.axis("off")
        tbl = tax.table(cellText=cells, colLabels=["Station", "Now", "Avg"],
                        loc="upper center", cellLoc="center",
                        colWidths=[0.34, 0.33, 0.33])
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(7)
        tbl.scale(1.0, 1.02)
        for (row, col), cell in tbl.get_celld().items():
            cell.set_edgecolor("#cccccc")
            if row == 0:
                cell.set_text_props(weight="bold")
                cell.set_facecolor("#f0f0f0")
            elif col == 0:
                cell.set_facecolor(staleness_color(latencies.get(chunk[row - 1])))
                cell.set_alpha(0.55)
    tmp = PDF_OUT.with_name(PDF_OUT.name + ".tmp.pdf")
    fig.savefig(tmp, format="pdf")
    plt.close(fig)
    tmp.replace(PDF_OUT)


def _archive_dump(body, now):
    """Permanent as-received archive of every dump (decided 2026-08-10):
    each minute's groundmag_h.json gzipped into a daily uncompressed tar
    (archive/groundmag/groundmag_YYYYMMDD.tar), so future models can be
    trained on the EXACT realtime latency and data distributions —
    including GIN backfills/revisions, which per-minute snapshots capture
    and a final archive cannot. Never pruned; never allowed to break the
    live products."""
    try:
        import gzip
        import io
        import tarfile
        day_dir = HERE / "archive" / "groundmag"
        day_dir.mkdir(parents=True, exist_ok=True)
        payload = gzip.compress(body.encode(), 6)
        with tarfile.open(day_dir / f"groundmag_{now:%Y%m%d}.tar", "a") as tf:
            info = tarfile.TarInfo(f"groundmag_{now:%Y%m%dT%H%M%S}.json.gz")
            info.size = len(payload)
            import calendar
            info.mtime = calendar.timegm(now.timetuple())
            tf.addfile(info, io.BytesIO(payload))
    except Exception:
        pass


def main():
    now = datetime.utcnow()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    STATE.mkdir(parents=True, exist_ok=True)
    start_date = (now - timedelta(days=1)).strftime("%Y-%m-%d")

    series = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=WORKERS) as ex:
        futs = {ex.submit(fetch_station, c, start_date): c for c in STATIONS}
        for fut in concurrent.futures.as_completed(futs):
            series[futs[fut]] = fut.result()

    # common 1-min grid covering the last WINDOW_H hours
    t0 = (now - timedelta(hours=WINDOW_H)).replace(second=0, microsecond=0)
    n = WINDOW_H * 60
    keys = [(t0 + timedelta(minutes=i)).strftime("%Y-%m-%dT%H:%M")
            for i in range(n + 1)]
    latencies, stations_doc = {}, {}
    for code, (lat, lon) in STATIONS.items():
        s = series.get(code, {})
        h = [None if (v := s.get(k)) is None else round(v, 1) for k in keys]
        last = max(s) if s else None
        age = None
        if last is not None:
            age = max(0.0, (now - datetime.strptime(last, "%Y-%m-%dT%H:%M")
                            ).total_seconds() / 60.0)
        latencies[code] = age
        stations_doc[code] = {"lat": lat, "lon": lon, "h": h,
                              "last_valid": last,
                              "age_min": None if age is None else round(age, 1)}

    doc = {"generated_utc": now.strftime("%Y-%m-%dT%H:%M:%S"),
           "window_start": t0.strftime("%Y-%m-%dT%H:%M"),
           "cadence_min": 1, "stations": stations_doc}
    body = json.dumps(doc)
    tmp = JSON_OUT.with_suffix(".tmp")
    tmp.write_text(body)
    tmp.replace(JSON_OUT)
    _archive_dump(body, now)

    # rolling 45-day H store for the corrector's station features (quiet-day
    # baselines need weeks of history); never let it break the health product
    try:
        from station_features_rt import update_h_store
        update_h_store(series)
    except Exception:
        pass

    stamp = now.strftime("%Y-%m-%dT%H:%M:%S")
    with open(HISTORY, "a") as f:
        for code in STATIONS:
            a = latencies[code]
            f.write(json.dumps({"t": stamp, "station": code,
                                "latency_min": None if a is None
                                else round(a, 1)}) + "\n")
    rows = load_prune_history(now)
    n_runs = len({r["t"] for r in rows})
    averages = {}
    for code in STATIONS:
        vals = [r["latency_min"] for r in rows
                if r["station"] == code and r["latency_min"] is not None]
        averages[code] = sum(vals) / len(vals) if vals else None
    render_pdf(latencies, averages, now, n_runs)

    ok = sum(1 for v in latencies.values() if v is not None)
    fresh = sum(1 for v in latencies.values() if v is not None and v <= 30)
    line = (f"{stamp} OK stations={len(STATIONS)} reporting={ok} "
            f"fresh(<=30m)={fresh}")
    LOGS.mkdir(parents=True, exist_ok=True)
    with open(LOGS / f"groundmag_{now.strftime('%Y%m')}.log", "a") as fh:
        fh.write(line + "\n")
    print(line)
    return 0


if __name__ == "__main__":
    sys.exit(main())
