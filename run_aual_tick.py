#!/usr/bin/env python3
"""One tick of the AU/AL corrector service (LAUREN).

scan realtime-geodgp -> archive new grids -> build features for new or
updated valid times -> predict -> maintain state/predictions.csv ->
write realtime-aual/aual.dat (columns: year mo dy hr mn sc AU AL, nT;
rolling 24 h window incl. future-valid rows up to the SW frontier).

Idempotent micro-batch: a skipped or killed tick costs nothing but
latency. Model absent => archives only and logs 'no-model'. Runs from
cron every minute via dev/aual_tick.sh (flock + timeout).
"""
import json
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

import archive_grids
from predict_aual import Corrector

ROOT = Path(__file__).resolve().parent
GRIDS = ROOT / "realtime-geodgp"
MODEL = ROOT / "models" / "corrector_grid_2deg_v01.joblib"
STORE = ROOT / "state" / "predictions.csv"
OUT = ROOT / "realtime-aual" / "aual.dat"
LOGS = ROOT / "logs"
ARCHIVE = ROOT / "archive" / "aual"

BACKFILL_H = 6          # how far back an opening tick will predict
STORE_KEEP_H = 36 * 24  # feeds the website's 35 d view (+ slack)
OUT_WINDOW_H = 24


def scan_grids():
    """{valid_time: (path, mtime)} for every grid file currently landed."""
    out = {}
    for p in GRIDS.glob("deltaB_*_real4.out"):
        try:
            t = pd.Timestamp(datetime.strptime(p.name.split("_")[1],
                                               "%Y%m%d%H%M%S"))
            out[t] = (p, p.stat().st_mtime)
        except (ValueError, OSError):
            continue
    return out


LOG_KEEP_MONTHS = 6


def archive_new(grids, summary):
    n, bad = 0, 0
    for path, _ in grids.values():
        try:
            n += archive_one(path)
        except Exception:
            bad += 1                       # one bad file never blocks
    archive_grids.prune(ROOT / "grid_archive", archive_grids.RETENTION_DAYS)
    prune_logs()
    summary.append(f"archived={n}" + (f" bad={bad}" if bad else ""))


def prune_logs():
    """Monthly log files (tick_YYYYMM.log, push_YYYYMM.log) never rotate on
    their own — drop them after LOG_KEEP_MONTHS so logs/ stays bounded."""
    now = datetime.now(timezone.utc)
    cutoff = (now.year * 12 + now.month - 1 - LOG_KEEP_MONTHS)
    for f in LOGS.glob("*_2?????.log"):
        try:
            ym = int(f.stem.rsplit("_", 1)[1])
            if (ym // 100) * 12 + (ym % 100) - 1 < cutoff:
                f.unlink()
        except (ValueError, OSError):
            continue


def archive_one(path):
    return archive_grids.archive_one(path, ROOT / "grid_archive")


def archive_events(events, now):
    """Permanent append-only ledger of every prediction EVENT (decided
    2026-08-10): one row per (predicted_at, valid_time) — re-predictions
    of the same valid time appear again, preserving the revision/lead
    structure for future training and lead-dependent skill studies.
    ~100 MB/year; never pruned."""
    if not events:
        return
    ARCHIVE.mkdir(parents=True, exist_ok=True)
    path = ARCHIVE / f"aual_events_{now.strftime('%Y%m')}.csv"
    new = not path.exists()
    with open(path, "a") as fh:
        if new:
            fh.write("predicted_at,valid_time,AU,AL\n")
        ts = now.strftime("%Y-%m-%dT%H:%M:%S")
        for t, au, al in events:
            fh.write(f"{ts},{t.strftime('%Y-%m-%dT%H:%M:%S')},"
                     f"{au:.1f},{al:.1f}\n")


def load_store():
    if STORE.exists():
        return pd.read_csv(STORE, index_col=0, parse_dates=True)
    df = pd.DataFrame(columns=["AU", "AL", "src_mtime", "predicted_at"])
    df.index.name = "valid_time"
    return df


def save_store(df):
    STORE.parent.mkdir(parents=True, exist_ok=True)
    tmp = STORE.with_suffix(".tmp")
    df.sort_index().to_csv(tmp)
    tmp.replace(STORE)


def write_aual(df, now):
    keep = df.loc[df.index >= now - pd.Timedelta(hours=OUT_WINDOW_H)]
    keep = keep.sort_index()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    tmp = OUT.with_suffix(".tmp")
    with open(tmp, "w") as fh:
        for t, r in keep.iterrows():
            fh.write(f"{t.year:4d} {t.month:2d} {t.day:2d} {t.hour:2d} "
                     f"{t.minute:2d} {t.second:2d} "
                     f"{r.AU:8.1f} {r.AL:8.1f}\n")
    tmp.replace(OUT)
    return len(keep)


# Website JSON views (SWORD-Web products.js laurenDraw contract):
# {generated_utc, t[], au[], al[]} — rows with t > generated_utc are drawn
# as the forecast. Long windows are resampled to keep payloads small.
VIEWS = (("last_24h.json", pd.Timedelta(hours=24), None),
         ("last_7d.json", pd.Timedelta(days=7), "5min"),
         ("last_35d.json", pd.Timedelta(days=35), "15min"))


def write_views(df, now):
    stamp = now.strftime("%Y-%m-%dT%H:%M:%SZ")
    for name, span, rule in VIEWS:
        keep = df.loc[df.index >= now - span, ["AU", "AL"]].sort_index()
        if rule:
            keep = keep.resample(rule).mean().dropna()
        doc = {"generated_utc": stamp,
               "t": [t.strftime("%Y-%m-%dT%H:%M:%SZ") for t in keep.index],
               "au": [round(v, 1) for v in keep["AU"]],
               "al": [round(v, 1) for v in keep["AL"]]}
        tmp = OUT.parent / (name + ".tmp")
        tmp.write_text(json.dumps(doc))
        tmp.replace(OUT.parent / name)


def run(now, summary):
    grids = scan_grids()
    summary.append(f"grids={len(grids)}")
    archive_new(grids, summary)
    try:
        # refresh the realtime-f107/ product every tick (cheap: 3 h cache
        # TTL, written only on change) — GITM-RT's F10.7 feed must stay
        # fresh even when there is nothing to predict
        from fetch_f107_rt import f107_latest
        f107_latest()
    except Exception:
        summary.append("f107-fail")
    if not MODEL.exists():
        summary.append("no-model")
        return
    store = load_store()
    todo = [t for t, (p, m) in grids.items()
            if t >= now - pd.Timedelta(hours=BACKFILL_H)
            and (t not in store.index or m > store.loc[t, "src_mtime"] + 0.5)]
    predicted = 0
    events = []
    if todo:
        from build_features_rt import build_rows
        feats = build_rows(todo)
        c = Corrector(MODEL)
        # NaN for every model column we don't supply, then overlay live
        # station features from the groundmag store (per-station NaN =
        # station dark — the model tolerates that, as in training) and the
        # exact time encodings. Station build failure is non-fatal: the
        # tick degrades to the NaN-station configuration.
        extra = [col for _, col in c.other_cols if col not in feats.columns]
        sfeats = None
        if len(feats):
            try:
                from fetch_groundmag_rt import STATIONS as SF_COORDS
                from station_features_rt import build as sf_build
                from station_features_rt import model_stations
                sfeats = sf_build(
                    feats.index,
                    model_stations([col for _, col in c.other_cols]),
                    SF_COORDS)
            except Exception:
                summary.append("stations-fail")
        for t in feats.index:
            row = feats.loc[t].to_dict()
            row.update(dict.fromkeys(extra, np.nan))
            if sfeats is not None:
                row.update(sfeats.loc[t].to_dict())
            # time encodings, verbatim from the training matrix assembler
            # (assemble_year.py): minute-of-day/1440, dayofyear/365.25
            mod = t.hour * 60 + t.minute
            row["ut_sin"] = np.sin(2 * np.pi * mod / 1440)
            row["ut_cos"] = np.cos(2 * np.pi * mod / 1440)
            row["doy_sin"] = np.sin(2 * np.pi * t.dayofyear / 365.25)
            row["doy_cos"] = np.cos(2 * np.pi * t.dayofyear / 365.25)
            au, al = c.predict(grids[t][0], row)
            store.loc[t, ["AU", "AL", "src_mtime", "predicted_at"]] = \
                [au, al, grids[t][1], now.strftime("%Y-%m-%dT%H:%M:%S")]
            events.append((t, au, al))
            predicted += 1
    archive_events(events, now)
    store = store.loc[store.index >= now - pd.Timedelta(hours=STORE_KEEP_H)]
    save_store(store)
    n_out = write_aual(store, now)
    write_views(store, now)
    summary.append(f"predicted={predicted} pending={len(todo) - predicted} "
                   f"rows={n_out}")


def main():
    t0 = time.monotonic()
    now = pd.Timestamp(datetime.now(timezone.utc).replace(tzinfo=None))
    summary = []
    try:
        run(now, summary)
        ok = True
    except Exception:
        ok = False
    ms = int((time.monotonic() - t0) * 1000)
    line = (f"{now.strftime('%Y-%m-%dT%H:%M:%S')} {'OK' if ok else 'FAIL'} "
            + " ".join(summary) + f" ms={ms}\n")
    LOGS.mkdir(parents=True, exist_ok=True)
    with open(LOGS / f"tick_{now.strftime('%Y%m')}.log", "a") as fh:
        fh.write(line)
        if not ok:
            fh.write(traceback.format_exc())
    sys.stdout.write(line)
    if not ok:
        sys.stdout.write(traceback.format_exc())
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
