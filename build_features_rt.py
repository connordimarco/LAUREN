#!/usr/bin/env python3
"""Assemble the corrector's 93 driver columns for realtime valid times.

Reuses the golden-tested hindcast functions from reference_geodgp_common
verbatim (gap_clean / add_tilt / join_dst / build_driver_features), so the
realtime rows follow the training conventions exactly — including the
deliberate quirks (positional Dst shifts, ffill-after-drop; see the
reference module's header: do not "fix" these). Only the loaders differ
from the hindcast:

- SW drivers: MIDL-RT's IMF_14Re.dat (the training driver distribution),
  read via the realtime-midl symlink. The file carries ~37 h of history
  plus the ballistic transit lead, so future grid valid times up to the
  SW frontier (~+1 h) are coverable.
- Dst: fetch_dst_rt (Kyoto WDC realtime, HH:30 stamps).
- F10.7: fetch_f107_rt (latest observed value, constant over the frame —
  equivalent to the hindcast's daily ffill since f10.7 is never lagged).
"""
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from fetch_dst_rt import dst_hourly
from fetch_f107_rt import f107_latest
from reference_geodgp_common import (SW_COLS, add_tilt, build_driver_features,
                                     gap_clean, join_dst)

HERE = Path(__file__).resolve().parent
MIDL_STAGING = HERE / "realtime-midl"

# 13 h: 12 h of Dst lags for Dst_diff_1..12 + 1 h slack (lagmd needs 1 h).
HISTORY = pd.Timedelta(hours=13)


def load_midl_14re(staging=MIDL_STAGING):
    """Newest IMF_14Re file -> 6-col SW frame (OMNI names), 1-min reindexed.

    The plain-named IMF_14Re.dat vanishes for ~1 min on MIDL publish
    minutes (staging prune); fall back to the newest stamped generation
    (stamps sort chronologically). Rows MIDL omitted (missing variables)
    come back as NaN via the reindex, which is what gap_clean expects.
    """
    p = staging / "IMF_14Re.dat"
    if not p.exists():
        cands = sorted(staging.glob("IMF_14Re.*.dat"))
        if not cands:
            raise FileNotFoundError(f"no IMF_14Re data in {staging}")
        p = cands[-1]
    times, rows = [], []
    for line in p.read_text().splitlines():
        t = line.split()
        # data rows: year mn dy hr mn sc msc bx by bz ux uy uz rho T
        if len(t) != 15 or len(t[0]) != 4 or not t[0].isdigit():
            continue
        times.append(datetime(int(t[0]), int(t[1]), int(t[2]),
                              int(t[3]), int(t[4]), int(t[5])))
        v = [float(x) for x in t[7:]]
        rows.append([v[0], v[1], v[2], v[3], v[6], v[7]])   # bx by bz ux rho T
    df = pd.DataFrame(rows, index=pd.DatetimeIndex(times), columns=SW_COLS)
    df = df[~df.index.duplicated(keep="last")].sort_index()
    df = df.reindex(pd.date_range(df.index[0], df.index[-1], freq="1min"))
    df.index.name = "Datetime"
    return df


def build_rows(targets, sw=None, dst=None, f107=None):
    """93-driver feature rows for the given valid times.

    Returns a DataFrame indexed by the subset of targets that lie inside
    driver coverage; callers treat missing targets as pending (beyond the
    SW frontier, or inside a dropped data gap) and retry next tick.
    """
    targets = pd.DatetimeIndex(sorted(targets))
    sw = load_midl_14re() if sw is None else sw
    start = targets.min() - HISTORY
    sw = gap_clean(sw.loc[sw.index >= start])
    # Tilt: the tilt column is never lagged, so only the extracted rows'
    # values are ever consumed. Computing geopack tilt at the targets only
    # (0.0 elsewhere, to satisfy build_driver_features' final dropna) is
    # byte-identical to the reference add_tilt for the returned rows, at
    # ~1% of the cost (recalc is ~0.1 s per minute of frame).
    # geopack's import runs an IGRF HTTP refresh with NO timeout (harmless
    # on failure — local igrf14 file wins either way); cap it so a wedged
    # NOAA server can't hang the tick.
    import socket
    socket.setdefaulttimeout(20)
    import geopack.geopack as gp
    frame = sw.copy()
    frame["Dipole Tilt Angle"] = 0.0
    tt = sw.index.intersection(targets)
    ut = (tt - pd.Timestamp("1970-01-01")).total_seconds().values
    frame.loc[tt, "Dipole Tilt Angle"] = [gp.recalc(t) for t in ut]
    frame["f10.7_index"] = float(f107_latest() if f107 is None else f107)
    if dst is None:
        now = datetime.now(timezone.utc).replace(tzinfo=None)
        hb = int((now - start.to_pydatetime()).total_seconds() // 3600) + 3
        dst = dst_hourly(hours_back=hb)
    if isinstance(dst, pd.Series):
        dst = dst.to_frame()
    frame = join_dst(frame, dst)
    feats = build_driver_features(frame)
    return feats.loc[feats.index.intersection(targets)]


if __name__ == "__main__":
    import sys
    import time as _time
    sw = load_midl_14re()
    frontier = sw.index[-1]
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    targets = sw.index[-n:]
    t0 = _time.perf_counter()
    rows = build_rows(targets)
    dt = _time.perf_counter() - t0
    print(f"SW frontier {frontier}, built {len(rows)}/{n} rows "
          f"x {rows.shape[1]} cols in {dt:.2f}s")
    print(rows.iloc[-1].loc[["Bx, nT (GSE, GSM)", "Bz, nT (GSM)",
                             "Vx Velocity, km/s, GSE", "Dipole Tilt Angle",
                             "f10.7_index", "Dst", "Dst_diff_1"]].to_string())
