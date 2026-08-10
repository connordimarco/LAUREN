#!/usr/bin/env python3
"""Archive GeoDGP realtime deltaB grids before their ~90-min retention
window deletes them, so corrector validation has history to work with.

Reads (READ-ONLY) ajivani's realtime output and stores, per file, the dBN
variable at lat >= 30N as float32 npz in grid_archive/YYYYMMDD/. ~90 KB
per file, ~60 MB/day at the ~2-min cadence. Idempotent and cheap; safe
to cron every few minutes. Skips files younger than 60 s (may still be
mid-write on their side).

Usage: venv/bin/python archive_grids.py [--src DIR] [--dst DIR]
"""
import argparse
import time
from pathlib import Path

import numpy as np

import swmf_datafile as swmf

SRC = Path("/data/tuija/cdimarco/LAUREN/realtime-geodgp")
DST = Path(__file__).resolve().parent / "grid_archive"
LAT_MIN = 30.0
RETENTION_DAYS = 45


def archive_one(path, dst_root):
    stamp = path.name.split("_")[1]          # deltaB_YYYYMMDDHHMMSS_real4.out
    day_dir = dst_root / stamp[:8]
    out = day_dir / f"dBN_{stamp}.npz"
    if out.exists():
        return False
    r = swmf.read_file(str(path))
    names = r["name"].split()[2:]
    dbn = r["state"][names.index("dBN")]
    lat = r["coord"][1][:, 0]
    lon = r["coord"][0][0, :]
    keep = lat >= LAT_MIN
    day_dir.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".tmp.npz")
    np.savez_compressed(tmp, dbn=dbn[keep, :].astype(np.float32),
                        lat=lat[keep].astype(np.float32),
                        lon=lon.astype(np.float32), head=r["head"])
    tmp.replace(out)
    return True


def prune(dst_root, days):
    cutoff = time.strftime("%Y%m%d", time.gmtime(time.time() - days * 86400))
    for d in sorted(dst_root.glob("20*")):
        if d.is_dir() and d.name < cutoff:
            for f in d.iterdir():
                f.unlink()
            d.rmdir()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, default=SRC)
    ap.add_argument("--dst", type=Path, default=DST)
    a = ap.parse_args()
    now = time.time()
    n = 0
    for p in sorted(a.src.glob("deltaB_*_real4.out")):
        try:
            if now - p.stat().st_mtime < 60:
                continue
            n += archive_one(p, a.dst)
        except Exception as e:                    # one bad file never blocks
            print(f"skip {p.name}: {e}")
    prune(a.dst, RETENTION_DAYS)
    print(f"archived {n} new files")


if __name__ == "__main__":
    main()
