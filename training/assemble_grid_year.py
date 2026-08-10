#!/usr/bin/env python3
"""Build one year's grid-feature training matrix.

Reads the sword-config daily tars (out/sword_hist/archive2deg/), extracts the
dBN field (31x180 = 5,580 float32 columns named dBNg_{lat}_{lon}), and joins
it onto the year's EXISTING certified matrix (data/matrix/matrix_Y.parquet)
with the old 288 ring columns dropped — so truth, drivers, stations, and time
encodings are reused unchanged from the validated pipeline.

Usage: python assemble_grid_year.py --year 2015
Output: data/matrix_grid/matrix_{Y}.parquet
"""
import argparse
import tarfile
from pathlib import Path

import numpy as np
import pandas as pd

import geodgp_common as gc
import swmf_datafile as swmf

ARCH = gc.HERE / "out" / "sword_hist" / "archive2deg"
OUT = gc.HERE / "data" / "matrix_grid"

LATS = np.arange(30, 91, 2)
LONS = np.arange(0, 360, 2)
GCOLS = [f"dBNg_{la}_{lo:03d}" for la in LATS for lo in LONS]


def read_member(tf, member):
    import io, tempfile, os
    data = tf.extractfile(member).read()
    # swmf reader wants a path; use an in-memory-ish temp
    tmp = Path("/tmp") / f"agy_{os.getpid()}.out"
    tmp.write_bytes(data)
    r = swmf.read_file(str(tmp))
    names = r["name"].split()[2:]
    z = r["state"][names.index("dBN")][:, :-1]  # drop wrap column
    when = pd.to_datetime(r["head"].split()[-1], format="%Y%m%d%H%M%S")
    return when, z.astype(np.float32).ravel()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--year", type=int, required=True)
    y = ap.parse_args().year
    OUT.mkdir(parents=True, exist_ok=True)
    dest = OUT / f"matrix_{y}.parquet"
    if dest.exists():
        print(f"{dest.name} exists, skip")
        return

    tars = sorted(p for p in ARCH.glob(f"deltaB2deg_{y}*.tar*")
                  if not p.name.endswith(".done"))
    if not tars:
        raise SystemExit(f"no tars for {y}")
    missing_done = [t for t in tars
                    if not (ARCH / (t.name.split(".tar")[0] + ".tar.done")).exists()]
    if missing_done:
        raise SystemExit(f"{len(missing_done)} days lack .done markers for {y} "
                         f"(first: {missing_done[0].name}) — year incomplete")

    times, rows = [], []
    for t in tars:
        with tarfile.open(t) as tf:
            for m in tf.getmembers():
                when, row = read_member(tf, m)
                times.append(when)
                rows.append(row)
    grid = pd.DataFrame(np.vstack(rows), index=pd.DatetimeIndex(times, name="Datetime"),
                        columns=GCOLS, dtype=np.float32).sort_index()
    grid = grid[~grid.index.duplicated(keep="last")]
    print(f"{y}: grid {grid.shape}", flush=True)

    base = pd.read_parquet(gc.DATA / "matrix" / f"matrix_{y}.parquet")
    ring_cols = [c for c in base.columns if c.startswith("dBN_")]
    base = base.drop(columns=ring_cols)
    mat = base.join(grid, how="inner")
    n_lost = len(base) - len(mat)
    print(f"{y}: base {base.shape} -> joined {mat.shape} (lost {n_lost} rows)",
          flush=True)
    mat.to_parquet(dest)
    print(f"WROTE {dest.name}", flush=True)


if __name__ == "__main__":
    main()
