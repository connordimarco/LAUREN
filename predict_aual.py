#!/usr/bin/env python3
"""Realtime AU/AL corrector — load-and-call interface for the sword machine.

    from predict_aual import Corrector
    c = Corrector("corrector_grid_2deg_v01.joblib")
    au, al = c.predict(grid_file, features)

`grid_file`: path to a deltaB_*_real4.out file (any grid resolution >= the
model's — the model file's own column list defines which geographic nodes
are sampled, so swapping in a future model trained on finer grids requires
NO plumbing changes).
`features`: dict or pandas Series holding the non-grid features by name
(the 93 solar-wind drivers your Input pipeline already computes, plus the
station-feature columns; missing station columns may be NaN — the model
handles NaN natively. Missing DRIVER columns raise).

Dependencies: numpy, pandas, joblib, lightgbm==4.x, scikit-learn 1.3.x,
plus swmf_datafile.py (bundled).
"""
import re

import joblib
import numpy as np

import swmf_datafile as swmf

_GCOL = re.compile(r"dBNg_(-?[\d.]+)_(\d{3})$")


class Corrector:
    def __init__(self, model_path):
        b = joblib.load(model_path)
        self.models = b["models"]
        self.fcols = b["fcols"]
        self.meta = {k: v for k, v in b.items() if k not in ("models",)}
        self.grid_cols = [(i, float(m.group(1)), float(m.group(2)))
                          for i, c in enumerate(self.fcols)
                          if (m := _GCOL.match(c))]
        self.other_cols = [(i, c) for i, c in enumerate(self.fcols)
                           if not _GCOL.match(c)]

    def predict(self, grid_file, features):
        r = swmf.read_file(str(grid_file))
        names = r["name"].split()[2:]
        dbn = r["state"][names.index("dBN")]
        lon = r["coord"][0][0, :]   # first row: all longitudes
        lat = r["coord"][1][:, 0]   # first col: all latitudes
        x = np.empty(len(self.fcols), dtype=np.float32)
        x[:] = np.nan
        for i, la, lo in self.grid_cols:
            ii = int(np.argmin(np.abs(lat - la)))
            jj = int(np.argmin(np.abs(lon - lo)))
            if abs(lat[ii] - la) > 1.01 or abs(lon[jj] - lo) > 1.01:
                raise ValueError(f"grid lacks node near ({la},{lo})")
            x[i] = dbn[ii, jj]
        for i, c in self.other_cols:
            v = features.get(c) if hasattr(features, "get") else features[c]
            if v is None and not c.startswith(("ABK", "NAQ")) and "_" not in c:
                raise KeyError(f"missing driver feature {c}")
            x[i] = np.nan if v is None else v
        X = x.reshape(1, -1)
        au = float(self.models["AU"].predict(X)[0])
        al = float(self.models["AL"].predict(X)[0])
        return au, al


if __name__ == "__main__":
    import argparse
    import pandas as pd
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("grid_file")
    ap.add_argument("features_parquet",
                    help="1-row parquet/csv of driver+station features")
    a = ap.parse_args()
    f = (pd.read_parquet(a.features_parquet) if a.features_parquet.endswith("parquet")
         else pd.read_csv(a.features_parquet)).iloc[0]
    au, al = Corrector(a.model).predict(a.grid_file, f)
    print(f"AU={au:.1f} nT  AL={al:.1f} nT")
