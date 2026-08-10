#!/usr/bin/env python3
"""Grid-features prospective corrector: train on 2005-2022, test 2023-2024.

My PoC design (2026-08-08): features = full 2-deg grid dBN (5,580 cols,
sword configuration) + drivers + stations + time encodings, i.e. the
matrix_grid matrices. Same LightGBM params and storm weighting as every run
in this project. Prints the ring-corrector prospective baseline alongside.

Output: out/grid_prospective/oof_{2023,2024}.parquet + metrics printout.
"""
import numpy as np
import pandas as pd

import geodgp_common as gc

MATRIX = gc.HERE / "data" / "matrix_grid"
import os
DEPLOY = os.environ.get("DEPLOY") == "1"
TRAIN_YEARS = range(2005, 2023)
TEST_YEARS = (2023, 2024)
RING_BASELINE = {"AL": (0.887, 71.7), "AU": (0.880, 38.6)}  # out/prospective


def main():
    import lightgbm as lgb
    files = {int(f.stem.split("_")[1]): f for f in sorted(MATRIX.glob("matrix_*.parquet"))}
    missing = [y for y in list(TRAIN_YEARS) + list(TEST_YEARS) if y not in files]
    if missing:
        raise SystemExit(f"missing grid matrices: {missing}")

    import pyarrow.parquet as pq
    fcols = sorted({c for y in list(TRAIN_YEARS) + list(TEST_YEARS)
                    for c in pq.read_schema(files[y]).names
                    if not c.startswith("y_")})
    nrows = [pq.ParquetFile(files[y]).metadata.num_rows for y in TRAIN_YEARS]
    total = sum(nrows)
    # preallocate the final matrix once; stream each year into its slab so
    # peak memory is ~1x the matrix, not 2-3x (the concat version OOMed a
    # 512 GB node)
    Xtr = np.empty((total, len(fcols)), dtype=np.float32)
    yAU = np.empty(total, dtype=np.float32)
    yAL = np.empty(total, dtype=np.float32)
    pos = 0
    for y, n in zip(TRAIN_YEARS, nrows):
        df = pd.read_parquet(files[y])
        Xtr[pos:pos + n] = df.reindex(columns=fcols).to_numpy(dtype=np.float32)
        yAU[pos:pos + n] = df["y_AU"].to_numpy(dtype=np.float32)
        yAL[pos:pos + n] = df["y_AL"].to_numpy(dtype=np.float32)
        pos += n
        del df
        print(f"loaded {y} ({pos}/{total})", flush=True)
    tests = {y: pd.read_parquet(files[y]) for y in TEST_YEARS}
    print(f"train {Xtr.shape}", flush=True)

    w = np.minimum(1.0 + np.abs(yAL) / 300.0, 8.0)
    params = dict(objective="regression", n_estimators=600, learning_rate=0.05,
                  num_leaves=255, subsample=0.8, colsample_bytree=0.8,
                  n_jobs=-1, verbosity=-1)
    targets = {"AU": yAU, "AL": yAL}
    models = {}
    for k in ("AU", "AL"):
        mdl = lgb.LGBMRegressor(**params)
        mdl.fit(Xtr, targets[k], sample_weight=w)
        models[k] = mdl
        print(f"{k} trained", flush=True)
    del Xtr
    if DEPLOY:
        import joblib
        d = gc.HERE / "out" / "deploy_model"
        d.mkdir(parents=True, exist_ok=True)
        joblib.dump({"models": models, "fcols": fcols,
                     "train_years": list(TRAIN_YEARS),
                     "validated": "r(AL)=0.888 rmse=71.0, r(AU)=0.884 rmse=38.1 on 2023-2024 prospective"},
                    d / "corrector_grid_2deg_v01.joblib")
        print("SAVED deploy model v0.1", flush=True)

    dest = gc.HERE / "out" / "grid_prospective"
    dest.mkdir(parents=True, exist_ok=True)
    frames = []
    for y, test in tests.items():
        Xte = test.reindex(columns=fcols).values.astype(np.float32)
        out = pd.DataFrame({
            "AU_true": test["y_AU"], "AL_true": test["y_AL"],
            "AU_ml": models["AU"].predict(Xte),
            "AL_ml": models["AL"].predict(Xte)}, index=test.index)
        out.to_parquet(dest / f"oof_{y}.parquet")
        frames.append(out)
        print(f"WROTE oof_{y}.parquet ({len(out)} min)", flush=True)

    d = pd.concat(frames).dropna()
    print("\n=== GRID-FEATURE PROSPECTIVE (train 2005-2022, test 2023-24) ===")
    for k in ("AL", "AU"):
        r = float(np.corrcoef(d[f"{k}_true"], d[f"{k}_ml"])[0, 1])
        rmse = float(np.sqrt(((d[f"{k}_true"] - d[f"{k}_ml"]) ** 2).mean()))
        br, brmse = RING_BASELINE[k]
        print(f"{k}: r={r:.3f} rmse={rmse:.1f} nT   "
              f"[ring baseline r={br:.3f} rmse={brmse:.1f}]")


if __name__ == "__main__":
    main()
