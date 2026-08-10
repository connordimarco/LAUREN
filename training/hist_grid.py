#!/usr/bin/env python3
"""Historical GeoDGP grid runs in the sword deployment's exact configuration.

Two modes:
  --step0            Reproduce the sample file from the deployment copy's own
                     ACE/Dst inputs (their Input-prep code, full globe) and
                     report max/rms deviation vs sample_output/*.out.
  --year Y --month M Production: drive their models with our golden-tested
                     driver93_midl features; domain 30..90N x 0..359 at 1 deg;
                     per-minute real4 files; inline 288-pt SM ring extraction;
                     2 deg decimated archive appended to a daily tar.

Their code is used directly: GeoDGP/SuperMAGDataset from nn_utils, their
scalers, their _10T nowcast weights, their SM transform (astropy ITRS ->
sunpy SolarMagnetic igrf13), their real4 writer dict.
"""
import argparse
import sys
import tarfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

SWORD = Path("/home7/cdimarco/geodgp_web_from_sword")
sys.path.insert(0, str(SWORD / "scripts"))

import astropy.coordinates
import astropy.units as u
from astropy.utils import iers
from sunpy.coordinates import frames

iers.conf.auto_download = False  # compute nodes have no outbound network

import swmf_datafile as swmf
from nn_utils import GeoDGP, SuperMAGDataset_fromtensor

HERE = Path(__file__).resolve().parent
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# gpytorch DGP inference collapses under thread oversubscription on
# many-core nodes — cap like geodgp_common.load_model does.
import os as _os
torch.set_num_threads(int(_os.environ.get("SLURM_CPUS_PER_TASK", 32)))
QOIS = ["dBH", "dBN", "dBE"]
WINDOW_MAX = 10
X_SHAPE = 96
BATCH = 4096

BASE8 = ['Bx, nT (GSE, GSM)', 'By, nT (GSM)', 'Bz, nT (GSM)',
         'Vx Velocity, km/s, GSE', 'Proton Density, n/cc', 'Temperature, K',
         'Dipole Tilt Angle', 'f10.7_index']
LAG_COVS = BASE8[:6]

SM_LATS = (63.5, 66.0, 68.5, 71.0)
SM_LONS = np.arange(0, 360, 5)
RING_COLS = [f"dBN_{lat}_{lon:03d}" for lat in SM_LATS for lon in SM_LONS]


def their_column_order():
    """The 93-column Input order their script produces."""
    cols = BASE8 + ["Dst"]
    for lag in [i * 5 for i in range(12)]:
        for cov in LAG_COVS:
            cols.append(f"{cov}_lagmd_{lag}")
    cols += [f"Dst_diff_{j}" for j in range(1, 13)]
    return cols


def load_models():
    import pickle
    scalers = {}
    for q in QOIS:
        with open(SWORD / "data" / "scaler" / f"scalerY_{q}_{WINDOW_MAX}.pkl", "rb") as f:
            scalers[q] = pickle.load(f)
    with open(SWORD / "data" / "scaler" / f"scalerX_{WINDOW_MAX}.pkl", "rb") as f:
        scaler_x = pickle.load(f)
    models = {}
    for q in QOIS:
        m = GeoDGP(X_SHAPE, DEVICE=DEVICE)
        sd = torch.load(SWORD / "data" / "model" / f"model_state_{q}_10T.pth",
                        map_location=DEVICE)
        m.load_state_dict(sd)
        m.to(DEVICE).eval()
        models[q] = m
    return models, scalers, scaler_x


def make_domain(lat_min, lat_max, step=1.0):
    lat = np.arange(lat_min, lat_max + 0.5, step)
    lon = np.arange(0.0, 360.0, step)
    geolon, geolat = np.meshgrid(lon, lat)
    loc = astropy.coordinates.EarthLocation(lat=geolat * u.deg, lon=geolon * u.deg)
    return geolon, geolat, loc


def sm_of(loc, when):
    itrs = astropy.coordinates.ITRS(x=loc.x, y=loc.y, z=loc.z, obstime=when)
    sm = itrs.transform_to(frames.SolarMagnetic(magnetic_model="igrf13", obstime=when))
    return sm.lat.value, sm.lon.value


def eval_chunk(models, scalers, scaler_x, blocks):
    """blocks: list of (base93, sm_lat, sm_lon), one per timestamp. All
    timestamps are stacked into one matrix so the fixed per-forward cost
    amortizes (the ring-tile pipeline's trick — the GP is pointwise, so
    row-stacking is exact). Returns one {QoI: field} dict per timestamp."""
    Xs, shapes = [], []
    for base93, sm_lat, sm_lon in blocks:
        smf = np.column_stack([sm_lat.ravel(),
                               np.cos(np.deg2rad(sm_lon.ravel())),
                               np.sin(np.deg2rad(sm_lon.ravel()))])
        base = np.repeat(base93.reshape(1, -1), smf.shape[0], axis=0)
        Xs.append(np.hstack([base, smf]))
        shapes.append(sm_lat.shape)
    npts = [x.shape[0] for x in Xs]
    X = torch.tensor(scaler_x.transform(np.vstack(Xs)), dtype=torch.float32)
    outs = [dict() for _ in blocks]
    with torch.no_grad():
        for q in QOIS:
            mus = []
            for i in range(0, len(X), BATCH):
                xb = X[i:i + BATCH].to(DEVICE)
                preds = models[q].likelihood(models[q](xb))
                mus.append(preds.mean.cpu())
            mean = torch.cat(mus, dim=-1).mean(0).numpy()
            pos = 0
            for k, (n, shp) in enumerate(zip(npts, shapes)):
                outs[k][q] = scalers[q].inverse_transform(
                    mean[pos:pos + n].reshape(-1, 1)).reshape(shp)
                pos += n
    return outs


def write_real4(path, when, geolon, geolat, z):
    zaug = {q: np.hstack([z[q], z[q][:, :1]]) for q in QOIS}
    glon = np.hstack([geolon, np.full((geolon.shape[0], 1), 360.0)])
    glat = np.hstack([geolat, geolat[:, :1]])
    swmf.write_file({
        "head": f"DeltaB Prediction Data {when.strftime('%Y%m%d%H%M%S')}",
        "step": 0, "time": when.year, "ndim": 2, "npar": 0, "nvar": len(QOIS),
        "name": "GeoLon GeoLat " + " ".join(QOIS),
        "coord": np.stack([glon, glat]),
        "state": [zaug[q] for q in QOIS]}, str(path), "real4")


def ring_inline(sm_lat, sm_lon, dbn):
    from scipy.interpolate import griddata
    sl, sn = sm_lat.ravel(), sm_lon.ravel() % 360.0
    v = dbn.ravel()
    keep = sl > 40
    pts = np.column_stack([np.concatenate([sn[keep] - 360, sn[keep], sn[keep] + 360]),
                           np.tile(sl[keep], 3)])
    tlat = np.repeat(SM_LATS, len(SM_LONS))
    tlon = np.tile(SM_LONS, len(SM_LATS)).astype(float)
    return griddata(pts, np.tile(v[keep], 3),
                    np.column_stack([tlon, tlat]), method="linear")


def their_input_prep(ace_input, dst):
    """Verbatim port of their Input construction (realt_deltaB_export lines
    148-209): tilt via geopack, join Dst, 72 lag-medians, 12 Dst diffs."""
    import geopack.geopack as gp
    ut = (ace_input.index - pd.Timestamp("1970-01-01")).total_seconds()
    ace_input = ace_input.copy()
    ace_input["Dipole Tilt Angle"] = [gp.recalc(t) for t in ut.values]
    ace_input = ace_input[BASE8]
    dst = dst.resample("1min").ffill()
    Input = ace_input.join(dst, how="outer").ffill()
    lagged = []
    for lag in [i * 5 for i in range(12)]:
        for cov in LAG_COVS:
            lagged.append(Input[cov].shift(lag).rolling("5min", min_periods=3)
                          .median().rename(f"{cov}_lagmd_{lag}"))
    lagged_dst = [Input["Dst"].shift(l).rename(f"Dst_lag_{l}")
                  for l in [i * 60 for i in range(1, 13)]]
    Input = pd.concat([Input] + lagged_dst, axis=1)
    Input = pd.concat([Input] + lagged, axis=1)
    dmin = [i * 60 for i in range(1, 13)]
    diffs = [(Input["Dst"] - Input["Dst_lag_60"]).rename("Dst_diff_1")]
    for j in range(1, len(dmin)):
        diffs.append((Input[f"Dst_lag_{dmin[j-1]}"] -
                      Input[f"Dst_lag_{dmin[j]}"]).rename(f"Dst_diff_{j+1}"))
    Input = pd.concat([Input] + diffs, axis=1)
    Input = Input.drop(columns=[f"Dst_lag_{l}" for l in dmin])
    return Input.dropna()


def step0():
    sample = sorted((SWORD / "sample_output").glob("deltaB_*_real4.out"))[-1]
    ref = swmf.read_file(str(sample))
    when = pd.to_datetime(ref["head"].split()[-1], format="%Y%m%d%H%M%S")
    print(f"step0 target: {sample.name} valid {when}", flush=True)

    ace = pd.read_parquet(SWORD / "input_data" / "ace_input" / "input.parquet.gzip")
    dst = pd.read_parquet(SWORD / "input_data" / "dst" / "dst.parquet.gzip")
    Input = their_input_prep(ace, dst)
    if when not in Input.index:
        raise SystemExit(f"{when} not in prepared Input range "
                         f"{Input.index.min()}..{Input.index.max()}")
    row = Input[their_column_order()].loc[when].to_numpy()

    models, scalers, scaler_x = load_models()
    geolon, geolat, loc = make_domain(-90, 90)
    sm_lat, sm_lon = sm_of(loc, when)
    z = eval_grid(models, scalers, scaler_x, row, sm_lat, sm_lon)

    names = ref["name"].split()[2:]
    for q in QOIS:
        mine = z[q]
        theirs = ref["state"][names.index(q)][:, :-1]  # drop wrap col
        d = mine - theirs
        print(f"{q}: max|d|={np.abs(d).max():.3f} nT  rms={np.sqrt((d**2).mean()):.3f} nT"
              f"  (field rms {np.sqrt((theirs**2).mean()):.1f} nT)", flush=True)
    out = HERE / "out" / "step0_repro.out"
    write_real4(out, when, geolon, geolat, z)
    print(f"wrote {out}", flush=True)


def production(year, month, outroot, day=None, limit=None):
    feats = pd.read_parquet(HERE / "data" / "features" /
                            f"driver93_midl_{year}.parquet")
    cols = their_column_order()
    missing = [c for c in cols if c not in feats.columns]
    if missing:
        raise SystemExit(f"driver93 columns missing vs their order: {missing[:5]}")
    feats = feats[cols]
    sel = feats.index[(feats.index.year == year) & (feats.index.month == month)]
    if day is not None:
        sel = sel[sel.day == day]
    if limit is not None:
        sel = sel[:limit]
    print(f"{year}-{month:02d}{f'-{day:02d}' if day else ''}: "
          f"{len(sel)} timestamps", flush=True)

    models, scalers, scaler_x = load_models()
    geolon, geolat, loc = make_domain(30, 90, step=2.0)  # native 2-deg production
    print("models loaded, domain "
          f"{geolat.shape[0]}x{geolon.shape[1]} nodes", flush=True)

    day_dir = Path(outroot) / "tmp_files"
    day_dir.mkdir(parents=True, exist_ok=True)
    arch_dir = Path(outroot) / "archive2deg"
    arch_dir.mkdir(parents=True, exist_ok=True)
    cur_day, tar, tmp_files = None, None, []

    def close_day():
        nonlocal tar, tmp_files
        if tar is not None:
            tar.close()
        for f in tmp_files:
            f.unlink()
        tmp_files = []

    done_days = set()
    for p in arch_dir.glob("deltaB2deg_*.tar.done"):
        done_days.add(p.stem.split("_")[1].split(".")[0])

    import time as _time
    t0 = _time.time()
    ndone = 0
    todo = [w for w in sel if w.strftime("%Y%m%d") not in done_days]
    CHUNK = 16
    for c0 in range(0, len(todo), CHUNK):
        chunk = todo[c0:c0 + CHUNK]
        blocks = [(feats.loc[w].to_numpy(),) + sm_of(loc, w) for w in chunk]
        zs = eval_chunk(models, scalers, scaler_x, blocks)
        for when, z in zip(chunk, zs):
            dstr = when.strftime("%Y%m%d")
            if dstr != cur_day:
                if cur_day is not None:
                    (arch_dir / f"deltaB2deg_{cur_day}.tar.done").touch()
                close_day()
                cur_day = dstr
                # restart day fresh if a partial tar exists from a killed job
                part = arch_dir / f"deltaB2deg_{dstr}.tar"
                if part.exists():
                    part.unlink()
                tar = tarfile.open(part, "a")
            # field is computed natively at 2 deg; archive as-is
            fname = f"deltaB_{when.strftime('%Y%m%d%H%M%S')}_real4.out"
            tmp = day_dir / fname
            write_real4(tmp, when, geolon, geolat, z)
            tar.add(tmp, arcname=fname)
            tmp_files.append(tmp)
            ndone += 1
        if (c0 // CHUNK) % 4 == 0 and ndone:
            print(f"{ndone}/{len(todo)} ts, {(_time.time()-t0)/ndone:.2f} s/ts",
                  flush=True)
    if cur_day is not None and limit is None:
        # a --limit run may end mid-day; never mark partial days done
        (arch_dir / f"deltaB2deg_{cur_day}.tar.done").touch()
    close_day()
    print(f"DONE {year}-{month:02d}: {ndone} evaluated "
          f"({len(sel) - len(todo)} skipped as done)", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--step0", action="store_true")
    ap.add_argument("--year", type=int)
    ap.add_argument("--month", type=int)
    ap.add_argument("--day", type=int, default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--outroot", default=str(HERE / "out" / "sword_hist"))
    a = ap.parse_args()
    if a.step0:
        step0()
    else:
        production(a.year, a.month, a.outroot, a.day, a.limit)
