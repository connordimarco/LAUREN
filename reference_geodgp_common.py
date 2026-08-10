#!/usr/bin/env python3
"""Shared pipeline pieces for the GeoDGP -> AU/AL 2024 hindcast.

Replicates GeoDGP's training feature assembly (GeoDGP/scripts/OMNI_download.py
Steps 3-6) exactly, for either MIDL or OMNI 1-min drivers, and evaluates the
dBN models at the 12 official Kyoto AE observatories.

Conventions replicated from the GeoDGP repo (do not "fix" these):
- Fill values 999.99 / 9999.99 / 99999.9 / 9999999 -> NaN.
- Per column, rows inside NaN runs of >= 15 consecutive minutes are DROPPED
  (iteratively, column order matters), then remaining gaps are ffilled.
- Dipole tilt from geopack.recalc(ut), radians, per minute.
- F10.7 from OMNI2 hourly file column index 50, 999.9 -> NaN, ffill,
  resampled to 1-min by ffill, left-joined.
- Kyoto Dst hourly values stamped at HH:30 (pyspedas convention), resampled
  to 1-min by ffill, left-joined; then global ffill and dropna.
- 72 lagged 5-min medians, lag-major over the 6 SW covariates;
  Dst_diff_1..12 from hourly-lagged Dst; final column order = 9 base + 72
  lagmd + 12 Dst_diff = 93 driver columns (matches OMNI_paper_5m_feature.pkl).
- Station features appended per row: SM_lat, CosLon, SinLon from sunpy
  Geomagnetic(AACGM mlon/mlat) -> SolarMagnetic(igrf13) at each minute.
"""
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
sys.path.insert(0, str(HERE / "GeoDGP" / "src"))

SW_COLS = ['Bx, nT (GSE, GSM)', 'By, nT (GSM)', 'Bz, nT (GSM)',
           'Vx Velocity, km/s, GSE', 'Proton Density, n/cc', 'Temperature, K']

MIDL_RENAME = {'Bx': SW_COLS[0], 'By': SW_COLS[1], 'Bz': SW_COLS[2],
               'Ux': SW_COLS[3], 'rho': SW_COLS[4], 'T': SW_COLS[5]}

# The 12 official Kyoto AE observatories (WDC Table 1). station_info.csv
# (Deep Blue) carries their AACGM coords under these IAGA codes.
AE_STATIONS = ['ABK', 'DIK', 'CCS', 'TIK', 'CWE', 'BRW',
               'CMO', 'YKC', 'FCC', 'PBQ', 'NAQ', 'LRV']

# OMNI 1-min .asc column names after dropping the first 4 time columns
# (verbatim from OMNI_download.py Step 2).
OMNI_MIN_COLS = ['ID for IMF spacecraft', 'ID for SW Plasma spacecraft',
                 '# of points in IMF averages', '# of points in Plasma averages',
                 'Percent interp', 'Timeshift, sec', 'RMS, Timeshift',
                 'RMS, Phase front normal', 'Time btwn observations, sec',
                 'Field magnitude average', 'Bx, nT (GSE, GSM)', 'By, nT (GSE)',
                 'Bz, nT (GSE)', 'By, nT (GSM)', 'Bz, nT (GSM)', 'RMS SD B scalar',
                 'RMS SD field vector', 'Flow speed, km/s', 'Vx Velocity, km/s, GSE',
                 'Vy Velocity, km/s, GSE', 'Vz Velocity, km/s, GSE',
                 'Proton Density, n/cc', 'Temperature, K', 'Flow pressure, nPa',
                 'Electric field, mV/m', 'Plasma beta', 'Alfven mach number',
                 'X(s/c), GSE, Re', 'Y(s/c), GSE, Re', 'Z(s/c), GSE, Re',
                 'BSN location, Xgse, Re', 'BSN location, Ygse, Re',
                 'BSN location, Zgse, Re', 'AE-index, nT', 'AL-index, nT',
                 'AU-index, nT', 'SYM/D index, nT', 'SYM/H index, nT',
                 'ASY/D index, nT', 'ASY/H index, nT', 'Na/Np Ratio',
                 'Magnetosonic mach number']


def gap_clean(df):
    """GeoDGP Step 3 gap policy: drop rows in >=15-min NaN runs, then ffill."""
    for col in df.columns:
        mask = df[col].notna()
        a = mask.ne(mask.shift()).cumsum()
        df = df[(a.groupby(a).transform('size') < 15) | mask]
    return df.ffill()


def load_midl_sw():
    """MIDL 2024 -> 6-column SW frame with OMNI naming, gap-cleaned."""
    parts = [pd.read_parquet(p) for p in sorted((DATA / "midl").glob("midl_2024*.parquet"))]
    df = pd.concat(parts)
    df = df[~df.index.duplicated(keep='first')].sort_index()
    df = df.rename(columns=MIDL_RENAME)[SW_COLS]
    df.index.name = 'Datetime'
    return gap_clean(df)


def load_omni_sw(year=2024):
    """OMNI 1-min monthlies -> same 6-column frame (control driver)."""
    frames_ = []
    for p in sorted((DATA / "omni").glob(f"omni_min{year}*.asc")):
        raw = np.loadtxt(p)
        idx = pd.to_datetime(dict(year=raw[:, 0].astype(int), month=1, day=1)) \
            + pd.to_timedelta((raw[:, 1] - 1).astype(int), unit='D') \
            + pd.to_timedelta(raw[:, 2].astype(int), unit='h') \
            + pd.to_timedelta(raw[:, 3].astype(int), unit='m')
        d = pd.DataFrame(raw[:, 4:], index=idx, columns=OMNI_MIN_COLS)
        frames_.append(d[SW_COLS])
    df = pd.concat(frames_).sort_index()
    df.index.name = 'Datetime'
    for fill in (999.99, 9999.99, 99999.9, 9999999.):
        df = df.replace(fill, np.nan)
    return gap_clean(df)


def add_tilt(df):
    """GeoDGP Step 3: dipole tilt via geopack, per minute (radians)."""
    import geopack.geopack as gp
    ut = (df.index - pd.Timestamp("1970-01-01")).total_seconds().values
    df = df.copy()
    df["Dipole Tilt Angle"] = [gp.recalc(t) for t in ut]
    return df


def add_f107(df, omni2_paths):
    """GeoDGP Step 4: F10.7 from OMNI2 hourly files, col index 50."""
    parts = []
    for p in omni2_paths:
        raw = np.loadtxt(p)
        idx = pd.to_datetime(dict(year=raw[:, 0].astype(int), month=1, day=1)) \
            + pd.to_timedelta((raw[:, 1] - 1).astype(int), unit='D') \
            + pd.to_timedelta(raw[:, 2].astype(int), unit='h')
        parts.append(pd.DataFrame({"f10.7_index": raw[:, 50]}, index=idx))
    f = pd.concat(parts).sort_index()
    f = f.replace(999.9, np.nan).ffill()
    f = f.resample('1min').ffill()
    return df.join(f)


def load_dst_wdc(paths):
    """Kyoto WDC .for.request Dst files -> hourly series stamped HH:30.

    WDC format: 'DST2401*01...' records, 24 hourly values in fixed-width
    4-char fields at positions 20..116 (base value at 16..20 applies +
    hundreds via the base field: value = base*100 + field? No — Kyoto WDC
    hourly fields are absolute except fill 9999). We parse the standard
    layout: cols 20:116 as 24 x 4-char ints, fill 9999 -> NaN.
    """
    times, vals = [], []
    for p in paths:
        for line in Path(p).read_text().splitlines():
            if not line.startswith("DST"):
                continue
            yy = int(line[3:5])
            mm = int(line[5:7])
            dd = int(line[8:10])
            century = int(line[14:16]) if line[14:16].strip() else 20
            year = century * 100 + yy
            base = int(line[16:20])
            for h in range(24):
                fld = line[20 + 4 * h:24 + 4 * h]
                v = int(fld)
                if v == 9999:
                    val = np.nan
                else:
                    val = base * 100 + v
                times.append(pd.Timestamp(year, mm, dd, h, 30))
                vals.append(val)
    s = pd.DataFrame({"Dst": vals}, index=pd.DatetimeIndex(times)).sort_index()
    return s


def join_dst(df, dst_hourly):
    """GeoDGP Step 5 join: 1-min ffill of Dst, then global ffill + dropna."""
    dst = dst_hourly.resample('1min').ffill()
    out = df.join(dst)
    return out.ffill().dropna()


def build_driver_features(omni):
    """GeoDGP Step 6: 93-column driver feature table, exact column order."""
    lagged_data = []
    for lag in [i * 5 for i in range(0, 12)]:
        for covariate in omni.columns[0:len(omni.columns) - 3]:
            lagged_data.append(
                omni[covariate].shift(lag).rolling('5min', min_periods=3)
                .median().rename(f'{covariate}_lagmd_{1 * lag}'))
    lagged_dst = []
    for lag in [i * 60 for i in range(1, 13)]:
        lagged_dst.append(omni['Dst'].shift(lag).rename(f'Dst_lag_{1 * lag}'))
    omni = pd.concat([omni] + lagged_dst, axis=1)
    omni = pd.concat([omni] + lagged_data, axis=1)
    diff_dst = []
    dst_minutes = [i * 60 for i in range(1, 13)]
    for j in range(len(dst_minutes)):
        if j == 0:
            diff_dst.append((omni['Dst'] - omni[f'Dst_lag_{dst_minutes[j]}'])
                            .rename(f'Dst_diff_{j + 1}'))
        else:
            diff_dst.append((omni[f'Dst_lag_{dst_minutes[j - 1]}'] -
                             omni[f'Dst_lag_{dst_minutes[j]}'])
                            .rename(f'Dst_diff_{j + 1}'))
    omni = pd.concat([omni] + diff_dst, axis=1)
    for lag in [i * 60 for i in range(1, 13)]:
        omni = omni.drop(f'Dst_lag_{1 * lag}', axis=1)
    return omni.dropna()


def station_aacgm():
    """AACGM (SuperMAG-style) mlat/mlon for the 12 AE observatories."""
    info = pd.read_csv(DATA / "weights" / "station_info.csv")
    info = info.set_index("IAGA")
    out = {}
    for code in AE_STATIONS:
        row = info.loc[code]
        out[code] = (float(row["AACGMLAT"]), float(row["AACGMLON"]))
    return out


def sm_coords(mlat, mlon, times, chunk=50000):
    """Per-minute SM lat/lon for a station (sunpy MAG->SM, igrf13)."""
    import astropy.units as u
    from sunpy.coordinates import frames as sframes
    lats, lons = [], []
    for i in range(0, len(times), chunk):
        t = times[i:i + chunk]
        mag = sframes.Geomagnetic(lon=np.full(len(t), mlon) * u.deg,
                                  lat=np.full(len(t), mlat) * u.deg,
                                  obstime=t)
        sm = mag.transform_to(sframes.SolarMagnetic(magnetic_model='igrf13',
                                                    obstime=t))
        lats.append(sm.lat.to_value(u.deg))
        lons.append(sm.lon.to_value(u.deg))
    return np.concatenate(lats), np.concatenate(lons)


def load_model(qoi="dBN", variant=""):
    """variant '' (nowcast) or '_1h_ahead'."""
    import torch
    from model import DeepGaussianProcess
    torch.set_num_threads(8)
    model = DeepGaussianProcess(96, torch.device('cpu'), num_hidden1_dims=20,
                                num_hidden2_dims=10, num_hidden3_dims=10)
    sd = torch.load(DATA / "weights" / f"model_state_{qoi}_10T{variant}.pth",
                    map_location='cpu')
    model.load_state_dict(sd)
    model.eval()
    scaler_x = pickle.load(open(DATA / "weights" / "scalerX_10.pkl", 'rb'))
    scaler_y = pickle.load(open(DATA / "weights" / f"scalerY_{qoi}_10.pkl", 'rb'))
    return model, scaler_x, scaler_y


def predict_batches(model, scaler_x, scaler_y, X96, batch=4096):
    """X96: (n,96) float ndarray in feature space -> (pred_nT, sigma_nT)."""
    import torch
    from torch.utils.data import DataLoader
    from utils import SuperMAGDataset_fromtensor, sigma_GMM
    Xs = torch.from_numpy(scaler_x.transform(X96)).float()
    ds = SuperMAGDataset_fromtensor(Xs, torch.zeros(len(Xs)))
    with torch.no_grad():
        mus, variances, _ = model.predict(DataLoader(ds, batch_size=batch))
    sigma = sigma_GMM(mus, variances)
    pred = scaler_y.inverse_transform(mus.mean(0).numpy().reshape(-1, 1)).ravel()
    sig = (np.asarray(sigma).ravel() * scaler_y.scale_[0])
    return pred, sig
