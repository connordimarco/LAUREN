#!/usr/bin/env python3
"""Adapter: sword-machine GeoDGP grid files -> 288-point SM ring features.

Input: deltaB_YYYYMMDDhhmmss_real4.out (SWMF IDL real4 binary; GeoLon GeoLat
dBH dBN dBE dBD on a 1x1 deg geographic grid; header carries the valid time).

Method: forward-transform every grid node to Solar Magnetic coordinates with
the SAME astropy/sunpy path the deployment uses (ITRS ->
sunpy SolarMagnetic, igrf13), then linearly interpolate the scattered
SM-space dBN onto the corrector's fixed ring points
(lats 63.5/66/68.5/71 SM x lons 0..355 step 5). Longitude wrap handled by
tiling +-360. Output columns match the training matrices: dBN_{lat}_{lon:03d}.

Usage: python sword_adapter.py FILE [FILE ...] [--out ring.parquet]
"""
import argparse
import struct
from datetime import datetime

import numpy as np
import pandas as pd

SM_LATS = (63.5, 66.0, 68.5, 71.0)
SM_LONS = np.arange(0, 360, 5)
RING_COLS = [f"dBN_{lat}_{lon:03d}" for lat in SM_LATS for lon in SM_LONS]


def read_real4(path):
    with open(path, "rb") as f:
        buf = f.read()
    pos = 0

    def rec():
        nonlocal pos
        n = struct.unpack("<i", buf[pos:pos + 4])[0]
        data = buf[pos + 4:pos + 4 + n]
        n2 = struct.unpack("<i", buf[pos + 4 + n:pos + 8 + n])[0]
        if n != n2:
            raise ValueError(f"record marker mismatch {n} != {n2}")
        pos += 8 + n
        return data

    head = rec().decode().strip()
    step, time, ndim, npar, nvar = struct.unpack("<ifiii", rec())
    dims = np.frombuffer(rec(), dtype="<i4")
    if npar > 0:
        rec()
    names = rec().decode().strip().split()
    coords = np.frombuffer(rec(), dtype="<f4").reshape(ndim, *dims[::-1])
    state = []
    while pos < len(buf):
        state.append(np.frombuffer(rec(), dtype="<f4").reshape(*dims[::-1]))
    when = datetime.strptime(head.split()[-1], "%Y%m%d%H%M%S")
    out = {nm: st for nm, st in zip(names[2:], state)}
    out.update(GeoLon=coords[0], GeoLat=coords[1], when=when)
    return out


def ring_from_grid(g):
    import astropy.coordinates
    import astropy.units as u
    from scipy.interpolate import griddata
    from sunpy.coordinates import frames

    # drop the duplicated lon=360 column before transforming
    lon, lat, dbn = g["GeoLon"][:, :-1], g["GeoLat"][:, :-1], g["dBN"][:, :-1]
    loc = astropy.coordinates.EarthLocation(lat=lat * u.deg, lon=lon * u.deg)
    itrs = astropy.coordinates.ITRS(x=loc.x, y=loc.y, z=loc.z, obstime=g["when"])
    sm = itrs.transform_to(frames.SolarMagnetic(magnetic_model="igrf13",
                                                obstime=g["when"]))
    sm_lat = sm.lat.value.ravel()
    sm_lon = sm.lon.value.ravel() % 360.0
    vals = dbn.ravel()

    # tile in longitude so ring points near 0/360 interpolate across the seam
    pts = np.column_stack([
        np.concatenate([sm_lon - 360.0, sm_lon, sm_lon + 360.0]),
        np.tile(sm_lat, 3)])
    v3 = np.tile(vals, 3)
    keep = (pts[:, 1] > 40)  # auroral zone only; keeps griddata small
    tlat = np.repeat(SM_LATS, len(SM_LONS))
    tlon = np.tile(SM_LONS, len(SM_LATS)).astype(float)
    ring = griddata(pts[keep], v3[keep], np.column_stack([tlon, tlat]),
                    method="linear")
    return pd.Series(ring, index=RING_COLS, name=g["when"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    rows = [ring_from_grid(read_real4(f)) for f in args.files]
    df = pd.DataFrame(rows)
    df.index.name = "Datetime"
    if args.out:
        df.to_parquet(args.out)
        print(f"wrote {args.out} ({df.shape})")
    else:
        print(df.iloc[:, :4])
        print(f"... {df.shape[1]} ring columns, {len(df)} timestamps")


if __name__ == "__main__":
    main()
