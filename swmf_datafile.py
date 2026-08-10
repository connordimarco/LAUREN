"""Minimal swmf_datafile replacement: read/write SWMF IDL-format binary
files ("real4"), matching the sword deployment's output layout
(Fortran sequential records, little-endian, 79-char padded strings).

Record layout:
  1: head (79-char padded string)
  2: step(i4), time(f4), ndim(i4), npar(i4), nvar(i4)
  3: dims (ndim x i4)  -- e.g. [361, 181]
  4: params (npar x f4), only if npar > 0
  5: names (79-char padded string)
  6: coords (f4, ndim * prod(dims))
  7+: one record per state variable (f4, prod(dims))

write_file takes the dict the deployment passes: keys head, step, time,
ndim, npar, nvar, name, coord, state.
"""
import struct

import numpy as np

_STR_LEN = 79


def _pad(s):
    return s.ljust(_STR_LEN)[:_STR_LEN].encode()


def _wrec(f, payload):
    n = struct.pack("<i", len(payload))
    f.write(n)
    f.write(payload)
    f.write(n)


def write_file(data, path, fmt="real4"):
    if fmt != "real4":
        raise NotImplementedError(f"fmt={fmt}")
    coords = np.asarray(data["coord"], dtype="<f4")
    state = [np.asarray(s, dtype="<f4") for s in data["state"]]
    ndim = int(data["ndim"])
    dims = coords.shape[1:][::-1]  # array is (ndim, n2, n1); dims written [n1, n2]
    with open(path, "wb") as f:
        _wrec(f, _pad(data["head"]))
        _wrec(f, struct.pack("<ifiii", int(data["step"]), float(data["time"]),
                             ndim, int(data["npar"]), int(data["nvar"])))
        _wrec(f, np.asarray(dims, dtype="<i4").tobytes())
        if int(data["npar"]) > 0:
            _wrec(f, np.asarray(data["par"], dtype="<f4").tobytes())
        _wrec(f, _pad(data["name"]))
        _wrec(f, coords.tobytes())
        for s in state:
            _wrec(f, s.tobytes())


def read_file(path):
    with open(path, "rb") as f:
        buf = f.read()
    pos = 0

    def rec():
        nonlocal pos
        n = struct.unpack("<i", buf[pos:pos + 4])[0]
        payload = buf[pos + 4:pos + 4 + n]
        n2 = struct.unpack("<i", buf[pos + 4 + n:pos + 8 + n])[0]
        if n != n2:
            raise ValueError(f"record marker mismatch {n} != {n2}")
        pos += 8 + n
        return payload

    head = rec().decode().strip()
    step, time, ndim, npar, nvar = struct.unpack("<ifiii", rec())
    dims = np.frombuffer(rec(), dtype="<i4")
    par = np.frombuffer(rec(), dtype="<f4") if npar > 0 else np.array([])
    name = rec().decode().strip()
    coords = np.frombuffer(rec(), dtype="<f4").reshape(ndim, *dims[::-1])
    state = []
    while pos < len(buf):
        state.append(np.frombuffer(rec(), dtype="<f4").reshape(*dims[::-1]))
    return dict(head=head, step=step, time=time, ndim=ndim, npar=npar,
                nvar=nvar, dims=dims, par=par, name=name, coord=coords,
                state=np.array(state))
