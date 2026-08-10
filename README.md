# LAUREN

**L**ower **A**nd **U**pper **R**eal-time **E**lectrojet i**N**dices — a
machine-learning approach to predicting the AU and AL auroral indices in
real time.

LAUREN corrects GeoDGP's realtime ΔB grid predictions into Kyoto-style
AU/AL using a LightGBM model trained on 20 years (2005–2022) of GeoDGP
reruns driven by MIDL solar wind. Validated fully prospectively on
2023–2024 (solar max, Gannon storm included): **r(AL) = 0.888 / RMSE
71 nT, r(AU) = 0.884 / 38 nT**. Because the solar wind is measured at L1
and ballistically propagated, the output includes a free short-range
forecast (~1 h, shrinking with wind speed).

## How it runs

A per-minute cron tick (`aual_tick.sh` → `run_aual_tick.py`):

1. reads GeoDGP ΔB grids rsynced in from the SWORD machine
   (`realtime-geodgp/`),
2. assembles the model's 93 solar-wind driver features from MIDL's
   realtime 14 Re product, realtime Kyoto Dst, and SWPC F10.7 — following
   the GeoDGP training conventions exactly (`build_features_rt.py`),
3. predicts AU/AL for every new grid valid time, and
4. publishes products, pushed back to SWORD for GITM-RT:
   - `realtime-aual/aual.dat` — `year mo dy hr mn sc AU AL` (nT),
     rolling 24 h incl. future-valid rows
   - `realtime-f107/f107.json` — current F10.7 + 81-day average +
     90-day daily series (replaces SWORD's own SWPC fetch)

## Layout

| | |
|---|---|
| `predict_aual.py`, `swmf_datafile.py` | model interface + grid reader (Pleiades contract — don't edit) |
| `models/` | the corrector `.joblib` (drop-in swaps for retrains) |
| `validate_model.py` | acceptance checks for any new model file |
| `build_features_rt.py`, `fetch_*_rt.py` | realtime feature assembly |
| `reference_geodgp_common.py` | golden-tested hindcast feature builder the realtime path must match |
| `run_aual_tick.py`, `aual_tick.sh` | the per-minute service |
| `archive_grids.py`, `grid_archive/` | 45-day ΔB grid archive for retraining/validation |
| `docs/` | Pleiades handoff + prospective skill evidence |

## Setup

```bash
bash setup.sh                                        # builds ./venv
venv/bin/python validate_model.py models/<model>.joblib
```

Known limitations and provenance: `docs/HANDOFF_SWORD.md`.
