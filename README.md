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

## Inputs

| Input | Source | Details |
|---|---|---|
| ΔB grid predictions | [GeoDGP](https://csem.engin.umich.edu/GeoDGP/) realtime | SWMF `real4` binary grid files of predicted ground magnetic perturbation (`dBN`), 1-min valid times incl. future-valid; the model samples 5,580 integer-degree nodes ≥ 30°N |
| Solar wind at 14 R<sub>E</sub> | [MIDL](https://csem.engin.umich.edu/MIDL/) realtime | Multi-spacecraft L1 merge ballistically propagated to the bow-shock nose (OMNI convention), 1-min GSM: B<sub>x</sub>, B<sub>y</sub>, B<sub>z</sub>, V<sub>x</sub>, density, temperature — the transit lead past wall clock is what powers the forecast tail |
| Dst index | [WDC Kyoto](https://wdc.kugi.kyoto-u.ac.jp/) realtime | Hourly quicklook Dst (`dst_realtime` service), stamped HH:30, forward-filled over the 1–2 h publication lag |
| F10.7 solar flux | [NOAA SWPC](https://services.swpc.noaa.gov/) + [CelesTrak](https://celestrak.org/SpaceData/) | Latest observed 10.7 cm flux (SWPC) plus the daily history (CelesTrak) for 81-day averaging |
| Ground magnetometer H | [INTERMAGNET GIN](https://imag-data.bgs.ac.uk/) (BGS) | 1-min H-component from ~50 high-latitude observatories incl. the reachable AE-ring stations, with per-station data age. |

From the solar wind, Dst, and F10.7 the service derives the model's 93
driver features exactly as in training: the 6 solar-wind covariates,
geopack dipole tilt, F10.7, Dst, 72 lagged 5-minute medians (0–55 min),
and 12 hourly Dst differences.

## Layout

| | |
|---|---|
| `predict_aual.py`, `swmf_datafile.py` | model interface + grid reader |
| `models/` | the corrector `.joblib` (drop-in swaps for retrains) |
| `validate_model.py` | acceptance checks for any new model file |
| `build_features_rt.py`, `fetch_*_rt.py` | realtime feature assembly |
| `reference_geodgp_common.py` | golden-tested hindcast feature builder the realtime path must match |
| `run_aual_tick.py` | the per-minute service |
| `archive_grids.py`, `grid_archive/` | 45-day ΔB grid archive for retraining/validation |
| `training/` | model training pipeline (GeoDGP rerun features → LightGBM corrector) |
