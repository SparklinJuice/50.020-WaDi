# WaDi Preprocessing Data Interface

## Paths

- Raw normal file: `dataset/WADI_14days_new.csv`
- Raw attack file: `dataset/WADI_attackdataLABLE.csv`
- Feature config: `wadi_feature_config.json`
- Processed outputs: `data/processed/train.parquet`, `val.parquet`, `test.parquet`
- Scaler: `artifacts/scaler.joblib`
- Ordered features: `artifacts/feature_names.json`
- Explicit-rule sensors: `artifacts/rule_sensors.json`

## Data layout

Each processed parquet file contains `timestamp`, `label`, and the ordered model feature
columns plus `segment_id`. Timestamps are UTC-naive pandas timestamps; normal data may
contain calendar gaps. A positive diff that is not 10 seconds starts a gap and a new
segment. Labels use `1` for attack and `0` for normal.

The config currently declares 86 modelling features: 67 continuous plus 26 discrete
entries overlap conceptually; after all-NaN, constant-normal, and monotonic-totaliser
removal, `artifacts/feature_names.json` is the authoritative final column count.

## Windows

`make_windows(df, size=20, stride=1)` creates 200-second windows on default settings.
Windows are generated independently within train, validation, and test, and never cross
`segment_id` boundaries. Segments shorter than 20 rows are excluded before windowing.
A window is labelled `1` if any 10-second step inside it is attacked. `flatten_window_summary(X)`
returns mean, standard deviation, minimum, maximum, and last-step summaries for
Isolation Forest.

For the current real-data outputs, default-window counts are: train 62,729 windows
(0.00% attack-labelled), validation 15,673 windows (0.00%), and test 17,262 windows
(7.39% attack-labelled).

## Planned score file

Write model score CSV files with this layout:

```text
timestamp,label,score,err_<feature1>,err_<feature2>,...
```

- `timestamp`: window-end timestamp
- `label`: evaluation label
- `score`: model anomaly score
- `err_<feature>`: one reconstruction/error column per final feature

## Post-split degenerate feature drop
After the chronological split, `preprocess.py` scans every feature on TRAIN statistics only.
Continuous features with <= 5 unique train values or raw train std < 1e-4 are dropped, plus
`3_AIT_001_PV` explicitly (near-constant in train, scaled values up to ~58,825 in val, flat in test,
KS_attack 0.028). Dropped features and reasons are in `artifacts/dropped_after_split.json`.
The final feature count is printed by `python preprocess.py` and stored in `artifacts/feature_names.json`
(expected 85 or fewer, was 86). Features with max |scaled| > 20 in val/test are printed for manual review.
