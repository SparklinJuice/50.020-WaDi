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
columns. Timestamps are UTC-naive pandas timestamps at 10-second intervals. Labels use
`1` for attack and `0` for normal.

The config currently declares 86 modelling features: 67 continuous plus 26 discrete
entries overlap conceptually; after all-NaN, constant-normal, and monotonic-totaliser
removal, `artifacts/feature_names.json` is the authoritative final column count.

## Windows

`make_windows(df, size=20, stride=1)` creates 200-second windows on default settings.
Windows are generated independently within train, validation, and test. A window is
labelled `1` if any 10-second step inside it is attacked. `flatten_window_summary(X)`
returns mean, standard deviation, minimum, maximum, and last-step summaries for
Isolation Forest.

## Planned score file

Write model score CSV files with this layout:

```text
timestamp,label,score,err_<feature1>,err_<feature2>,...
```

- `timestamp`: window-end timestamp
- `label`: evaluation label
- `score`: model anomaly score
- `err_<feature>`: one reconstruction/error column per final feature
