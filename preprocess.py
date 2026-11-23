"""Reusable WaDi preprocessing pipeline.

The raw 1 Hz files are expected in dataset/.  Missing raw data is reported as a
clear CLI error; generated files are written only after a successful run.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Iterable, Sequence

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)


def load_wadi_csv(path: Path | str, nrows: int | None = None) -> pd.DataFrame:
    """Load a WaDi CSV, handling the attack file's extra header line."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Required WaDi CSV not found: {path}")

    encoding = "utf-8-sig"
    head: list[str] = []
    for candidate in ("utf-8-sig", "cp1252", "latin1"):
        try:
            with path.open("r", encoding=candidate, newline="", errors="strict") as handle:
                head = [next(handle, "") for _ in range(10)]
            encoding = candidate
            break
        except UnicodeDecodeError:
            continue

    header_idx = next(
        (i for i, line in enumerate(head) if "Row" in line and "Date" in line),
        0,
    )
    header_line = head[header_idx]
    delimiter = csv.Sniffer().sniff(header_line, delimiters=",;\t").delimiter

    df = pd.read_csv(
        path,
        sep=delimiter,
        skiprows=header_idx,
        encoding=encoding,
        low_memory=False,
        nrows=nrows,
    )
    df.columns = df.columns.astype(str).str.strip()
    df = df.dropna(how="all")
    for col in df.columns:
        if col not in ("Date", "Time"):
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("float32")

    if {"Row", "Date"}.issubset(df.columns):
        bad = df["Row"].isna() | df["Date"].isna()
        if bad.any():
            df = df.loc[~bad].copy()
    return df.reset_index(drop=True)


def strip_column_prefixes(columns: Iterable[str]) -> list[str]:
    """Remove optional packaging prefixes while preserving sensor identifiers."""
    cleaned = []
    for col in columns:
        name = str(col).strip()
        name = re.sub(r"^(?:WADI|WaDi)[_. -]+", "", name, flags=re.IGNORECASE)
        name = re.sub(r"^.*::", "", name)
        cleaned.append(name)
    return cleaned


def rebuild_timestamp(df: pd.DataFrame, name: str) -> pd.DataFrame:
    """Rebuild a 1 Hz timestamp from Row and the first date rollover."""
    df = df.copy()
    if "Row" not in df or "Date" not in df:
        raise ValueError(f"{name}: required Row and Date columns are missing")

    date = pd.to_datetime(df["Date"], format="mixed", errors="coerce").dt.normalize()
    row = df["Row"].astype("float64").to_numpy()
    valid = date.notna() & np.isfinite(row)
    if not valid.all():
        raise ValueError(f"{name}: malformed or missing Date/Row values")

    rollovers = np.flatnonzero(date.to_numpy()[1:] != date.to_numpy()[:-1])
    if len(rollovers):
        anchor_index = int(rollovers[0] + 1)
        anchor_time = date.iloc[anchor_index]
    else:
        anchor_time = date.iloc[0]
        mmss = df["Time"].astype(str).str.extract(r"^(\d{1,2}):(\d{2})")
        if mmss.isna().all().all():
            anchor_time = anchor_time + pd.Timedelta(seconds=0)
        else:
            minutes = pd.to_numeric(mmss[0], errors="coerce").fillna(0)
            seconds = pd.to_numeric(mmss[1], errors="coerce").fillna(0)
            anchor_time = date.iloc[0] + pd.to_timedelta(
                minutes.iloc[0] * 60 + seconds.iloc[0], unit="s"
            )

    anchor_row = float(row[0 if len(rollovers) == 0 else anchor_index])
    df["timestamp"] = anchor_time + pd.to_timedelta(row - anchor_row, unit="s")
    df = df.sort_values("timestamp").reset_index(drop=True)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df


def find_label_column(df: pd.DataFrame) -> str:
    matches = [
        col for col in df.columns
        if "attack" in str(col).lower() or "label" in str(col).lower()
    ]
    if not matches:
        raise ValueError("Attack file has no attack/label column")
    return matches[0]


def align_feature_names(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = strip_column_prefixes(df.columns)
    duplicates = df.columns[df.columns.duplicated()].tolist()
    if duplicates:
        raise ValueError(f"Duplicate columns after prefix removal: {duplicates}")
    return df


def downsample(df: pd.DataFrame, continuous: Sequence[str], discrete: Sequence[str]) -> pd.DataFrame:
    """Downsample to exact 10-second bins; attack labels use max/any."""
    data = df.set_index("timestamp").sort_index()
    feature_names = [c for c in continuous + discrete if c in data.columns]
    aggregations = {c: "mean" for c in continuous if c in data.columns}
    aggregations.update({c: "max" for c in discrete if c in data.columns})
    if "label" in data.columns:
        aggregations["label"] = "max"

    result = data[list(feature_names) + (["label"] if "label" in data.columns else [])].resample("10s").agg(aggregations)
    result = result.dropna(how="all", subset=feature_names)
    result = result.ffill().bfill()
    result.index.name = "timestamp"
    return result.reset_index()


def make_windows(
    df_or_array: pd.DataFrame | np.ndarray,
    labels: Sequence[int] | None = None,
    size: int = 20,
    stride: int = 1,
) -> tuple[np.ndarray, np.ndarray, pd.DatetimeIndex]:
    """Build windows inside one split only."""
    if size < 1 or stride < 1:
        raise ValueError("size and stride must be positive")

    if isinstance(df_or_array, pd.DataFrame):
        if "timestamp" not in df_or_array:
            raise ValueError("DataFrame input requires a timestamp column")
        timestamps = pd.to_datetime(df_or_array["timestamp"])
        if "label" in df_or_array:
            labels = df_or_array["label"].to_numpy()
            feature_columns = [c for c in df_or_array.columns if c not in ("timestamp", "label")]
        else:
            if labels is None:
                raise ValueError("labels are required when DataFrame has no label column")
            feature_columns = [c for c in df_or_array.columns if c != "timestamp"]
        values = df_or_array[feature_columns].to_numpy(dtype=np.float32)
    else:
        values = np.asarray(df_or_array, dtype=np.float32)
        timestamps = pd.DatetimeIndex(
            labels if isinstance(labels, pd.DatetimeIndex) else []
        )
        if isinstance(labels, pd.Series):
            timestamps = pd.DatetimeIndex(labels)
        if len(timestamps) == 0:
            timestamps = pd.date_range("1970-01-01", periods=len(values), freq="10s")

    if labels is None:
        raise ValueError("labels are required")
    if len(values) != len(labels):
        raise ValueError("Feature rows and labels must have the same length")
    label_values = np.asarray(labels, dtype=np.int8)

    X, y, ends = [], [], []
    for start in range(0, len(values) - size + 1, stride):
        stop = start + size
        X.append(values[start:stop])
        y.append(int(label_values[start:stop].max()))
        ends.append(timestamps[stop - 1])
    return (
        np.asarray(X, dtype=np.float32),
        np.asarray(y, dtype=np.int8),
        pd.DatetimeIndex(ends),
    )


def flatten_window_summary(X: np.ndarray) -> np.ndarray:
    """Flatten mean, std, minimum, maximum, and last step for Isolation Forest."""
    if X.ndim != 3:
        raise ValueError("X must have shape (n_windows, size, n_features)")
    return np.concatenate(
        (
            X.mean(axis=1),
            X.std(axis=1),
            X.min(axis=1),
            X.max(axis=1),
            X[:, -1, :],
        ),
        axis=1,
    )


def fill_split_with_median(df: pd.DataFrame, feature_names: Sequence[str], train_medians: pd.Series | None = None) -> tuple[pd.DataFrame, pd.Series]:
    df = df.copy()
    df[feature_names] = df[feature_names].ffill().bfill()
    if train_medians is None:
        train_medians = df[feature_names].median()
    df[feature_names] = df[feature_names].fillna(train_medians)
    return df, train_medians


def preprocess(root: Path | None = None) -> dict[str, pd.DataFrame]:
    root = (root or Path(__file__).resolve().parent).resolve()
    dataset_dir = root / "dataset"
    normal_path = dataset_dir / "WADI_14days_new.csv"
    attack_path = dataset_dir / "WADI_attackdataLABLE.csv"

    normal = align_feature_names(load_wadi_csv(normal_path))
    attack = align_feature_names(load_wadi_csv(attack_path))
    normal_rows = len(normal)
    attack_rows = len(attack)

    normal = rebuild_timestamp(normal, "normal")
    attack = rebuild_timestamp(attack, "attack")
    label_col = find_label_column(attack)
    attack["label"] = (pd.to_numeric(attack[label_col], errors="coerce") == -1).astype("int8")
    attack = attack.drop(columns=[label_col])
    normal["label"] = np.int8(0)

    with (root / "wadi_feature_config.json").open("r", encoding="utf-8") as handle:
        config = json.load(handle)

    feature_names = [c for c in config["features"] if c in normal.columns and c in attack.columns]
    before_count = len(config["features"])
    drop_names = set(config.get("dropped_all_nan", []))
    drop_names.update(config.get("dropped_constant_normal", []))
    drop_names.update(config.get("totalisers_use_diff_or_drop", []))
    final_features = [c for c in feature_names if c not in drop_names]

    rule_sensors = [
        c for c in config.get("constant_normal_but_attacked", [])
        if c in normal.columns and c in attack.columns
    ]

    continuous = [c for c in config["continuous"] if c in final_features]
    discrete = [c for c in config["discrete"] if c in final_features]

    normal = normal[["timestamp"] + final_features + ["label"]]
    attack = attack[["timestamp"] + final_features + ["label"]]
    normal_down = downsample(normal, continuous, discrete)
    attack_down = downsample(attack, continuous, discrete)

    split_index = int(len(normal_down) * 0.8)
    train = normal_down.iloc[:split_index].copy()
    val = normal_down.iloc[split_index:].copy()
    test = attack_down.copy()

    train, train_medians = fill_split_with_median(train, final_features)
    val, _ = fill_split_with_median(val, final_features, train_medians)
    test, _ = fill_split_with_median(test, final_features, train_medians)

    # StandardScaler is appropriate here because the EDA mixes continuous ranges
    # and train-only scaling prevents validation/test distribution leakage.
    scaler = StandardScaler()
    train_values = scaler.fit_transform(train[final_features].to_numpy(dtype=np.float64))
    val_values = scaler.transform(val[final_features].to_numpy(dtype=np.float64))
    test_values = scaler.transform(test[final_features].to_numpy(dtype=np.float64))
    train[final_features] = train_values.astype(np.float32)
    val[final_features] = val_values.astype(np.float32)
    test[final_features] = test_values.astype(np.float32)

    processed_dir = root / "data" / "processed"
    artifact_dir = root / "artifacts"
    processed_dir.mkdir(parents=True, exist_ok=True)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    train.to_parquet(processed_dir / "train.parquet", index=False)
    val.to_parquet(processed_dir / "val.parquet", index=False)
    test.to_parquet(processed_dir / "test.parquet", index=False)
    joblib.dump(scaler, artifact_dir / "scaler.joblib")
    (artifact_dir / "feature_names.json").write_text(
        json.dumps(final_features, indent=2), encoding="utf-8"
    )
    (artifact_dir / "rule_sensors.json").write_text(
        json.dumps(rule_sensors, indent=2), encoding="utf-8"
    )

    catalogue = pd.read_csv(root / "wadi_attack_catalogue.csv")
    shortest_seconds = float(catalogue["duration_min"].min() * 60)
    if shortest_seconds < 10:
        print(
            f"WARNING: shortest catalogued attack is {shortest_seconds:.2f}s; "
            "it is shorter than the 10-second downsampling interval."
        )

    print(
        f"Rows (before -> after 10s): normal {normal_rows:,} -> {len(normal_down):,}; "
        f"attack {attack_rows:,} -> {len(attack_down):,}"
    )
    print(f"Features (before -> after dropping): {before_count} -> {len(final_features)}")
    print(f"Test attack-window fraction: {test['label'].mean():.2%}")
    print(
        f"Saved train={len(train):,}, val={len(val):,}, test={len(test):,}; "
        f"train/val/test paths under {processed_dir}"
    )
    catalogue_matches = count_attack_catalogue_matches(test, root)
    print(
        "Attack counts: "
        f"catalogue={catalogue_matches['catalogue_attacks']}, "
        f"contiguous-labelled-runs-in-test={catalogue_matches['test_attack_runs']}"
    )
    if catalogue_matches["catalogue_attacks"] != catalogue_matches["test_attack_runs"]:
        print(
            "WARNING: catalogue attack count differs from contiguous runs in test labels; "
            "verify merged/split attack episodes and label boundaries."
        )
    return {"train": train, "val": val, "test": test}


def count_attack_catalogue_matches(test: pd.DataFrame, root: Path | None = None) -> dict[str, int]:
    root = (root or Path(__file__).resolve().parent).resolve()
    catalogue = pd.read_csv(root / "wadi_attack_catalogue.csv")
    labels = test["label"].to_numpy(dtype=np.int8)
    raw_runs = int(np.sum((labels[1:] == 1) & (labels[:-1] == 0))) + int(labels[0] == 1) if len(labels) else 0
    return {"catalogue_attacks": len(catalogue), "test_attack_runs": raw_runs}


def main() -> None:
    parser = argparse.ArgumentParser(description="Preprocess the WaDi A2 dataset")
    parser.add_argument("--root", type=Path, default=None)
    args = parser.parse_args()
    preprocess(args.root)


if __name__ == "__main__":
    main()
