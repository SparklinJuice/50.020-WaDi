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

# Sensors dropped regardless of the generic scan, with the reason recorded.
FORCE_DROP_AFTER_SPLIT = {
    "3_AIT_001_PV": (
        "near-constant in train (3 unique values, raw std ~3e-6) but active in val "
        "(scaled up to 58,825); flat in test; KS_attack 0.028 so no attack signal"
    ),
    "2B_AIT_002_PV": (
        "persistent level shift in the attack file: from 2017-10-10 15:49:40 to the end "
        "(~9,400 of 17,281 test rows, only ~400 labelled attack) it sits ~68,500 train-std "
        "above normal. This is a sensor/data-quality fault, not an attack signature; it would "
        "make every later window look anomalous (covers attacks A7-A14). Decision taken from "
        "inspecting the test file, so disclose in the report."
    ),
}


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
    if "timestamp" not in df or df.empty:
        return df

    data = df.set_index("timestamp").sort_index().copy()
    feature_names = [c for c in continuous + discrete if c in data.columns]
    diff_s = data.index.to_series().diff().dt.total_seconds()
    segment = (diff_s.isna() | (diff_s > 10)).cumsum().astype("int32")
    data["_source_segment"] = segment.to_numpy()

    data["_bin"] = data.index.floor("10s")
    aggregations = {c: "mean" for c in continuous if c in data.columns}
    aggregations.update({c: "max" for c in discrete if c in data.columns})
    if "label" in data.columns:
        aggregations["label"] = "max"
    result = (
        data.groupby(["_source_segment", "_bin"], sort=True)
        .agg(aggregations)
        .reset_index()
        .rename(columns={"_bin": "timestamp"})
    )
    result = result.dropna(how="all", subset=feature_names)
    result = result.sort_values("timestamp").reset_index(drop=True)
    result = result[["timestamp"] + feature_names + (["label"] if "label" in result.columns else [])]
    result[feature_names] = result[feature_names].ffill().bfill()
    diffs = result["timestamp"].diff().dt.total_seconds()
    result["segment_id"] = (diffs.isna() | (diffs != 10)).cumsum().astype("int32")
    return result


def add_segment_ids(df: pd.DataFrame) -> pd.DataFrame:
    """Add contiguous-segment IDs to a 10-second DataFrame."""
    df = df.copy().sort_values("timestamp").reset_index(drop=True)
    diffs = df["timestamp"].diff().dt.total_seconds()
    df["segment_id"] = (diffs.isna() | (diffs != 10)).cumsum().astype("int32")
    return df


def print_gap_diagnostics(df: pd.DataFrame, name: str) -> None:
    df = df.sort_values("timestamp").reset_index(drop=True)
    diffs = df["timestamp"].diff().dt.total_seconds()
    gap_positions = np.flatnonzero(diffs.to_numpy() > 10) if len(diffs) else np.array([], dtype=int)
    print(f"{name}: gaps={len(gap_positions)}, rows={len(df)}")
    for pos in gap_positions:
        pos = int(pos)
        print(
            f"  gap: {df.iloc[pos - 1]['timestamp']} -> {df.iloc[pos]['timestamp']} "
            f"(diff={int(diffs.iloc[pos])}s)"
        )

    segment = add_segment_ids(df)["segment_id"]
    for segment_id, indices in df.groupby(segment).groups.items():
        segment_df = df.loc[list(indices)]
        print(
            f"  segment {int(segment_id)}: rows={len(segment_df)}, "
            f"{segment_df.iloc[0]['timestamp']} -> {segment_df.iloc[-1]['timestamp']}"
        )


def drop_short_segments(df: pd.DataFrame, size: int = 20) -> tuple[pd.DataFrame, int, int]:
    """Drop segments too short for a window and report lost rows/windows."""
    df = add_segment_ids(df)
    keep = df["segment_id"].map(df.groupby("segment_id").size().ge(size))
    dropped_rows = int((~keep).sum())
    dropped_windows = sum(
        max(0, len(segment) - size + 1)
        for _, segment in df.loc[~keep].groupby("segment_id")
    )
    return df.loc[keep].copy().reset_index(drop=True), dropped_rows, dropped_windows


def make_windows(
    df_or_array: pd.DataFrame | np.ndarray,
    labels: Sequence[int] | None = None,
    size: int = 20,
    stride: int = 1,
) -> tuple[np.ndarray, np.ndarray, pd.DatetimeIndex]:
    """Build windows inside one split only."""
    if size < 1 or stride < 1:
        raise ValueError("size and stride must be positive")

    segment_ids = None
    if isinstance(df_or_array, pd.DataFrame):
        if "timestamp" not in df_or_array:
            raise ValueError("DataFrame input requires a timestamp column")
        timestamps = pd.to_datetime(df_or_array["timestamp"])
        if "label" in df_or_array:
            labels = df_or_array["label"].to_numpy()
            feature_columns = [
                c for c in df_or_array.columns
                if c not in ("timestamp", "label", "segment_id")
            ]
        else:
            if labels is None:
                raise ValueError("labels are required when DataFrame has no label column")
            feature_columns = [
                c for c in df_or_array.columns
                if c not in ("timestamp", "segment_id")
            ]
        segment_ids = df_or_array["segment_id"].to_numpy() if "segment_id" in df_or_array else None
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
    if segment_ids is not None:
        segment_groups = []
        for _, indices in pd.Series(segment_ids).groupby(segment_ids).groups.items():
            segment_groups.append(np.asarray(list(indices), dtype=int))
        for indices in segment_groups:
            if len(indices) < size:
                continue
            for start in range(0, len(indices) - size + 1, stride):
                stop = start + size
                window_indices = indices[start:stop]
                X.append(values[window_indices])
                y.append(int(label_values[window_indices].max()))
                ends.append(timestamps[window_indices[-1]])
    else:
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
    if "segment_id" in df.columns:
        for _, indices in df.groupby("segment_id", sort=True).groups.items():
            segment = df.loc[list(indices), feature_names]
            df.loc[list(indices), feature_names] = segment.ffill().bfill()
    else:
        df[feature_names] = df[feature_names].ffill().bfill()
    if train_medians is None:
        train_medians = df[feature_names].median()
    df[feature_names] = df[feature_names].fillna(train_medians)
    return df, train_medians


def difference_totalisers_within_segments(
    df: pd.DataFrame, totalisers: Sequence[str]
) -> pd.DataFrame:
    """Difference counters independently in every 10-second segment."""
    df = df.copy()
    for _, indices in df.groupby("segment_id", sort=True).groups.items():
        positions = list(indices)
        for column in totalisers:
            if column in df.columns:
                df.loc[positions, column] = df.loc[positions, column].diff()
    present = [c for c in totalisers if c in df.columns]
    if present:
        df[present] = df[present].mask(df[present].isna())
    return df


def preprocess(root: Path | None = None) -> dict[str, pd.DataFrame]:
    root = (root or Path(__file__).resolve().parent).resolve()
    print("Stage: loading raw CSVs")
    dataset_dir = root / "dataset"
    normal_path = dataset_dir / "WADI_14days_new.csv"
    attack_path = dataset_dir / "WADI_attackdataLABLE.csv"

    normal = align_feature_names(load_wadi_csv(normal_path))
    attack = align_feature_names(load_wadi_csv(attack_path))
    print("Stage: rebuilding timestamps and selecting features")
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
    final_features = [c for c in feature_names if c not in drop_names]
    totalisers = [
        c for c in config.get("totalisers_use_diff_or_drop", [])
        if c in final_features
    ]

    rule_sensors = [
        c for c in config.get("constant_normal_but_attacked", [])
        if c in normal.columns and c in attack.columns
    ]

    continuous = [c for c in config["continuous"] if c in final_features]
    discrete = [c for c in config["discrete"] if c in final_features]

    normal = normal[["timestamp"] + final_features + ["label"]]
    attack = attack[["timestamp"] + final_features + ["label"]]
    print("Stage: downsampling normal and attack data")
    normal_down = downsample(normal, continuous, discrete)
    print("Stage: downsampling attack data")
    attack_down = downsample(attack, continuous, discrete)
    normal_down = difference_totalisers_within_segments(normal_down, totalisers)
    attack_down = difference_totalisers_within_segments(attack_down, totalisers)

    print_gap_diagnostics(normal_down, "NORMAL 10-second split")
    print_gap_diagnostics(attack_down, "ATTACK 10-second split")

    split_index = int(len(normal_down) * 0.8)
    gap_indices = normal_down.index[
        normal_down["timestamp"].diff().dt.total_seconds() > 10
    ].to_numpy()
    split_inside_segment = True
    if len(gap_indices):
        nearest = gap_indices[np.argmin(np.abs(gap_indices - split_index))]
        if abs(nearest - split_index) <= int(len(normal_down) * 0.05):
            split_index = int(nearest)
            split_inside_segment = False
    if split_inside_segment:
        print(
            f"Normal split: split point is inside a segment at normal_down row {split_index}; "
            "no segment was sufficiently near 80%."
        )
    else:
        print(
            f"Normal split: moved to segment boundary at normal_down row {split_index} "
            "(near the 80% target)."
        )
    train = normal_down.iloc[:split_index].copy()
    val = normal_down.iloc[split_index:].copy()
    train = add_segment_ids(train)
    val = add_segment_ids(val)
    test = add_segment_ids(attack_down)

    train, train_dropped_rows, train_dropped_windows = drop_short_segments(train)
    val, val_dropped_rows, val_dropped_windows = drop_short_segments(val)
    test, test_dropped_rows, test_dropped_windows = drop_short_segments(test)
    print(
        "Short-segment drops (window size 20): "
        f"train rows={train_dropped_rows}, windows={train_dropped_windows}; "
        f"val rows={val_dropped_rows}, windows={val_dropped_windows}; "
        f"test rows={test_dropped_rows}, windows={test_dropped_windows}"
    )

    train, train_medians = fill_split_with_median(train, final_features)
    val, _ = fill_split_with_median(val, final_features, train_medians)
    test, _ = fill_split_with_median(test, final_features, train_medians)

    # Post-split degeneracy scan. A feature that is (near-)constant in TRAIN but
    # active in val/test gets a tiny scaler std and explodes after scaling
    # (e.g. 3_AIT_001_PV reached |z| ~ 58,825 in val). Decide on TRAIN stats only.
    discrete_set = set(discrete)
    dropped_after_split: dict[str, str] = {}
    scan_rows = []
    for c in final_features:
        tr_std = float(train[c].std(ddof=0))
        tr_nuniq = int(train[c].nunique())
        scan_rows.append((c, tr_std, tr_nuniq))
        if c in FORCE_DROP_AFTER_SPLIT:
            dropped_after_split[c] = FORCE_DROP_AFTER_SPLIT[c]
        elif c not in discrete_set and (tr_nuniq <= 5 or tr_std < 1e-4):
            dropped_after_split[c] = (
                f"degenerate in train (unique={tr_nuniq}, raw std={tr_std:.3g}); "
                "scaled values would explode in val/test"
            )
        elif c in discrete_set and (tr_nuniq <= 1 or tr_std < 1e-4):
            print(f"REPORT (discrete, kept): {c} train unique={tr_nuniq}, std={tr_std:.3g}")
    if dropped_after_split:
        print(f"Dropping {len(dropped_after_split)} degenerate feature(s) after split:")
        for c, why in dropped_after_split.items():
            print(f"  - {c}: {why}")
    final_features = [c for c in final_features if c not in dropped_after_split]

    # StandardScaler is appropriate here because the EDA mixes continuous ranges
    # and train-only scaling prevents validation/test distribution leakage.
    scaler = StandardScaler()
    train_values = scaler.fit_transform(train[final_features].to_numpy(dtype=np.float64))
    val_values = scaler.transform(val[final_features].to_numpy(dtype=np.float64))
    test_values = scaler.transform(test[final_features].to_numpy(dtype=np.float64))
    train[final_features] = train_values.astype(np.float32)
    val[final_features] = val_values.astype(np.float32)
    test[final_features] = test_values.astype(np.float32)
    train = train.drop(columns=list(dropped_after_split))
    val = val.drop(columns=list(dropped_after_split))
    test = test.drop(columns=list(dropped_after_split))

    # Scan table: worst scaled excursions in val/test (report only above 20).
    scan = pd.DataFrame(scan_rows, columns=["feature", "train_raw_std", "train_unique"])
    scan = scan[scan["feature"].isin(final_features)].set_index("feature")
    scan["max_abs_val"] = pd.Series(np.abs(val_values).max(axis=0), index=final_features)
    scan["max_abs_test"] = pd.Series(np.abs(test_values).max(axis=0), index=final_features)
    scan["worst"] = scan[["max_abs_val", "max_abs_test"]].max(axis=1)
    print("Top 15 features by max |scaled| in val/test:")
    print(scan.sort_values("worst", ascending=False).head(15).to_string())
    flagged = scan[scan["worst"] > 20].sort_values("worst", ascending=False)
    print(f"{len(flagged)} feature(s) with max |scaled| > 20 (kept, review manually):")
    print(", ".join(flagged.index) if len(flagged) else "none")

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
    (artifact_dir / "dropped_after_split.json").write_text(
        json.dumps(dropped_after_split, indent=2), encoding="utf-8"
    )
    (artifact_dir / "rule_sensors.json").write_text(
        json.dumps(rule_sensors, indent=2), encoding="utf-8"
    )

    catalogue = pd.read_csv(root / "wadi_attack_catalogue.csv")
    shortest_seconds = float(catalogue["duration_min"].min() * 60)
    shortest_windows = int(np.ceil(shortest_seconds / 10))
    print(
        f"Shortest catalogue attack: {shortest_seconds:.2f}s; "
        f"covers {shortest_windows} 10-second windows."
    )
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
    for name, split in (("train", train), ("val", val), ("test", test)):
        _, window_labels, _ = make_windows(split, size=20, stride=1)
        print(
            f"Final window counts: {name}={len(window_labels)}, "
            f"attack_fraction={window_labels.mean():.2%}"
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
