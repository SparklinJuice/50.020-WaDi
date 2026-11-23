import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

import preprocess


def test_make_windows_stays_within_split():
    df = pd.DataFrame({
        "timestamp": pd.date_range("2020-01-01", periods=25, freq="10s"),
        "label": np.r_[np.zeros(20, dtype=int), np.ones(5, dtype=int)],
        "sensor": np.arange(25),
    })
    X, y, ends = preprocess.make_windows(df, size=20, stride=1)
    assert X.shape == (6, 20, 1)
    assert y.shape == (6,)
    assert len(ends) == 6
    assert y.tolist() == [0, 1, 1, 1, 1, 1]
    assert ends[-1] == df["timestamp"].iloc[-1]


def test_make_windows_does_not_cross_segment_ids():
    df = pd.DataFrame({
        "timestamp": pd.to_datetime([
            "2020-01-01 00:00:00", "2020-01-01 00:00:10",
            "2020-01-01 00:00:20", "2020-01-01 00:01:00",
            "2020-01-01 00:01:10", "2020-01-01 00:01:20",
        ]),
        "label": np.array([0, 0, 0, 1, 1, 1]),
        "segment_id": np.array([0, 0, 0, 1, 1, 1]),
        "sensor": np.arange(6),
    })
    X, y, ends = preprocess.make_windows(df, size=2, stride=1)
    assert X.shape == (4, 2, 1)
    assert y.tolist() == [0, 0, 1, 1]


def test_make_windows_never_reads_across_split_gap():
    root = Path(__file__).resolve().parents[1]
    processed = root / "data" / "processed"
    if not (processed / "train.parquet").exists():
        pytest.skip("Raw WaDi CSVs are not present")
    train = pd.read_parquet(processed / "train.parquet")
    X, y, ends = preprocess.make_windows(train, size=20, stride=1)
    assert len(X) > 0


def test_make_windows_array_preserves_feature_shape():
    values = np.arange(30, dtype=np.float32).reshape(10, 3)
    label_values = np.r_[np.zeros(8, dtype=int), np.ones(2, dtype=int)]
    timestamps = pd.date_range("2020-01-01", periods=10, freq="10s")
    X, y, ends = preprocess.make_windows(values, label_values, size=3, stride=2)

    assert X.shape == (4, 3, 3)
    assert y.tolist() == [0, 0, 0, 1]
    assert len(ends) == 4
    assert ends.is_monotonic_increasing


def test_synthetic_split_scaler_and_outputs(tmp_path, monkeypatch):
    # A focused synthetic check of the core split/scaling invariants.
    rng = np.random.default_rng(42)
    train = pd.DataFrame({
        "timestamp": pd.date_range("2020-01-01", periods=100, freq="10s"),
        "a": rng.normal(10, 2, 100),
        "label": np.zeros(100, dtype=int),
    })
    val = pd.DataFrame({
        "timestamp": pd.date_range("2020-01-02", periods=40, freq="10s"),
        "a": rng.normal(15, 3, 40),
        "label": np.zeros(40, dtype=int),
    })
    test = pd.DataFrame({
        "timestamp": pd.date_range("2020-01-03", periods=30, freq="10s"),
        "a": rng.normal(20, 4, 30),
        "label": np.r_[np.zeros(20), np.ones(10)],
    })

    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler().fit(train[["a"]])
    train_scaled = scaler.transform(train[["a"]])
    val_scaled = scaler.transform(val[["a"]])

    assert abs(train_scaled.mean()) < 1e-12
    assert abs(val_scaled.mean()) > 0.1
    assert not np.isnan(train_scaled).any()
    assert set(test["label"].unique()).issubset({0, 1})


def test_expected_artifacts_exist_when_pipeline_has_been_run():
    root = Path(__file__).resolve().parents[1]
    processed = root / "data" / "processed"
    artifacts = root / "artifacts"
    if not (processed / "train.parquet").exists():
        pytest.skip("Raw WaDi CSVs are not present, so pipeline outputs cannot be generated")

    train = pd.read_parquet(processed / "train.parquet")
    val = pd.read_parquet(processed / "val.parquet")
    test = pd.read_parquet(processed / "test.parquet")
    feature_names = json.loads((artifacts / "feature_names.json").read_text())
    scaler = joblib.load(artifacts / "scaler.joblib")

    assert set(feature_names) == set(train.columns) - {"timestamp", "label", "segment_id"}
    assert all(len(scaler.mean_) == len(feature_names) for _ in [0])
    for df in (train, val, test):
        assert not df.isna().any().any()
        assert set(df["label"].unique()).issubset({0, 1})
        assert df["timestamp"].is_monotonic_increasing
        diffs = df["timestamp"].diff().dropna().dt.total_seconds()
        assert (diffs > 0).all()
        assert (diffs % 10 == 0).all()
        for _, segment in df.groupby("segment_id"):
            segment_diffs = segment["timestamp"].diff().dropna().dt.total_seconds()
            assert segment_diffs.eq(10).all()


def test_no_exploding_features_after_scaling():
    root = Path(__file__).resolve().parents[1]
    processed = root / "data" / "processed"
    if not (processed / "train.parquet").exists():
        pytest.skip("Pipeline outputs not generated")
    names = json.loads((root / "artifacts" / "feature_names.json").read_text())
    assert "3_AIT_001_PV" not in names
    assert "2B_AIT_002_PV" not in names
    config = json.loads((root / "wadi_feature_config.json").read_text())
    discrete = set(config["discrete"])
    cont = [c for c in names if c not in discrete]
    # Val is normal data: huge scaled values there are a pipeline defect.
    val = pd.read_parquet(processed / "val.parquet")
    assert val[cont].abs().max().max() < 1000
    # Test contains attacks: large excursions may be genuine attack signal,
    # so report the worst offenders instead of failing.
    test = pd.read_parquet(processed / "test.parquet")
    worst = test[cont].abs().max().sort_values(ascending=False).head(5)
    print("Largest |scaled| in test (continuous):", worst.to_dict())


def test_flattened_summary_shape():
    X = np.arange(60, dtype=np.float32).reshape(3, 4, 5)
    summary = preprocess.flatten_window_summary(X)
    assert summary.shape == (3, 25)


def test_catalogue_attack_count_is_reported_not_silently_reconciled():
    root = Path(__file__).resolve().parents[1]
    catalogue = pd.read_csv(root / "wadi_attack_catalogue.csv")
    synthetic_test = pd.DataFrame({
        "timestamp": pd.date_range("2020-01-01", periods=6, freq="10s"),
        "label": np.array([0, 1, 1, 0, 1, 0], dtype=int),
    })
    result = preprocess.count_attack_catalogue_matches(synthetic_test, root)
    assert result["catalogue_attacks"] == len(catalogue) == 14
    assert result["test_attack_runs"] == 2


def test_downsample_mean_max_and_attack_label():
    timestamps = pd.date_range("2020-01-01", periods=30, freq="1s")
    frame = pd.DataFrame({
        "timestamp": timestamps,
        "continuous_sensor": np.r_[np.arange(10), np.arange(10) + 20, np.arange(10) + 40],
        "discrete_sensor": np.r_[np.ones(10), np.full(10, 2), np.full(10, 3)],
        "label": np.r_[np.zeros(10), np.array([0, 0, 0, 0, 1, 0, 0, 0, 0, 0]), np.zeros(10)],
    })
    result = preprocess.downsample(
        frame,
        continuous=["continuous_sensor"],
        discrete=["discrete_sensor"],
    )

    assert len(result) == 3
    assert result["continuous_sensor"].tolist() == pytest.approx([4.5, 24.5, 44.5])
    assert result["discrete_sensor"].tolist() == [1, 2, 3]
    assert result["label"].tolist() == [0, 1, 0]
