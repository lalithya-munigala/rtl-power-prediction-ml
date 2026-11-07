import os
import pickle

import numpy as np
import pandas as pd
import pytest

from rtl_power.pipeline import (
    CATEGORICAL_COLS, LEAKAGE_COLS, MODEL_FACTORIES, TARGET_COL,
    build_pipeline, build_preprocessor, detect_schema, evaluate,
    is_boolean_series, load_dataset, split_features_target, train_and_evaluate,
)

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET = os.path.join(REPO_ROOT, "rtl_power_modeling_dataset2.csv")


# ---------- fixtures ----------

@pytest.fixture
def small_df():
    """Small synthetic dataset with the same columns as the real one."""
    rng = np.random.default_rng(0)
    n = 300
    df = pd.DataFrame({
        "cycle_id": np.arange(1, n + 1),
        "benchmark": rng.choice(["sort", "crc32", "matmul"], n),
        "block": rng.choice(["Fetch", "LSU", "Execute"], n),
        "instruction": rng.choice(["ALU", "NOP", "LOAD"], n),
        "toggle_count": rng.integers(0, 100, n),
        "hamming_dist": rng.integers(0, 40, n),
        "control_activity": rng.integers(0, 2, n),
        "data_bus_activity": rng.uniform(0, 20, n),
        "stall": rng.integers(0, 2, n),
    })
    df["static_power_mW"] = 0.4 + 0.005 * df["toggle_count"]
    df["dynamic_power_mW"] = 0.03 * df["toggle_count"] + 0.02 * df["data_bus_activity"]
    df[TARGET_COL] = df["static_power_mW"] + df["dynamic_power_mW"]
    df["total_check"] = df[TARGET_COL]
    return df


# ---------- data loading & feature split ----------

def test_load_dataset_rejects_missing_target(tmp_path):
    path = tmp_path / "bad.csv"
    pd.DataFrame({"a": [1, 2]}).to_csv(path, index=False)
    with pytest.raises(ValueError, match=TARGET_COL):
        load_dataset(str(path))


def test_split_removes_target_id_and_check_columns(small_df):
    X, y = split_features_target(small_df)
    for col in [TARGET_COL, "cycle_id", "total_check"]:
        assert col not in X.columns
    assert y.name == TARGET_COL
    assert len(X) == len(y) == len(small_df)


def test_split_excludes_leakage_columns_by_default(small_df):
    X, _ = split_features_target(small_df)
    assert not set(LEAKAGE_COLS) & set(X.columns)


def test_split_can_keep_leakage_columns(small_df):
    X, _ = split_features_target(small_df, exclude_leakage=False)
    assert set(LEAKAGE_COLS) <= set(X.columns)


# ---------- preprocessing ----------

def test_preprocessor_output_shape(small_df):
    X, _ = split_features_target(small_df)
    out = build_preprocessor(X).fit_transform(X)
    n_onehot = sum(X[c].nunique() for c in CATEGORICAL_COLS)
    n_numeric = X.shape[1] - len(CATEGORICAL_COLS)
    assert out.shape == (len(X), n_onehot + n_numeric)


def test_preprocessor_scales_numeric_features(small_df):
    X, _ = split_features_target(small_df)
    out = build_preprocessor(X).fit_transform(X)
    out = out.toarray() if hasattr(out, "toarray") else out
    n_onehot = sum(X[c].nunique() for c in CATEGORICAL_COLS)
    numeric = out[:, n_onehot:]
    assert np.allclose(numeric.mean(axis=0), 0, atol=1e-8)
    assert np.allclose(numeric.std(axis=0), 1, atol=1e-8)


def test_preprocessor_ignores_unseen_category(small_df):
    X, _ = split_features_target(small_df)
    prep = build_preprocessor(X).fit(X)
    row = X.iloc[[0]].copy()
    row["benchmark"] = "never_seen_benchmark"
    assert prep.transform(row).shape[0] == 1  # no exception


# ---------- models ----------

def test_build_pipeline_rejects_unknown_model(small_df):
    X, _ = split_features_target(small_df)
    with pytest.raises(ValueError, match="Unknown model"):
        build_pipeline("XGBoost", X)


@pytest.mark.parametrize("model_name", sorted(MODEL_FACTORIES))
def test_every_model_trains_and_predicts(small_df, model_name):
    pipe, metrics = train_and_evaluate(small_df, model_name)
    X, _ = split_features_target(small_df)
    preds = pipe.predict(X.head(5))
    assert preds.shape == (5,)
    assert np.all(np.isfinite(preds))
    assert set(metrics) == {"r2", "mae", "rmse"}


def test_random_forest_learns_signal(small_df):
    _, metrics = train_and_evaluate(small_df, "Random Forest")
    assert metrics["r2"] > 0.9


def test_training_is_reproducible(small_df):
    _, m1 = train_and_evaluate(small_df, "Decision Tree")
    _, m2 = train_and_evaluate(small_df, "Decision Tree")
    assert m1 == m2


def test_saved_model_gives_same_predictions(small_df, tmp_path):
    pipe, _ = train_and_evaluate(small_df, "Decision Tree")
    X, _ = split_features_target(small_df)
    path = tmp_path / "model.pkl"
    with open(path, "wb") as f:
        pickle.dump(pipe, f)
    with open(path, "rb") as f:
        loaded = pickle.load(f)
    np.testing.assert_allclose(pipe.predict(X), loaded.predict(X))


# ---------- metrics ----------

def test_evaluate_perfect_predictions():
    y = np.array([1.0, 2.0, 3.0])
    assert evaluate(y, y) == {"r2": 1.0, "mae": 0.0, "rmse": 0.0}


def test_evaluate_known_errors():
    m = evaluate(np.array([0.0, 0.0]), np.array([1.0, -3.0]))
    assert m["mae"] == pytest.approx(2.0)
    assert m["rmse"] == pytest.approx(np.sqrt(5.0))


# ---------- GUI schema helpers ----------

@pytest.mark.parametrize("values, expected", [
    ([0, 1, 1, 0], True),
    (["yes", "no"], True),
    ([True, False], True),
    ([0, 1, 2], False),
    (["ALU", "NOP"], False),
])
def test_is_boolean_series(values, expected):
    assert is_boolean_series(pd.Series(values)) is expected


def test_detect_schema_types(small_df):
    schema, features = detect_schema(small_df)
    types = {item["name"]: item["type"] for item in schema}
    assert types["benchmark"] == "category"
    assert types["stall"] == "bool"
    assert types["data_bus_activity"] == "number"
    assert TARGET_COL not in features


# ---------- checks on the real dataset ----------

real_data = pytest.mark.skipif(not os.path.exists(DATASET), reason="dataset not present")


@real_data
def test_real_dataset_schema():
    df = load_dataset(DATASET)
    expected = {"benchmark", "block", "instruction", "toggle_count", "hamming_dist",
                "control_activity", "data_bus_activity", "stall", *LEAKAGE_COLS, TARGET_COL}
    assert expected <= set(df.columns)
    assert df[TARGET_COL].notna().all()
    assert (df[TARGET_COL] >= 0).all()


@real_data
def test_real_dataset_target_is_sum_of_components():
    """Documents why LEAKAGE_COLS are excluded from the features."""
    df = load_dataset(DATASET)
    np.testing.assert_allclose(
        df[TARGET_COL], df["static_power_mW"] + df["dynamic_power_mW"], atol=1e-9)
