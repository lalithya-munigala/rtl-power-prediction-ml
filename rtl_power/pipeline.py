"""Core, GUI-free logic for RTL power prediction.

Kept separate from the Tkinter dashboard so it can be imported and unit-tested.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor

TARGET_COL = "total_power_mW"
ID_COLS = ["cycle_id"]
EXCLUDE_COLS = ["total_check"]
CATEGORICAL_COLS = ["benchmark", "block", "instruction"]
# total_power_mW == static_power_mW + dynamic_power_mW, so feeding both
# components to a model leaks the target.
LEAKAGE_COLS = ["static_power_mW", "dynamic_power_mW"]
LOW_CARDINALITY_THRESHOLD = 20

MODEL_FACTORIES = {
    "SVR": lambda: SVR(),
    "KNN": lambda: KNeighborsRegressor(n_neighbors=5),
    "Decision Tree": lambda: DecisionTreeRegressor(random_state=42),
    "Random Forest": lambda: RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
    "MLP": lambda: MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=300, random_state=42),
}


def load_dataset(path: str) -> pd.DataFrame:
    """Read the dataset and check that the target column is present."""
    df = pd.read_csv(path)
    if TARGET_COL not in df.columns:
        raise ValueError(f"Target column '{TARGET_COL}' missing from {path}")
    return df


def split_features_target(df: pd.DataFrame, exclude_leakage: bool = True):
    """Return (X, y) with ID, check, target and (optionally) leaking columns removed."""
    drop = [TARGET_COL] + ID_COLS + EXCLUDE_COLS
    if exclude_leakage:
        drop += LEAKAGE_COLS
    X = df.drop(columns=[c for c in drop if c in df.columns])
    y = df[TARGET_COL]
    return X, y


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    """One-hot encode categorical columns and standard-scale numeric ones."""
    cat_cols = [c for c in CATEGORICAL_COLS if c in X.columns]
    num_cols = [c for c in X.columns if c not in cat_cols]
    return ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
        ("num", StandardScaler(), num_cols),
    ])


def build_pipeline(model_name: str, X: pd.DataFrame) -> Pipeline:
    """Preprocessing + regressor in one object, so training and inference match."""
    if model_name not in MODEL_FACTORIES:
        raise ValueError(f"Unknown model '{model_name}'. Choose from {sorted(MODEL_FACTORIES)}")
    return Pipeline([
        ("preprocess", build_preprocessor(X)),
        ("model", MODEL_FACTORIES[model_name]()),
    ])


def evaluate(y_true, y_pred) -> dict:
    """R2, MAE and RMSE."""
    return {
        "r2": float(r2_score(y_true, y_pred)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
    }


def train_and_evaluate(df: pd.DataFrame, model_name: str, test_size: float = 0.2,
                       random_state: int = 42, exclude_leakage: bool = True):
    """Split, fit one model and return (fitted pipeline, test metrics)."""
    X, y = split_features_target(df, exclude_leakage=exclude_leakage)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=test_size, random_state=random_state)
    pipe = build_pipeline(model_name, X_tr)
    pipe.fit(X_tr, y_tr)
    return pipe, evaluate(y_te, pipe.predict(X_te))


def is_boolean_series(s: pd.Series) -> bool:
    """True if the column only holds 0/1, true/false or yes/no values."""
    if s.dtype == bool:
        return True
    vals = {str(v).lower() for v in s.dropna().unique()}
    return bool(vals) and vals.issubset({"0", "1", "true", "false", "yes", "no"})


def detect_schema(df: pd.DataFrame, exclude_leakage: bool = True):
    """Describe each input feature (bool / category / number) for the GUI form."""
    X, _ = split_features_target(df, exclude_leakage=exclude_leakage)
    schema = []
    for col in X.columns:
        s = X[col]
        if is_boolean_series(s):
            default = str(int(pd.to_numeric(s, errors="coerce").fillna(0).median()))
            schema.append({"name": col, "type": "bool", "values": ["0", "1"], "default": default})
        elif not pd.api.types.is_numeric_dtype(s) or s.nunique(dropna=True) <= LOW_CARDINALITY_THRESHOLD:
            cats = sorted(pd.Series(s.dropna().unique()).astype(str).tolist()) or [""]
            schema.append({"name": col, "type": "category", "values": cats, "default": cats[0]})
        else:
            med = pd.to_numeric(s, errors="coerce").median()
            default = "" if np.isnan(med) else str(round(float(med), 6))
            schema.append({"name": col, "type": "number", "values": None, "default": default})
    return schema, list(X.columns)
