from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBRegressor

from research_pipeline.config import PipelineConfig


@dataclass
class ModelSpec:
    name: str
    estimator_name: str
    feature_names: list[str]


def estimator_catalog(config: PipelineConfig) -> dict[str, object]:
    return {
        "linear": LinearRegression(),
        "random_forest": RandomForestRegressor(
            n_estimators=250,
            max_depth=10,
            min_samples_leaf=5,
            n_jobs=-1,
            random_state=config.random_state,
        ),
        "xgboost": XGBRegressor(
            n_estimators=350,
            max_depth=4,
            learning_rate=0.04,
            subsample=0.9,
            colsample_bytree=0.85,
            reg_lambda=1.5,
            objective="reg:squarederror",
            random_state=config.random_state,
        ),
    }


def build_pipeline(estimator_name: str, feature_names: list[str], config: PipelineConfig) -> Pipeline:
    estimator = estimator_catalog(config)[estimator_name]
    numeric_features = [column for column in feature_names if column != "asset"]
    categorical_features = [column for column in feature_names if column == "asset"]

    numeric_steps = [("imputer", SimpleImputer(strategy="median"))]
    if estimator_name == "linear":
        numeric_steps.append(("scaler", StandardScaler()))

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", Pipeline(steps=numeric_steps), numeric_features),
            (
                "cat",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("encoder", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical_features,
            ),
        ],
        remainder="drop",
    )
    return Pipeline([("preprocessor", preprocessor), ("model", estimator)])


def fit_predict(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_names: list[str],
    estimator_name: str,
    config: PipelineConfig,
) -> tuple[Pipeline, np.ndarray]:
    pipeline = build_pipeline(estimator_name, feature_names, config)
    pipeline.fit(train_df[feature_names], train_df["target_forward_volatility"])
    prediction = pipeline.predict(test_df[feature_names])
    return pipeline, prediction


def transformed_matrix(pipeline: Pipeline, frame: pd.DataFrame, feature_names: list[str]) -> tuple[np.ndarray, list[str]]:
    matrix = pipeline.named_steps["preprocessor"].transform(frame[feature_names])
    output_names = pipeline.named_steps["preprocessor"].get_feature_names_out().tolist()
    return matrix, output_names
