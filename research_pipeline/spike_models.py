from __future__ import annotations

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from research_pipeline.config import PipelineConfig


def classifier_catalog(config: PipelineConfig) -> dict[str, object]:
    return {
        "logistic": LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=config.random_state,
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=300,
            max_depth=10,
            min_samples_leaf=4,
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=config.random_state,
        ),
        "lightgbm": LGBMClassifier(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=5,
            subsample=0.9,
            colsample_bytree=0.85,
            objective="binary",
            class_weight="balanced",
            random_state=config.random_state,
            verbosity=-1,
        ),
    }


def build_classifier(estimator_name: str, feature_names: list[str], config: PipelineConfig) -> Pipeline:
    estimator = classifier_catalog(config)[estimator_name]
    numeric_features = [column for column in feature_names if column != "asset"]
    categorical_features = [column for column in feature_names if column == "asset"]

    numeric_steps = [("imputer", SimpleImputer(strategy="median"))]
    if estimator_name == "logistic":
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


def fit_predict_proba(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_names: list[str],
    label_name: str,
    estimator_name: str,
    config: PipelineConfig,
) -> tuple[Pipeline, np.ndarray]:
    pipeline = build_classifier(estimator_name, feature_names, config)
    pipeline.fit(train_df[feature_names], train_df[label_name])
    probability = pipeline.predict_proba(test_df[feature_names])[:, 1]
    return pipeline, probability


def transformed_matrix(pipeline: Pipeline, frame: pd.DataFrame, feature_names: list[str]) -> tuple[np.ndarray, list[str]]:
    matrix = pipeline.named_steps["preprocessor"].transform(frame[feature_names])
    if hasattr(matrix, "toarray"):
        matrix = matrix.toarray()
    output_names = pipeline.named_steps["preprocessor"].get_feature_names_out().tolist()
    return matrix, output_names
