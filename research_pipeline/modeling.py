from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import shap
from matplotlib import pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBRegressor

from research_pipeline.config import PipelineConfig

BASE_FEATURES = [
    "asset",
    "log_return",
    "realized_volatility",
    "volume_change",
    "range_pct",
    "close_to_open",
    "vix_lag_1",
    "vix_lag_2",
    "vix_lag_5",
    "oil_volatility_proxy_lag_1",
    "oil_volatility_proxy_lag_2",
    "oil_volatility_proxy_lag_5",
    "gpr_lag_1",
    "gpr_lag_2",
    "gpr_lag_5",
    "wti_price_lag_1",
    "wti_price_lag_2",
    "wti_price_lag_5",
]

POLY_FEATURES = [
    "poly_probability_level",
    "poly_probability_change",
    "poly_probability_volatility",
    "poly_volume_zscore",
    "poly_order_imbalance",
    "poly_daily_volume",
    "poly_trade_count",
    "poly_market_count",
]

SENTIMENT_FEATURES = [
    "sentiment",
    "sentiment_rolling_mean",
    "sentiment_change",
]


@dataclass
class ModelRun:
    name: str
    model: object
    features: list[str]
    predictions: pd.DataFrame
    metrics: pd.DataFrame


def train_test_split_timewise(dataset: pd.DataFrame, config: PipelineConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    unique_dates = dataset["date"].sort_values().drop_duplicates().to_list()
    split_index = int(len(unique_dates) * config.train_fraction)
    split_date = unique_dates[split_index]
    train = dataset.loc[dataset["date"] < split_date].copy()
    test = dataset.loc[dataset["date"] >= split_date].copy()
    return train, test


def _baseline_model() -> Pipeline:
    numeric_features = [feature for feature in BASE_FEATURES if feature != "asset"]
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric_features,
            ),
            (
                "cat",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("encoder", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                ["asset"],
            ),
        ]
    )
    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", LinearRegression()),
        ]
    )


def _augmented_model(config: PipelineConfig) -> XGBRegressor:
    return XGBRegressor(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.04,
        subsample=0.9,
        colsample_bytree=0.85,
        reg_lambda=1.5,
        objective="reg:squarederror",
        random_state=config.random_state,
    )


def evaluate_predictions(name: str, truth: pd.Series, pred: pd.Series, labels: pd.Series | None = None) -> pd.DataFrame:
    rmse = float(np.sqrt(mean_squared_error(truth, pred)))
    rows = [
        {
            "model": name,
            "segment": "overall",
            "rmse": rmse,
            "mae": mean_absolute_error(truth, pred),
        }
    ]
    if labels is not None:
        for regime in ["low", "medium", "high"]:
            mask = labels.eq(regime)
            if mask.sum() == 0:
                continue
            rows.append(
                {
                    "model": name,
                    "segment": regime,
                    "rmse": float(np.sqrt(mean_squared_error(truth[mask], pred[mask]))),
                    "mae": mean_absolute_error(truth[mask], pred[mask]),
                }
            )
    return pd.DataFrame(rows)


def fit_baseline(train: pd.DataFrame, test: pd.DataFrame) -> ModelRun:
    model = _baseline_model()
    model.fit(train[BASE_FEATURES], train["target_forward_volatility"])
    test_pred = model.predict(test[BASE_FEATURES])
    predictions = test[["date", "asset", "regime", "target_forward_volatility"]].copy()
    predictions["prediction"] = test_pred
    metrics = evaluate_predictions(
        "A_baseline", predictions["target_forward_volatility"], predictions["prediction"], predictions["regime"]
    )
    return ModelRun("A_baseline", model, BASE_FEATURES, predictions, metrics)


def fit_augmented(
    name: str, train: pd.DataFrame, test: pd.DataFrame, features: list[str], config: PipelineConfig
) -> ModelRun:
    model = _augmented_model(config)
    feature_frame_train = pd.get_dummies(train[features], columns=["asset"], dummy_na=False)
    feature_frame_test = pd.get_dummies(test[features], columns=["asset"], dummy_na=False)
    feature_frame_test = feature_frame_test.reindex(columns=feature_frame_train.columns, fill_value=0)
    model.fit(feature_frame_train, train["target_forward_volatility"])
    test_pred = model.predict(feature_frame_test)
    predictions = test[["date", "asset", "regime", "target_forward_volatility"]].copy()
    predictions["prediction"] = test_pred
    metrics = evaluate_predictions(
        name, predictions["target_forward_volatility"], predictions["prediction"], predictions["regime"]
    )
    predictions.attrs["feature_columns"] = list(feature_frame_train.columns)
    predictions.attrs["design_matrix"] = feature_frame_test
    return ModelRun(name, model, features, predictions, metrics)


def run_ablation(dataset: pd.DataFrame, config: PipelineConfig) -> tuple[list[ModelRun], pd.DataFrame, pd.DataFrame]:
    train, test = train_test_split_timewise(dataset, config)
    model_a = fit_baseline(train, test)
    model_b = fit_augmented("B_plus_polymarket", train, test, BASE_FEATURES + POLY_FEATURES, config)
    model_c = fit_augmented(
        "C_plus_polymarket_sentiment",
        train,
        test,
        BASE_FEATURES + POLY_FEATURES + SENTIMENT_FEATURES,
        config,
    )
    metrics = pd.concat([model_a.metrics, model_b.metrics, model_c.metrics], ignore_index=True)
    predictions = pd.concat(
        [
            model_a.predictions.assign(model=model_a.name),
            model_b.predictions.assign(model=model_b.name),
            model_c.predictions.assign(model=model_c.name),
        ],
        ignore_index=True,
    )
    return [model_a, model_b, model_c], metrics, predictions


def create_shap_outputs(model_run: ModelRun, config: PipelineConfig) -> pd.DataFrame:
    feature_matrix = model_run.predictions.attrs["design_matrix"]
    feature_names = model_run.predictions.attrs["feature_columns"]
    explainer = shap.TreeExplainer(model_run.model)
    shap_values = explainer.shap_values(feature_matrix)
    mean_abs = np.abs(shap_values).mean(axis=0)
    importance = pd.DataFrame({"feature": feature_names, "mean_abs_shap": mean_abs}).sort_values(
        "mean_abs_shap", ascending=False
    )

    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, feature_matrix, show=False, max_display=15)
    plt.tight_layout()
    plt.savefig(config.plot_dir / "shap_summary.png", dpi=config.plot_dpi, bbox_inches="tight")
    plt.close()

    poly_feature = "poly_probability_level"
    if poly_feature in feature_matrix.columns:
        plt.figure(figsize=(8, 5))
        shap.dependence_plot(poly_feature, shap_values, feature_matrix, show=False, interaction_index=None)
        plt.tight_layout()
        plt.savefig(config.plot_dir / "shap_dependence_polymarket_probability.png", dpi=config.plot_dpi, bbox_inches="tight")
        plt.close()
    return importance
