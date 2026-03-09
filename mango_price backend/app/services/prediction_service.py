import numpy as np
import pandas as pd
from typing import Any, Optional, Dict
import os
from sklearn.preprocessing import OneHotEncoder

from .model_loader import get_models
from .feature_engineering import prepare_features
from ..config import Config


def _predict_single_model(model, feature_df: pd.DataFrame, key: str) -> Optional[np.ndarray]:
    """
    Predict using a model, handling the sklearn serialization issue by manually
    applying preprocessing to bypass the broken ColumnTransformer.
    """
    try:
        print(f"[PredictionService] Predicting with '{key}' model. Features: {feature_df.columns.tolist()}")
        print(f"[PredictionService] Feature shape: {feature_df.shape}")

        # Extract the preprocessor and estimator from the pipeline
        if hasattr(model, "named_steps") and "pre" in model.named_steps:
            # It's a Pipeline - we need to preprocess manually
            preprocessor = model.named_steps["pre"]
            estimator = model.named_steps.get("clf") or model.named_steps.get("reg")

            # Apply preprocessing manually
            try:
                # Get numeric and categorical column names from the preprocessor
                numeric_cols = None
                categorical_cols = None
                ohe = None

                if hasattr(preprocessor, "transformers_"):
                    for name, transformer, columns in preprocessor.transformers_:
                        if name == "num":
                            numeric_cols = list(columns)
                        elif name == "cat":
                            categorical_cols = list(columns)
                            ohe = transformer

                if numeric_cols is None or categorical_cols is None:
                    raise ValueError("Could not extract numeric/categorical columns from preprocessor")

                # Prepare numeric features (passthrough)
                X_num = feature_df[numeric_cols].values.astype(np.float64)

                # Prepare categorical features (OneHotEncode)
                X_cat = feature_df[categorical_cols]
                if ohe is not None:
                    try:
                        X_cat_encoded = ohe.transform(X_cat)
                        # Handle both sparse and dense output
                        if hasattr(X_cat_encoded, 'toarray'):
                            X_cat_encoded = X_cat_encoded.toarray()
                        else:
                            X_cat_encoded = np.asarray(X_cat_encoded)
                    except Exception as e:
                        print(f"[PredictionService] OneHotEncoder transform failed: {e}, trying fit_transform")
                        X_cat_encoded = ohe.fit_transform(X_cat)
                        if hasattr(X_cat_encoded, 'toarray'):
                            X_cat_encoded = X_cat_encoded.toarray()
                        else:
                            X_cat_encoded = np.asarray(X_cat_encoded)
                else:
                    X_cat_encoded = X_cat.values

                # Combine
                X_processed = np.hstack([X_num, X_cat_encoded])

                # Predict
                pred = estimator.predict(X_processed)
                print(f"[PredictionService] '{key}' prediction succeeded: {pred}")
                return pred.flatten()

            except Exception as preprocess_error:
                print(f"[PredictionService] Manual preprocessing failed: {preprocess_error}")
                raise
        else:
            # Not a pipeline, try direct prediction
            pred = model.predict(feature_df)
            print(f"[PredictionService] '{key}' prediction succeeded: {pred}")
            return pred.flatten()

    except Exception as e:
        print(f"[PredictionService] Prediction failed for '{key}': {type(e).__name__}: {e}", flush=True)
        import traceback
        traceback.print_exc()
        return None


def predict(input_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Run all available models on input_df and return structured predictions.
    Loads historical data for proper feature engineering (lags, rolling averages).
    """
    models = get_models()
    if not models:
        raise RuntimeError("No models are loaded. Check model files and MODEL_DIR config.")

    # Load historical data if available for proper lag/rolling feature calculation
    historical_df = None
    if hasattr(Config, "DATA_FILE") and Config.DATA_FILE:
        try:
            data_path = Config.DATA_FILE
            if os.path.exists(data_path):
                print(f"[PredictionService] Loading historical data from: {data_path}")
                historical_df = pd.read_csv(data_path)
                print(f"[PredictionService] Loaded {len(historical_df)} historical records")
            else:
                print(f"[PredictionService] Data file not found at {data_path}, proceeding without historical data")
        except Exception as e:
            print(f"[PredictionService] Failed to load historical data: {e}, proceeding without it")

    feature_df, original_df = prepare_features(input_df, historical_df=historical_df)

    raw_preds: Dict[str, np.ndarray] = {}

    if "local" in models:
        result = _predict_single_model(models["local"], feature_df, "local")
        if result is not None:
            raw_preds["local_predicted"] = result

    if "export" in models:
        result = _predict_single_model(models["export"], feature_df, "export")
        if result is not None:
            raw_preds["export_predicted"] = result

    if "multi_cat" in models:
        result = _predict_single_model(models["multi_cat"], feature_df, "multi_cat")
        if result is not None:
            if result.ndim > 1 or (len(result.shape) == 1 and len(result) > 1):
                if len(result.shape) > 1:
                    raw_preds["multi_cat_local"] = result[:, 0].flatten()
                    raw_preds["multi_cat_export"] = result[:, 1].flatten()
                else:
                    raw_preds["multi_cat"] = result.flatten()

    if "multi_xgb" in models:
        result = _predict_single_model(models["multi_xgb"], feature_df, "multi_xgb")
        if result is not None:
            if result.ndim > 1 or (len(result.shape) == 1 and len(result) > 1):
                if len(result.shape) > 1:
                    raw_preds["multi_xgb_local"] = result[:, 0].flatten()
                    raw_preds["multi_xgb_export"] = result[:, 1].flatten()
                else:
                    raw_preds["multi_xgb"] = result.flatten()

    return _build_response(original_df, raw_preds)


def _build_response(df: pd.DataFrame, raw_preds: dict) -> dict:
    records = []
    n = len(df)

    for i in range(n):
        row = {
            "date": str(df["Date"].iloc[i].date()),
            "region": str(df["Region"].iloc[i]),
            "mango_age_days": int(df["Mango_Age_Days"].iloc[i]),
            "actual_local_price_lkr": round(float(df["Local_Price_LKR"].iloc[i]), 2),
            "actual_export_price_usd": round(float(df["Export_Price_USD"].iloc[i]), 4),
        }

        if "local_predicted" in raw_preds:
            row["predicted_local_price_lkr"] = round(float(raw_preds["local_predicted"][i]), 2)

        if "export_predicted" in raw_preds:
            row["predicted_export_price_usd"] = round(float(raw_preds["export_predicted"][i]), 4)

        if "Harvesting_after_3months_price" in df.columns:
            harv_value = df["Harvesting_after_3months_price"].iloc[i]
            if pd.notna(harv_value):
                row["forecast_3month_price_lkr"] = round(float(harv_value), 2)
            else:
                row["forecast_3month_price_lkr"] = None

        records.append(row)

    summary = _build_summary(df, raw_preds)
    return {"predictions": records, "summary": summary}


def _build_summary(df: pd.DataFrame, raw_preds: dict) -> dict:
    summary: Dict[str, Any] = {}

    if "local_predicted" in raw_preds:
        local_avg = float(np.mean(raw_preds["local_predicted"]))
        summary["avg_predicted_local_price_lkr"] = round(local_avg, 2)

    if "export_predicted" in raw_preds:
        export_avg = float(np.mean(raw_preds["export_predicted"]))
        summary["avg_predicted_export_price_usd"] = round(export_avg, 4)

    if "local_predicted" in raw_preds and "export_predicted" in raw_preds:
        # Simple recommendation: export is generally higher value
        # This can be enhanced with USD→LKR conversion rate
        export_avg = float(np.mean(raw_preds["export_predicted"]))
        summary["market_recommendation"] = "export" if export_avg > 1.2 else "local"
        summary["recommendation_note"] = (
            "Export market shows stronger predicted prices. Consider export channels."
            if summary["market_recommendation"] == "export"
            else "Local market prices are competitive. Consider local distribution."
        )

    return summary