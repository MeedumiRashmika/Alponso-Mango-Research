"""
Market Prediction Service - SOLID-principled implementation.

Provides full market classification and price prediction from lightweight 7-field input.
Abstractions allow easy testing and extension without modifying the orchestrator.
"""
import os
import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple

from .model_loader import get_models, get_encoders
from .feature_engineering import prepare_features
from ..config import Config


# =============================================================================
# INTERFACE ABSTRACTIONS (SOLID: I - Interface Segregation, D - Dependency Inversion)
# =============================================================================

class MarketClassifierProtocol(ABC):
    """Interface for market classifiers."""

    @abstractmethod
    def predict_market(self, feature_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Predict market classification.

        Returns:
            {
                "market": str,
                "confidence": float (0-1),
                "all_probabilities": {label: float, ...}
            }
        """


class PriceRegressorProtocol(ABC):
    """Interface for price regressors."""

    @abstractmethod
    def predict_prices(self, feature_df: pd.DataFrame) -> Dict[str, float]:
        """
        Predict three price targets.

        Returns:
            {
                "local_price_lkr": float,
                "export_price_usd": float,
                "harvest_3m_price": float
            }
        """


# =============================================================================
# PREPROCESSING HELPER (SOLID: S - Single Responsibility)
# =============================================================================

class PipelinePreprocessor:
    """
    Handles manual preprocessing for sklearn Pipeline objects with version-mismatch issues.
    Extracts the trained OneHotEncoder and numeric column list from a ColumnTransformer,
    then manually applies preprocessing to bypass the broken passthrough reference.

    Single responsibility: sklearn workaround only.
    """

    def transform(self, model, feature_df: pd.DataFrame) -> np.ndarray:
        """
        Manually preprocess features from a Pipeline model.

        Args:
            model: A sklearn Pipeline with 'pre' (ColumnTransformer) and 'clf'/'reg' steps
            feature_df: DataFrame with all required feature columns

        Returns:
            numpy array: [numeric_cols | one_hot_encoded_categorical_cols]
        """
        preprocessor = model.named_steps["pre"]
        numeric_cols, categorical_cols, ohe = self._extract_transformers(preprocessor)

        X_num = feature_df[numeric_cols].values.astype(np.float64)
        X_cat = feature_df[categorical_cols]

        try:
            X_cat_encoded = ohe.transform(X_cat)
        except Exception:
            # Fallback: fit_transform if transform fails
            X_cat_encoded = ohe.fit_transform(X_cat)

        # Handle both sparse and dense outputs
        if hasattr(X_cat_encoded, "toarray"):
            X_cat_encoded = X_cat_encoded.toarray()
        else:
            X_cat_encoded = np.asarray(X_cat_encoded)

        return np.hstack([X_num, X_cat_encoded])

    def _extract_transformers(self, preprocessor) -> Tuple[list, list, Any]:
        """Extract numeric columns, categorical columns, and OHE from ColumnTransformer."""
        numeric_cols = None
        categorical_cols = None
        ohe = None

        for name, transformer, columns in preprocessor.transformers_:
            if name == "num":
                numeric_cols = list(columns)
            elif name == "cat":
                categorical_cols = list(columns)
                ohe = transformer

        if numeric_cols is None or categorical_cols is None or ohe is None:
            raise ValueError(
                "Cannot extract transformer columns and encoders from preprocessor. "
                "Ensure model has 'num' and 'cat' transformers."
            )
        return numeric_cols, categorical_cols, ohe


# =============================================================================
# CONCRETE CLASSIFIERS (SOLID: L - Liskov Substitution, O - Open/Closed)
# =============================================================================

class LocalMarketClassifier(MarketClassifierProtocol):
    """
    Local market classifier using CatBoost.

    Single responsibility: classify local market destination.
    """

    def __init__(self, model, label_encoder, preprocessor: PipelinePreprocessor):
        self._model = model
        self._le = label_encoder
        self._preprocessor = preprocessor

    def predict_market(self, feature_df: pd.DataFrame) -> Dict[str, Any]:
        X = self._preprocessor.transform(self._model, feature_df)
        estimator = self._model.named_steps["clf"]

        raw_class = estimator.predict(X)[0]
        probabilities = estimator.predict_proba(X)[0]
        # Extract scalar - works for both numpy arrays and Python scalars
        try:
            raw_class_int = int(raw_class.item())
        except (AttributeError, TypeError):
            raw_class_int = int(raw_class)
        label = self._le.inverse_transform([raw_class_int])[0]

        return {
            "market": str(label),
            "confidence": round(float(probabilities[int(raw_class_int)]), 4),
            "all_probabilities": {
                str(self._le.classes_[i]): round(float(probabilities[int(i)]), 4)
                for i in range(len(self._le.classes_))
            },
        }


class ExportMarketClassifier(MarketClassifierProtocol):
    """
    Export market classifier using XGBoost.

    Single responsibility: classify export market destination.
    """

    def __init__(self, model, label_encoder, preprocessor: PipelinePreprocessor):
        self._model = model
        self._le = label_encoder
        self._preprocessor = preprocessor

    def predict_market(self, feature_df: pd.DataFrame) -> Dict[str, Any]:
        X = self._preprocessor.transform(self._model, feature_df)
        estimator = self._model.named_steps["clf"]

        raw_class = estimator.predict(X)[0]
        probabilities = estimator.predict_proba(X)[0]
        # Extract scalar - works for both numpy arrays and Python scalars
        try:
            raw_class_int = int(raw_class.item())
        except (AttributeError, TypeError):
            raw_class_int = int(raw_class)
        label = self._le.inverse_transform([raw_class_int])[0]

        return {
            "market": str(label),
            "confidence": round(float(probabilities[int(raw_class_int)]), 4),
            "all_probabilities": {
                str(self._le.classes_[i]): round(float(probabilities[int(i)]), 4)
                for i in range(len(self._le.classes_))
            },
        }


# =============================================================================
# CONCRETE REGRESSORS (SOLID: L, O)
# =============================================================================

class CatBoostPriceRegressor(PriceRegressorProtocol):
    """
    Multi-output price regressor using CatBoost.

    Single responsibility: predict prices using CatBoost.
    """

    def __init__(self, model, preprocessor: PipelinePreprocessor):
        self._model = model
        self._preprocessor = preprocessor

    def predict_prices(self, feature_df: pd.DataFrame) -> Dict[str, float]:
        X = self._preprocessor.transform(self._model, feature_df)
        estimator = self._model.named_steps["reg"]
        prices = estimator.predict(X)[0]

        return {
            "local_price_lkr": round(float(prices[0]), 2),
            "export_price_usd": round(float(prices[1]), 4),
            "harvest_3m_price": round(float(prices[2]), 2),
        }


class XGBoostPriceRegressor(PriceRegressorProtocol):
    """
    Multi-output price regressor using XGBoost.

    Single responsibility: predict prices using XGBoost.
    """

    def __init__(self, model, preprocessor: PipelinePreprocessor):
        self._model = model
        self._preprocessor = preprocessor

    def predict_prices(self, feature_df: pd.DataFrame) -> Dict[str, float]:
        X = self._preprocessor.transform(self._model, feature_df)
        estimator = self._model.named_steps["reg"]
        prices = estimator.predict(X)[0]

        return {
            "local_price_lkr": round(float(prices[0]), 2),
            "export_price_usd": round(float(prices[1]), 4),
            "harvest_3m_price": round(float(prices[2]), 2),
        }


# =============================================================================
# HISTORICAL DATA PROVIDER (SOLID: S)
# =============================================================================

class HistoricalDataProvider:
    """
    Loads and caches historical mango price data.

    Single responsibility: provide historical context for feature engineering.
    """

    def __init__(self, data_file_path: str):
        self._path = data_file_path
        self._cache: Optional[pd.DataFrame] = None

    def get(self) -> Optional[pd.DataFrame]:
        """Load (or return cached) historical DataFrame."""
        if self._cache is not None:
            return self._cache

        if not os.path.exists(self._path):
            return None

        try:
            df = pd.read_csv(self._path)
            df["Date"] = pd.to_datetime(df["Date"])
            self._cache = df
            return self._cache
        except Exception as e:
            print(f"[HistoricalDataProvider] Failed to load {self._path}: {e}")
            return None

    def get_last_prices_for_region(self, region: str) -> Dict[str, float]:
        """
        Get the most recent prices for a given region.
        Falls back to dataset medians if region not found.

        Returns:
            {
                "Local_Price_LKR": float,
                "Export_Price_USD": float,
                "Harvesting_after_3months_price": float
            }
        """
        hist = self.get()
        if hist is None:
            return {
                "Local_Price_LKR": 0.0,
                "Export_Price_USD": 0.0,
                "Harvesting_after_3months_price": 0.0,
            }

        region_data = hist[hist["Region"] == region]
        if len(region_data) > 0:
            last = region_data.sort_values("Date").iloc[-1]
        else:
            # Fallback: use dataset medians
            last = hist

        try:
            local_price = (
                float(last["Local_Price_LKR"].median())
                if hasattr(last["Local_Price_LKR"], "median")
                else float(last["Local_Price_LKR"])
            )
            export_price = (
                float(last["Export_Price_USD"].median())
                if hasattr(last["Export_Price_USD"], "median")
                else float(last["Export_Price_USD"])
            )
            harvest_price = (
                float(last["Harvesting_after_3months_price"].median())
                if hasattr(last["Harvesting_after_3months_price"], "median")
                else float(last["Harvesting_after_3months_price"])
            )
        except Exception as e:
            print(f"[HistoricalDataProvider] Error extracting prices: {e}")
            local_price = export_price = harvest_price = 0.0

        return {
            "Local_Price_LKR": local_price,
            "Export_Price_USD": export_price,
            "Harvesting_after_3months_price": harvest_price,
        }


# =============================================================================
# REQUEST ROW BUILDER (SOLID: S)
# =============================================================================

class MarketRequestRowBuilder:
    """
    Enriches a 7-field request with price data from history.

    Single responsibility: build complete DataFrame row from lightweight input.
    """

    def __init__(self, historical_provider: HistoricalDataProvider):
        self._provider = historical_provider

    def build(self, normalized_input: dict) -> pd.DataFrame:
        """
        Take 7 fields and enrich with prices from history.

        Args:
            normalized_input: validated dict with Date, Region, etc.

        Returns:
            1-row DataFrame ready for feature engineering.
        """
        row = normalized_input.copy()
        prices = self._provider.get_last_prices_for_region(row["Region"])
        row.update(prices)
        return pd.DataFrame([row])


# =============================================================================
# RESPONSE BUILDER (SOLID: S)
# =============================================================================

class MarketResponseBuilder:
    """
    Assembles the final API response from prediction results.

    Single responsibility: format and structure the output.
    """

    def build(
        self,
        input_data: dict,
        local_result: Dict[str, Any],
        export_result: Dict[str, Any],
        catboost_prices: Dict[str, float],
        xgboost_prices: Dict[str, float],
    ) -> dict:
        """Build the complete response JSON."""
        avg_local = round(
            (catboost_prices["local_price_lkr"] + xgboost_prices["local_price_lkr"]) / 2,
            2,
        )
        avg_export = round(
            (catboost_prices["export_price_usd"] + xgboost_prices["export_price_usd"]) / 2,
            4,
        )

        recommendation = self._recommend(
            local_result, export_result, avg_local, avg_export
        )

        return {
            "input": {
                "date": input_data["Date"],
                "region": input_data["Region"],
                "mango_age_days": input_data["Mango_Age_Days"],
                "days_to_maturity": input_data["Days_To_Maturity"],
                "temp_c": input_data["Temp_C"],
                "humidity_percent": input_data["Humidity_%"],
                "weather": input_data["weather"],
            },
            "market_classification": {
                "local_market": local_result,
                "export_market": export_result,
            },
            "price_predictions": {
                "catboost": catboost_prices,
                "xgboost": xgboost_prices,
                "ensemble_avg": {
                    "local_price_lkr": avg_local,
                    "export_price_usd": avg_export,
                },
            },
            "recommendation": recommendation,
        }

    def _recommend(
        self,
        local: Dict[str, Any],
        export: Dict[str, Any],
        avg_local_lkr: float,
        avg_export_usd: float,
    ) -> dict:
        """Generate a market recommendation based on confidence and prices."""
        if export["confidence"] >= 0.6 and avg_export_usd > 1.2:
            channel = "export"
            note = "High confidence in export market. Prioritise export channels."
        elif local["confidence"] >= 0.6:
            channel = "local"
            note = "Local market shows strong confidence. Distribute locally."
        else:
            channel = "undecided"
            note = "Model confidence is low. Consider waiting for better market conditions."

        return {
            "suggested_channel": channel,
            "note": note,
            "local_market_confidence": local["confidence"],
            "export_market_confidence": export["confidence"],
        }


# =============================================================================
# ORCHESTRATOR (SOLID: D - Dependency Inversion)
# =============================================================================

class MarketPredictionOrchestrator:
    """
    Coordinates market prediction for a single request.

    SOLID-D: Depends on abstractions (MarketClassifierProtocol, PriceRegressorProtocol),
             not concrete implementations. All dependencies injected via constructor.
    SOLID-S: Only orchestrates; knows nothing about sklearn, filesystems, or formatting.
    SOLID-O: New model types added by extending protocols and updating factory, not this class.
    """

    def __init__(
        self,
        row_builder: MarketRequestRowBuilder,
        local_classifier: MarketClassifierProtocol,
        export_classifier: MarketClassifierProtocol,
        catboost_regressor: PriceRegressorProtocol,
        xgboost_regressor: PriceRegressorProtocol,
        historical_provider: HistoricalDataProvider,
        response_builder: MarketResponseBuilder,
    ):
        self._row_builder = row_builder
        self._local_clf = local_classifier
        self._export_clf = export_classifier
        self._cat_reg = catboost_regressor
        self._xgb_reg = xgboost_regressor
        self._history = historical_provider
        self._response_builder = response_builder

    def predict(self, normalized_input: dict) -> dict:
        """
        Execute the full market prediction pipeline.

        Args:
            normalized_input: validated dict from MarketPredictionRequest schema

        Returns:
            Structured response dict with market classifications and price predictions

        Raises:
            ValueError: if input data is invalid
            RuntimeError: if models fail to predict
        """
        # 1. Build complete row with price columns (needed for lag features)
        input_row_df = self._row_builder.build(normalized_input)

        # 2. Load historical data for feature engineering (lags, rolling means, etc.)
        historical_df = self._history.get()

        # 3. Engineer all 39 features (reuses existing prepare_features function)
        feature_df, _ = prepare_features(input_row_df, historical_df=historical_df)

        # 4. Run all 4 models in parallel (conceptually)
        local_result = self._local_clf.predict_market(feature_df)
        export_result = self._export_clf.predict_market(feature_df)
        catboost_prices = self._cat_reg.predict_prices(feature_df)
        xgboost_prices = self._xgb_reg.predict_prices(feature_df)

        # 5. Build and return structured response
        return self._response_builder.build(
            normalized_input,
            local_result,
            export_result,
            catboost_prices,
            xgboost_prices,
        )


# =============================================================================
# FACTORY FUNCTION (SOLID: D - Composition Root)
# =============================================================================

def build_market_orchestrator() -> MarketPredictionOrchestrator:
    """
    Factory: constructs the MarketPredictionOrchestrator with all dependencies resolved.

    This is the ONLY place where concrete classes are instantiated (new LocalMarketClassifier, etc.).
    To add a new model type, update only this function and the model_loader.

    Raises:
        RuntimeError: if required models are not loaded
    """
    models = get_models()
    encoders = get_encoders()

    # Verify all required models are loaded
    required = ["local", "export", "multi_cat", "multi_xgb"]
    missing = [k for k in required if k not in models]
    if missing:
        raise RuntimeError(f"Required models not loaded: {missing}")

    required_encoders = ["local", "export"]
    missing_encoders = [k for k in required_encoders if k not in encoders]
    if missing_encoders:
        raise RuntimeError(f"Required encoders not loaded: {missing_encoders}")

    # Instantiate dependencies
    preprocessor = PipelinePreprocessor()
    historical_provider = HistoricalDataProvider(data_file_path=Config.DATA_FILE)

    # Wire everything together
    return MarketPredictionOrchestrator(
        row_builder=MarketRequestRowBuilder(historical_provider),
        local_classifier=LocalMarketClassifier(
            model=models["local"],
            label_encoder=encoders["local"],
            preprocessor=preprocessor,
        ),
        export_classifier=ExportMarketClassifier(
            model=models["export"],
            label_encoder=encoders["export"],
            preprocessor=preprocessor,
        ),
        catboost_regressor=CatBoostPriceRegressor(
            model=models["multi_cat"],
            preprocessor=preprocessor,
        ),
        xgboost_regressor=XGBoostPriceRegressor(
            model=models["multi_xgb"],
            preprocessor=preprocessor,
        ),
        historical_provider=historical_provider,
        response_builder=MarketResponseBuilder(),
    )
