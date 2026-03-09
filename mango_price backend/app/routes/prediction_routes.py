from flask import Blueprint, request, jsonify
import pandas as pd

from ..schemas.prediction_schema import validate_batch_payload
from ..schemas.market_schema import validate_market_payload
from ..services.prediction_service import predict
from ..services.market_prediction_service import build_market_orchestrator

prediction_bp = Blueprint("prediction", __name__)


@prediction_bp.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "service": "mango-price-prediction"}), 200


@prediction_bp.route("/predict", methods=["POST"])
def predict_prices():
    """
    Predict local and export mango prices for a batch of records.
    """
    payload = request.get_json(silent=True)
    if payload is None:
        return jsonify({"error": "Invalid JSON body"}), 400

    records, errors = validate_batch_payload(payload)
    if errors:
        return jsonify({"errors": errors}), 422

    try:
        df = pd.DataFrame(records)
        result = predict(df)
        return jsonify(result), 200
    except RuntimeError as e:
        return jsonify({"error": str(e)}), 503
    except Exception as e:
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500


@prediction_bp.route("/predict/single", methods=["POST"])
def predict_single():
    """
    Predict prices for a single mango record.
    """
    payload = request.get_json(silent=True)
    if payload is None:
        return jsonify({"error": "Invalid JSON body"}), 400

    # Wrap single record into batch format
    wrapped, errors = validate_batch_payload({"records": [payload]})
    if errors:
        return jsonify({"errors": errors}), 422

    try:
        df = pd.DataFrame(wrapped)
        print(f"[DebugPredictSingle] DataFrame columns: {df.columns.tolist()}")
        print(f"[DebugPredictSingle] DataFrame shape: {df.shape}")
        result = predict(df)
        print(f"[DebugPredictSingle] Prediction result: {result}")
        # Return first record directly
        response = {
            "prediction": result["predictions"][0] if result["predictions"] else {},
            "summary": result["summary"],
        }
        return jsonify(response), 200
    except RuntimeError as e:
        return jsonify({"error": str(e)}), 503
    except Exception as e:
        print(f"[ErrorPredictSingle] {str(e)}", exc_info=True)
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500


@prediction_bp.route("/models/status", methods=["GET"])
def model_status():
    """
    Check which models are currently loaded.
    """
    from ..services.model_loader import get_models, get_encoders
    models = get_models()
    encoders = get_encoders()
    return jsonify({
        "models_loaded": list(models.keys()),
        "encoders_loaded": list(encoders.keys()),
        "total_models": len(models),
    }), 200


# =============================================================================
# MARKET PREDICTION ENDPOINT - SOLID Design
# =============================================================================

# Module-level lazy-initialized orchestrator (built once, reused for all requests)
_market_orchestrator = None


def _get_market_orchestrator():
    """Lazy initialization of the orchestrator on first request."""
    global _market_orchestrator
    if _market_orchestrator is None:
        _market_orchestrator = build_market_orchestrator()
    return _market_orchestrator


@prediction_bp.route("/predict/market", methods=["POST"])
def predict_market():
    """
    Predict market classification and prices from 7 lightweight input fields.

    No need to supply actual prices - they are inferred from historical data by Region.
    This endpoint returns:
    - Local market classification (Colombo/Dambulla/Pettah) with probabilities
    - Export market classification (Dubai/London/Paris/Doha) with probabilities
    - Price predictions from both CatBoost and XGBoost regressors
    - Ensemble average and market recommendation

    Request body:
    {
        "Date": "2025-11-15",
        "Mango_Age_Days": 40,
        "Days_To_Maturity": 40,
        "Temp_C": 30.2,
        "Humidity_percent": 72,
        "Region": "Colombo",
        "weather": "Clear"
    }
    """
    payload = request.get_json(silent=True)
    if payload is None:
        return jsonify({"error": "Invalid or missing JSON body"}), 400

    # Validate 7-field input
    normalized, errors = validate_market_payload(payload)
    if errors:
        return jsonify({"errors": errors}), 422

    try:
        # Get or initialize the orchestrator
        orchestrator = _get_market_orchestrator()
        # Run full prediction pipeline
        result = orchestrator.predict(normalized)
        return jsonify(result), 200
    except RuntimeError as e:
        # Models not loaded
        return jsonify({"error": str(e)}), 503
    except Exception as e:
        # Unexpected error during prediction
        import traceback
        print(f"[ErrorMarketPredict] {str(e)}")
        traceback.print_exc()
        return jsonify({"error": f"Market prediction failed: {str(e)}"}), 500