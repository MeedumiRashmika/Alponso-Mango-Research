import os
import joblib
import warnings

warnings.filterwarnings("ignore")

# Mac M1/M2 XGBoost fix
os.environ["DYLD_LIBRARY_PATH"] = "/opt/homebrew/Cellar/libomp/21.1.3/lib"

_models = {}
_encoders = {}
_loaded = False


def load_models(model_dir: str) -> dict:
    global _models, _encoders, _loaded

    if _loaded:
        return {"models": _models, "encoders": _encoders}

    model_files = {
        "local": f"{model_dir}/local_catboost_weighted.joblib",
        "export": f"{model_dir}/export_xgboost.joblib",
        "multi_cat": f"{model_dir}/regressor_cat_multi.joblib",
        "multi_xgb": f"{model_dir}/regressor_xgb_multi.joblib",
    }

    encoder_files = {
        "local": f"{model_dir}/label_encoder_local.joblib",
        "export": f"{model_dir}/label_encoder_export.joblib",
    }

    for name, path in model_files.items():
        try:
            _models[name] = joblib.load(path)
        except Exception as e:
            print(f"[ModelLoader] Failed to load model '{name}': {e}")

    for name, path in encoder_files.items():
        try:
            _encoders[name] = joblib.load(path)
        except Exception as e:
            print(f"[ModelLoader] Failed to load encoder '{name}': {e}")

    _loaded = True
    return {"models": _models, "encoders": _encoders}


def get_models():
    return _models


def get_encoders():
    return _encoders