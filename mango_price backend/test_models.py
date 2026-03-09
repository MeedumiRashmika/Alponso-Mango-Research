#!/usr/bin/env python3
"""
Test script to load saved models and make predictions with dummy data
"""
import os
import json
import joblib
import numpy as np
import pandas as pd
from datetime import datetime

# ===========================
# Model paths
# ===========================
MODEL_DIR = "output/models"
CSV_PATH = "dataset/mango_price_dataset.csv"

print("=" * 80)
print("MANGO PRICE PREDICTION - TEST SCRIPT")
print("=" * 80)

# Check if models exist
print("\n[1] Checking model files...")
models_to_load = {
    "local_classifier": "local_catboost_weighted.joblib",
    "export_classifier": "export_xgboost.joblib",
    "regressor_cat": "regressor_cat_multi.joblib",
    "regressor_xgb": "regressor_xgb_multi.joblib",
    "le_local": "label_encoder_local.joblib",
    "le_export": "label_encoder_export.joblib",
}

for key, filename in models_to_load.items():
    path = os.path.join(MODEL_DIR, filename)
    exists = os.path.exists(path)
    status = "✓" if exists else "✗"
    print(f"  {status} {filename}")

# ===========================
# Load models
# ===========================
print("\n[2] Loading models...")
try:
    local_classifier = joblib.load(os.path.join(MODEL_DIR, "local_catboost_weighted.joblib"))
    export_classifier = joblib.load(os.path.join(MODEL_DIR, "export_xgboost.joblib"))
    regressor_cat = joblib.load(os.path.join(MODEL_DIR, "regressor_cat_multi.joblib"))
    regressor_xgb = joblib.load(os.path.join(MODEL_DIR, "regressor_xgb_multi.joblib"))
    le_local = joblib.load(os.path.join(MODEL_DIR, "label_encoder_local.joblib"))
    le_export = joblib.load(os.path.join(MODEL_DIR, "label_encoder_export.joblib"))
    print("  ✓ All models loaded successfully")
except Exception as e:
    print(f"  ✗ Error loading models: {e}")
    exit(1)

# ===========================
# Feature Engineering Function
# ===========================
def prepare_full_features(df):
    """Create engineered features (same as training notebook)"""
    d = df.copy()
    d['Date'] = pd.to_datetime(d['Date'])
    d = d.sort_values(['Region', 'Date']).reset_index(drop=True)

    # Time-based features
    d['day'] = d['Date'].dt.day
    d['month'] = d['Date'].dt.month
    d['dayofyear'] = d['Date'].dt.dayofyear
    d['year'] = d['Date'].dt.year
    d['month_sin'] = np.sin(2 * np.pi * d['month'] / 12)
    d['month_cos'] = np.cos(2 * np.pi * d['month'] / 12)
    d['doy_sin'] = np.sin(2 * np.pi * d['dayofyear'] / 365)
    d['doy_cos'] = np.cos(2 * np.pi * d['dayofyear'] / 365)

    # Lag features
    d['local_price_lag1'] = d.groupby('Region')['Local_Price_LKR'].shift(1)
    d['export_price_lag1'] = d.groupby('Region')['Export_Price_USD'].shift(1)

    # Rolling means
    d['local_price_roll7'] = d.groupby('Region')['Local_Price_LKR'].transform(lambda s: s.rolling(7, min_periods=1).mean())
    d['local_price_roll14'] = d.groupby('Region')['Local_Price_LKR'].transform(lambda s: s.rolling(14, min_periods=1).mean())
    d['export_price_roll7'] = d.groupby('Region')['Export_Price_USD'].transform(lambda s: s.rolling(7, min_periods=1).mean())
    d['export_price_roll14'] = d.groupby('Region')['Export_Price_USD'].transform(lambda s: s.rolling(14, min_periods=1).mean())

    # Momentum
    d['local_price_mom1'] = d['Local_Price_LKR'] - d['local_price_lag1']
    d['export_price_mom1'] = d['Export_Price_USD'] - d['export_price_lag1']

    # Fill lags/rolling with regional medians
    grp_med_local = d.groupby('Region')['Local_Price_LKR'].transform('median')
    grp_med_export = d.groupby('Region')['Export_Price_USD'].transform('median')

    for col in ['local_price_lag1', 'local_price_roll7', 'local_price_roll14']:
        d[col] = d[col].fillna(grp_med_local).fillna(d['Local_Price_LKR'].median())
    for col in ['export_price_lag1', 'export_price_roll7', 'export_price_roll14']:
        d[col] = d[col].fillna(grp_med_export).fillna(d['Export_Price_USD'].median())
    d['local_price_mom1'] = d['local_price_mom1'].fillna(0)
    d['export_price_mom1'] = d['export_price_mom1'].fillna(0)

    # New engineered features
    d['local_price_region_mean'] = d.groupby('Region')['Local_Price_LKR'].transform('mean')
    d['price_dev'] = d['Local_Price_LKR'] - d['local_price_region_mean']
    d['price_to_age_ratio'] = d['Local_Price_LKR'] / (d['Mango_Age_Days'].replace(0, 1))
    d['local_price_vol7'] = d.groupby('Region')['Local_Price_LKR'].transform(lambda s: s.rolling(7, min_periods=1).std()).fillna(0)
    d['local_price_vol14'] = d.groupby('Region')['Local_Price_LKR'].transform(lambda s: s.rolling(14, min_periods=1).std()).fillna(0)
    d['local_mom_region_mean'] = d.groupby('Region')['local_price_mom1'].transform('mean')
    d['momentum_dev'] = d['local_price_mom1'] - d['local_mom_region_mean']

    # Interactions
    d['dev_x_ratio'] = d['price_dev'] * d['price_to_age_ratio']
    d['vol_x_mom'] = d['local_price_vol14'] * d['momentum_dev']
    d['age_x_price'] = d['Mango_Age_Days'] * d['local_price_lag1']
    d['roll14_x_mom'] = d['local_price_roll14'] * d['local_price_mom1']

    # Fill remaining NaNs
    fill_cols = ['price_dev', 'price_to_age_ratio', 'local_price_vol7', 'local_price_vol14',
                 'momentum_dev', 'dev_x_ratio', 'vol_x_mom', 'age_x_price', 'roll14_x_mom']
    for c in fill_cols:
        if c in d.columns:
            d[c] = d[c].fillna(0)

    return d

# ===========================
# Load historical data for lags
# ===========================
print(f"\n[3] Loading historical data from {CSV_PATH}...")
try:
    hist = pd.read_csv(CSV_PATH)
    hist['Date'] = pd.to_datetime(hist['Date'])
    print(f"  ✓ Loaded {len(hist)} historical records")
except Exception as e:
    print(f"  ✗ Error loading CSV: {e}")
    exit(1)

# ===========================
# Create dummy inputs
# ===========================
print("\n[4] Creating dummy test inputs...")

dummy_inputs = [
    {
        "Date": "2025-11-15",
        "Mango_Age_Days": 40,
        "Days_To_Maturity": 40,
        "Temp_C": 30.2,
        "Humidity_%": 72,
        "Region": "Colombo",
        "weather": "Clear"
    },
    {
        "Date": "2025-12-20",
        "Mango_Age_Days": 80,
        "Days_To_Maturity": 15,
        "Temp_C": 28.5,
        "Humidity_%": 65,
        "Region": "Hambantota",
        "weather": "Partly Cloudy"
    },
    {
        "Date": "2025-10-05",
        "Mango_Age_Days": 25,
        "Days_To_Maturity": 60,
        "Temp_C": 31.0,
        "Humidity_%": 75,
        "Region": "Colombo",
        "weather": "Rain"
    }
]

print(f"  ✓ Created {len(dummy_inputs)} test samples")

# ===========================
# Process each input
# ===========================
print("\n[5] Processing predictions...")
print("=" * 80)

all_results = []

for idx, inp in enumerate(dummy_inputs, 1):
    print(f"\n📊 SAMPLE {idx}")
    print("-" * 80)
    print(f"Input: {json.dumps(inp, indent=2)}")

    # Prepare row with dummy prices for lag calculation
    row = inp.copy()
    row['Date'] = pd.to_datetime(row['Date'])

    # Get latest prices from same region
    region_hist = hist[hist['Region'] == row['Region']]
    if len(region_hist) > 0:
        last_row = region_hist.sort_values('Date').iloc[-1]
        row['Local_Price_LKR'] = float(last_row['Local_Price_LKR'])
        row['Export_Price_USD'] = float(last_row['Export_Price_USD'])
        row['Harvesting_after_3months_price'] = float(last_row['Harvesting_after_3months_price'])
    else:
        row['Local_Price_LKR'] = float(hist['Local_Price_LKR'].median())
        row['Export_Price_USD'] = float(hist['Export_Price_USD'].median())
        row['Harvesting_after_3months_price'] = float(hist['Harvesting_after_3months_price'].median())

    # Combine with history and engineer features
    combined = pd.concat([hist, pd.DataFrame([row])], ignore_index=True)
    combined = prepare_full_features(combined)

    # Extract the last row (our prediction sample)
    X_pred = combined.iloc[[-1]]

    # ===========================
    # Make predictions
    # ===========================
    try:
        # WORKAROUND: Manual preprocessing due to broken ColumnTransformer
        # (sklearn 1.3.0 -> 1.5.0 version mismatch)

        # Extract preprocessor and get numeric/categorical columns + OHE
        preprocessor = local_classifier.named_steps['pre']
        numeric_cols = None
        categorical_cols = None
        ohe = None

        if hasattr(preprocessor, 'transformers_'):
            for name, transformer, columns in preprocessor.transformers_:
                if name == 'num':
                    numeric_cols = list(columns)
                elif name == 'cat':
                    categorical_cols = list(columns)
                    ohe = transformer

        # Extract numeric features
        X_numeric = X_pred[numeric_cols].values.astype(np.float64)

        # One-hot encode categorical features using the trained OHE
        X_cat = X_pred[categorical_cols]
        if ohe is not None:
            try:
                X_cat_encoded = ohe.transform(X_cat)
            except Exception:
                # Fallback: fit_transform
                X_cat_encoded = ohe.fit_transform(X_cat)
            # Handle both sparse and dense output
            if hasattr(X_cat_encoded, 'toarray'):
                X_cat_encoded = X_cat_encoded.toarray()
            else:
                X_cat_encoded = np.asarray(X_cat_encoded)
        else:
            X_cat_encoded = X_cat.values

        # Combine
        X_processed = np.hstack([X_numeric, X_cat_encoded])

        # Local Market Classification
        local_pred_class = local_classifier.named_steps['clf'].predict(X_processed)[0]
        local_pred_proba = local_classifier.named_steps['clf'].predict_proba(X_processed)[0]
        local_label = le_local.inverse_transform([local_pred_class])[0]

        # Export Market Classification
        export_pred_class = export_classifier.named_steps['clf'].predict(X_processed)[0]
        export_pred_proba = export_classifier.named_steps['clf'].predict_proba(X_processed)[0]
        export_label = le_export.inverse_transform([export_pred_class])[0]

        # Price Predictions (using CatBoost regressor)
        prices_cat = regressor_cat.named_steps['reg'].predict(X_processed)[0]
        prices_xgb = regressor_xgb.named_steps['reg'].predict(X_processed)[0]

        # Build result
        result = {
            "sample_number": idx,
            "input": inp,
            "predictions": {
                "local_market": {
                    "market": local_label,
                    "confidence": float(local_pred_proba[int(local_pred_class)]),
                    "all_probabilities": {
                        le_local.classes_[i]: float(local_pred_proba[i])
                        for i in range(len(le_local.classes_))
                    }
                },
                "export_market": {
                    "market": export_label,
                    "confidence": float(export_pred_proba[int(export_pred_class)]),
                    "all_probabilities": {
                        le_export.classes_[i]: float(export_pred_proba[i])
                        for i in range(len(le_export.classes_))
                    }
                },
                "price_predictions_catboost": {
                    "local_price_lkr": float(prices_cat[0]),
                    "export_price_usd": float(prices_cat[1]),
                    "harvesting_after_3months_price": float(prices_cat[2])
                },
                "price_predictions_xgboost": {
                    "local_price_lkr": float(prices_xgb[0]),
                    "export_price_usd": float(prices_xgb[1]),
                    "harvesting_after_3months_price": float(prices_xgb[2])
                }
            }
        }

        all_results.append(result)

        # Print formatted output
        print("\n🎯 PREDICTIONS:")
        print(f"\n  Local Market: {local_label}")
        print(f"  └─ Confidence: {result['predictions']['local_market']['confidence']:.2%}")
        print(f"  └─ All probabilities:")
        for market, prob in result['predictions']['local_market']['all_probabilities'].items():
            print(f"     • {market}: {prob:.4f}")

        print(f"\n  Export Market: {export_label}")
        print(f"  └─ Confidence: {result['predictions']['export_market']['confidence']:.2%}")
        print(f"  └─ All probabilities:")
        for market, prob in result['predictions']['export_market']['all_probabilities'].items():
            print(f"     • {market}: {prob:.4f}")

        print(f"\n  💰 Price Predictions (CatBoost):")
        print(f"  ├─ Local Price (LKR): {prices_cat[0]:.2f}")
        print(f"  ├─ Export Price (USD): {prices_cat[1]:.4f}")
        print(f"  └─ Harvesting (3 months): {prices_cat[2]:.2f}")

        print(f"\n  💰 Price Predictions (XGBoost):")
        print(f"  ├─ Local Price (LKR): {prices_xgb[0]:.2f}")
        print(f"  ├─ Export Price (USD): {prices_xgb[1]:.4f}")
        print(f"  └─ Harvesting (3 months): {prices_xgb[2]:.2f}")

    except Exception as e:
        print(f"\n  ✗ Error making predictions: {e}")
        import traceback
        traceback.print_exc()

# ===========================
# Final Summary
# ===========================
print("\n" + "=" * 80)
print("COMPLETE JSON OUTPUT")
print("=" * 80)
print(json.dumps(all_results, indent=2))

# Save to file
output_file = "predictions_output.json"
with open(output_file, "w") as f:
    json.dump(all_results, f, indent=2)
print(f"\n✓ Results saved to {output_file}")
