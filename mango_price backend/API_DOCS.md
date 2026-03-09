# Mango Price Prediction API

REST API for predicting local and export mango market prices using trained ML models.

**Base URL:** `http://localhost:5002/api/v1`

---

## Table of Contents

- [Authentication](#authentication)
- [Endpoints](#endpoints)
  - [Health Check](#1-health-check)
  - [Model Status](#2-model-status)
  - [Predict (Batch)](#3-predict-batch)
  - [Predict (Single)](#4-predict-single)
  - [Predict (Market Classification & Prices)](#5-predict-market-classification--prices-lightweight)
- [Request Schema](#request-schema)
- [Response Schema](#response-schema)
- [Error Handling](#error-handling)
- [Example Requests](#example-requests)

---

## Authentication

No authentication is required in the current version.

---

## Endpoints

### 1. Health Check

Check if the API is running.

```
GET /api/v1/health
```

**Response `200 OK`**
```json
{
  "status": "ok",
  "service": "mango-price-prediction"
}
```

---

### 2. Model Status

Check which ML models and encoders are currently loaded in memory.

```
GET /api/v1/models/status
```

**Response `200 OK`**
```json
{
  "models_loaded": ["local", "export", "multi_cat", "multi_xgb"],
  "encoders_loaded": ["local", "export"],
  "total_models": 4
}
```

---

### 3. Predict (Batch)

Submit multiple mango records and receive predictions for all of them.

```
POST /api/v1/predict
Content-Type: application/json
```

**Request Body**

```json
{
  "records": [
    {
      "Date": "2024-03-15",
      "Local_Market": "Colombo",
      "Local_Price_LKR": 320.50,
      "Export_Market": "Dubai",
      "Export_Price_USD": 1.45,
      "Mango_Age_Days": 75,
      "Days_To_Maturity": 20,
      "Temp_C": 31.5,
      "Humidity_percent": 72.0,
      "Region": "Monaragala",
      "weather": "Sunny",
      "Harvesting_after_3months_price": 410.00
    }
  ]
}
```

**Response `200 OK`**

```json
{
  "predictions": [
    {
      "date": "2024-03-15",
      "region": "Monaragala",
      "mango_age_days": 75,
      "actual_local_price_lkr": 320.50,
      "actual_export_price_usd": 1.4500,
      "predicted_local_price_lkr": 334.20,
      "predicted_export_price_usd": 1.5210,
      "forecast_3month_price_lkr": 410.00
    }
  ],
  "summary": {
    "avg_predicted_local_price_lkr": 334.20,
    "avg_predicted_export_price_usd": 1.5210,
    "market_recommendation": "export",
    "recommendation_note": "Export market shows stronger predicted prices. Consider export channels."
  }
}
```

---

### 4. Predict (Single)

Submit a single mango record and receive a prediction.

```
POST /api/v1/predict/single
Content-Type: application/json
```

**Request Body**

Same fields as a single record object (no wrapping in `"records"` array):

```json
{
  "Date": "2024-03-15",
  "Local_Market": "Pettah",
  "Local_Price_LKR": 290.00,
  "Export_Market": "London",
  "Export_Price_USD": 1.60,
  "Mango_Age_Days": 60,
  "Days_To_Maturity": 35,
  "Temp_C": 28.0,
  "Humidity_percent": 68.0,
  "Region": "Kandy",
  "weather": "Clouds"
}
```

**Response `200 OK`**

```json
{
  "prediction": {
    "date": "2024-03-15",
    "region": "Kandy",
    "mango_age_days": 60,
    "actual_local_price_lkr": 290.00,
    "actual_export_price_usd": 1.6000,
    "predicted_local_price_lkr": 305.80,
    "predicted_export_price_usd": 1.6430
  },
  "summary": {
    "avg_predicted_local_price_lkr": 305.80,
    "avg_predicted_export_price_usd": 1.6430,
    "market_recommendation": "export",
    "recommendation_note": "Export market shows stronger predicted prices. Consider export channels."
  }
}
```

---

### 5. Predict (Market Classification & Prices) — Lightweight

**⭐ NEW ENDPOINT:** Simplified 7-field input for market classification and price prediction without pre-supplying actual prices. Ideal for real-time market analysis.

```
POST /api/v1/predict/market
Content-Type: application/json
```

**Key Differences from `/predict/single`:**
- Only 7 required fields (no `Local_Market`, `Export_Market`, or price fields needed)
- Prices are automatically inferred from historical data by Region
- Returns market classification from both local and export classifiers
- Returns price predictions from both CatBoost and XGBoost models
- Includes ensemble average and market recommendation

**Request Body**

```json
{
  "Date": "2025-11-15",
  "Mango_Age_Days": 40,
  "Days_To_Maturity": 40,
  "Temp_C": 30.2,
  "Humidity_percent": 72,
  "Region": "Colombo",
  "weather": "Clear"
}
```

**Response `200 OK`**

```json
{
  "input": {
    "date": "2025-11-15",
    "region": "Colombo",
    "mango_age_days": 40,
    "days_to_maturity": 40,
    "temp_c": 30.2,
    "humidity_percent": 72.0,
    "weather": "Clear"
  },
  "market_classification": {
    "local_market": {
      "market": "Colombo",
      "confidence": 0.602,
      "all_probabilities": {
        "Colombo": 0.602,
        "Dambulla": 0.0399,
        "Pettah": 0.3581
      }
    },
    "export_market": {
      "market": "London",
      "confidence": 0.6388,
      "all_probabilities": {
        "Doha": 0.0017,
        "Dubai": 0.0088,
        "London": 0.6388,
        "Paris": 0.3506
      }
    }
  },
  "price_predictions": {
    "catboost": {
      "local_price_lkr": 285.26,
      "export_price_usd": 1.5199,
      "harvest_3m_price": 306.03
    },
    "xgboost": {
      "local_price_lkr": 287.93,
      "export_price_usd": 1.5307,
      "harvest_3m_price": 310.11
    },
    "ensemble_avg": {
      "local_price_lkr": 286.6,
      "export_price_usd": 1.5253
    }
  },
  "recommendation": {
    "suggested_channel": "export",
    "note": "High confidence in export market. Prioritise export channels.",
    "local_market_confidence": 0.602,
    "export_market_confidence": 0.6388
  }
}
```

---

## Request Schema

### For `/predict` and `/predict/single` endpoints

| Field | Type | Required | Description |
|---|---|---|---|
| `Date` | `string` | ✅ | Date in `YYYY-MM-DD` format |
| `Local_Market` | `string` | ✅ | Local market name (e.g., `"Colombo"`, `"Pettah"`, `"Dambulla"`) |
| `Local_Price_LKR` | `float` | ✅ | Current local price per kg in LKR |
| `Export_Market` | `string` | ✅ | Export destination (e.g., `"Dubai"`, `"London"`, `"Doha"`) |
| `Export_Price_USD` | `float` | ✅ | Current export price per kg in USD |
| `Mango_Age_Days` | `integer` | ✅ | Age of mango crop in days |
| `Days_To_Maturity` | `integer` | ✅ | Estimated days remaining until harvest |
| `Temp_C` | `float` | ✅ | Current temperature in Celsius |
| `Humidity_percent` | `float` | ✅ | Humidity percentage (0–100) |
| `Region` | `string` | ✅ | Growing region (e.g., `"Monaragala"`, `"Matara"`, `"Kandy"`) |
| `weather` | `string` | ✅ | Weather condition (e.g., `"Sunny"`, `"Rain"`, `"Clouds"`, `"Drizzle"`) |
| `Harvesting_after_3months_price` | `float` | ❌ | Known/expected price after 3 months (optional, used for accuracy comparison) |

### For `/predict/market` endpoint (Lightweight)

| Field | Type | Required | Description |
|---|---|---|---|
| `Date` | `string` | ✅ | Date in `YYYY-MM-DD` format |
| `Mango_Age_Days` | `integer` | ✅ | Age of mango crop in days (0–150) |
| `Days_To_Maturity` | `integer` | ✅ | Estimated days remaining until harvest (0–150) |
| `Temp_C` | `float` | ✅ | Current temperature in Celsius (20–40) |
| `Humidity_percent` | `float` | ✅ | Humidity percentage (0–100) |
| `Region` | `string` | ✅ | Growing region ⬇️ |
| `weather` | `string` | ✅ | Weather condition ⬇️ |

**Note:** Price fields (`Local_Price_LKR`, `Export_Price_USD`) are NOT required for `/predict/market`. They are automatically inferred from historical data for the given Region.

#### Accepted Region Values
```
"Colombo"
"Kandy"
"Galle"
"Hambantota"
"Jaffna"
"Matara"
"Kurunegala"
"Anuradhapura"
"Monaragala"
```

#### Accepted Weather Values
```
"Clear"
"Rain"
"Partly Cloudy"
"Overcast"
"Thunderstorm"
"Sunny"
"Clouds"
"Drizzle"
```

---

## Response Schema

### For `/predict/market` endpoint

| Field | Type | Description |
|---|---|---|
| `input` | `object` | Echo of the input parameters |
| `market_classification.local_market` | `object` | Local market prediction with confidence scores |
| `market_classification.export_market` | `object` | Export market prediction with confidence scores |
| `price_predictions.catboost` | `object` | Price predictions from CatBoost regressor |
| `price_predictions.xgboost` | `object` | Price predictions from XGBoost regressor |
| `price_predictions.ensemble_avg` | `object` | Ensemble average of both models |
| `recommendation` | `object` | Market channel recommendation and reasoning |

**Market Classification Object:**
| Field | Type | Description |
|---|---|---|
| `market` | `string` | Predicted market/destination |
| `confidence` | `float` | Model confidence (0–1) |
| `all_probabilities` | `object` | Probability distribution across all classes |

**Price Prediction Object:**
| Field | Type | Description |
|---|---|---|
| `local_price_lkr` | `float` | Predicted local price (LKR/kg) |
| `export_price_usd` | `float` | Predicted export price (USD/kg) |
| `harvest_3m_price` | `float` | Predicted price after 3 months |

**Recommendation Object:**
| Field | Type | Description |
|---|---|---|
| `suggested_channel` | `string` | `"export"`, `"local"`, or `"undecided"` |
| `note` | `string` | Human-readable explanation |
| `local_market_confidence` | `float` | Confidence in local market prediction |
| `export_market_confidence` | `float` | Confidence in export market prediction |

### Prediction Object (for `/predict` and `/predict/single`)

| Field | Type | Description |
|---|---|---|
| `date` | `string` | Input date |
| `region` | `string` | Input region |
| `mango_age_days` | `integer` | Input mango age |
| `actual_local_price_lkr` | `float` | Input local price (LKR/kg) |
| `actual_export_price_usd` | `float` | Input export price (USD/kg) |
| `predicted_local_price_lkr` | `float` | Model-predicted local price (LKR/kg) |
| `predicted_export_price_usd` | `float` | Model-predicted export price (USD/kg) |
| `forecast_3month_price_lkr` | `float` | 3-month price forecast (if input provided) |

### Summary Object

| Field | Type | Description |
|---|---|---|
| `avg_predicted_local_price_lkr` | `float` | Average predicted local price across batch |
| `avg_predicted_export_price_usd` | `float` | Average predicted export price across batch |
| `market_recommendation` | `string` | `"local"` or `"export"` |
| `recommendation_note` | `string` | Human-readable explanation of recommendation |

---

## Error Handling

| HTTP Status | Meaning |
|---|---|
| `400` | Invalid or missing JSON body |
| `422` | Validation error — missing required fields |
| `500` | Internal prediction error |
| `503` | Models not loaded (check `MODEL_DIR` config) |

**Example Error Response `422`**
```json
{
  "errors": [
    "Record[0] missing fields: ['Temp_C', 'weather']"
  ]
}
```

**Example Error Response `503`**
```json
{
  "error": "No models are loaded. Check model files and MODEL_DIR config."
}
```

---

## Example Requests

### cURL — Single Prediction (Full Data)

```bash
curl -X POST http://localhost:5002/api/v1/predict/single \
  -H "Content-Type: application/json" \
  -d '{
    "Date": "2024-03-15",
    "Local_Market": "Colombo",
    "Local_Price_LKR": 320.50,
    "Export_Market": "Dubai",
    "Export_Price_USD": 1.45,
    "Mango_Age_Days": 75,
    "Days_To_Maturity": 20,
    "Temp_C": 31.5,
    "Humidity_percent": 72.0,
    "Region": "Monaragala",
    "weather": "Sunny"
  }'
```

### cURL — Market Prediction (Lightweight - 7 Fields)

```bash
curl -X POST http://localhost:5002/api/v1/predict/market \
  -H "Content-Type: application/json" \
  -d '{
    "Date": "2025-11-15",
    "Mango_Age_Days": 40,
    "Days_To_Maturity": 40,
    "Temp_C": 30.2,
    "Humidity_percent": 72,
    "Region": "Colombo",
    "weather": "Clear"
  }'
```

### cURL — Batch Prediction (Full Data)

```bash
curl -X POST http://localhost:5002/api/v1/predict \
  -H "Content-Type: application/json" \
  -d '{
    "records": [
      {
        "Date": "2024-03-15",
        "Local_Market": "Colombo",
        "Local_Price_LKR": 320.50,
        "Export_Market": "Dubai",
        "Export_Price_USD": 1.45,
        "Mango_Age_Days": 75,
        "Days_To_Maturity": 20,
        "Temp_C": 31.5,
        "Humidity_percent": 72.0,
        "Region": "Monaragala",
        "weather": "Sunny"
      },
      {
        "Date": "2024-04-01",
        "Local_Market": "Pettah",
        "Local_Price_LKR": 290.00,
        "Export_Market": "London",
        "Export_Price_USD": 1.60,
        "Mango_Age_Days": 60,
        "Days_To_Maturity": 35,
        "Temp_C": 28.0,
        "Humidity_percent": 68.0,
        "Region": "Kandy",
        "weather": "Clouds"
      }
    ]
  }'
```

### Python — Market Prediction (Lightweight)

```python
import requests

url = "http://localhost:5002/api/v1/predict/market"

payload = {
    "Date": "2025-11-15",
    "Mango_Age_Days": 40,
    "Days_To_Maturity": 40,
    "Temp_C": 30.2,
    "Humidity_percent": 72,
    "Region": "Colombo",
    "weather": "Clear"
}

response = requests.post(url, json=payload)
result = response.json()

# Extract predictions
local_market = result["market_classification"]["local_market"]["market"]
export_market = result["market_classification"]["export_market"]["market"]
local_price = result["price_predictions"]["ensemble_avg"]["local_price_lkr"]
export_price = result["price_predictions"]["ensemble_avg"]["export_price_usd"]
recommendation = result["recommendation"]["suggested_channel"]

print(f"Local Market: {local_market} (Confidence: {result['market_classification']['local_market']['confidence']:.2%})")
print(f"Export Market: {export_market} (Confidence: {result['market_classification']['export_market']['confidence']:.2%})")
print(f"Predicted Local Price: LKR {local_price:.2f}/kg")
print(f"Predicted Export Price: USD {export_price:.4f}/kg")
print(f"Recommendation: {recommendation.upper()} - {result['recommendation']['note']}")
```

### Python — Full Prediction (All Data)

```python
import requests

url = "http://localhost:5002/api/v1/predict/single"

payload = {
    "Date": "2024-03-15",
    "Local_Market": "Colombo",
    "Local_Price_LKR": 320.50,
    "Export_Market": "Dubai",
    "Export_Price_USD": 1.45,
    "Mango_Age_Days": 75,
    "Days_To_Maturity": 20,
    "Temp_C": 31.5,
    "Humidity_percent": 72.0,
    "Region": "Monaragala",
    "weather": "Sunny"
}

response = requests.post(url, json=payload)
print(response.json())
```

---

## Architecture & Design

### Model Pipeline

The API uses a sophisticated ML pipeline with feature engineering and multiple models:

1. **Feature Engineering (39 features)**
   - Cyclic time encoding (month_sin/cos, doy_sin/cos)
   - Lag features (7-day and 14-day rolling means)
   - Momentum indicators
   - Derived features (price deviation, volatility, ratio features)
   - Interaction terms

2. **Classification Models**
   - **Local Market Classifier** (CatBoost): Predicts which local market (Colombo/Dambulla/Pettah)
   - **Export Market Classifier** (XGBoost): Predicts which export destination (Dubai/London/Paris/Doha)

3. **Regression Models**
   - **CatBoost Multi-Output Regressor**: Predicts 3 prices simultaneously
   - **XGBoost Multi-Output Regressor**: Predicts 3 prices simultaneously
   - Both models predict: local price (LKR), export price (USD), 3-month forecast

### `/predict/market` Endpoint (SOLID Design)

The `/predict/market` endpoint implements SOLID principles for clean, maintainable code:

- **S (Single Responsibility)**: Each class has one job (preprocessing, data loading, response formatting)
- **O (Open/Closed)**: Add new models without modifying existing code
- **L (Liskov Substitution)**: Model implementations are interchangeable via protocols
- **I (Interface Segregation)**: Separate classifier and regressor interfaces
- **D (Dependency Inversion)**: Depends on abstractions, not concrete implementations

**Architecture:**
```
MarketPredictionRequest (7 fields)
    ↓
MarketRequestRowBuilder (enrich with prices)
    ↓
Feature Engineering (39 features)
    ↓
PipelinePreprocessor (manual sklearn workaround)
    ↓
Predictions:
  - LocalMarketClassifier (CatBoost)
  - ExportMarketClassifier (XGBoost)
  - CatBoostPriceRegressor
  - XGBoostPriceRegressor
    ↓
MarketResponseBuilder (format response)
    ↓
JSON Response
```

---

## Running the API

```bash
# Install dependencies
pip install -r requirements.txt

# Run server
python run.py
```

The server runs on **port 5002** by default.

Environment variables:

| Variable | Default | Description |
|---|---|---|
| `MODEL_DIR` | `output/models` | Path to directory containing `.joblib` model files |
| `DATA_FILE` | `dataset/mango_price_dataset.csv` | Path to historical dataset CSV (12,000 records used for lag features) |
| `FLASK_DEBUG` | `false` | Enable Flask debug mode |
| `SECRET_KEY` | `mango-secret-key` | Flask secret key |

---

## Comparison: Which Endpoint to Use?

| Use Case | Endpoint | Input Fields | Output |
|---|---|---|---|
| Quick market analysis | `/predict/market` | 7 | Market classification + prices from both models |
| Full prediction with known prices | `/predict/single` | 11 | Price predictions + market recommendation |
| Batch processing with known prices | `/predict` | 11 (per record) | Array of predictions + summary |
| Just check if API is up | `/health` | None | Status |
| Check model status | `/models/status` | None | Loaded models & encoders |

---

## Error Responses Reference

**400 - Invalid JSON**
```json
{"error": "Invalid or missing JSON body"}
```

**422 - Validation Error (Missing Fields)**
```json
{"errors": ["Missing required fields: ['Humidity_percent', 'Region', 'weather']"]}
```

**422 - Validation Error (Invalid Type)**
```json
{"errors": ["'Date' must be in YYYY-MM-DD format"]}
```

**503 - Models Not Loaded**
```json
{"error": "Required models not loaded: ['local', 'export']"}
```

**500 - Prediction Error**
```json
{"error": "Market prediction failed: <error detail>"}
```