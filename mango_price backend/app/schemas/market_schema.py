"""
Schema for the /predict/market endpoint.
Validates 7-field lightweight market prediction requests.
"""
from dataclasses import dataclass
from datetime import datetime
from typing import Tuple, List

MARKET_REQUIRED_FIELDS = [
    "Date",
    "Mango_Age_Days",
    "Days_To_Maturity",
    "Temp_C",
    "Humidity_percent",
    "Region",
    "weather",
]


@dataclass
class MarketPredictionRequest:
    """Represents a valid market prediction request (7 fields)."""
    Date: str                  # YYYY-MM-DD
    Mango_Age_Days: int
    Days_To_Maturity: int
    Temp_C: float
    Humidity_percent: float    # API field name; maps to "Humidity_%" internally
    Region: str
    weather: str


def validate_market_payload(payload: dict) -> Tuple[dict, List[str]]:
    """
    Validate a single market prediction request.

    Returns:
        (normalized_record, errors) where:
        - normalized_record: dict with normalized values and "Humidity_%" key
        - errors: list of validation error messages (empty if valid)
    """
    errors: List[str] = []

    if not isinstance(payload, dict):
        return {}, ["Request body must be a JSON object"]

    # Check all required fields are present
    missing = [f for f in MARKET_REQUIRED_FIELDS if f not in payload]
    if missing:
        return {}, [f"Missing required fields: {missing}"]

    normalized = {}

    # Validate Date (YYYY-MM-DD format)
    try:
        date_str = str(payload["Date"]).strip()
        datetime.strptime(date_str, "%Y-%m-%d")
        normalized["Date"] = date_str
    except ValueError:
        errors.append("'Date' must be in YYYY-MM-DD format")

    # Validate Mango_Age_Days (non-negative int)
    try:
        age = int(payload["Mango_Age_Days"])
        if age < 0:
            errors.append("'Mango_Age_Days' must be a non-negative integer")
        else:
            normalized["Mango_Age_Days"] = age
    except (TypeError, ValueError):
        errors.append("'Mango_Age_Days' must be an integer")

    # Validate Days_To_Maturity (non-negative int)
    try:
        days = int(payload["Days_To_Maturity"])
        if days < 0:
            errors.append("'Days_To_Maturity' must be a non-negative integer")
        else:
            normalized["Days_To_Maturity"] = days
    except (TypeError, ValueError):
        errors.append("'Days_To_Maturity' must be an integer")

    # Validate Temp_C (float)
    try:
        normalized["Temp_C"] = float(payload["Temp_C"])
    except (TypeError, ValueError):
        errors.append("'Temp_C' must be a number")

    # Validate Humidity_percent (float) and rename to Humidity_%
    try:
        normalized["Humidity_%"] = float(payload["Humidity_percent"])
    except (TypeError, ValueError):
        errors.append("'Humidity_percent' must be a number")

    # Validate Region (non-empty string)
    if not isinstance(payload.get("Region"), str) or not payload["Region"].strip():
        errors.append("'Region' must be a non-empty string")
    else:
        normalized["Region"] = payload["Region"].strip()

    # Validate weather (non-empty string)
    if not isinstance(payload.get("weather"), str) or not payload["weather"].strip():
        errors.append("'weather' must be a non-empty string")
    else:
        normalized["weather"] = payload["weather"].strip()

    if errors:
        return {}, errors

    return normalized, []
