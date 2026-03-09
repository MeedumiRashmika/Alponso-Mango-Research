from dataclasses import dataclass, field
from typing import Optional


@dataclass
class PredictionRecord:
    Date: str
    Local_Market: str
    Local_Price_LKR: float
    Export_Market: str
    Export_Price_USD: float
    Mango_Age_Days: int
    Days_To_Maturity: int
    Temp_C: float
    Humidity_percent: float   # maps to "Humidity_%" column
    Region: str
    weather: str
    Harvesting_after_3months_price: Optional[float] = None


REQUIRED_FIELDS = [
    "Date", "Local_Market", "Local_Price_LKR", "Export_Market",
    "Export_Price_USD", "Mango_Age_Days", "Days_To_Maturity",
    "Temp_C", "Humidity_percent", "Region", "weather",
]


def validate_batch_payload(payload: dict) -> tuple[list[dict], list[str]]:
    """
    Validate incoming JSON payload.
    Returns (clean_records, errors).
    """
    errors = []

    if "records" not in payload:
        errors.append("Missing required key: 'records'")
        return [], errors

    if not isinstance(payload["records"], list) or len(payload["records"]) == 0:
        errors.append("'records' must be a non-empty list")
        return [], errors

    clean = []
    for idx, rec in enumerate(payload["records"]):
        missing = [f for f in REQUIRED_FIELDS if f not in rec]
        if missing:
            errors.append(f"Record[{idx}] missing fields: {missing}")
            continue
        # Rename Humidity_percent → Humidity_%
        normalized = dict(rec)
        normalized["Humidity_%"] = normalized.pop("Humidity_percent")
        clean.append(normalized)

    return clean, errors