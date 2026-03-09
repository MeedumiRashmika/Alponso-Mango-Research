import numpy as np
import pandas as pd
import os

REQUIRED_FEATURES = [
    "month_sin", "month_cos", "doy_sin", "doy_cos",
    "Mango_Age_Days", "Days_To_Maturity", "Temp_C", "Humidity_%",
    "Region", "weather",
    "local_price_lag1", "local_price_roll7", "local_price_roll14", "local_price_mom1",
    "export_price_lag1", "export_price_roll7", "export_price_roll14", "export_price_mom1",
    "price_dev", "price_to_age_ratio", "local_price_vol7", "local_price_vol14", "momentum_dev",
    "dev_x_ratio", "vol_x_mom", "age_x_price", "roll14_x_mom",
]


def prepare_full_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform raw input DataFrame into model-ready feature DataFrame.
    Matches the exact training procedure from model.ipynb.
    """
    d = df.copy()
    d["Date"] = pd.to_datetime(d["Date"])
    d = d.sort_values(["Region", "Date"]).reset_index(drop=True)

    # Time-based features
    d["day"] = d["Date"].dt.day
    d["month"] = d["Date"].dt.month
    d["dayofyear"] = d["Date"].dt.dayofyear
    d["year"] = d["Date"].dt.year
    d["month_sin"] = np.sin(2 * np.pi * d["month"] / 12)
    d["month_cos"] = np.cos(2 * np.pi * d["month"] / 12)
    d["doy_sin"] = np.sin(2 * np.pi * d["dayofyear"] / 365)
    d["doy_cos"] = np.cos(2 * np.pi * d["dayofyear"] / 365)

    # Lags
    d["local_price_lag1"] = d.groupby("Region")["Local_Price_LKR"].shift(1)
    d["export_price_lag1"] = d.groupby("Region")["Export_Price_USD"].shift(1)

    # Rolling means (7 and 14 day windows)
    d["local_price_roll7"] = d.groupby("Region")["Local_Price_LKR"].transform(lambda s: s.rolling(7, min_periods=1).mean())
    d["local_price_roll14"] = d.groupby("Region")["Local_Price_LKR"].transform(lambda s: s.rolling(14, min_periods=1).mean())
    d["export_price_roll7"] = d.groupby("Region")["Export_Price_USD"].transform(lambda s: s.rolling(7, min_periods=1).mean())
    d["export_price_roll14"] = d.groupby("Region")["Export_Price_USD"].transform(lambda s: s.rolling(14, min_periods=1).mean())

    # Momentum
    d["local_price_mom1"] = d["Local_Price_LKR"] - d["local_price_lag1"]
    d["export_price_mom1"] = d["Export_Price_USD"] - d["export_price_lag1"]

    # Fills for lag/rolling
    grp_med_local = d.groupby("Region")["Local_Price_LKR"].transform("median")
    grp_med_export = d.groupby("Region")["Export_Price_USD"].transform("median")

    for col in ["local_price_lag1", "local_price_roll7", "local_price_roll14"]:
        d[col] = d[col].fillna(grp_med_local).fillna(d["Local_Price_LKR"].median())
    for col in ["export_price_lag1", "export_price_roll7", "export_price_roll14"]:
        d[col] = d[col].fillna(grp_med_export).fillna(d["Export_Price_USD"].median())
    d["local_price_mom1"] = d["local_price_mom1"].fillna(0)
    d["export_price_mom1"] = d["export_price_mom1"].fillna(0)

    # New features
    d["local_price_region_mean"] = d.groupby("Region")["Local_Price_LKR"].transform("mean")
    d["price_dev"] = d["Local_Price_LKR"] - d["local_price_region_mean"]
    d["price_to_age_ratio"] = d["Local_Price_LKR"] / (d["Mango_Age_Days"].replace(0, 1))
    d["local_price_vol7"] = d.groupby("Region")["Local_Price_LKR"].transform(lambda s: s.rolling(7, min_periods=1).std()).fillna(0)
    d["local_price_vol14"] = d.groupby("Region")["Local_Price_LKR"].transform(lambda s: s.rolling(14, min_periods=1).std()).fillna(0)
    d["local_mom_region_mean"] = d.groupby("Region")["local_price_mom1"].transform("mean")
    d["momentum_dev"] = d["local_price_mom1"] - d["local_mom_region_mean"]

    # Interactions
    d["dev_x_ratio"] = d["price_dev"] * d["price_to_age_ratio"]
    d["vol_x_mom"] = d["local_price_vol14"] * d["momentum_dev"]
    d["age_x_price"] = d["Mango_Age_Days"] * d["local_price_lag1"]
    d["roll14_x_mom"] = d["local_price_roll14"] * d["local_price_mom1"]

    # Fill residual NaNs
    fill_cols = ["price_dev", "price_to_age_ratio", "local_price_vol7", "local_price_vol14", "momentum_dev", "dev_x_ratio", "vol_x_mom", "age_x_price", "roll14_x_mom"]
    for c in fill_cols:
        if c in d.columns:
            d[c] = d[c].fillna(0)

    return d


def prepare_features(df: pd.DataFrame, historical_df: pd.DataFrame = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Transform raw input DataFrame into model-ready feature DataFrame.
    If historical_df is provided, concatenate and compute features together for proper lags.
    Returns (feature_df, original_df).
    """
    # If historical data is provided, concatenate and compute features on combined data
    if historical_df is not None:
        historical_df = historical_df.copy()
        df_copy = df.copy()

        # Ensure Date is datetime in both
        historical_df["Date"] = pd.to_datetime(historical_df["Date"])
        df_copy["Date"] = pd.to_datetime(df_copy["Date"])

        # Add missing price columns to historical if needed
        for col in ["Local_Price_LKR", "Export_Price_USD", "Harvesting_after_3months_price"]:
            if col not in historical_df.columns:
                historical_df[col] = np.nan

        # Concatenate
        combined = pd.concat([historical_df, df_copy], ignore_index=True, sort=False)

        # Prepare features on combined data
        combined_features = prepare_full_features(combined)

        # Extract the rows corresponding to new data (last N rows where N = len(df))
        new_features = combined_features.iloc[-len(df):].reset_index(drop=True)
        new_original = combined.iloc[-len(df):].reset_index(drop=True)

        feature_df = new_features[REQUIRED_FEATURES].copy()
        return feature_df, new_original
    else:
        # If no historical data, just prepare the input data
        df_prepared = prepare_full_features(df)
        feature_df = df_prepared[REQUIRED_FEATURES].copy()
        return feature_df, df_prepared