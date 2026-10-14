"""Temporal and seasonal feature extraction tailored to Kolkata."""
import pandas as pd
import numpy as np

def extract_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """Extracts cyclic and categorical time drivers from crash timestamps."""
    df_feat = df.copy()
    ts = pd.to_datetime(df_feat["timestamp"])

    df_feat["feat_hour"] = ts.dt.hour
    df_feat["feat_dayofweek"] = ts.dt.dayofweek
    df_feat["feat_is_weekend"] = (ts.dt.dayofweek >= 5).astype(int)

    # Cyclical hour encoding (sin/cos)
    df_feat["feat_hour_sin"] = np.sin(2 * np.pi * df_feat["feat_hour"] / 24.0)
    df_feat["feat_hour_cos"] = np.cos(2 * np.pi * df_feat["feat_hour"] / 24.0)

    # Night flag: 18:00 to 06:00 (Daytime is 06:00 to 18:00)
    df_feat["feat_is_night"] = ((df_feat["feat_hour"] >= 18) | (df_feat["feat_hour"] < 6)).astype(int)

    # Monsoon season in Kolkata (June through September)
    df_feat["feat_is_monsoon"] = ts.dt.month.isin([6, 7, 8, 9]).astype(int)

    # Kolkata Festive Season (Durga Puja / Kali Puja: typically October/November)
    df_feat["feat_is_festive_season"] = ts.dt.month.isin([10, 11]).astype(int)

    return df_feat
