"""Road network attribute transformation and encoding."""
import pandas as pd
import numpy as np

def encode_road_features(df: pd.DataFrame) -> pd.DataFrame:
    """Transforms raw segment road attributes into numerical ML features."""
    df_feat = df.copy()

    # Highway type one-hot or ordinal ranking (expressway/trunk > primary > secondary > tertiary)
    highway_rank = {
        "trunk": 4,
        "motorway": 4,
        "primary": 3,
        "secondary": 2,
        "tertiary": 1,
        "residential": 0,
        "unclassified": 0
    }
    df_feat["feat_highway_hierarchy"] = df_feat["segment_highway_type"].str.lower().map(highway_rank).fillna(1)
    df_feat["feat_speed_limit"] = df_feat["segment_speed_limit"].astype(float)
    df_feat["feat_is_high_speed"] = (df_feat["feat_speed_limit"] >= 50).astype(int)
    df_feat["feat_lanes"] = df_feat["segment_lanes"].astype(float)
    df_feat["feat_has_median"] = df_feat["segment_has_median"].astype(int)
    df_feat["feat_is_lit"] = df_feat["segment_is_lit"].astype(int)
    df_feat["feat_is_junction"] = df_feat["segment_is_junction"].astype(int)
    df_feat["feat_junction_density"] = df_feat["segment_junction_density"].astype(float)

    return df_feat
