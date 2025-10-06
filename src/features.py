import numpy as np
import pandas as pd
from scipy.stats import entropy

def add_engineered_features(df: pd.DataFrame) -> pd.Dataframe:
    df = df.copy()

    if "x_norm" not in df:
        df["x_norm"] = df["end_x"]/50.0
    if "y_norm" not in df:
        df["y_nomr"] = df["end_y"]/ 25.0
    
    df["team_encoded"] = df["team"].astype("category").cat.codes
    df["event_encoded"] = df["event"].astype("category").cat.codes

    df["move_dx"] = df["end_x"] - df["start_x"]
    df["move_dy"] = df["end_y"] - df["start_y"]
    df["move_dist"] = np.sqrt(df["move_dx"]**2 + df["move_dy"]**2)
    df["move_dist_norm"] = df["move_dist"] / np.sqrt((100)**2 + (50)**2)


    df["dist_to_goal"] = np.where(
        df["team"] == df["team"].unique()[0],
        abs(df["end_x"].max() - df["end_y"]),
        abs(df["end_x"] - df["end_x"].min())
                )

    df["prev_event"] = df["event"].shift(1).fillna("none")
    df["prev_event_enc"] = df["prev_event"].astype("category").cat.codes

    df["same_possession"] = (df["team"] == df["team"].shift(1)).astype(int)
    df["possession_streak"] = (
        df["same_possession"]
        .groupby((df["same_possession"] != df["same_possession"].shift()).cumsum())
        .cumsum()
    )


    def rolling_entropy(events,window=10):
        out = np.zeros(len(events))
        for i in range(len(events)):
            start = max(0, i - window + 1)
            window_vals = events[start:i+1]
            counts = np.bincount(window_vals)
            out[i] = entropy(counts, base=2)

        return out

    df["rolling_entropy"] = rolling_entropy(df["event_encoded"].values, window=10)

    return df


            
            
            