import pandas as pd
import os

# File paths
RAW_DATA_PATH = "data/raw/simulated_game_data.csv"
CLEAN_DATA_PATH = "data/processed/cleaned_data.csv"

def clean_data():
    # 1. Load raw data
    if not os.path.exists(RAW_DATA_PATH):
        raise FileNotFoundError(f"Raw data file not found at {RAW_DATA_PATH}")
    
    df = pd.read_csv(RAW_DATA_PATH)
    if "event" in df.columns and "x" in df.columns:
        df = df[~((df["event"] == "goal") & (df["x"] < 50))]

  
    if "x" in df.columns and "y" in df.columns:
        df["x_norm"] = (df["x"] - df["x"].min()) / (df["x"].max() - df["x"].min()) * 100
        df["y_norm"] = (df["y"] - df["y"].min()) / (df["y"].max() - df["y"].min()) * 100
    if "team" in df.columns:
        df["team_encoded"] = df["team"].astype("category").cat.codes

    if "event" in df.columns:  
        df["event_encoded"] = df["event"].astype("category").cat.codes

    os.makedirs(os.path.dirname(CLEAN_DATA_PATH), exist_ok=True)
    df.to_csv(CLEAN_DATA_PATH, index=False)
    print(f"✅ Cleaned data saved to {CLEAN_DATA_PATH}")
    print(f"Columns available: {list(df.columns)}")


if __name__ == "__main__":
    clean_data()

