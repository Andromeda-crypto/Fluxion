import numpy as np
import pandas as pd
from scipy.stats import entropy
import os

# ---------------- Parameters ----------------
NUM_ROWS = 5000  # number of events in the dataset
TEAMS = ["TeamA", "TeamB"]
EVENTS = ["pass", "shot", "dribble", "tackle", "foul", "goal"]
OUTPUT_PATH = "data/processed/cleaned_data.csv"

# ---------------- Generate base data ----------------
np.random.seed(42)

df = pd.DataFrame({
    "start_x": np.random.uniform(0, 100, NUM_ROWS),
    "start_y": np.random.uniform(0, 50, NUM_ROWS),
    "end_x": np.random.uniform(0, 100, NUM_ROWS),
    "end_y": np.random.uniform(0, 50, NUM_ROWS),
    "team": np.random.choice(TEAMS, NUM_ROWS),
    "event": np.random.choice(EVENTS, NUM_ROWS)
})

# ---------------- Engineered Features ----------------
# Normalized coordinates
df["x_norm"] = df["end_x"] / 100
df["y_norm"] = df["end_y"] / 50

# Encoding categorical columns
df["team_encoded"] = df["team"].astype("category").cat.codes
df["event_encoded"] = df["event"].astype("category").cat.codes

# Movement features
df["move_dx"] = df["end_x"] - df["start_x"]
df["move_dy"] = df["end_y"] - df["start_y"]
df["move_dist"] = np.sqrt(df["move_dx"]**2 + df["move_dy"]**2)
df["move_dist_norm"] = df["move_dist"] / np.sqrt(100**2 + 50**2)

# Distance to goal (using x only, assuming TeamA attacks right, TeamB left)
df["dist_to_goal"] = np.where(
    df["team"] == "TeamA",
    100 - df["end_x"],
    df["end_x"]
)

# Previous event and possession streaks
df["prev_event"] = df["event"].shift(1).fillna("none")
df["prev_event_enc"] = df["prev_event"].astype("category").cat.codes

df["same_possession"] = (df["team"] == df["team"].shift(1)).astype(int)
df["possession_streak"] = (
    df["same_possession"]
    .groupby((df["same_possession"] != df["same_possession"].shift()).cumsum())
    .cumsum()
)

# Rolling entropy over the last 10 events
def rolling_entropy(events, window=10):
    out = np.zeros(len(events))
    for i in range(len(events)):
        start = max(0, i - window + 1)
        window_vals = events[start:i+1]
        counts = np.bincount(window_vals.astype(int))
        out[i] = entropy(counts, base=2)
    return out

df["rolling_entropy"] = rolling_entropy(df["event_encoded"].values, window=10)

# ---------------- Save dataset ----------------
os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
df.to_csv(OUTPUT_PATH, index=False)
print(f"✅ Full-fledged dataset generated: {OUTPUT_PATH}")
print(f"Columns: {list(df.columns)}")
print(df.head())
