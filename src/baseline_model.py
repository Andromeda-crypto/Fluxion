import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
import joblib

CLEAN_FILE_PATH = "data/processed/cleaned_data.csv"
MODEL_PATH = "models/baseline_model.joblib"

def train_baseline():
    if not os.path.exists(CLEAN_FILE_PATH):
        raise FileNotFoundError(f"Cleaned data not found at {CLEAN_FILE_PATH}")

    df = pd.read_csv(CLEAN_FILE_PATH)
    print(f"Loaded dataset with {len(df)} rows and columns: {list(df.columns)}")

    # Ensure x_norm / y_norm exist
    if "x_norm" not in df.columns or "y_norm" not in df.columns:
        if "end_x" in df.columns and "end_y" in df.columns:
            df["x_norm"] = df["end_x"] / 50.0
            df["y_norm"] = df["end_y"] / 25.0
        else:
            raise ValueError("Neither normalized (x_norm, y_norm) nor raw (end_x, end_y) coordinates found.")

    # Encode team if missing
    if "team_encoded" not in df.columns and "team" in df.columns:
        df["team_encoded"] = df["team"].astype("category").cat.codes

    # Encode events if missing
    if "event_encoded" not in df.columns and "event" in df.columns:
        df["event_encoded"] = df["event"].astype("category").cat.codes

    # Define features and target
    features = ["x_norm", "y_norm", "team_encoded"]
    target = "event_encoded"

    missing = [col for col in features + [target] if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    x = df[features]
    y = df[target]

    # Train/test split
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    # Train model
    model = LogisticRegression(max_iter=1000, class_weight="balanced")
    model.fit(x_train, y_train)

    # Evaluate
    y_pred = model.predict(x_test)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))

    # Save model
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    print(f"Baseline model saved at {MODEL_PATH}")

    # Show class distribution
    print("\nEvent distribution in dataset:")
    print(df["event"].value_counts())


if __name__ == "__main__":
    train_baseline()

