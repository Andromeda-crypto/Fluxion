import os
import joblib
import pandas as pd
from src.features import add_engineered_features

def demo_predict():
    ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    model_path = os.path.join(ROOT, "models", "stack.joblib")
    data_path = os.path.join(ROOT, "data", "processed", "cleaned_data.csv")

    # Load the trained stack model
    model = joblib.load(model_path)

    # Load a small sample from your processed dataset
    df = pd.read_csv(data_path)
    df = add_engineered_features(df)

    # Same feature list as used in train_models
    features = ["x_norm", "y_norm", "team_encoded", "move_dist", "dist_to_goal",
                "possession_streak", "prev_event_enc", "rolling_entropy"]

    # Take one random example to simulate a real-time prediction
    sample = df.sample(1, random_state=42)
    X_sample = sample[features]

    # Make prediction
    pred = model.predict(X_sample)[0]

    # Decode back to human-readable label if possible
    event_mapping = dict(enumerate(df["event"].astype("category").cat.categories))
    predicted_event = event_mapping.get(pred, f"Unknown ({pred})")

    print("🎯 Demo Prediction:")
    print("------------------")
    print(f"Input features:\n{X_sample.to_dict(orient='records')[0]}")
    print(f"\nPredicted Event: {predicted_event} (encoded: {pred})")


if __name__ == "__main__":
    demo_predict()