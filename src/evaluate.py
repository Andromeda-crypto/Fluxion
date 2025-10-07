import os
import joblib
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

from src.features import add_engineered_features


def evaluate_models(models_dir="models", data_path="data/processed/cleaned_data.csv", results_dir="results"):
    """
    Loads all trained models, evaluates them on the dataset,
    and saves performance metrics and confusion matrices.
    """

    # --- Setup ---
    ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    models_dir = os.path.join(ROOT, models_dir)
    data_path = os.path.join(ROOT, data_path)
    results_dir = os.path.join(ROOT, results_dir)

    os.makedirs(results_dir, exist_ok=True)

    # --- Load and prepare data ---
    df = pd.read_csv(data_path)
    df = add_engineered_features(df)

    # Pick the same features you used in training
    features = ["x_norm", "y_norm", "team_encoded", "move_dist", "dist_to_goal",
                "possession_streak", "prev_event_enc", "rolling_entropy"]
    target = "event_encoded"

    X = df[features]
    y = df[target]

    # --- Evaluate each saved model ---
    results = []

    for model_file in os.listdir(models_dir):
        if model_file.endswith(".joblib"):
            model_path = os.path.join(models_dir, model_file)
            model = joblib.load(model_path)

            preds = model.predict(X)

            acc = accuracy_score(y, preds)
            prec = precision_score(y, preds, average='weighted', zero_division=0)
            rec = recall_score(y, preds, average='weighted', zero_division=0)
            f1 = f1_score(y, preds, average='weighted', zero_division=0)

            results.append({
                "Model": model_file.replace(".joblib", ""),
                "Accuracy": round(acc, 4),
                "Precision": round(prec, 4),
                "Recall": round(rec, 4),
                "F1 Score": round(f1, 4)
            })

            # --- Confusion Matrix Plot ---
            plt.figure(figsize=(6, 5))
            sns.heatmap(confusion_matrix(y, preds), annot=True, fmt="d", cmap="Blues")
            plt.title(f"Confusion Matrix: {model_file}")
            plt.xlabel("Predicted")
            plt.ylabel("Actual")
            plt.tight_layout()
            plt.savefig(os.path.join(results_dir, f"{model_file}_confusion.png"))
            plt.close()

    # --- Save metrics to CSV ---
    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join(results_dir, "model_comparison.csv"), index=False)

    print("\n✅ Model Evaluation Complete. Results:")
    print(results_df)


if __name__ == "__main__":
    evaluate_models()


