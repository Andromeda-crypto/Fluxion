import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from src.utils import save_json_safe, load_model_safe, get_project_root, ensure_dir, print_banner
from src.features import add_engineered_features


class ChaosAnalyzer:
    """
    Analyzes model robustness under chaotic or noisy data conditions.
    Helps identify which models perform best in different environments.
    """

    def __init__(self, models_dir=None, results_path=None):
        ROOT = get_project_root()
        self.models_dir = models_dir or os.path.join(ROOT, "models")
        self.results_path = results_path or os.path.join(ROOT, "results", "chaos_analysis.json")
        ensure_dir(os.path.dirname(self.results_path))
        self.models = self._load_all_models()

    def _load_all_models(self):
        """Load all saved models from the models directory."""
        models = {}
        for file in os.listdir(self.models_dir):
            if file.endswith(".joblib"):  # your models are saved as .joblib
                name = file.replace(".joblib", "")
                model = load_model_safe(os.path.join(self.models_dir, file))
                if model:
                    models[name] = model
        print(f"✅ Loaded models for chaos analysis: {list(models.keys())}")
        return models

    def inject_chaos(self, X, level=0.1):
        """
        Add random noise to input features to simulate chaotic conditions.
        """
        X_noisy = X.copy()
        numeric_cols = X_noisy.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            noise = np.random.normal(0, level, size=len(X_noisy))
            X_noisy[col] = np.clip(X_noisy[col] + noise, 0, 1)  # keep normalized features valid
        return X_noisy

    def compute_chaos_score(self, X):
        """
        Compute a 'chaos score' based on variability of key features.
        Higher = more chaotic environment.
        """
        if isinstance(X, pd.DataFrame):
            variability = X.std(axis=1)
        else:
            variability = np.std(list(X.values()), axis=1)
        return float(np.mean(variability))

    def analyze(self, X, y):
        """
        Evaluate how model performance changes with increasing chaos.
        """
        print_banner("CHAOS ANALYSIS STARTED")
        chaos_levels = [0.0, 0.05, 0.1, 0.2, 0.4]
        results = {}

        for model_name, model in self.models.items():
            print(f"\n🔍 Evaluating model: {model_name}")
            perf_by_level = []
            for chaos in chaos_levels:
                X_chaotic = self.inject_chaos(X, level=chaos)
                X_train, X_test, y_train, y_test = train_test_split(X_chaotic, y, test_size=0.2, random_state=42)
                preds = model.predict(X_test)
                acc = accuracy_score(y_test, preds)
                perf_by_level.append({"chaos_level": chaos, "accuracy": acc})
                print(f"Chaos={chaos:.2f} → Accuracy={acc:.4f}")

            best_condition = max(perf_by_level, key=lambda x: x["accuracy"])
            results[model_name] = {
                "performance": perf_by_level,
                "best_condition": best_condition
            }

        save_json_safe(self.results_path, results)
        print_banner("CHAOS ANALYSIS COMPLETE ✅")
        print(f"Results saved to: {self.results_path}\n")
        return results


# ---------------- DEMO EXECUTION ----------------
if __name__ == "__main__":
    ROOT = get_project_root()
    data_path = os.path.join(ROOT, "data", "processed", "cleaned_data.csv")
    df = pd.read_csv(data_path)
    df = add_engineered_features(df)

    features = [
        "x_norm", "y_norm", "team_encoded", "move_dist", "dist_to_goal",
        "possession_streak", "prev_event_enc", "rolling_entropy"
    ]
    target = "event_encoded"

    X = df[features]
    y = df[target]

    analyzer = ChaosAnalyzer()
    analyzer.analyze(X, y)





    


