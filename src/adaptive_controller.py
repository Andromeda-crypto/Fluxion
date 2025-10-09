import numpy as np
import joblib
from src.chaos_analysis import ChaosAnalyzer
from src.utils import load_data, log_event

class AdaptiveController:
    """
    Dynamically selects the best model based on chaos metrics and past performance.
    """

    def __init__(self, model_paths, chaos_data_path, threshold=0.5):
        """
        Initialize the controller with multiple trained models and chaos analyzer.
        """
        self.models = {name: joblib.load(path) for name, path in model_paths.items()}
        self.chaos_analyzer = ChaosAnalyzer(chaos_data_path)
        self.threshold = threshold
        self.performance_cache = {}  # stores adaptive history

    def evaluate_chaos(self, data):
        """
        Compute chaos metrics for the given input data.
        Returns a dict of metrics: variance, noise, entropy, etc.
        """
        chaos_metrics = self.chaos_analyzer.analyze_data(data)
        log_event("CHAOS_METRICS", chaos_metrics)
        return chaos_metrics

    def select_model(self, chaos_metrics):
        """
        Select model based on chaos level.
        - Low chaos → simpler / faster models (RandomForest)
        - High chaos → complex / robust models (XGBoost / Stacking)
        """
        variance = chaos_metrics.get('variance', 0)
        noise = chaos_metrics.get('noise', 0)

        if variance < 0.3 and noise < 0.2:
            chosen = "random_forest"
        elif variance < 0.6:
            chosen = "gradient_boosting"
        else:
            chosen = "xgboost"

        log_event("MODEL_SELECTED", {"model": chosen, "variance": variance, "noise": noise})
        return chosen

    def predict(self, data):
        """
        Full adaptive prediction pipeline:
        1. Analyze chaos
        2. Select model dynamically
        3. Predict
        4. Update internal logs
        """
        chaos_metrics = self.evaluate_chaos(data)
        chosen_model_name = self.select_model(chaos_metrics)
        chosen_model = self.models[chosen_model_name]

        prediction = chosen_model.predict(data)
        self.performance_cache[chosen_model_name] = self.performance_cache.get(chosen_model_name, []) + [prediction]

        log_event("PREDICTION", {
            "model_used": chosen_model_name,
            "chaos_level": chaos_metrics,
            "prediction": prediction.tolist() if isinstance(prediction, np.ndarray) else prediction
        })

        return prediction

    def adaptive_update(self, data, true_values):
        """
        Optional adaptive feedback loop: update model weights or decision thresholds
        based on error patterns and chaos metrics.
        """
        chaos_metrics = self.evaluate_chaos(data)
        errors = {}
        for name, model in self.models.items():
            preds = model.predict(data)
            mse = np.mean((preds - true_values) ** 2)
            errors[name] = mse

        # Lower error = better
        best_model = min(errors, key=errors.get)
        log_event("ADAPTIVE_UPDATE", {"best_model": best_model, "errors": errors})

        # Adjust threshold dynamically
        avg_error = np.mean(list(errors.values()))
        self.threshold = max(0.1, min(1.0, avg_error))

        return best_model, errors
