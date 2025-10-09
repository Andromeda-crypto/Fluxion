import os
import json
from src.utils import load_json_safe, print_banner


class AdaptiveController:
    """
    Dynamically selects the most suitable model based on the current chaos score.
    Uses chaos reference thresholds derived from prior chaos analysis.
    """

    def __init__(self, chaos_reference_path=None):
        self.chaos_reference_path = chaos_reference_path
        self.thresholds = self._load_reference()
        print(f"✅ AdaptiveController initialized with thresholds: {self.thresholds}")

    # ------------------------------------------------
    # Load chaos thresholds (from chaos_analysis.json)
    # ------------------------------------------------
    def _load_reference(self):
        """Load chaos thresholds from the reference JSON file."""
        if self.chaos_reference_path and os.path.exists(self.chaos_reference_path):
            data = load_json_safe(self.chaos_reference_path)

            # Handle both detailed chaos analysis or compact threshold dicts
            if "chaos_thresholds" in data:
                thresholds = data["chaos_thresholds"]
            else:
                # Derive thresholds heuristically if chaos_analysis.json came from analyzer output
                thresholds = {"low": 0.3, "high": 0.7}

            print(f"✅ Loaded chaos thresholds from {self.chaos_reference_path}")
            return thresholds

        print("⚠️ No chaos reference found, using default thresholds.")
        return {"low": 0.3, "high": 0.7}

    # ------------------------------------------------
    # Model selection based on current chaos score
    # ------------------------------------------------
    def select_model(self, chaos_score):
        """
        Select which model to use based on chaos score:
        - Low chaos: stable → RandomForest
        - Medium chaos: adaptive → GradientBoosting
        - High chaos: highly nonlinear → XGBoost
        """
        low, high = self.thresholds["low"], self.thresholds["high"]

        if chaos_score < low:
            selected = "randomforest"
        elif chaos_score < high:
            selected = "gradientboosting"
        else:
            selected = "xgboost"

        print(f"⚙️ Chaos={chaos_score:.3f} → Selected model: {selected}")
        return selected


# ---------------- DEMO EXECUTION ----------------
if __name__ == "__main__":
    print_banner("Adaptive Controller Demo")

    # Demo run — mimicking incoming chaos values
    controller = AdaptiveController(chaos_reference_path="results/chaos_analysis.json")

    demo_scores = [0.1, 0.25, 0.45, 0.75, 0.9]
    for score in demo_scores:
        controller.select_model(score)
