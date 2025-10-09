import os
import pandas as pd
import matplotlib.pyplot as plt
from src.features import add_engineered_features
from src.utils import get_project_root, load_model_safe, print_banner, ensure_dir, save_plot
from src.chaos_analysis import ChaosAnalyzer
from src.adaptive_controller import AdaptiveController


def run_adaptive_demo():
    print_banner("Adaptive Model Demo (Real Dataset Simulation) !")
    ROOT = get_project_root()
    data_path = os.path.join(ROOT, "data","processed","cleaned_data.csv")
    model_dir = os.path.join(ROOT, "models")
    results_dir = os.path.join(ROOT, "results")
    ensure_dir(results_dir)

    df = pd.read_csv(data_path)
    # --- Add engineered features if possible ---
    try:
        df = add_engineered_features(df)
    except Exception as e:
        print(f"⚠️ Could not add engineered features: {e}")

# --- Option 1: Fill missing columns ---
    required_features = [
    "x_norm", "y_norm", "team_encoded", "move_dist", "dist_to_goal",
    "possession_streak", "prev_event_enc", "rolling_entropy"
]

    for col in required_features:
        if col not in df.columns:
            print(f"⚠️ Missing column '{col}' added with default value 0")
            df[col] = 0

# --- Now select features safely ---
    X = df[required_features]
    y = df["event_encoded"]

    feature_cols = [
    "x_norm", "y_norm", "team_encoded", "move_dist", "dist_to_goal",
    "possession_streak", "prev_event_enc", "rolling_entropy"
]
    target = "event_encoded"

# --- Validate columns ---
    missing_cols = [col for col in feature_cols if col not in df.columns]
    if missing_cols:
        print(f"⚠️ Missing columns in dataset: {missing_cols}")
        print("→ They will be skipped for now.")
    feature_cols = [col for col in feature_cols if col in df.columns]

    if target not in df.columns:
        raise ValueError(f"❌ Target column '{target}' not found in dataset!")

    if not feature_cols:
        raise ValueError("❌ No valid feature columns found in dataset!")

# --- Build input matrices safely ---
    X = df[feature_cols]
    y = df[target]
    print(f"✅ Using {len(feature_cols)} features: {feature_cols}")

    # Intitialize compponents

    analyzer = ChaosAnalyzer(models_dir=model_dir)
    controller = AdaptiveController(chaos_reference_path=os.path.join(results_dir, "chaos_analysis.json"))

    print(f"Components ready : ChaosAnalyzer + AdapotiveController")

    # --  Run the loop -- 

    records = []
    for i, row in X.iterrows():
        x_row = row.to_frame().T
        chaos_score = analyzer.compute_chaos_score(x_row)

        selected_model_name = controller.select_model(chaos_score)
        model_path = os.path.join(model_dir, f"{selected_model_name}.joblib")
        model = load_model_safe(model_path)

        if model is None:
            continue

        prediction = model.predict(x_row)[0]
        true_value = y.iloc[i]

        records.append({
            "index": i,
            "chaos_score": chaos_score,
            "selected_model": selected_model_name,
            "predicted": prediction,
            "actual": true_value
        })

        if i % 100 == 0:
            print(f"[{i}/{len(X)}] Chaos={chaos_score:.3f} → {selected_model_name} → pred={prediction}")

    # --- Convert to DataFrame ---
    results_df = pd.DataFrame(records)
    results_csv = os.path.join(results_dir, "adaptive_demo.csv")
    results_df.to_csv(results_csv, index=False)
    print(f"\n✅ Results saved: {results_csv}")

    # --- Calculate summary stats ---
    accuracy = (results_df["predicted"] == results_df["actual"]).mean()
    model_usage = results_df["selected_model"].value_counts(normalize=True) * 100

    print_banner("📊 Adaptive Performance Summary")
    print(f"Overall Accuracy: {accuracy:.4f}")
    print(model_usage)

    # --- Visualization 1: Chaos vs Model Choice ---
    plt.figure(figsize=(8, 5))
    plt.scatter(results_df["chaos_score"], results_df["selected_model"], alpha=0.6)
    plt.title("Model Switching vs. Chaos Level")
    plt.xlabel("Chaos Score")
    plt.ylabel("Selected Model")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "adaptive_switch_plot.png"))
    plt.close()

    # --- Visualization 2: Model Usage Frequency ---
    model_usage.plot(kind='bar', color='skyblue', figsize=(6, 4))
    plt.title("Model Selection Frequency")
    plt.ylabel("Usage (%)")
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "model_usage_plot.png"))
    plt.close()

    print("📈 Visualizations saved: adaptive_switch_plot.png, model_usage_plot.png")

    print_banner("✅ Adaptive Demo Complete")
    print("This phase shows real-time adaptive control under chaotic conditions.")


if __name__ == "__main__":
    run_adaptive_demo()






