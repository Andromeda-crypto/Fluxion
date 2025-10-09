import numpy as np
import joblib
import os
import pandas as pd
from sklearn.metrics import accuracy_score
from src.features import add_engineered_features



def perturb_inputs(X, noise_level=0.05):
    X_peturbed = X.copy()
    for col in X.columns:
        if np.issubdtype(X[col].dtype, np.number):
            noise = np.random.normal(0, noise_level, size=len(X))
            X_peturbed[col] = X[col] * (1 + noise)
    
    return X_peturbed

def eval_chaos_stability(model, X, y, noise_levels=[0.01,0.05,0.1,0.2,0.5]):
    base_prediction = model.predict(X)
    results = []
    for nl in noise_levels:
        X_perturbed = perturb_inputs(X, noise_level = nl)
        perturbed_preds = model.predict(X_perturbed)

        change_rate = np.mean(base_prediction != perturbed_preds) * 100 # percent of prediction changed
        noisy_acc = accuracy_score(y, perturbed_preds)

        results.append({
            "noise_level" : nl,
            "Precision change(%)" : change_rate,
            "Noisy Accuracy": noisy_acc
        })

    return pd.DataFrame(results)


def main():
    ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    model_dir = os.path.join(ROOT, "models")
    data_path = os.path.join(ROOT, "data", "processed", "cleaned_data.csv")
    results_dir = os.path.join(ROOT, "results")
    os.makedirs(results_dir, exist_ok=True)

    # load data

    df = pd.read_csv(data_path)
    df = add_engineered_features(df)

    features = ["x_norm", "y_norm", "team_encoded", "move_dist", "dist_to_goal",
                "possession_streak", "prev_event_enc", "rolling_entropy"]
    target = "event_encoded"

    X = df[features]
    y = df[target]

    models = {
        "randomforest": os.path.join(model_dir, "randomforest.joblib"),
        "gradientboosting": os.path.join(model_dir, "gradientboosting.joblib"),
        "xgboost": os.path.join(model_dir, "xgboost.joblib"),
        "stack": os.path.join(model_dir, "stack.joblib")
    }

    summary = []
    for name, path in models.items():
        if not os.path.exists(path):
            print(f"⚠️ Model file not found: {path}")
            continue

        model = joblib.load(path)
        chaos_df = eval_chaos_stability(model, X, y)
        chaos_df["model"] = name
        summary.append(chaos_df)

        # Save individual model results
        chaos_df.to_csv(os.path.join(results_dir, f"chaos_{name}.csv"), index=False)
        print(f"\n📊 Chaos results for {name}:")
        print(chaos_df)

    if summary:
        final_df = pd.concat(summary, ignore_index=True)
        final_df.to_csv(os.path.join(results_dir, "chaos_summary.csv"), index=False)

        print("\n✅ Chaos Analysis Complete. Summary saved to results/chaos_summary.csv")

        # Compute and show mean change per model
        chaos_summary = (
            final_df.groupby("model")["Precision change(%)"]
            .mean()
            .reset_index()
            .rename(columns={"Precision change(%)": "Avg Prediction Change (%)"})
        )

        print("\n📈 Average Stability per Model:")
        print(chaos_summary)
    else:
        print("No chaos results generated — check model paths.")



if __name__ == "__main__":
    main()





    


