import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score, classification_report

from src.features import add_engineered_features


def train_models(clean_file_path=None, model_dir="models"):
    ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    if clean_file_path is None:
        clean_file_path = os.path.join(ROOT, "data", "processed", "cleaned_data.csv")

    os.makedirs(model_dir, exist_ok=True)
    
    df = pd.read_csv(clean_file_path)
    df = add_engineered_features(df)
     # --- Fallbacks for missing columns ---
    if "x_norm" not in df.columns and "end_x" in df.columns:
        df["x_norm"] = df["end_x"] / 50.0

    if "y_norm" not in df.columns and "end_y" in df.columns:
        df["y_norm"] = df["end_y"] / 25.0

    if "team_encoded" not in df.columns and "team" in df.columns:
        df["team_encoded"] = df["team"].astype("category").cat.codes

    if "event_encoded" not in df.columns and "event" in df.columns:
        df["event_encoded"] = df["event"].astype("category").cat.codes



    features = ["x_norm", "y_norm", "team_encoded", "move_dist", "dist_to_goal",
                "possession_streak", "prev_event_enc", "rolling_entropy"]
    target = "event_encoded"

    


    X = df[features]
    y = df[target]
   


    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Random Forest with tuning
    rf_pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", RandomForestClassifier(class_weight="balanced", random_state=42))
    ])

    rf_param_dist = {
        "clf__n_estimators": [100, 200, 500],
        "clf__max_depth": [None, 8, 16, 32],
        "clf__max_features": ["sqrt", "log2", 0.5]
    }

    rf_search = RandomizedSearchCV(
        rf_pipe, rf_param_dist, n_iter=10, cv=3,
        scoring="f1_weighted", random_state=42
    )
    rf_search.fit(X_train, y_train)
    best_rf = rf_search.best_estimator_
    joblib.dump(best_rf, os.path.join(model_dir, "randomforest.joblib"))

    # Gradient Boosting
    gb_model = GradientBoostingClassifier(n_estimators=200, random_state=42)
    gb_model.fit(X_train, y_train)
    joblib.dump(gb_model, os.path.join(model_dir, "gradientboosting.joblib"))

    # XGBoost
    xgb_model = xgb.XGBClassifier(
        n_estimators=200,
        learning_rate=0.1,
        max_depth=5,
        objective="multi:softmax",
        num_class=len(df[target].unique()),
        random_state=42
    )
    xgb_model.fit(X_train, y_train)
    joblib.dump(xgb_model, os.path.join(model_dir, "xgboost.joblib"))

    # Stacking
    stack = StackingClassifier(
        estimators=[("rf", best_rf), ("xgb", xgb_model)],
        final_estimator=LogisticRegression(max_iter=500),
        n_jobs=-1
    )
    stack.fit(X_train, y_train)
    joblib.dump(stack, os.path.join(model_dir, "stack.joblib"))

    # Evaluation
    models = {
        "RandomForest": best_rf,
        "GradientBoosting": gb_model,
        "XGBoost": xgb_model,
        "Stack": stack
    }

    for name, model in models.items():
        y_pred = model.predict(X_test)
        print(f"\n{name} Results:")
        print("Accuracy:", accuracy_score(y_test, y_pred))
        print("F1 Score:", f1_score(y_test, y_pred, average="weighted"))
        print(classification_report(y_test, y_pred))

    print("\n✅ All models trained and saved successfully.")


if __name__ == "__main__":
    train_models()


    