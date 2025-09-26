import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
import joblib

clean_file_path = "data/processed/cleaned_data.csv"
model_path = "models/baseline_model.joblib"
def train_baseline():
    df = pd.read_csv(clean_file_path)

    df["x_norm"] = df["end_x"] / 50.0
    df["y_norm"] = df["end_y"] / 25.0
    df["team_encoded"] = df["team"].astype("category").cat.codes
    df["event_encoded"] = df["event"].astype("category").cat.codes


    features = ["x_norm", "y_norm", "team_encoded"]
    target = "event_encoded"

    x = df[features]
    y = df[target]

    x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2, random_state=42)

    model = LogisticRegression(max_iter=1000, class_weight="balanced")
    model.fit(x_train,y_train)

    y_pred = model.predict(x_test)
    print("Accuracy", accuracy_score(y_test,y_pred))
    print(classification_report(y_test, y_pred))

    
    os.makedirs("models", exist_ok=True)
    #joblib.dump(model, model_path)
    print(df["event"].value_counts())
    print(f"Baseline model saved at {model_path}")



if __name__ == "__main__":
    train_baseline()

