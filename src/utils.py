import os
import json
import joblib
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

# =============== PATH UTILITIES ===============
def ensure_dir(path):
    """Create a directory if it doesn't exist."""
    os.makedirs(path, exist_ok=True)


def get_project_root():
    """Return the absolute path to the project root."""
    return os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))


# =============== MODEL UTILITIES ===============
def save_model_safe(model, path):
    """Safely save a model to disk using joblib."""
    ensure_dir(os.path.dirname(path))
    try:
        joblib.dump(model, path)
        print(f"✅ Model saved: {path}")
    except Exception as e:
        print(f"❌ Failed to save model at {path}: {e}")


def load_model_safe(path):
    """Safely load a model from disk."""
    if not os.path.exists(path):
        print(f"⚠️ Model file not found: {path}")
        return None
    try:
        model = joblib.load(path)
        return model
    except Exception as e:
        print(f"❌ Failed to load model {path}: {e}")
        return None


# =============== JSON UTILITIES ===============
def save_json_safe(path, data):
    """Save a dictionary to a JSON file safely."""
    ensure_dir(os.path.dirname(path))
    try:
        with open(path, 'w') as f:
            json.dump(data, f, indent=4)
        print(f"✅ JSON saved: {path}")
    except Exception as e:
        print(f"❌ Error saving JSON at {path}: {e}")


def load_json_safe(path):
    """Load JSON data safely."""
    if not os.path.exists(path):
        print(f"⚠️ JSON file not found: {path}")
        return {}
    try:
        with open(path, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"❌ Failed to load JSON from {path}: {e}")
        return {}


# =============== DATA UTILITIES ===============
def load_data(path):
    """
    Load CSV or JSON data into a pandas DataFrame.
    Handles both training and chaos input data gracefully.
    """
    if not os.path.exists(path):
        print(f"⚠️ Data file not found: {path}")
        return pd.DataFrame()

    try:
        if path.endswith(".csv"):
            df = pd.read_csv(path)
        elif path.endswith(".json"):
            df = pd.read_json(path)
        else:
            raise ValueError("Unsupported file format. Use CSV or JSON.")
        print(f"📂 Loaded data: {path} (shape={df.shape})")
        return df
    except Exception as e:
        print(f"❌ Failed to load data from {path}: {e}")
        return pd.DataFrame()


# =============== LOGGING UTILITIES ===============
def log_event(event_type, details):
    """
    Append key events (like chaos metrics, model choices, predictions) to a central log file.
    """
    log_path = os.path.join(get_project_root(), "logs", "adaptive_log.json")
    ensure_dir(os.path.dirname(log_path))

    entry = {
        "timestamp": datetime.utcnow().isoformat(),
        "event_type": event_type,
        "details": details
    }

    # Load existing logs
    logs = []
    if os.path.exists(log_path):
        try:
            with open(log_path, "r") as f:
                logs = json.load(f)
        except json.JSONDecodeError:
            logs = []

    # Append and save
    logs.append(entry)
    with open(log_path, "w") as f:
        json.dump(logs, f, indent=4)

    print(f"🧠 Logged event: {event_type}")


def print_banner(title):
    """Print a clean, visually distinct section header."""
    print("\n" + "=" * 60)
    print(f"🧩 {title}")
    print("=" * 60 + "\n")


# =============== PLOTTING UTILITIES ===============
def save_plot(df, x, y, title, save_path):
    """Save a simple line plot (used in chaos analysis visuals)."""
    ensure_dir(os.path.dirname(save_path))
    try:
        plt.figure(figsize=(6, 4))
        plt.plot(df[x], df[y], marker='o')
        plt.title(title)
        plt.xlabel(x)
        plt.ylabel(y)
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
        print(f"📊 Plot saved to {save_path}")
    except Exception as e:
        print(f"❌ Failed to save plot: {e}")
