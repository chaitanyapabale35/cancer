import os
import json
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, roc_auc_score
import joblib


SYMPTOM_FEATURES = [
    "Persistent cough",
    "Chest pain",
    "Unexplained weight loss",
    "Fatigue",
    "Shortness of breath",
    "Blood in sputum",
    "Frequent infections",
    "Hoarseness",
    "Difficulty swallowing",
    "Night sweats",
    "Loss of appetite",
    "Smoking history",
]


def generate_synthetic_data(n_samples: int = 5000, random_state: int = 42):
    rng = np.random.default_rng(random_state)

    base_probs = np.array([
        0.2,  # Persistent cough
        0.15, # Chest pain
        0.1,  # Unexplained weight loss
        0.25, # Fatigue
        0.12, # Shortness of breath
        0.05, # Blood in sputum
        0.2,  # Frequent infections
        0.08, # Hoarseness
        0.07, # Difficulty swallowing
        0.18, # Night sweats
        0.22, # Loss of appetite
        0.35, # Smoking history
    ])

    X = (rng.random((n_samples, len(base_probs))) < base_probs).astype(float)

    weights = np.array([
        1.2,  # Persistent cough
        1.1,  # Chest pain
        1.5,  # Unexplained weight loss
        0.8,  # Fatigue
        1.3,  # Shortness of breath
        2.0,  # Blood in sputum
        0.9,  # Frequent infections
        0.7,  # Hoarseness
        0.9,  # Difficulty swallowing
        0.6,  # Night sweats
        1.0,  # Loss of appetite
        1.4,  # Smoking history
    ])

    linear = X @ weights + rng.normal(0, 0.5, size=n_samples) - 2.5
    prob = 1 / (1 + np.exp(-linear))
    y = (rng.random(n_samples) < prob).astype(int)

    return X, y


def main():
    X, y = generate_synthetic_data()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    pipe = Pipeline(
        steps=[
            ("scaler", StandardScaler(with_mean=True, with_std=True)),
            ("clf", LogisticRegression(max_iter=500, solver="liblinear")),
        ]
    )

    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    y_prob = pipe.predict_proba(X_test)[:, 1]
    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)

    models_dir = os.path.join(os.path.dirname(__file__), "models")
    os.makedirs(models_dir, exist_ok=True)

    model_path = os.path.join(models_dir, "model.joblib")
    joblib.dump(pipe, model_path)

    meta = {
        "selected_feature_names": SYMPTOM_FEATURES,
        "target_names": ["no_cancer", "cancer"],
        "metrics": {"accuracy": float(acc), "roc_auc": float(auc)},
        "target_encoding": {"0": "no_cancer", "1": "cancer"},
    }
    with open(os.path.join(models_dir, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"Saved model to {model_path}")
    print(f"Held-out accuracy: {acc:.4f}, ROC-AUC: {auc:.4f}")


if __name__ == "__main__":
    main()
