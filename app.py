import os
import json
import numpy as np
import joblib
from flask import Flask, request, jsonify, render_template

BASE_DIR = os.path.dirname(__file__)
MODELS_DIR = os.path.join(BASE_DIR, "models")
MODEL_PATH = os.path.join(MODELS_DIR, "model.joblib")
META_PATH = os.path.join(MODELS_DIR, "metadata.json")

app = Flask(__name__, template_folder=os.path.join(BASE_DIR, "templates"), static_folder=os.path.join(BASE_DIR, "static"))

model = None
feature_order = None
label_map = None


def load_artifacts():
    global model, feature_order, label_map
    if not os.path.exists(MODEL_PATH) or not os.path.exists(META_PATH):
        return False
    model = joblib.load(MODEL_PATH)
    with open(META_PATH, "r", encoding="utf-8") as f:
        meta = json.load(f)
    feature_order = meta.get("selected_feature_names", [])
    label_map = {int(k): v for k, v in meta.get("target_encoding", {}).items()}
    return True


@app.route("/")
def index():
    if feature_order is None:
        return render_template("index.html", features=[], ready=False)
    return render_template("index.html", features=feature_order, ready=True)


@app.route("/predict", methods=["POST"])
def predict():
    if model is None or feature_order is None:
        return jsonify({"error": "Model not loaded. Train the model first by running train.py."}), 400

    payload = request.get_json(silent=True) or {}

    def parse_yes_no(v):
        if v is None:
            return None
        if isinstance(v, (int, float)):
            return 1.0 if float(v) >= 1 else 0.0
        s = str(v).strip().lower()
        if s in ["yes", "y", "true", "1"]:
            return 1.0
        if s in ["no", "n", "false", "0"]:
            return 0.0
        try:
            return 1.0 if float(s) >= 1 else 0.0
        except Exception:
            return None

    values = []
    for name in feature_order:
        parsed = parse_yes_no(payload.get(name))
        values.append(parsed)

    if any(v is None for v in values):
        return jsonify({"error": "All questions require a Yes/No answer."}), 400

    x = np.array(values, dtype=float).reshape(1, -1)
    proba = model.predict_proba(x)[0]
    risk = float(proba[1])
    risk_percent = round(risk * 100.0, 1)

    if risk_percent < 20:
        stage = "Very low risk"
    elif risk_percent < 40:
        stage = "Low risk"
    elif risk_percent < 60:
        stage = "Moderate risk"
    elif risk_percent < 80:
        stage = "High risk"
    else:
        stage = "Very high risk"

    return jsonify({
        "risk_percent": risk_percent,
        "stage": stage,
        "inputs": {name: int(v) for name, v in zip(feature_order, values)}
    })


if __name__ == "__main__":
    loaded = load_artifacts()
    if not loaded:
        print("Model artifacts not found. Run: python train.py")
    else:
        print("Model loaded.")
    app.run(host="0.0.0.0", port=5000, debug=True)
