# Cancer Detection ML Web App

- **Train**: Logistic Regression on scikit-learn Breast Cancer dataset using 10 mean-based features.
- **Serve**: Flask API with a simple HTML/CSS/JS UI for input and prediction.

## Setup

- **Python**: 3.9+

## Install dependencies

```bash
pip install -r requirements.txt
```

## Train the model

```bash
python train.py
```

This saves `models/model.joblib` and `models/metadata.json`.

## Run the web app

```bash
python app.py
```

Open http://localhost:5000/

## API

- `POST /predict`

Request JSON keys must match these feature names exactly:

```
[
  "mean radius",
  "mean texture",
  "mean perimeter",
  "mean area",
  "mean smoothness",
  "mean compactness",
  "mean concavity",
  "mean concave points",
  "mean symmetry",
  "mean fractal dimension"
]
```

Example:

```json
{
  "mean radius": 14.3,
  "mean texture": 19.2,
  "mean perimeter": 93.6,
  "mean area": 655.7,
  "mean smoothness": 0.09,
  "mean compactness": 0.11,
  "mean concavity": 0.07,
  "mean concave points": 0.06,
  "mean symmetry": 0.18,
  "mean fractal dimension": 0.06
}
```
