import json
from pathlib import Path
from flask import Flask, request, jsonify, render_template
import pandas as pd
from xgboost import XGBClassifier

# -----------------------------
# Flask App Initialization
# -----------------------------
app = Flask(__name__)

# -----------------------------
# Model Loading (Heroku Safe)
# -----------------------------
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "xgb_model.json"

xgbmodel = XGBClassifier()

if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")

xgbmodel.load_model(str(MODEL_PATH))

# -----------------------------
# Feature Columns (Fixed Order)
# -----------------------------
FEATURE_COLS = [
    'tenure',
    'MonthlyCharges',
    'Contract_Month-to-month',
    'TotalCharges',
    'InternetService_Fiber optic',
    'PaperlessBilling',
    'PaymentMethod_Electronic check',
    'PaymentMethod_Bank transfer (automatic)',
    'TotalServicesSubscribed',
    'Contract_Two year',
    'DeviceProtection',
    'SeniorCitizen'
]

THRESHOLD = 0.35


# -----------------------------
# Routes
# -----------------------------
@app.route('/')
def home():
    return render_template('home.html')


@app.route('/predict_api', methods=['POST'])
def predict_api():
    try:
        payload = request.get_json(force=True)
        data = payload.get("data", {})

        # Create empty DataFrame with correct columns
        X_single = pd.DataFrame(0, index=[0], columns=FEATURE_COLS)

        # Fill provided features only
        for k, v in data.items():
            if k in X_single.columns:
                X_single.loc[0, k] = v

        # Predict probability
        churn_prob = float(xgbmodel.predict_proba(X_single)[:, 1][0])
        churn_pred = int(churn_prob >= THRESHOLD)

        return jsonify({
            "churn_probability": round(churn_prob, 4),
            "threshold": THRESHOLD,
            "prediction": churn_pred
        })

    except Exception as e:
        return jsonify({
            "error": str(e)
        }), 500


# -----------------------------
# Local Run Only (NOT used by Heroku)
# -----------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
