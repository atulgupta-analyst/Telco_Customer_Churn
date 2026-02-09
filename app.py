# import pickle
import json
from flask import Flask, request, jsonify, render_template
import pandas as pd
from xgboost import XGBClassifier

app = Flask(__name__)

# XGBoost model in native format, avoiding pickle due to version issues
xgbmodel = XGBClassifier()
xgbmodel.load_model("xgb_model.json")  

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

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict_api', methods=['POST'])
def predict_api():
    payload = request.get_json(force=True)
    data = payload.get("data", {})

    # Build a 1-row DataFrame with correct columns/order
    X_single = pd.DataFrame(0, index=[0], columns=FEATURE_COLS)

    # Fill only provided keys (ignore unknown keys)
    for k, v in data.items():
        if k in X_single.columns:
            X_single.loc[0, k] = v

    # Predict probability + apply threshold
    churn_prob = float(xgbmodel.predict_proba(X_single)[:, 1][0])
    churn_pred = int(churn_prob >= THRESHOLD)

    return jsonify({
        "churn_probability": round(churn_prob, 4),
        "threshold": THRESHOLD,
        "prediction": churn_pred  # 1 = churn, 0 = no churn
    })

if __name__ == "__main__":
    app.run(debug=True)
