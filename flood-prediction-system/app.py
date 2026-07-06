"""
Rising Waters — Flood Prediction Web Application
==================================================
Epic 5: Application Building

Routes:
    /            -> home.html        (landing page)
    /Predict     -> index.html       (input form)
    /result      -> POST handler that runs the model, then redirects to
                     chance.html or no_chance.html
"""
import os
import joblib
import pandas as pd
from flask import Flask, render_template, request

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "floods.save")
SCALER_PATH = os.path.join(BASE_DIR, "models", "transform.save")

app = Flask(__name__)

model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

FEATURE_ORDER = ["Annual_Rainfall", "Cloud_Visibility", "Seasonal_Rainfall",
                 "Temperature", "Humidity"]


@app.route("/")
def home():
    return render_template("home.html")


@app.route("/Predict")
def predict_form():
    return render_template("index.html")


@app.route("/result", methods=["POST"])
def result():
    try:
        input_data = {
            "Annual_Rainfall": float(request.form["annual_rainfall"]),
            "Cloud_Visibility": float(request.form["cloud_visibility"]),
            "Seasonal_Rainfall": float(request.form["seasonal_rainfall"]),
            "Temperature": float(request.form["temperature"]),
            "Humidity": float(request.form["humidity"]),
        }
    except (KeyError, ValueError):
        return render_template("index.html", error="Please fill in all fields with valid numbers.")

    df = pd.DataFrame([input_data], columns=FEATURE_ORDER)
    scaled = scaler.transform(df)
    prediction = model.predict(scaled)[0]

    probability = None
    if hasattr(model, "predict_proba"):
        probability = round(float(model.predict_proba(scaled)[0][1]) * 100, 1)

    if prediction == 1:
        return render_template("chance.html", probability=probability, inputs=input_data)
    else:
        return render_template("no_chance.html", probability=probability, inputs=input_data)


if __name__ == "__main__":
    app.run(debug=True)
