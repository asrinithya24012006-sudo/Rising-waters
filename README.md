# Rising Waters — Flood Prediction System

A complete flood-risk prediction project: data preprocessing, model
comparison (Decision Tree, Random Forest, KNN, XGBoost), and a Flask web
app that serves live predictions from the winning model.

## Project layout

```
flood-prediction-system/
├── app.py                       # Flask application (Epic 5)
├── requirements.txt
├── data/
│   ├── generate_dataset.py      # builds a synthetic dataset (see note below)
│   └── flood_dataset.csv        # generated dataset
├── notebooks/
│   ├── model_training.py        # EDA, preprocessing, training, comparison (Epics 2–4)
│   └── figures/                 # saved EDA plots (distributions, box plots, heat map)
├── models/
│   ├── floods.save              # best trained model (joblib)
│   ├── transform.save           # fitted StandardScaler (joblib)
│   └── model_info.txt           # which model won and its accuracy
├── templates/
│   ├── home.html
│   ├── index.html
│   ├── chance.html
│   └── no_chance.html
└── static/
    ├── main.css
    └── main.js
```

## About the dataset

The project brief points to a Kaggle dataset
(`arbethi/rainfall-dataset`). This environment has no internet access,
so `data/generate_dataset.py` builds a realistic **synthetic**
stand-in with the same five features (Annual Rainfall, Cloud
Visibility, Seasonal Rainfall, Temperature, Humidity → flood class).

**To use the real data:** download the Kaggle file yourself, save it as
`data/flood_dataset_real.csv` with those same column names, and rerun
`model_training.py` — it automatically prefers the real file if present.

## About XGBoost

`xgboost` isn't installed in this sandbox and there's no network
access to `pip install` it. `model_training.py` tries to import it
first and, only if that fails, falls back to scikit-learn's
`GradientBoostingClassifier` — a very close relative in the same
gradient-boosted-tree family — so the pipeline still runs end to end.
On your own machine, `pip install xgboost` and rerun the script to use
the real thing; no code changes needed.

## Setup

```bash
cd flood-prediction-system
python -m venv venv
source venv/bin/activate        # venv\Scripts\activate on Windows
pip install -r requirements.txt
```

## 1. Generate the dataset (already done, rerun anytime)

```bash
python data/generate_dataset.py
```

## 2. Train and compare models

```bash
python notebooks/model_training.py
```

This runs EDA (saving plots to `notebooks/figures/`), handles missing
values, caps outliers with the IQR method, encodes/scales features,
trains all four models, prints accuracy/confusion-matrix/classification
reports for each, and saves the best-performing model + scaler to
`models/floods.save` and `models/transform.save`.

## 3. Run the web app

```bash
python app.py
```

Then open **http://127.0.0.1:5000/** in your browser:

- `/` — landing page
- `/Predict` — enter the 5 weather readings
- submitting the form runs the saved model and redirects to a
  flood-risk or no-risk result page with the predicted probability

## Notes

- Missing values are imputed with the column median.
- Outliers are capped (not removed) using the IQR method to preserve
  dataset size.
- Features are standardized with `StandardScaler`; the fitted scaler is
  saved alongside the model so real-time form inputs are transformed
  identically to training data.
