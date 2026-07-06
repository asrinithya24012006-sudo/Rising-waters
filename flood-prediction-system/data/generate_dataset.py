"""
Generates a synthetic flood-prediction dataset.

The public Kaggle dataset referenced in the project spec
(arbethi/rainfall-dataset) is not reachable from this environment
(no internet access), so this script builds a realistic synthetic
stand-in with the same five features described in the ER diagram:

    Annual_Rainfall, Cloud_Visibility, Seasonal_Rainfall,
    Temperature, Humidity  ->  class (0 = No Flood, 1 = Flood)

If you have the real Kaggle file, just drop it in this folder as
`flood_dataset.csv` with the same column names and re-run
`model_training.py` — it will use the real file automatically.
"""
import numpy as np
import pandas as pd

np.random.seed(42)
N = 2000

annual_rainfall = np.random.normal(1200, 350, N).clip(200, 3000)
seasonal_rainfall = (annual_rainfall * np.random.uniform(0.45, 0.75, N)).clip(100, 2200)
cloud_visibility = np.random.normal(6, 2.5, N).clip(0, 10)
temperature = np.random.normal(27, 4, N).clip(10, 45)
humidity = np.random.normal(70, 12, N).clip(20, 100)

# Introduce some missing values and outliers, like a real-world dataset
missing_idx = np.random.choice(N, size=int(N * 0.02), replace=False)
humidity_with_na = humidity.copy()
humidity_with_na[missing_idx] = np.nan

outlier_idx = np.random.choice(N, size=int(N * 0.01), replace=False)
annual_rainfall[outlier_idx] *= 2.5

# Flood risk score combines the features (higher rainfall/humidity,
# lower cloud visibility -> higher flood risk)
risk_score = (
    0.0025 * annual_rainfall
    + 0.0035 * seasonal_rainfall
    + 0.03 * humidity
    - 0.15 * cloud_visibility
    + 0.01 * (temperature - 27)
    + np.random.normal(0, 1.0, N)
)

threshold = np.percentile(risk_score, 55)
flood_class = (risk_score > threshold).astype(int)

df = pd.DataFrame({
    "Annual_Rainfall": annual_rainfall.round(2),
    "Cloud_Visibility": cloud_visibility.round(2),
    "Seasonal_Rainfall": seasonal_rainfall.round(2),
    "Temperature": temperature.round(2),
    "Humidity": humidity_with_na.round(2),
    "class": flood_class,
})

out_path = "flood_dataset.csv"
df.to_csv(out_path, index=False)
print(f"Saved {len(df)} rows to {out_path}")
print(df["class"].value_counts())
