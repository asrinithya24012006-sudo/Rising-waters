"""
Rising Waters — Flood Prediction Model Training Pipeline
=========================================================
Follows the epics from the project spec:
  Epic 2: Visualizing & Analysing the Data
  Epic 3: Data Pre-Processing
  Epic 4: Model Building (Decision Tree, Random Forest, KNN, XGBoost/GBM)

Run from the project root:
    python notebooks/model_training.py
"""
import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # headless, no display needed
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Try real XGBoost; fall back to sklearn's GradientBoostingClassifier
# (same gradient-boosted-tree family) if the package isn't installed.
try:
    from xgboost import XGBClassifier
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "models")
FIG_DIR = os.path.join(BASE_DIR, "notebooks", "figures")
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)


# ---------------------------------------------------------------------
# Epic 1 & 2: Data Collection + Visualizing & Analysing the Data
# ---------------------------------------------------------------------
def load_data():
    real_path = os.path.join(DATA_DIR, "flood_dataset_real.csv")
    synth_path = os.path.join(DATA_DIR, "flood_dataset.csv")
    path = real_path if os.path.exists(real_path) else synth_path
    df = pd.read_csv(path)
    print(f"Loaded dataset from: {path}")
    print("Shape:", df.shape)
    print("\nHead:\n", df.head())
    print("\nInfo:")
    df.info()
    print("\nDescribe:\n", df.describe())
    return df


def eda(df):
    numeric_cols = [c for c in df.columns if c != "class"]

    # Univariate: distribution plots
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for ax, col in zip(axes.flat, numeric_cols):
        sns.histplot(df[col].dropna(), kde=True, ax=ax)
        ax.set_title(f"Distribution: {col}")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "univariate_distributions.png"))
    plt.close()

    # Univariate: box plots (outlier detection)
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for ax, col in zip(axes.flat, numeric_cols):
        sns.boxplot(x=df[col], ax=ax)
        ax.set_title(f"Box Plot: {col}")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "boxplots.png"))
    plt.close()

    # Multivariate: heat map
    plt.figure(figsize=(8, 6))
    sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Feature Correlation Heat Map")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "heatmap.png"))
    plt.close()

    print(f"\nEDA figures saved to {FIG_DIR}")


# ---------------------------------------------------------------------
# Epic 3: Data Pre-Processing
# ---------------------------------------------------------------------
def handle_missing_values(df):
    print("\nMissing values before handling:\n", df.isnull().sum())
    for col in df.columns:
        if df[col].isnull().any():
            df[col] = df[col].fillna(df[col].median())
    print("\nMissing values after handling:\n", df.isnull().sum())
    return df


def cap_outliers_iqr(df, cols):
    for col in cols:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        df[col] = np.where(df[col] < lower, lower, df[col])
        df[col] = np.where(df[col] > upper, upper, df[col])
    print(f"\nOutliers capped (IQR method) for columns: {cols}")
    return df


# ---------------------------------------------------------------------
# Epic 4: Model Building
# ---------------------------------------------------------------------
def evaluate(name, model, X_test, y_test, y_pred):
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    print(f"\n--- {name} ---")
    print("Accuracy:", round(acc * 100, 2), "%")
    print("Confusion Matrix:\n", cm)
    print("Classification Report:\n", report)
    return acc


def decisiontree(X_train, X_test, y_train, y_test):
    model = DecisionTreeClassifier(random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = evaluate("Decision Tree", model, X_test, y_test, y_pred)
    return model, y_pred, acc


def randomForest(X_train, X_test, y_train, y_test):
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = evaluate("Random Forest", model, X_test, y_test, y_pred)
    return model, y_pred, acc


def KNN(X_train, X_test, y_train, y_test):
    model = KNeighborsClassifier(n_neighbors=5)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = evaluate("K-Nearest Neighbors", model, X_test, y_test, y_pred)
    return model, y_pred, acc


def boosted_model(X_train, X_test, y_train, y_test):
    if HAS_XGBOOST:
        model = XGBClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.1,
            use_label_encoder=False, eval_metric="logloss", random_state=42,
        )
        label = "XGBoost"
    else:
        # Fallback used in this sandbox because xgboost isn't installed
        # and there's no internet access to pip install it. Functionally
        # equivalent gradient-boosted-tree algorithm.
        model = GradientBoostingClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.1, random_state=42,
        )
        label = "GradientBoosting (XGBoost fallback)"
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = evaluate(label, model, X_test, y_test, y_pred)
    return model, y_pred, acc, label


def compareModel(results):
    print("\n=== Model Comparison ===")
    for name, acc in results.items():
        print(f"{name}: {round(acc * 100, 2)}%")
    best_name = max(results, key=results.get)
    print(f"\nBest model: {best_name} ({round(results[best_name] * 100, 2)}%)")
    return best_name


def main():
    df = load_data()
    eda(df)

    df = handle_missing_values(df)
    feature_cols = ["Annual_Rainfall", "Cloud_Visibility", "Seasonal_Rainfall",
                     "Temperature", "Humidity"]
    df = cap_outliers_iqr(df, feature_cols)

    X = df[feature_cols]
    y = df["class"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    models = {}
    results = {}

    dt_model, _, dt_acc = decisiontree(X_train_scaled, X_test_scaled, y_train, y_test)
    models["Decision Tree"] = dt_model
    results["Decision Tree"] = dt_acc

    rf_model, _, rf_acc = randomForest(X_train_scaled, X_test_scaled, y_train, y_test)
    models["Random Forest"] = rf_model
    results["Random Forest"] = rf_acc

    knn_model, _, knn_acc = KNN(X_train_scaled, X_test_scaled, y_train, y_test)
    models["K-Nearest Neighbors"] = knn_model
    results["K-Nearest Neighbors"] = knn_acc

    xgb_model, _, xgb_acc, xgb_label = boosted_model(X_train_scaled, X_test_scaled, y_train, y_test)
    models[xgb_label] = xgb_model
    results[xgb_label] = xgb_acc

    best_name = compareModel(results)
    best_model = models[best_name]

    # Save final model + scaler for deployment (Epic 4, Story 6)
    joblib.dump(best_model, os.path.join(MODEL_DIR, "floods.save"))
    joblib.dump(scaler, os.path.join(MODEL_DIR, "transform.save"))
    with open(os.path.join(MODEL_DIR, "model_info.txt"), "w") as f:
        f.write(f"Best model: {best_name}\n")
        f.write(f"Accuracy: {round(results[best_name] * 100, 2)}%\n")
        f.write(f"Features (in order): {feature_cols}\n")

    print(f"\nSaved best model ({best_name}) to models/floods.save")
    print("Saved scaler to models/transform.save")


if __name__ == "__main__":
    main()
