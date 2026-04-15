"""
Pre-training script — executed once during Docker image build.
Trains 4 supervised ML models with 5-fold cross-validation on the IBM Telco
Customer Churn dataset, then serializes all artifacts to models/artifacts.pkl.

The Streamlit app loads from that pkl at startup (instant, no re-training).
"""

import os
import warnings
import joblib

import numpy as np
import pandas as pd
import shap
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

# ── Column groups ──────────────────────────────────────────────────────────────
NUMERIC_COLS = ["tenure", "MonthlyCharges", "TotalCharges", "SeniorCitizen"]
BINARY_COLS  = ["gender", "Partner", "Dependents", "PhoneService", "PaperlessBilling"]
MULTI_COLS   = [
    "MultipleLines", "InternetService", "OnlineSecurity", "OnlineBackup",
    "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies",
    "Contract", "PaymentMethod",
]


def load_data() -> pd.DataFrame:
    df = pd.read_csv("data/Telco-Customer-Churn.csv")
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    median_tc = df["TotalCharges"].median()
    df["TotalCharges"] = df["TotalCharges"].fillna(median_tc)
    df = df.drop(columns=["customerID"])
    df["Churn"] = (df["Churn"] == "Yes").astype(int)
    return df


def build_preprocessor() -> ColumnTransformer:
    numeric_tf = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
    ])
    binary_tf = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("ohe",     OneHotEncoder(drop="first", sparse_output=False)),
    ])
    cat_tf = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("ohe",     OneHotEncoder(
            drop="first", sparse_output=False, handle_unknown="ignore"
        )),
    ])
    return ColumnTransformer([
        ("num", numeric_tf, NUMERIC_COLS),
        ("bin", binary_tf,  BINARY_COLS),
        ("cat", cat_tf,     MULTI_COLS),
    ])


def clean_feature_names(names) -> list[str]:
    return [n.split("__", 1)[1] if "__" in n else n for n in names]


def main() -> None:
    print("── Loading data ──────────────────────────────────────")
    df = load_data()
    X  = df.drop(columns=["Churn"])
    y  = df["Churn"]
    print(f"   Shape: {df.shape}  |  Churn rate: {y.mean()*100:.1f}%")

    preprocessor = build_preprocessor()

    pos_weight = float((y == 0).sum()) / float((y == 1).sum())

    MODELS = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000, class_weight="balanced", random_state=42,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=100, class_weight="balanced",
            random_state=42, n_jobs=-1,
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=100, random_state=42,
        ),
        "XGBoost": XGBClassifier(
            n_estimators=100, random_state=42, eval_metric="logloss",
            scale_pos_weight=pos_weight, n_jobs=-1,
        ),
    }

    cv      = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    SCORING = ["accuracy", "f1_weighted", "roc_auc"]

    cv_results   = {}
    final_models = {}

    print("\n── Cross-validation (5-fold stratified) ─────────────")
    for name, clf in MODELS.items():
        pipe   = Pipeline([("prep", preprocessor), ("clf", clf)])
        scores = cross_validate(pipe, X, y, cv=cv, scoring=SCORING)
        cv_results[name] = {
            "Accuracy":      round(float(scores["test_accuracy"].mean()),      4),
            "F1 (Weighted)": round(float(scores["test_f1_weighted"].mean()),   4),
            "ROC-AUC":       round(float(scores["test_roc_auc"].mean()),       4),
        }
        print(f"   {name:<22}  AUC={cv_results[name]['ROC-AUC']:.4f}  "
              f"Acc={cv_results[name]['Accuracy']:.4f}  "
              f"F1={cv_results[name]['F1 (Weighted)']:.4f}")
        # Fit final model on full dataset
        pipe.fit(X, y)
        final_models[name] = pipe

    # Train/test split for evaluation plots
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    eval_models = {}
    print("\n── Training eval models (80/20 split) ───────────────")
    for name, clf in MODELS.items():
        pipe = Pipeline([("prep", preprocessor), ("clf", clf)])
        pipe.fit(X_tr, y_tr)
        eval_models[name] = pipe
        print(f"   {name} ✓")

    # SHAP values — XGBoost TreeExplainer on 500-row sample
    print("\n── Computing SHAP values (XGBoost, 500 samples) ────")
    xgb_pipe   = final_models["XGBoost"]
    prep_fit   = xgb_pipe.named_steps["prep"]
    X_sample   = X.sample(500, random_state=42)
    X_trans    = prep_fit.transform(X_sample)
    feat_names = clean_feature_names(list(prep_fit.get_feature_names_out()))
    explainer  = shap.TreeExplainer(xgb_pipe.named_steps["clf"])
    shap_vals  = explainer.shap_values(X_trans)
    print(f"   SHAP values shape: {shap_vals.shape}")

    # Persist everything
    print("\n── Saving artifacts ─────────────────────────────────")
    os.makedirs("models", exist_ok=True)
    artifacts = {
        "df":           df,
        "X":            X,
        "y":            y,
        "X_te":         X_te,
        "y_te":         y_te,
        "cv_results":   cv_results,
        "final_models": final_models,
        "eval_models":  eval_models,
        "shap_vals":    shap_vals,
        "feat_names":   feat_names,
        "X_trans":      X_trans,
    }
    joblib.dump(artifacts, "models/artifacts.pkl", compress=3)
    print("   Saved → models/artifacts.pkl")
    print("\n✓ Training complete.")


if __name__ == "__main__":
    main()
