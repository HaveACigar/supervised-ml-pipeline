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
from sklearn.datasets import fetch_openml
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

try:
    from xgboost import XGBClassifier
    XGB_AVAILABLE = True
except Exception:
    XGB_AVAILABLE = False

warnings.filterwarnings("ignore")

def load_telco_data() -> tuple[pd.DataFrame, dict]:
    df = pd.read_csv("data/Telco-Customer-Churn.csv")
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    median_tc = df["TotalCharges"].median()
    df["TotalCharges"] = df["TotalCharges"].fillna(median_tc)
    df = df.drop(columns=["customerID"])
    df["Churn"] = (df["Churn"] == "Yes").astype(int)
    numeric_cols = ["tenure", "MonthlyCharges", "TotalCharges", "SeniorCitizen"]
    binary_cols = ["gender", "Partner", "Dependents", "PhoneService", "PaperlessBilling"]
    multi_cols = [
        "MultipleLines", "InternetService", "OnlineSecurity", "OnlineBackup",
        "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies",
        "Contract", "PaymentMethod",
    ]
    meta = {
        "dataset_key": "telco_ibm",
        "dataset_name": "IBM Telco Customer Churn",
        "dataset_description": "7,043 telecom subscribers, 20 features.",
        "numeric_cols": numeric_cols,
        "binary_cols": binary_cols,
        "multi_cols": multi_cols,
        "categorical_cols": binary_cols + multi_cols,
        "prediction_mode": "form",
        "cv_folds": 5,
    }
    return df, meta


def load_kdd_data() -> tuple[pd.DataFrame, dict]:
    ds = fetch_openml(data_id=1112, as_frame=True)
    X = ds.data.copy()
    y = ds.target.astype(str).str.strip().replace({"1": 1, "-1": 0}).astype(int)
    df = X.copy()
    df["Churn"] = y

    numeric_cols = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
    categorical_cols = [c for c in X.columns if c not in numeric_cols]
    meta = {
        "dataset_key": "kddcup09",
        "dataset_name": "KDDCup09 Churn (OpenML)",
        "dataset_description": "50,000 rows and 230 predictor columns.",
        "numeric_cols": numeric_cols,
        "binary_cols": [],
        "multi_cols": categorical_cols,
        "categorical_cols": categorical_cols,
        "prediction_mode": "sample_row",
        "cv_folds": 3,
    }
    return df, meta


def load_data() -> tuple[pd.DataFrame, dict]:
    dataset_key = os.getenv("CHURN_DATASET", "kddcup09").strip().lower()
    if dataset_key in {"telco", "telco_ibm", "ibm"}:
        return load_telco_data()
    if dataset_key in {"kdd", "kddcup09", "openml_kdd"}:
        return load_kdd_data()
    raise ValueError(
        "Unsupported CHURN_DATASET value. Use one of: telco_ibm, kddcup09"
    )


def build_preprocessor(
    numeric_cols: list[str],
    categorical_cols: list[str],
    dataset_key: str,
) -> ColumnTransformer:
    numeric_tf = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
    ])
    if dataset_key == "kddcup09":
        # KDD has many high-cardinality categoricals; ordinal encoding keeps feature space bounded.
        cat_tf = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("ordinal", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
        ])
    else:
        cat_tf = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("ohe",     OneHotEncoder(
                drop="if_binary", sparse_output=False, handle_unknown="ignore"
            )),
        ])
    transformers = []
    if numeric_cols:
        transformers.append(("num", numeric_tf, numeric_cols))
    if categorical_cols:
        transformers.append(("cat", cat_tf, categorical_cols))
    return ColumnTransformer(transformers)


def clean_feature_names(names) -> list[str]:
    return [n.split("__", 1)[1] if "__" in n else n for n in names]


def main() -> None:
    print("── Loading data ──────────────────────────────────────")
    df, dataset_meta = load_data()
    X  = df.drop(columns=["Churn"])
    y  = df["Churn"]
    print(f"   Dataset: {dataset_meta['dataset_name']}")
    print(f"   Shape: {df.shape}  |  Churn rate: {y.mean()*100:.1f}%")

    preprocessor = build_preprocessor(
        dataset_meta["numeric_cols"],
        dataset_meta["categorical_cols"],
        dataset_meta["dataset_key"],
    )

    pos_weight = float((y == 0).sum()) / float((y == 1).sum())

    if dataset_meta["dataset_key"] == "kddcup09":
        MODELS = {
            "Logistic Regression": LogisticRegression(
                max_iter=1000, class_weight="balanced", random_state=42,
            ),
            "Random Forest": RandomForestClassifier(
                n_estimators=200, class_weight="balanced_subsample",
                random_state=42, n_jobs=-1,
            ),
            "Hist Gradient Boosting": HistGradientBoostingClassifier(
                max_iter=200, learning_rate=0.05, random_state=42,
            ),
        }
    else:
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
        }
    if XGB_AVAILABLE:
        MODELS["XGBoost"] = XGBClassifier(
            n_estimators=100, random_state=42, eval_metric="logloss",
            scale_pos_weight=pos_weight, n_jobs=-1,
        )
    else:
        print("   XGBoost unavailable in this environment; continuing without it.")

    cv_folds = int(dataset_meta.get("cv_folds", 5))
    cv      = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    SCORING = ["accuracy", "f1_weighted", "roc_auc"]

    cv_results   = {}
    final_models = {}

    print(f"\n── Cross-validation ({cv_folds}-fold stratified) ─────────────")
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

    # SHAP values — use XGBoost when available, otherwise Random Forest
    shap_model_name = "XGBoost" if "XGBoost" in final_models else "Random Forest"
    print(f"\n── Computing SHAP values ({shap_model_name}, 500 samples) ────")
    shap_pipe  = final_models[shap_model_name]
    prep_fit   = shap_pipe.named_steps["prep"]
    X_sample   = X.sample(500, random_state=42)
    X_trans    = prep_fit.transform(X_sample)
    feat_names = clean_feature_names(list(prep_fit.get_feature_names_out()))
    explainer  = shap.TreeExplainer(shap_pipe.named_steps["clf"])
    shap_vals  = explainer.shap_values(X_trans, check_additivity=False)
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
        "dataset_meta": dataset_meta,
        "shap_model_name": shap_model_name,
        "cv_folds": cv_folds,
    }
    joblib.dump(artifacts, "models/artifacts.pkl", compress=3)
    print("   Saved → models/artifacts.pkl")
    print("\n✓ Training complete.")


if __name__ == "__main__":
    main()
