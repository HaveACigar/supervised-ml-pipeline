"""
Supervised ML: Customer Churn Prediction Pipeline
IBM Telco Customer Churn dataset — 7,043 telecom customers, 20 features.
"""

import warnings
warnings.filterwarnings("ignore")

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shap
from sklearn.metrics import auc, classification_report, confusion_matrix, roc_curve

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Churn Prediction Pipeline",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Palette / helpers ──────────────────────────────────────────────────────────
CHURN_COLORS  = {"No": "#2196F3", "Yes": "#F44336"}
MODEL_COLORS  = ["#4CAF50", "#2196F3", "#FF9800", "#9C27B0"]
TEMPLATE      = "plotly_dark"

NUMERIC_COLS  = ["tenure", "MonthlyCharges", "TotalCharges", "SeniorCitizen"]
BINARY_COLS   = ["gender", "Partner", "Dependents", "PhoneService", "PaperlessBilling"]
MULTI_COLS    = [
    "MultipleLines", "InternetService", "OnlineSecurity", "OnlineBackup",
    "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies",
    "Contract", "PaymentMethod",
]


def _clean(names) -> list[str]:
    return [n.split("__", 1)[1] if "__" in n else n for n in names]


# ── Load pre-trained artifacts ──────────────────────────────────────────────────
@st.cache_resource
def load_artifacts() -> dict:
    return joblib.load("models/artifacts.pkl")


# ── Section: Overview ──────────────────────────────────────────────────────────
def render_overview(df: pd.DataFrame) -> None:
    st.subheader("Dataset Overview")

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Customers", f"{len(df):,}")
    m2.metric("Features", len(df.columns) - 1)
    m3.metric("Churn Rate", f"{df['Churn'].mean() * 100:.1f}%")
    m4.metric("Missing Values", int(df.isnull().sum().sum()))

    col1, col2 = st.columns([1, 2])

    with col1:
        counts = df["Churn"].value_counts().rename({0: "No", 1: "Yes"})
        fig = px.pie(
            names=counts.index,
            values=counts.values,
            color=counts.index,
            color_discrete_map=CHURN_COLORS,
            title="Churn Distribution",
            hole=0.45,
            template=TEMPLATE,
        )
        fig.update_traces(textinfo="percent+label")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        display = df.copy()
        display["Churn"] = display["Churn"].map({0: "No", 1: "Yes"})
        st.markdown("**Sample rows**")
        st.dataframe(display.head(8), use_container_width=True, height=310)

    st.markdown("**Numeric Feature Statistics**")
    st.dataframe(
        df[NUMERIC_COLS].describe().round(2), use_container_width=True
    )


# ── Section: Feature Explorer ──────────────────────────────────────────────────
def render_feature_explorer(df: pd.DataFrame) -> None:
    st.subheader("Feature Explorer")
    all_features = NUMERIC_COLS + BINARY_COLS + MULTI_COLS
    feature = st.selectbox("Select a feature to explore:", all_features)

    churn_label = df["Churn"].map({0: "No", 1: "Yes"})
    col1, col2  = st.columns(2)

    if feature in NUMERIC_COLS:
        with col1:
            fig = px.histogram(
                df.assign(Churn=churn_label),
                x=feature, color="Churn",
                barmode="overlay", nbins=40, opacity=0.7,
                color_discrete_map=CHURN_COLORS,
                title=f"{feature} — Distribution by Churn",
                template=TEMPLATE,
            )
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            fig = px.box(
                df.assign(Churn=churn_label),
                x="Churn", y=feature,
                color="Churn",
                color_discrete_map=CHURN_COLORS,
                title=f"{feature} — Box Plot by Churn",
                template=TEMPLATE,
            )
            st.plotly_chart(fig, use_container_width=True)
    else:
        tmp = (
            df.assign(Churn=churn_label)
            .groupby([feature, "Churn"])
            .size()
            .reset_index(name="Count")
        )
        with col1:
            fig = px.bar(
                tmp, x=feature, y="Count", color="Churn",
                barmode="group",
                color_discrete_map=CHURN_COLORS,
                title=f"{feature} — Count by Churn",
                template=TEMPLATE,
            )
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            rate = df.groupby(feature)["Churn"].mean().reset_index()
            rate.columns = [feature, "Churn Rate (%)"]
            rate["Churn Rate (%)"] = (rate["Churn Rate (%)"] * 100).round(1)
            fig = px.bar(
                rate.sort_values("Churn Rate (%)", ascending=False),
                x=feature, y="Churn Rate (%)",
                color="Churn Rate (%)",
                color_continuous_scale="RdYlGn_r",
                title=f"{feature} — Churn Rate (%)",
                template=TEMPLATE,
            )
            st.plotly_chart(fig, use_container_width=True)

    st.markdown("**Pearson Correlation with Churn (numeric features)**")
    corr = (
        df[NUMERIC_COLS + ["Churn"]]
        .corr()["Churn"]
        .drop("Churn")
        .reset_index()
    )
    corr.columns = ["Feature", "Correlation"]
    fig = px.bar(
        corr, x="Feature", y="Correlation",
        color="Correlation",
        color_continuous_scale="RdBu",
        color_continuous_midpoint=0,
        title="Pearson Correlation with Churn",
        template=TEMPLATE,
    )
    st.plotly_chart(fig, use_container_width=True)


# ── Section: Preprocessing ─────────────────────────────────────────────────────
def render_preprocessing() -> None:
    st.subheader("Preprocessing Pipeline")
    steps = [
        (
            "1. Drop customerID",
            "Removes the `customerID` column — a non-informative identifier that would "
            "leak row identity into model training.",
        ),
        (
            "2. Encode target",
            "`Churn` → binary integer (Yes → 1, No → 0).",
        ),
        (
            "3. Fix TotalCharges",
            "Some rows contain whitespace instead of a numeric value. "
            "Coerced to numeric; 11 resulting NaN rows filled with the column median.",
        ),
        (
            "4. Scale numeric features",
            "`tenure`, `MonthlyCharges`, `TotalCharges`, `SeniorCitizen` → "
            "`StandardScaler` (zero mean, unit variance). "
            "Median imputation applied first inside the pipeline.",
        ),
        (
            "5. Encode binary features",
            "`gender`, `Partner`, `Dependents`, `PhoneService`, `PaperlessBilling` → "
            "`OneHotEncoder(drop='first')` — produces a single binary column per feature.",
        ),
        (
            "6. Encode multi-category features",
            "`MultipleLines`, `InternetService`, `OnlineSecurity`, `OnlineBackup`, "
            "`DeviceProtection`, `TechSupport`, `StreamingTV`, `StreamingMovies`, "
            "`Contract`, `PaymentMethod` → `OneHotEncoder(drop='first', handle_unknown='ignore')`.",
        ),
        (
            "7. sklearn Pipeline + ColumnTransformer",
            "All steps are wrapped in a `ColumnTransformer` (applied per column group) "
            "and then a `Pipeline([('prep', preprocessor), ('clf', model)])` per model. "
            "This guarantees zero data leakage during cross-validation — "
            "the scaler and encoder are fit only on training folds.",
        ),
    ]

    for title, desc in steps:
        with st.expander(title, expanded=False):
            st.markdown(desc)

    st.code(
        """ColumnTransformer([
    ('num', Pipeline([SimpleImputer(median), StandardScaler()]),  numeric_cols),
    ('bin', Pipeline([SimpleImputer(mode),   OneHotEncoder(drop='first')]), binary_cols),
    ('cat', Pipeline([SimpleImputer(mode),   OneHotEncoder(drop='first')]), multi_cols),
])
# Wrapped per model:
Pipeline([('prep', ColumnTransformer), ('clf', <estimator>)])""",
        language="python",
    )


# ── Section: Model Comparison ──────────────────────────────────────────────────
def render_model_comparison(cv_results: dict) -> None:
    st.subheader("Model Comparison")
    st.caption("5-fold stratified cross-validation on the full dataset (n = 7,043)")

    df_res = (
        pd.DataFrame(cv_results)
        .T.reset_index()
        .rename(columns={"index": "Model"})
        .sort_values("ROC-AUC", ascending=False)
    )

    st.dataframe(
        df_res.style
        .format({"Accuracy": "{:.3f}", "F1 (Weighted)": "{:.3f}", "ROC-AUC": "{:.3f}"})
        .background_gradient(subset=["ROC-AUC"],       cmap="Greens")
        .background_gradient(subset=["Accuracy"],      cmap="Blues")
        .background_gradient(subset=["F1 (Weighted)"], cmap="Oranges"),
        use_container_width=True,
        hide_index=True,
    )

    df_melt = df_res.melt(id_vars="Model", var_name="Metric", value_name="Score")
    fig = px.bar(
        df_melt, x="Model", y="Score", color="Metric",
        barmode="group",
        color_discrete_sequence=MODEL_COLORS,
        title="Cross-Validation Performance by Model",
        template=TEMPLATE,
    )
    fig.update_layout(yaxis_range=[0.6, 1.0])
    st.plotly_chart(fig, use_container_width=True)

    best = df_res.iloc[0]["Model"]
    st.success(f"**Best model by ROC-AUC: {best}**")


# ── Section: Evaluation ────────────────────────────────────────────────────────
def render_evaluation(X_te, y_te, eval_models: dict) -> None:
    st.subheader("Model Evaluation — Hold-Out Test Set (20%)")

    model_names = list(eval_models.keys())
    default_idx = model_names.index("XGBoost") if "XGBoost" in model_names else 0
    selected    = st.selectbox("Select model:", model_names, index=default_idx)
    pipe        = eval_models[selected]

    y_pred  = pipe.predict(X_te)
    y_proba = pipe.predict_proba(X_te)[:, 1]

    col1, col2 = st.columns(2)

    with col1:
        cm  = confusion_matrix(y_te, y_pred)
        fig = px.imshow(
            cm,
            text_auto=True,
            x=["Predicted: No", "Predicted: Yes"],
            y=["Actual: No",    "Actual: Yes"],
            color_continuous_scale="Blues",
            title=f"Confusion Matrix — {selected}",
            template=TEMPLATE,
        )
        fig.update_coloraxes(showscale=False)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        report = classification_report(
            y_te, y_pred,
            target_names=["No Churn", "Churn"],
            output_dict=True,
        )
        df_rep = pd.DataFrame(report).T.drop("accuracy").round(3)
        st.markdown(f"**Classification Report — {selected}**")
        st.dataframe(df_rep, use_container_width=True)

    st.markdown("**ROC Curves — All Models**")
    fig = go.Figure()
    fig.add_shape(
        type="line", x0=0, y0=0, x1=1, y1=1,
        line=dict(dash="dash", color="gray"),
    )
    for i, (name, mdl) in enumerate(eval_models.items()):
        proba      = mdl.predict_proba(X_te)[:, 1]
        fpr, tpr, _ = roc_curve(y_te, proba)
        roc_auc    = auc(fpr, tpr)
        fig.add_trace(go.Scatter(
            x=fpr, y=tpr,
            name=f"{name}  (AUC = {roc_auc:.3f})",
            mode="lines",
            line=dict(width=2, color=MODEL_COLORS[i]),
        ))
    fig.update_layout(
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
        title="ROC Curves — All Models",
        template=TEMPLATE,
        legend=dict(x=0.55, y=0.05),
    )
    st.plotly_chart(fig, use_container_width=True)


# ── Section: Feature Importance ────────────────────────────────────────────────
def render_feature_importance(shap_vals, feat_names: list, X_trans) -> None:
    st.subheader("Feature Importance — SHAP (XGBoost)")
    st.caption(
        "SHAP TreeExplainer on a 500-row sample. "
        "Mean |SHAP| = average magnitude of a feature's contribution to the model output."
    )

    mean_shap = np.abs(shap_vals).mean(axis=0)
    imp_df    = (
        pd.DataFrame({"Feature": feat_names, "Mean |SHAP|": mean_shap})
        .sort_values("Mean |SHAP|", ascending=False)
        .head(20)
    )

    fig = px.bar(
        imp_df, x="Mean |SHAP|", y="Feature",
        orientation="h",
        color="Mean |SHAP|",
        color_continuous_scale="Viridis",
        title="Top 20 Features — Mean |SHAP| Value",
        template=TEMPLATE,
    )
    fig.update_layout(yaxis=dict(autorange="reversed"))
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("**SHAP Beeswarm Plot — Top 15 Features**")
    st.caption("Red = high feature value, Blue = low feature value. Points to the right increase churn probability.")
    top15_idx = np.argsort(mean_shap)[::-1][:15]
    fig_bee, _ = plt.subplots(figsize=(10, 6))
    shap.summary_plot(
        shap_vals[:, top15_idx],
        X_trans[:, top15_idx],
        feature_names=[feat_names[i] for i in top15_idx],
        show=False,
        plot_size=None,
    )
    plt.tight_layout()
    st.pyplot(fig_bee, use_container_width=True)
    plt.close(fig_bee)


# ── Section: Prediction Demo ───────────────────────────────────────────────────
def render_prediction(X, final_models: dict) -> None:
    st.subheader("Churn Probability Predictor")
    st.caption("Input a customer profile to receive a churn probability estimate from any trained model.")

    model_names  = list(final_models.keys())
    default_idx  = model_names.index("XGBoost") if "XGBoost" in model_names else 0
    chosen_model = st.selectbox("Model:", model_names, index=default_idx)
    pipe         = final_models[chosen_model]

    with st.form("predict_form"):
        c1, c2, c3 = st.columns(3)

        with c1:
            tenure      = st.slider("Tenure (months)", 0, 72, 12)
            monthly     = st.number_input("Monthly Charges ($)", 18.0, 120.0, 65.0, step=0.5)
            total       = st.number_input("Total Charges ($)", 0.0, 9000.0, 780.0, step=10.0)
            gender      = st.selectbox("Gender", ["Male", "Female"])
            senior      = st.selectbox("Senior Citizen", [0, 1], format_func=lambda x: "Yes" if x else "No")

        with c2:
            partner     = st.selectbox("Partner",        ["Yes", "No"])
            dependents  = st.selectbox("Dependents",     ["Yes", "No"])
            phone       = st.selectbox("Phone Service",  ["Yes", "No"])
            multi_lines = st.selectbox("Multiple Lines", ["Yes", "No", "No phone service"])
            internet    = st.selectbox("Internet Service", ["Fiber optic", "DSL", "No"])

        with c3:
            online_sec  = st.selectbox("Online Security",   ["Yes", "No", "No internet service"])
            online_bk   = st.selectbox("Online Backup",     ["Yes", "No", "No internet service"])
            device_prot = st.selectbox("Device Protection", ["Yes", "No", "No internet service"])
            tech_supp   = st.selectbox("Tech Support",      ["Yes", "No", "No internet service"])
            streaming_tv= st.selectbox("Streaming TV",      ["Yes", "No", "No internet service"])

        c4, c5 = st.columns(2)
        with c4:
            streaming_mv = st.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])
            contract     = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
        with c5:
            paperless    = st.selectbox("Paperless Billing", ["Yes", "No"])
            payment      = st.selectbox("Payment Method", [
                "Electronic check", "Mailed check",
                "Bank transfer (automatic)", "Credit card (automatic)",
            ])

        submitted = st.form_submit_button("Predict Churn Probability", use_container_width=True)

    if submitted:
        row = pd.DataFrame([{
            "tenure":          tenure,
            "MonthlyCharges":  monthly,
            "TotalCharges":    total,
            "SeniorCitizen":   senior,
            "gender":          gender,
            "Partner":         partner,
            "Dependents":      dependents,
            "PhoneService":    phone,
            "MultipleLines":   multi_lines,
            "InternetService": internet,
            "OnlineSecurity":  online_sec,
            "OnlineBackup":    online_bk,
            "DeviceProtection":device_prot,
            "TechSupport":     tech_supp,
            "StreamingTV":     streaming_tv,
            "StreamingMovies": streaming_mv,
            "Contract":        contract,
            "PaperlessBilling":paperless,
            "PaymentMethod":   payment,
        }])

        prob  = pipe.predict_proba(row)[0, 1]
        pred  = "Likely to Churn" if prob >= 0.5 else "Likely to Stay"
        color = "#F44336" if prob >= 0.5 else "#4CAF50"

        st.markdown(
            f"""<div style="padding:24px; border-radius:12px;
                background:{color}22; border:2px solid {color};
                text-align:center; margin-top:16px;">
  <h2 style="color:{color}; margin:0;">{pred}</h2>
  <h3 style="margin:8px 0 0 0;">Churn Probability: <strong>{prob * 100:.1f}%</strong></h3>
  <p style="opacity:.7; margin:8px 0 0 0;">Model: {chosen_model}</p>
</div>""",
            unsafe_allow_html=True,
        )

        if chosen_model == "XGBoost":
            prep  = pipe.named_steps["prep"]
            clf   = pipe.named_steps["clf"]
            X_row = prep.transform(row)
            names = _clean(list(prep.get_feature_names_out()))
            exp   = shap.TreeExplainer(clf)
            sv    = exp.shap_values(X_row)
            top_n = 10
            idx   = np.argsort(np.abs(sv[0]))[::-1][:top_n]
            df_force = pd.DataFrame({
                "Feature":    [names[i] for i in idx],
                "SHAP Value": sv[0][idx],
            })
            bar_colors = ["#F44336" if v > 0 else "#2196F3" for v in df_force["SHAP Value"]]
            fig = go.Figure(go.Bar(
                x=df_force["SHAP Value"],
                y=df_force["Feature"],
                orientation="h",
                marker_color=bar_colors,
            ))
            fig.update_layout(
                title=f"Top {top_n} Feature Contributions  "
                      "(Red = ↑ churn risk, Blue = ↓ churn risk)",
                template=TEMPLATE,
                xaxis_title="SHAP Value",
                yaxis=dict(autorange="reversed"),
            )
            st.plotly_chart(fig, use_container_width=True)


# ── Main ───────────────────────────────────────────────────────────────────────
def main() -> None:
    st.title("📡 Supervised ML: Customer Churn Prediction Pipeline")
    st.markdown(
        "**Dataset:** IBM Telco Customer Churn — 7,043 telecom subscribers, 20 features. "
        "**Objective:** Identify which customers are at risk of cancelling their subscription. "
        "Full sklearn pipeline: preprocessing → 5-fold CV → SHAP explainability."
    )

    arts = load_artifacts()
    df          = arts["df"]
    X           = arts["X"]
    X_te        = arts["X_te"]
    y_te        = arts["y_te"]
    cv_results  = arts["cv_results"]
    final_models= arts["final_models"]
    eval_models = arts["eval_models"]
    shap_vals   = arts["shap_vals"]
    feat_names  = arts["feat_names"]
    X_trans     = arts["X_trans"]

    tabs = st.tabs([
        "📊 Overview",
        "🔍 Feature Explorer",
        "⚙️ Preprocessing",
        "🏆 Model Comparison",
        "📈 Evaluation",
        "🔮 Feature Importance",
        "🎯 Prediction Demo",
    ])

    with tabs[0]: render_overview(df)
    with tabs[1]: render_feature_explorer(df)
    with tabs[2]: render_preprocessing()
    with tabs[3]: render_model_comparison(cv_results)
    with tabs[4]: render_evaluation(X_te, y_te, eval_models)
    with tabs[5]: render_feature_importance(shap_vals, feat_names, X_trans)
    with tabs[6]: render_prediction(X, final_models)


if __name__ == "__main__":
    main()
