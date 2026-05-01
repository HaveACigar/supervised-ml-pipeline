"""
Supervised ML: Customer Churn Prediction Pipeline
Supports IBM Telco baseline, KDDCup09 (OpenML), and optional KKBox pre-joined features.
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


def _clean(names) -> list[str]:
    return [n.split("__", 1)[1] if "__" in n else n for n in names]


def _extract_shap_array(shap_vals):
    if isinstance(shap_vals, list):
        return shap_vals[1] if len(shap_vals) > 1 else shap_vals[0]
    if getattr(shap_vals, "ndim", 0) == 3:
        return shap_vals[:, :, 1] if shap_vals.shape[2] > 1 else shap_vals[:, :, 0]
    return shap_vals


def _top_shap_features(shap_vals, feat_names: list[str], top_n: int = 5) -> pd.DataFrame:
    shap_array = _extract_shap_array(shap_vals)
    mean_abs = np.abs(shap_array).mean(axis=0)
    top_idx = np.argsort(mean_abs)[::-1][:top_n]
    return pd.DataFrame({
        "Feature": np.array(feat_names)[top_idx],
        "Mean |SHAP|": mean_abs[top_idx],
    })


# ── Load pre-trained artifacts ──────────────────────────────────────────────────
@st.cache_resource
def load_artifacts() -> dict:
    return joblib.load("models/artifacts.pkl")


# ── Section: Overview ──────────────────────────────────────────────────────────
def render_overview(df: pd.DataFrame, numeric_cols: list[str]) -> None:
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

    if numeric_cols:
        stats_cols = numeric_cols[:12]
        st.markdown("**Numeric Feature Statistics**")
        st.caption(f"Showing {len(stats_cols)} of {len(numeric_cols)} numeric features")
        st.dataframe(
            df[stats_cols].describe().round(2), use_container_width=True
        )


def render_model_card(
    df: pd.DataFrame,
    cv_results: dict,
    shap_vals,
    feat_names: list[str],
    dataset_name: str,
    dataset_key: str,
    shap_model_name: str,
) -> None:
    st.subheader("Model Card")

    results_df = (
        pd.DataFrame(cv_results)
        .T.reset_index()
        .rename(columns={"index": "Model"})
        .sort_values("ROC-AUC", ascending=False)
    )
    best_row = results_df.iloc[0]
    top_features = _top_shap_features(shap_vals, feat_names, top_n=5)

    c1, c2, c3 = st.columns(3)
    c1.metric("Dataset", dataset_name)
    c2.metric("Best ROC-AUC", f"{best_row['ROC-AUC']:.3f}")
    c3.metric("Churn Rate", f"{df['Churn'].mean() * 100:.1f}%")

    st.caption(
        f"Best model: {best_row['Model']}. SHAP summary generated with {shap_model_name}."
    )

    summary = [
        f"Rows: {len(df):,} | Predictors: {df.shape[1] - 1}",
        f"Class balance: {100 - df['Churn'].mean() * 100:.1f}% stay vs {df['Churn'].mean() * 100:.1f}% churn",
        (
            "KDDCup09 uses anonymized variables, so feature interpretation is relative importance rather than business-readable labels."
            if dataset_key == "kddcup09"
            else "IBM Telco uses business-readable columns, so feature effects are directly interpretable."
        ),
    ]
    for line in summary:
        st.markdown(f"- {line}")

    st.markdown("**Top SHAP Drivers**")
    st.dataframe(
        top_features.style.format({"Mean |SHAP|": "{:.4f}"}),
        use_container_width=True,
        hide_index=True,
    )


# ── Section: Feature Explorer ──────────────────────────────────────────────────
def render_feature_explorer(df: pd.DataFrame, numeric_cols: list[str], categorical_cols: list[str]) -> None:
    st.subheader("Feature Explorer")
    all_features = numeric_cols + categorical_cols
    feature = st.selectbox("Select a feature to explore:", all_features)

    churn_label = df["Churn"].map({0: "No", 1: "Yes"})
    col1, col2  = st.columns(2)

    if feature in numeric_cols:
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
    if numeric_cols:
        corr = (
            df[numeric_cols + ["Churn"]]
            .corr()["Churn"]
            .drop("Churn")
            .reset_index()
        )
        corr.columns = ["Feature", "Correlation"]
        corr = corr.reindex(corr["Correlation"].abs().sort_values(ascending=False).index).head(25)
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
def render_preprocessing(
    numeric_cols: list[str],
    categorical_cols: list[str],
    dataset_name: str,
    dataset_key: str,
) -> None:
    st.subheader("Preprocessing Pipeline")
    st.caption(f"Configured for: {dataset_name}")
    steps = [
        (
            "1. Encode target",
            "`Churn` is encoded as a binary integer (1 = churn, 0 = stay).",
        ),
        (
            "2. Missing value handling",
            "Numeric features use median imputation; categorical features use mode imputation.",
        ),
        (
            "3. Scale numeric features",
            f"{len(numeric_cols)} numeric columns are standardized with `StandardScaler`.",
        ),
        (
            "4. Encode categorical features",
            (
                f"{len(categorical_cols)} categorical columns use `OrdinalEncoder` with unknown-value handling."
                if dataset_key == "kddcup09"
                else f"{len(categorical_cols)} categorical columns use `OneHotEncoder(drop='if_binary', handle_unknown='ignore')`."
            ),
        ),
        (
            "5. sklearn Pipeline + ColumnTransformer",
            "All steps are wrapped in a `ColumnTransformer` (applied per column group) "
            "and then a `Pipeline([('prep', preprocessor), ('clf', model)])` per model. "
            "This guarantees zero data leakage during cross-validation — "
            "the scaler and encoder are fit only on training folds.",
        ),
    ]

    for title, desc in steps:
        with st.expander(title, expanded=False):
            st.markdown(desc)

    if dataset_key == "kddcup09":
        preprocess_code = """ColumnTransformer([
    ('num', Pipeline([SimpleImputer(median), StandardScaler()]), numeric_cols),
    ('cat', Pipeline([SimpleImputer(mode), OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)]), categorical_cols),
])
# Wrapped per model:
Pipeline([('prep', ColumnTransformer), ('clf', <estimator>)])"""
    else:
        preprocess_code = """ColumnTransformer([
    ('num', Pipeline([SimpleImputer(median), StandardScaler()]), numeric_cols),
    ('cat', Pipeline([SimpleImputer(mode), OneHotEncoder(drop='if_binary', handle_unknown='ignore')]), categorical_cols),
])
# Wrapped per model:
Pipeline([('prep', ColumnTransformer), ('clf', <estimator>)])"""

    st.code(preprocess_code, language="python")


# ── Section: Model Comparison ──────────────────────────────────────────────────
def render_model_comparison(cv_results: dict, n_rows: int, cv_folds: int) -> None:
    st.subheader("Model Comparison")
    st.caption(f"{cv_folds}-fold stratified cross-validation on the full dataset (n = {n_rows:,})")

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
def render_prediction(X, X_te, y_te, final_models: dict, dataset_meta: dict, shap_model_name: str) -> None:
    st.subheader("Churn Probability Predictor")
    st.caption("Estimate churn probability using any trained model.")

    model_names  = list(final_models.keys())
    default_idx  = model_names.index("XGBoost") if "XGBoost" in model_names else 0
    chosen_model = st.selectbox("Model:", model_names, index=default_idx)
    pipe         = final_models[chosen_model]

    if dataset_meta.get("prediction_mode") != "form":
        st.info("This dataset uses a high-dimensional schema, so prediction demo runs on holdout rows.")
        idx = st.slider("Choose a holdout row index", 0, len(X_te) - 1, 0)
        row = X_te.iloc[[idx]].copy()
        prob = pipe.predict_proba(row)[0, 1]
        pred = "Likely to Churn" if prob >= 0.5 else "Likely to Stay"
        actual = "Churn" if int(y_te.iloc[idx]) == 1 else "No Churn"
        color = "#F44336" if prob >= 0.5 else "#4CAF50"
        st.markdown(
            f"""<div style="padding:24px; border-radius:12px;
                background:{color}22; border:2px solid {color};
                text-align:center; margin-top:16px;">
  <h2 style="color:{color}; margin:0;">{pred}</h2>
  <h3 style="margin:8px 0 0 0;">Churn Probability: <strong>{prob * 100:.1f}%</strong></h3>
  <p style="opacity:.8; margin:8px 0 0 0;">Actual label: {actual} | Model: {chosen_model}</p>
</div>""",
            unsafe_allow_html=True,
        )
        st.markdown("**Selected holdout row (first 20 columns)**")
        st.dataframe(row.iloc[:, :20], use_container_width=True)
        return

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

        if chosen_model == shap_model_name:
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
    dataset_meta = arts.get("dataset_meta", {})
    shap_model_name = arts.get("shap_model_name", "XGBoost")
    cv_folds = int(arts.get("cv_folds", 5))
    numeric_cols = dataset_meta.get("numeric_cols", [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])])
    categorical_cols = dataset_meta.get("categorical_cols", [c for c in X.columns if c not in numeric_cols])
    dataset_name = dataset_meta.get("dataset_name", "Customer Churn Dataset")
    dataset_key = dataset_meta.get("dataset_key", "")
    dataset_desc = dataset_meta.get("dataset_description", "")

    st.markdown(
        f"**Dataset:** {dataset_name}. {dataset_desc} "
        "**Objective:** Identify which customers are at risk of cancelling their subscription. "
        "Full sklearn pipeline: preprocessing → 5-fold CV → SHAP explainability."
    )

    tabs = st.tabs([
        "🧾 Model Card",
        "📊 Overview",
        "🔍 Feature Explorer",
        "⚙️ Preprocessing",
        "🏆 Model Comparison",
        "📈 Evaluation",
        "🔮 Feature Importance",
        "🎯 Prediction Demo",
    ])

    with tabs[0]: render_model_card(df, cv_results, shap_vals, feat_names, dataset_name, dataset_key, shap_model_name)
    with tabs[1]: render_overview(df, numeric_cols)
    with tabs[2]: render_feature_explorer(df, numeric_cols, categorical_cols)
    with tabs[3]: render_preprocessing(numeric_cols, categorical_cols, dataset_name, dataset_key)
    with tabs[4]: render_model_comparison(cv_results, len(df), cv_folds)
    with tabs[5]: render_evaluation(X_te, y_te, eval_models)
    with tabs[6]: render_feature_importance(shap_vals, feat_names, X_trans)
    with tabs[7]: render_prediction(X, X_te, y_te, final_models, dataset_meta, shap_model_name)


if __name__ == "__main__":
    main()
