import io
import zipfile
from typing import List, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from mlxtend.frequent_patterns import apriori, association_rules
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

st.set_page_config(page_title="Pineapple Leather Analytics Studio", layout="wide")

TARGET_CLASS = "purchase_intent_binary"
TARGET_CLASS_5 = "purchase_intent_5level"
TARGET_REG = "estimated_spending_score"
ID_COL = "respondent_id"
LEAKY_COLS = ["hidden_persona", TARGET_CLASS_5, TARGET_CLASS, TARGET_REG, ID_COL]
BINARY_ASSOC_PREFIXES = ["interest_", "trust_", "concern_"]


@st.cache_data
def load_default_data() -> pd.DataFrame:
    return pd.read_csv("sample_training_data.csv")


def read_uploaded_file(uploaded_file) -> pd.DataFrame:
    if uploaded_file.name.endswith(".csv"):
        return pd.read_csv(uploaded_file)
    return pd.read_excel(uploaded_file)


def get_feature_columns(df: pd.DataFrame) -> List[str]:
    return [c for c in df.columns if c not in LEAKY_COLS]


def split_columns(df: pd.DataFrame, features: List[str]) -> Tuple[List[str], List[str]]:
    num_cols = [c for c in features if pd.api.types.is_numeric_dtype(df[c])]
    cat_cols = [c for c in features if c not in num_cols]
    return num_cols, cat_cols


def build_preprocessor(df: pd.DataFrame, features: List[str]) -> ColumnTransformer:
    num_cols, cat_cols = split_columns(df, features)
    return ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline([
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                ]),
                num_cols,
            ),
            (
                "cat",
                Pipeline([
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("onehot", OneHotEncoder(handle_unknown="ignore")),
                ]),
                cat_cols,
            ),
        ]
    )


def make_prescription(row: pd.Series) -> str:
    prob = row.get("predicted_interest_probability", 0)
    spend = row.get("predicted_spending_score", 0)
    cluster = row.get("assigned_segment", "Unknown")

    if prob >= 0.75 and spend >= 65:
        return "Target with premium launch messaging, limited discount, and exclusivity-led storytelling."
    if prob >= 0.75 and spend < 65:
        return "Target with starter bundle, first-purchase incentive, and value-plus-sustainability messaging."
    if 0.50 <= prob < 0.75:
        return "Nurture with education on durability, certifications, reviews, and touch-and-feel proof points."
    if "Skeptical" in cluster or prob < 0.50:
        return "Use awareness campaigns first; avoid deep discounting until trust improves."
    return "Run a light-test campaign with curated bundles and track response."


def label_cluster(summary_row: pd.Series) -> str:
    if summary_row["sustainability_importance"] >= 4 and summary_row[TARGET_REG] >= 60:
        return "Eco-conscious premium adopters"
    if summary_row["openness_to_try"] >= 4 and summary_row[TARGET_REG] >= 55:
        return "Design-first urban buyers"
    if summary_row[TARGET_REG] < 50 and summary_row["willingness_proxy"] <= 1.5:
        return "Value-seeking curious buyers"
    if summary_row["concern_index"] >= 2.2 and summary_row["openness_to_try"] <= 3:
        return "Skeptical traditional buyers"
    return "Conscious gifting buyers"


def add_willingness_proxy(df: pd.DataFrame) -> pd.DataFrame:
    mapping = {"0%": 0, "Up to 5%": 1, "6%-10%": 2, "11%-20%": 3, "Above 20%": 4}
    out = df.copy()
    out["willingness_proxy"] = out["willingness_to_pay_extra"].map(mapping).fillna(0)
    concern_cols = [c for c in out.columns if c.startswith("concern_")]
    out["concern_index"] = out[concern_cols].sum(axis=1) if concern_cols else 0
    return out


def get_feature_names(preprocessor: ColumnTransformer) -> List[str]:
    names = []
    for name, transformer, cols in preprocessor.transformers_:
        if name == "remainder":
            continue
        if hasattr(transformer, "named_steps") and "onehot" in transformer.named_steps:
            onehot = transformer.named_steps["onehot"]
            names.extend(onehot.get_feature_names_out(cols).tolist())
        else:
            names.extend(cols)
    return names


def render_download_button(df: pd.DataFrame, label: str, file_name: str):
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button(label, data=csv, file_name=file_name, mime="text/csv")


def main():
    st.title("Pineapple Leather Analytics Studio")
    st.caption("Descriptive, diagnostic, predictive, and prescriptive analytics for a sustainable luxury startup.")

    with st.sidebar:
        st.header("Data")
        uploaded_training = st.file_uploader("Upload training data (CSV or Excel)", type=["csv", "xlsx"])
        use_default = st.checkbox("Use bundled sample dataset", value=uploaded_training is None)
        st.markdown("---")
        st.write("Built-in sample data includes 2,500 synthetic respondents.")

    if uploaded_training is not None:
        df = read_uploaded_file(uploaded_training)
    elif use_default:
        df = load_default_data()
    else:
        st.info("Upload a dataset or use the sample dataset from the sidebar.")
        st.stop()

    df = add_willingness_proxy(df)

    required = {TARGET_CLASS, TARGET_CLASS_5, TARGET_REG}
    missing_required = required - set(df.columns)
    if missing_required:
        st.error(f"Dataset is missing required target columns: {sorted(missing_required)}")
        st.stop()

    features = get_feature_columns(df)
    preprocessor = build_preprocessor(df, features)

    tabs = st.tabs([
        "Overview",
        "Descriptive",
        "Diagnostic",
        "Predictive",
        "Association Rules",
        "Prescriptive",
        "Future Customer Scoring",
    ])

    with tabs[0]:
        st.subheader("Dataset Overview")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Rows", f"{len(df):,}")
        c2.metric("Columns", len(df.columns))
        c3.metric("Interested %", f"{df[TARGET_CLASS].mean()*100:.1f}%")
        c4.metric("Avg Spending Score", f"{df[TARGET_REG].mean():.1f}")
        st.dataframe(df.head(15), use_container_width=True)
        missing = df.isna().sum().reset_index()
        missing.columns = ["column", "missing_values"]
        st.write("Missing values summary")
        st.dataframe(missing[missing["missing_values"] > 0], use_container_width=True)
        render_download_button(df, "Download current dataset", "pineapple_leather_current_dataset.csv")

    with tabs[1]:
        st.subheader("Descriptive Analysis")
        col1, col2 = st.columns(2)
        with col1:
            fig = px.histogram(df, x="age_group", color=TARGET_CLASS_5, barmode="group", title="Purchase intent by age group")
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            fig = px.histogram(df, x="city_tier", color=TARGET_CLASS_5, barmode="group", title="Purchase intent by city tier")
            st.plotly_chart(fig, use_container_width=True)

        col3, col4 = st.columns(2)
        with col3:
            fig = px.box(df, x="purchase_intent_5level", y=TARGET_REG, title="Spending score by purchase intent")
            st.plotly_chart(fig, use_container_width=True)
        with col4:
            prod_cols = [c for c in df.columns if c.startswith("interest_")]
            prod_share = df[prod_cols].mean().sort_values(ascending=False).reset_index()
            prod_share.columns = ["product", "share"]
            fig = px.bar(prod_share, x="product", y="share", title="Most preferred product categories")
            st.plotly_chart(fig, use_container_width=True)

        col5, col6 = st.columns(2)
        with col5:
            concern_cols = [c for c in df.columns if c.startswith("concern_")]
            concern_share = df[concern_cols].mean().sort_values(ascending=False).reset_index()
            concern_share.columns = ["concern", "share"]
            fig = px.bar(concern_share, x="concern", y="share", title="Top customer concerns")
            st.plotly_chart(fig, use_container_width=True)
        with col6:
            fig = px.histogram(df, x="preferred_brand_story", color=TARGET_CLASS_5, barmode="group", title="Story resonance by intent")
            st.plotly_chart(fig, use_container_width=True)

    with tabs[2]:
        st.subheader("Diagnostic Analysis")
        compare = df.groupby(TARGET_CLASS).agg({
            "sustainability_importance": "mean",
            "openness_to_try": "mean",
            "familiarity_with_sustainable_materials": "mean",
            "perceived_durability_vs_leather": "mean",
            TARGET_REG: "mean",
            "concern_index": "mean",
        }).reset_index()
        compare[TARGET_CLASS] = compare[TARGET_CLASS].map({0: "Not interested", 1: "Interested"})
        st.dataframe(compare, use_container_width=True)

        corr_cols = [
            "sustainability_importance", "familiarity_with_sustainable_materials", "openness_to_try",
            "perceived_durability_vs_leather", "likelihood_to_recommend", TARGET_REG, TARGET_CLASS,
            "willingness_proxy", "concern_index"
        ]
        corr = df[corr_cols].corr(numeric_only=True)
        heat = px.imshow(corr, text_auto=".2f", aspect="auto", title="Correlation heatmap")
        st.plotly_chart(heat, use_container_width=True)

        c1, c2 = st.columns(2)
        with c1:
            fig = px.box(df, x=TARGET_CLASS_5, y="sustainability_importance", title="Sustainability importance vs purchase intent")
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            fig = px.box(df, x=TARGET_CLASS_5, y="openness_to_try", title="Openness to try vs purchase intent")
            st.plotly_chart(fig, use_container_width=True)

        st.write("Conversion barriers among low-intent respondents")
        low_intent = df[df[TARGET_CLASS] == 0]
        concern_cols = [c for c in df.columns if c.startswith("concern_")]
        low_concern = low_intent[concern_cols].mean().sort_values(ascending=False).reset_index()
        low_concern.columns = ["barrier", "share"]
        fig = px.bar(low_concern, x="barrier", y="share", title="What holds low-intent customers back")
        st.plotly_chart(fig, use_container_width=True)

    with tabs[3]:
        st.subheader("Predictive Modeling")
        test_size = st.slider("Test set size", 0.15, 0.40, 0.25, 0.05)
        random_state = st.number_input("Random state", min_value=1, max_value=999, value=42)
        n_clusters = st.slider("K-Means clusters", 3, 6, 4)

        X = df[features]
        y = df[TARGET_CLASS]
        y_reg = df[TARGET_REG]

        X_train, X_test, y_train, y_test, y_train_reg, y_test_reg = train_test_split(
            X, y, y_reg, test_size=test_size, random_state=random_state, stratify=y
        )

        clf = Pipeline([
            ("preprocessor", preprocessor),
            ("model", RandomForestClassifier(n_estimators=250, max_depth=10, random_state=random_state, class_weight="balanced")),
        ])
        clf.fit(X_train, y_train)
        pred = clf.predict(X_test)
        prob = clf.predict_proba(X_test)[:, 1]

        acc = accuracy_score(y_test, pred)
        prec = precision_score(y_test, pred, zero_division=0)
        rec = recall_score(y_test, pred, zero_division=0)
        f1 = f1_score(y_test, pred, zero_division=0)
        roc = roc_auc_score(y_test, prob)

        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Accuracy", f"{acc:.3f}")
        m2.metric("Precision", f"{prec:.3f}")
        m3.metric("Recall", f"{rec:.3f}")
        m4.metric("F1-score", f"{f1:.3f}")
        m5.metric("ROC-AUC", f"{roc:.3f}")

        c1, c2 = st.columns(2)
        with c1:
            fpr, tpr, _ = roc_curve(y_test, prob)
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name="ROC curve"))
            fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Baseline", line=dict(dash="dash")))
            fig.update_layout(title="ROC Curve", xaxis_title="False Positive Rate", yaxis_title="True Positive Rate")
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            cm = confusion_matrix(y_test, pred)
            fig = px.imshow(cm, text_auto=True, title="Confusion Matrix", labels=dict(x="Predicted", y="Actual"))
            st.plotly_chart(fig, use_container_width=True)

        transformed = clf.named_steps["preprocessor"].fit_transform(X_train)
        model_for_importance = RandomForestClassifier(n_estimators=250, max_depth=10, random_state=random_state, class_weight="balanced")
        model_for_importance.fit(transformed, y_train)
        importances = pd.DataFrame({
            "feature": get_feature_names(clf.named_steps["preprocessor"]),
            "importance": model_for_importance.feature_importances_,
        }).sort_values("importance", ascending=False).head(20)
        fig = px.bar(importances, x="importance", y="feature", orientation="h", title="Top classification feature importances")
        st.plotly_chart(fig, use_container_width=True)
        st.text(classification_report(y_test, pred, zero_division=0))

        reg = Pipeline([
            ("preprocessor", build_preprocessor(df, features)),
            ("model", RandomForestRegressor(n_estimators=250, max_depth=10, random_state=random_state)),
        ])
        reg.fit(X_train, y_train_reg)
        pred_reg = reg.predict(X_test)
        mae = np.mean(np.abs(y_test_reg.values - pred_reg))
        rmse = float(np.sqrt(np.mean((y_test_reg.values - pred_reg) ** 2)))
        st.markdown("#### Regression summary")
        r1, r2 = st.columns(2)
        r1.metric("MAE", f"{mae:.2f}")
        r2.metric("RMSE", f"{rmse:.2f}")
        reg_plot = pd.DataFrame({"actual": y_test_reg.values, "predicted": pred_reg})
        fig = px.scatter(reg_plot, x="actual", y="predicted", title="Actual vs predicted spending score")
        st.plotly_chart(fig, use_container_width=True)

        cluster_data = df[[
            "sustainability_importance", "familiarity_with_sustainable_materials",
            "openness_to_try", "perceived_durability_vs_leather", TARGET_REG,
            "willingness_proxy", "concern_index"
        ]].copy()
        scaled_cluster = StandardScaler().fit_transform(cluster_data)
        kmeans = KMeans(n_clusters=n_clusters, n_init=20, random_state=random_state)
        df["cluster_id"] = kmeans.fit_predict(scaled_cluster)
        cluster_summary = df.groupby("cluster_id").agg({
            "sustainability_importance": "mean",
            "openness_to_try": "mean",
            TARGET_REG: "mean",
            TARGET_CLASS: "mean",
            "willingness_proxy": "mean",
            "concern_index": "mean",
        }).reset_index()
        cluster_summary["assigned_segment"] = cluster_summary.apply(label_cluster, axis=1)
        df["assigned_segment"] = df["cluster_id"].map(cluster_summary.set_index("cluster_id")["assigned_segment"])
        st.markdown("#### Cluster summary")
        st.dataframe(cluster_summary, use_container_width=True)
        fig = px.scatter(
            df, x="openness_to_try", y=TARGET_REG, color="assigned_segment",
            hover_data=["city_tier", "income_range", TARGET_CLASS_5], title="Customer segments"
        )
        st.plotly_chart(fig, use_container_width=True)

        st.session_state["clf"] = clf
        st.session_state["reg"] = reg
        st.session_state["kmeans"] = kmeans
        st.session_state["cluster_columns"] = cluster_data.columns.tolist()
        st.session_state["cluster_scaler"] = StandardScaler().fit(cluster_data)
        st.session_state["cluster_map"] = cluster_summary.set_index("cluster_id")["assigned_segment"].to_dict()
        st.session_state["features"] = features

    with tabs[4]:
        st.subheader("Association Rule Mining")
        min_support = st.slider("Minimum support", 0.03, 0.30, 0.08, 0.01)
        min_confidence = st.slider("Minimum confidence", 0.10, 0.95, 0.35, 0.05)
        min_lift = st.slider("Minimum lift", 1.00, 5.00, 1.20, 0.05)
        assoc_cols = [c for c in df.columns if any(c.startswith(prefix) for prefix in BINARY_ASSOC_PREFIXES)]
        basket = df[assoc_cols].astype(bool)
        frequent = apriori(basket, min_support=min_support, use_colnames=True)
        if frequent.empty:
            st.warning("No frequent itemsets found at this support threshold. Lower the threshold.")
        else:
            rules = association_rules(frequent, metric="confidence", min_threshold=min_confidence)
            rules = rules[rules["lift"] >= min_lift].copy()
            if rules.empty:
                st.warning("No rules matched the selected confidence and lift thresholds.")
            else:
                rules["antecedents"] = rules["antecedents"].apply(lambda x: ", ".join(sorted(list(x))))
                rules["consequents"] = rules["consequents"].apply(lambda x: ", ".join(sorted(list(x))))
                display_rules = rules[["antecedents", "consequents", "support", "confidence", "lift"]].sort_values(["lift", "confidence"], ascending=False)
                st.dataframe(display_rules.head(25), use_container_width=True)
                fig = px.scatter(display_rules.head(50), x="confidence", y="lift", size="support", hover_data=["antecedents", "consequents"], title="Association rules by confidence and lift")
                st.plotly_chart(fig, use_container_width=True)

    with tabs[5]:
        st.subheader("Prescriptive Analytics")
        if "assigned_segment" not in df.columns:
            st.info("Run the Predictive tab first to generate segments and prescriptions.")
        else:
            summary = df.groupby("assigned_segment").agg({
                TARGET_CLASS: "mean",
                TARGET_REG: "mean",
                "preferred_offer": lambda x: x.mode().iloc[0] if not x.mode().empty else "N/A",
                "preferred_brand_story": lambda x: x.mode().iloc[0] if not x.mode().empty else "N/A",
                "preferred_shopping_channel": lambda x: x.mode().iloc[0] if not x.mode().empty else "N/A",
            }).reset_index()
            summary["interest_rate"] = (summary[TARGET_CLASS] * 100).round(1)
            summary["recommended_action"] = summary.apply(
                lambda r: (
                    "Premium storytelling + limited discounts" if r[TARGET_CLASS] >= 0.7 and r[TARGET_REG] >= 60 else
                    "Bundle offers + first-order incentive" if r[TARGET_CLASS] >= 0.6 else
                    "Education-led trust building + proof of durability"
                ), axis=1
            )
            st.dataframe(summary[["assigned_segment", "interest_rate", TARGET_REG, "preferred_offer", "preferred_brand_story", "preferred_shopping_channel", "recommended_action"]], use_container_width=True)

            fig = px.bar(summary, x="assigned_segment", y=TARGET_REG, color="interest_rate", title="Which segments deserve priority")
            st.plotly_chart(fig, use_container_width=True)

    with tabs[6]:
        st.subheader("Future Customer Scoring")
        st.write("Upload future customer records with the same feature columns as the training data. The app will score inclination, segment, budget, and recommended action.")
        new_file = st.file_uploader("Upload future customer data", type=["csv", "xlsx"], key="new_customer_uploader")

        if new_file is None:
            st.info("Upload a file to score new customers.")
        else:
            if not all(k in st.session_state for k in ["clf", "reg", "kmeans", "cluster_scaler", "cluster_columns", "cluster_map", "features"]):
                st.warning("Run the Predictive tab first so the models can be trained.")
            else:
                new_df = read_uploaded_file(new_file)
                required_features = st.session_state["features"]
                new_df = add_willingness_proxy(new_df)
                missing_cols = [c for c in required_features if c not in new_df.columns]
                if missing_cols:
                    st.error(f"Uploaded file is missing required feature columns: {missing_cols[:10]}{'...' if len(missing_cols) > 10 else ''}")
                else:
                    X_new = new_df[required_features]
                    clf = st.session_state["clf"]
                    reg = st.session_state["reg"]
                    scaler = st.session_state["cluster_scaler"]
                    kmeans = st.session_state["kmeans"]
                    cluster_cols = st.session_state["cluster_columns"]
                    cluster_map = st.session_state["cluster_map"]

                    new_df["predicted_interest_probability"] = clf.predict_proba(X_new)[:, 1]
                    new_df["predicted_interest_class"] = (new_df["predicted_interest_probability"] >= 0.5).astype(int)
                    new_df["predicted_spending_score"] = reg.predict(X_new).round(1)
                    new_cluster_matrix = scaler.transform(new_df[cluster_cols])
                    new_df["cluster_id"] = kmeans.predict(new_cluster_matrix)
                    new_df["assigned_segment"] = new_df["cluster_id"].map(cluster_map)
                    new_df["recommended_marketing_action"] = new_df.apply(make_prescription, axis=1)

                    preview_cols = [col for col in [ID_COL, "predicted_interest_probability", "predicted_interest_class", "predicted_spending_score", "assigned_segment", "recommended_marketing_action"] if col in new_df.columns]
                    st.dataframe(new_df[preview_cols], use_container_width=True)
                    render_download_button(new_df, "Download scored customers", "scored_future_customers.csv")


if __name__ == "__main__":
    main()
