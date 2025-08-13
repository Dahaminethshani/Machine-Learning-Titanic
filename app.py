
import os
import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px

st.set_page_config(page_title="Titanic: ML Demo", layout="wide")

# ---------- Helper paths ----------
DATA_POSSIBLE_PATHS = ['Data/Titanic-Dataset.csv']

def find_data_path():
    for p in DATA_POSSIBLE_PATHS:
        if os.path.exists(p):
            return p
    return None

# ---------- Load assets ----------
@st.cache_resource(show_spinner=False)
def load_model():
    try:
        model = joblib.load('model.pkl')
        return model
    except Exception as e:
        return None

@st.cache_resource(show_spinner=False)
def load_feature_info():
    try:
        with open('feature_info.json', 'r') as f:
            return json.load(f)
    except Exception:
        return None

@st.cache_data(show_spinner=False)
def load_data():
    data_path = find_data_path()
    if data_path:
        df = pd.read_csv(data_path)
        return df, data_path
    return None, None

@st.cache_data(show_spinner=False)
def load_metrics():
    try:
        with open('metrics.json', 'r') as f:
            return json.load(f)
    except Exception:
        return None

model = load_model()
feature_info = load_feature_info()
df, data_path = load_data()
metrics = load_metrics()

# ---------- Sidebar Navigation ----------
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Overview", "Data Exploration", "Visualizations", "Model Prediction", "Model Performance"])

st.sidebar.markdown("---")
st.sidebar.caption("Dataset path: " + (data_path if data_path else "Not found"))
st.sidebar.caption("Model: " + ("Loaded" if model else "Missing"))

# ---------- Overview ----------
if page == "Overview":
    st.title("Titanic Survival Prediction — ML Demo")
    st.markdown("""
    This Streamlit app demonstrates a complete ML pipeline on the classic **Titanic** dataset:
    - EDA and data exploration
    - Multiple models with cross‑validation
    - Best model selection
    - Interactive predictions with probability
    - Performance metrics and charts
    """)
    st.info("Tip: Ensure `model.pkl`, `metrics.json`, `feature_info.json`, and the dataset CSV exist in your project.")

    if not model:
        st.warning("`model.pkl` not found. Run the training notebook to generate it.")

# ---------- Data Exploration ----------
elif page == "Data Exploration":
    st.title("Data Exploration")
    if df is None:
        st.error("Dataset not found. Place `Titanic-Dataset.csv` in the `Data/` folder.")
    else:
        st.subheader("Dataset Overview")
        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("Rows", len(df))
        with c2:
            st.metric("Columns", len(df.columns))
        with c3:
            st.metric("Missing Cells", int(df.isna().sum().sum()))

        with st.expander("Show sample data", expanded=True):
            st.dataframe(df.head(20), use_container_width=True)

        with st.expander("Column types & missing values"):
            st.write(pd.DataFrame({
                "dtype": df.dtypes.astype(str),
                "missing": df.isna().sum(),
                "unique": df.nunique()
            }))

        st.subheader("Interactive Filter")
        # Simple interactive filtering controls
        sex_col = "Sex" if "Sex" in df.columns else None
        pclass_col = "Pclass" if "Pclass" in df.columns else None

        sex_options = ["All"] + (sorted([x for x in df[sex_col].dropna().unique()]) if sex_col else [])
        sex_sel = st.selectbox("Sex", sex_options) if sex_col else "All"

        pclass_options = ["All"] + (sorted([int(x) for x in df[pclass_col].dropna().unique()]) if pclass_col else [])
        pclass_sel = st.selectbox("Passenger Class", pclass_options) if pclass_col else "All"

        age_min = int(df["Age"].min(skipna=True)) if "Age" in df.columns else 0
        age_max = int(df["Age"].max(skipna=True)) if "Age" in df.columns else 100
        age_range = st.slider("Age range", min_value=age_min, max_value=age_max, value=(age_min, age_max))

        filtered = df.copy()
        if sex_col and sex_sel != "All":
            filtered = filtered[filtered[sex_col] == sex_sel]
        if pclass_col and pclass_sel != "All":
            filtered = filtered[filtered[pclass_col] == pclass_sel]
        if "Age" in filtered.columns:
            filtered = filtered[(filtered["Age"].fillna(age_min) >= age_range[0]) & (filtered["Age"].fillna(age_max) <= age_range[1])]

        st.caption(f"Filtered rows: {len(filtered)}")
        st.dataframe(filtered.head(20), use_container_width=True)

# ---------- Visualizations ----------
elif page == "Visualizations":
    st.title("Visualizations")
    if df is None:
        st.error("Dataset not found.")
    else:
        if "Survived" not in df.columns:
            st.error("Expected target column 'Survived' not in dataset.")
        else:
            st.subheader("Survival by Sex")
            if "Sex" in df.columns:
                fig1 = px.histogram(df, x="Sex", color="Survived", barmode="group")
                st.plotly_chart(fig1, use_container_width=True)
            else:
                st.info("Column 'Sex' not found.")

            st.subheader("Age Distribution by Survival")
            if "Age" in df.columns:
                fig2 = px.histogram(df, x="Age", color="Survived", nbins=40, marginal="box")
                st.plotly_chart(fig2, use_container_width=True)
            else:
                st.info("Column 'Age' not found.")

            st.subheader("Fare vs. Age")
            if "Age" in df.columns and "Fare" in df.columns:
                fig3 = px.scatter(df, x="Age", y="Fare", color=df["Survived"].astype(str), trendline="lowess")
                st.plotly_chart(fig3, use_container_width=True)
            else:
                st.info("Columns 'Age' or 'Fare' not found.")

# ---------- Model Prediction ----------
elif page == "Model Prediction":
    st.title("Make a Prediction")
    if (model is None) or (feature_info is None):
        st.error("Model or feature metadata missing. Run the training notebook first.")
    else:
        st.write("Enter passenger details to predict survival probability.")
        used_features = feature_info["used_features"]
        num_feats = feature_info["numeric_features"]
        cat_feats = feature_info["categorical_features"]

        # Build input widgets based on feature types
        input_data = {}
        cols = st.columns(2)
        for i, feat in enumerate(used_features):
            if feat in num_feats:
                # numeric
                with cols[i % 2]:
                    col_data = df[feat] if (df is not None and feat in df.columns) else None
                    if col_data is not None and pd.api.types.is_numeric_dtype(col_data):
                        minv = float(np.nanmin(col_data.values))
                        maxv = float(np.nanmax(col_data.values))
                        default = float(np.nanmedian(col_data.values))
                    else:
                        minv, maxv, default = 0.0, 100.0, 30.0
                    val = st.number_input(f"{feat}", value=float(default), min_value=float(minv), max_value=float(maxv))
                    input_data[feat] = val
            else:
                # categorical
                with cols[i % 2]:
                    options = [""]  # empty -> will be imputed
                    if df is not None and feat in df.columns:
                        options += sorted([str(x) for x in df[feat].dropna().unique()])
                    sel = st.selectbox(f"{feat}", options, index=0)
                    input_data[feat] = sel if sel != "" else None

        if st.button("Predict"):
            with st.spinner("Computing prediction..."):
                try:
                    X_input = pd.DataFrame([input_data], columns=used_features)
                    proba = model.predict_proba(X_input)[0, 1]
                    pred = int(proba >= 0.5)
                    c1, c2 = st.columns(2)
                    with c1:
                        st.metric("Prediction", "Survived" if pred == 1 else "Did not survive")
                    with c2:
                        st.metric("Probability of Survival", f"{proba:.2%}")
                except Exception as e:
                    st.error(f"Prediction failed: {e}")

# ---------- Model Performance ----------
elif page == "Model Performance":
    st.title("Model Performance")
    if metrics is None:
        st.warning("metrics.json not found. Run the training notebook to generate performance artifacts.")
    else:
        st.subheader("Best Model & Metrics")
        st.write(f"**Best model:** {metrics['best_model']}")

        # Cross‑validation comparison table
        rows = []
        for m, vals in metrics['cv_results'].items():
            rows.append({
                "Model": m,
                "CV Mean Accuracy": vals["cv_mean_accuracy"],
                "Test Accuracy": vals["test_accuracy"],
                "Test Precision": vals["test_precision"],
                "Test Recall": vals["test_recall"],
                "Test F1": vals["test_f1"],
                "Test ROC AUC": vals["test_roc_auc"],
                "Best Params": json.dumps(vals["best_params"])
            })
        st.dataframe(pd.DataFrame(rows).sort_values("CV Mean Accuracy", ascending=False), use_container_width=True)

        st.subheader("Confusion Matrix")
        if os.path.exists("confusion_matrix.png"):
            st.image("confusion_matrix.png", caption="Confusion Matrix (hold‑out test set)")
        else:
            st.info("Confusion matrix image not found.")

        st.subheader("ROC Curve")
        if os.path.exists("roc_curve.png"):
            st.image("roc_curve.png", caption="ROC Curve (hold‑out test set)")
        else:
            st.info("ROC curve image not found.")
