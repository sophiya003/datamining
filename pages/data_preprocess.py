import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import train_test_split

# ---------------------- Streamlit Setup ----------------------
st.set_page_config(page_title="Climate Data Explorer", layout="wide")
st.title("🌤️ Climate Data Preprocessing")

# ---------------------- Cached Data Load ----------------------
@st.cache_data
def load_data():
    if "data" in st.session_state:
        return st.session_state["data"]
    try:
        st.warning("⚠️ No uploaded data found. Using default dataset.")
        return pd.read_csv("data/nepal_gis_dailydata.csv")
    except FileNotFoundError:
        st.error("❌ Default dataset not found.")
        st.stop()

df = load_data()

# ---------------------- Column Selection ----------------------
with st.expander("🧩 Select Columns"):
    all_columns = df.columns.tolist()
    selected_columns = st.multiselect("Choose columns to include", all_columns, default=all_columns)
    if selected_columns:
        df = df[selected_columns]
    else:
        st.warning("⚠️ Please select at least one column to proceed.")

# ---------------------- Save Numeric Columns to Session ----------------------
if "numerical_columns" not in st.session_state:
    st.session_state["numerical_columns"] = df.select_dtypes(include=np.number).columns.tolist()

# ---------------------- User Configuration ----------------------
with st.expander("🧮 Select Numeric Columns for Processing"):
    numeric_cols = st.multiselect("Select numeric features", df.select_dtypes(include=np.number).columns.tolist(), default=st.session_state["numerical_columns"])
    st.session_state["numerical_columns"] = numeric_cols

# ---------------------- Target Variable ----------------------
possible_targets = [col for col in df.columns if col in numeric_cols]
target = st.selectbox("🎯 Select Target Variable", possible_targets, index=0 if possible_targets else None)
st.session_state["target_variable"] = target

# ---------------------- Filtering UI (Deferred Execution) ----------------------
with st.expander("🔍 Filter Rows"):
    apply_filter = st.checkbox("Enable Filtering", value=False)
    filter_col = st.selectbox("Select column to filter", df.columns)
    filter_params = {}

    if apply_filter:
        if pd.api.types.is_numeric_dtype(df[filter_col]):
            min_val, max_val = df[filter_col].min(), df[filter_col].max()
            filter_range = st.slider(f"Select range for {filter_col}", float(min_val), float(max_val), (float(min_val), float(max_val)))
            filter_params = {"type": "range", "col": filter_col, "range": filter_range}
        else:
            unique_vals = df[filter_col].dropna().unique().tolist()
            selected_vals = st.multiselect(f"Select values for {filter_col}", unique_vals, default=unique_vals)
            filter_params = {"type": "category", "col": filter_col, "values": selected_vals}

# ---------------------- Preprocessing Options ----------------------
missing_strategy = st.radio("🧹 Handle Missing Values", ["Drop rows", "Fill median", "Fill most frequent"], horizontal=True)

normalize = st.checkbox("🔧 Normalize/Standardize Features?")
scale_method = st.selectbox("Choose Scaling Method", ["MinMax", "Standard"]) if normalize else None
one_hot_encode = st.checkbox("🧬 One-Hot Encode Categorical Variables?")
apply_pca = st.checkbox("📉 Apply PCA?")
n_components = st.slider("Number of PCA Components", 1, min(len(numeric_cols) - 1, 20), 5) if apply_pca else None
apply_variance_filter = st.checkbox("📊 Remove Low Variance Features?")
threshold = st.slider("Variance Threshold", 0.0, 1.0, 0.01) if apply_variance_filter else None
randomize = st.checkbox("🔀 Shuffle Rows?")
sample_frac = st.slider("📦 Sample Fraction", 0.01, 1.0, 0.1)
stratified_sample = st.checkbox("📊 Use Stratified Sampling?")

# ---------------------- Processing Functions ----------------------
def apply_filtering(df, params):
    if not params:
        return df
    col = params["col"]
    if params["type"] == "range":
        return df[df[col].between(params["range"][0], params["range"][1])]
    elif params["type"] == "category":
        return df[df[col].isin(params["values"])]
    return df

@st.cache_data
def handle_correlation(df):
    corr = df.corr(numeric_only=True).fillna(0)
    return corr

def handle_duplicate(df):
    return df[df.duplicated()]

def preprocess(df, target):
    df = df.copy()

    # Filter
    if apply_filter:
        df = apply_filtering(df, filter_params)

    # Drop high null columns
    df.drop(columns=[col for col in df.columns if df[col].isnull().mean() > 0.45], inplace=True)
    df.dropna(axis=1, how='all', inplace=True)

    # One-hot encode
    if one_hot_encode:
        df = pd.get_dummies(df)

    numeric = df.select_dtypes(include=np.number).columns.tolist()
    features = [col for col in numeric if col != target]

    # Handle missing ONLY if needed
    if df[features + [target]].isnull().values.any():
        st.info("ℹ️ Missing values detected. Applying selected strategy.")
        if missing_strategy == "Drop rows":
            df.dropna(subset=features + [target], inplace=True)
        elif missing_strategy == "Fill median":
            df[features + [target]] = df[features + [target]].fillna(df[features + [target]].median())
        elif missing_strategy == "Fill most frequent":
            df[features + [target]] = df[features + [target]].fillna(df[features + [target]].mode().iloc[0])
    else:
        st.success("✅ No missing values found. Skipping missing value handling.")

    if df.empty:
        st.error("❌ All rows dropped after missing value handling.")
        return pd.DataFrame()

    # Normalize
    if normalize:
        scaler = MinMaxScaler() if scale_method == "MinMax" else StandardScaler()
        df[features] = scaler.fit_transform(df[features])

    # Variance filter
    if apply_variance_filter and threshold is not None:
        selector = VarianceThreshold(threshold)
        selected_data = selector.fit_transform(df[features])
        selected_features = [features[i] for i in selector.get_support(indices=True)]
        df = pd.concat([pd.DataFrame(selected_data, columns=selected_features, index=df.index), df[[target]]], axis=1)
        features = selected_features

    # PCA
    if apply_pca and n_components:
        pca = PCA(n_components=n_components)
        pca_data = pca.fit_transform(df[features])
        df = pd.concat([pd.DataFrame(pca_data, columns=[f"PC{i+1}" for i in range(n_components)], index=df.index), df[[target]]], axis=1)

    # Sampling
    if stratified_sample:
        try:
            df, _ = train_test_split(df, test_size=(1 - sample_frac), stratify=df[target], random_state=42)
        except ValueError:
            st.warning("⚠️ Stratified sampling failed. Using random sampling.")
            df = df.sample(frac=sample_frac, random_state=42)
    else:
        if randomize:
            df = df.sample(frac=1.0, random_state=42)
        df = df.sample(frac=sample_frac, random_state=42)

    return df

# ---------------------- Execution ----------------------
if st.button("🚀 Preprocess Data"):
    if df.empty:
        st.error("❌ No data to preprocess.")
    elif target not in df.columns:
        st.error("❌ Target column not found.")
    else:
        preprocessed_df = preprocess(df, target)
        if not preprocessed_df.empty:
            st.success("✅ Preprocessing Complete!")
            st.dataframe(preprocessed_df.head())
            st.info(f"🧾 Data shape: {preprocessed_df.shape}")
            st.session_state["preprocessed_data"] = preprocessed_df

            st.download_button(
                label="⬇️ Download Preprocessed CSV",
                data=preprocessed_df.to_csv(index=False).encode("utf-8"),
                file_name="preprocessed_climate_data.csv",
                mime="text/csv"
            )

# ---------------------- Post-Processing Expander ----------------------
with st.expander("🔁 View Duplicate Rows (After Preprocessing)"):
    if "preprocessed_data" in st.session_state:
        duplicates = handle_duplicate(st.session_state["preprocessed_data"])
        if not duplicates.empty:
            st.warning(f"{len(duplicates)} duplicate rows found.")
            st.dataframe(duplicates)
        else:
            st.success("✅ No duplicate rows found.")
    else:
        st.info("ℹ️ Preprocess data first.")

with st.expander("📊 Correlation Matrix (After Preprocessing)"):
    if "preprocessed_data" in st.session_state:
        corr = handle_correlation(st.session_state["preprocessed_data"])
        st.dataframe(corr.style.background_gradient(cmap="coolwarm").format(precision=2))
    else:
        st.info("ℹ️ Preprocess data first.")
