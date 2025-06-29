import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler


st.set_page_config(page_title="Climate Data Cleaner", layout="wide")
st.title("Climate Data Cleaning & Preprocessing")



if "data" in st.session_state:
    df = st.session_state["data"]
    st.success("Using uploaded or session data.")
else:
    try:
        df = pd.read_csv("data/nepal_gis_dailydata.csv")
        st.info("No uploaded data found. Using default dataset.")
    except FileNotFoundError:
        st.error("Default dataset not found.")
        st.stop()


def convert_datetime(df):
    for col in df.columns:
        
        if df[col].dtype == object:
            try:
                df[col] = pd.to_datetime(df[col], errors='raise')
                st.info(f"Converted column '{col}' to datetime.")
            except Exception:
                pass
    return df

df = convert_datetime(df)


with st.expander(" Select Columns"):
    all_columns = df.columns.tolist()
    selected_columns = st.multiselect("Choose columns to include", all_columns, default=all_columns)
    if selected_columns:
        df = df[selected_columns]
    else:
        st.warning(" Please select at least one column.")


def remove_duplicate_columns(df):
    duplicated_cols = []
    cols = df.columns
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            if df[cols[i]].equals(df[cols[j]]):
                duplicated_cols.append(cols[j])
    if duplicated_cols:
        df = df.drop(columns=duplicated_cols)
        st.warning(f"Removed duplicate columns: {duplicated_cols}")
    else:
        st.info("No duplicate columns found.")
    return df

df = remove_duplicate_columns(df)


with st.expander("🔍 Filter Rows"):
    apply_filter = st.checkbox("Enable Filtering", value=False)
    if apply_filter:
        filter_col = st.selectbox("Select column to filter", df.columns)
        if pd.api.types.is_numeric_dtype(df[filter_col]):
            min_val, max_val = df[filter_col].min(), df[filter_col].max()
            filter_range = st.slider(f"Range for {filter_col}", float(min_val), float(max_val), (float(min_val), float(max_val)))
            df = df[df[filter_col].between(filter_range[0], filter_range[1])]
        elif pd.api.types.is_datetime64_any_dtype(df[filter_col]):
            min_date, max_date = df[filter_col].min(), df[filter_col].max()
            date_range = st.date_input(f"Select date range for {filter_col}", [min_date, max_date])
            if len(date_range) == 2:
                df = df[(df[filter_col] >= pd.to_datetime(date_range[0])) & (df[filter_col] <= pd.to_datetime(date_range[1]))]
        else:
            unique_vals = df[filter_col].dropna().unique().tolist()
            selected_vals = st.multiselect(f"Values for {filter_col}", unique_vals, default=unique_vals)
            df = df[df[filter_col].isin(selected_vals)]


missing_strategy = st.radio("🧹 Handle Missing Values", ["Drop rows", "Fill median", "Fill most frequent"], horizontal=True)

if missing_strategy == "Drop rows":
    df = df.dropna()
elif missing_strategy == "Fill median":
    df = df.fillna(df.median(numeric_only=True))
elif missing_strategy == "Fill most frequent":
    df = df.fillna(df.mode().iloc[0])


with st.expander(" Outlier Detection & Removal"):
    enable_outlier = st.checkbox("Enable Outlier Removal?", value=False)
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    
    if enable_outlier and numeric_cols:
        outlier_cols = st.multiselect("Select numeric columns for outlier removal", numeric_cols, default=numeric_cols)
        k = st.number_input("IQR multiplier (k)", min_value=0.5, max_value=5.0, value=1.5, step=0.1)

        if st.button("Remove Outliers"):
            before_rows = df.shape[0]
            for col in outlier_cols:
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - k * IQR
                upper_bound = Q3 + k * IQR
                df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
            after_rows = df.shape[0]
            st.success(f"Removed {before_rows - after_rows} total outlier rows.")


normalize = st.checkbox(" Normalize/Standardize numeric columns?")
if normalize:
    scale_method = st.selectbox("Choose Scaling Method", ["MinMax", "Standard"])
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    if numeric_cols:
        scaler = MinMaxScaler() if scale_method == "MinMax" else StandardScaler()
        df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
    else:
        st.warning(" No numeric columns to scale.")


if st.button(" Process Data"):
    if df.empty:
        st.error(" No data available after preprocessing.")
    else:
        st.success(" Data ready for visualization!")
        st.dataframe(df)  
        st.info(f"Data shape: {df.shape}")

        
        st.session_state["preprocessed_data"] = df.copy()

        st.download_button(
            label=" Download Cleaned CSV",
            data=df.to_csv(index=False).encode("utf-8"),
            file_name="cleaned_climate_data.csv",
            mime="text/csv"
        )