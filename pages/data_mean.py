import streamlit as st
import pandas as pd
import numpy as np

# ----------------- Page Config -----------------
st.set_page_config(page_title="Climate Data Explorer", layout="wide")

# ----------------- Header -----------------
st.markdown("""
    <div style='background-color:#001F3F;padding:1.5rem;border-radius:12px;margin-bottom:1rem'>
        <h2 style='color:#00ffcc;text-align:center;'>Climate Data Explorer</h2>
        <p style='color:#cceeff;text-align:center;font-size:16px;'>Explore statistical summaries and insights from your climate dataset</p>
    </div>
""", unsafe_allow_html=True)

# ----------------- Load Data -----------------
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

# Drop unwanted index column
if "Unnamed: 0" in df.columns:
    df.drop(columns=["Unnamed: 0"], inplace=True)

# ----------------- Tabs -----------------
tab1, tab2, tab3, tab4 = st.tabs([
    "Data Table", 
    "Data Summary", 
    "Feature Statistics", 
    "Rank Features"
])

# ----------------- Tab 1: Raw Data Table -----------------
with tab1:
    st.subheader("Full Dataset")
    st.dataframe(df, use_container_width=True)

# ----------------- Tab 2: Data Summary -----------------
with tab2:
    st.subheader("Basic Information")
    st.markdown(f"- **Rows:** {df.shape[0]}")
    st.markdown(f"- **Columns:** {df.shape[1]}")
    st.markdown(f"- **Missing Values:** {'Yes' if df.isnull().any().any() else 'No'}")

    dtypes = pd.DataFrame({
        "Column": df.columns,
        "Data Type": df.dtypes.values,
        "Missing Values": df.isnull().sum().values
    })
    st.markdown("### Column Details")
    st.dataframe(dtypes, use_container_width=True)


binary_cols = [col for col in df.select_dtypes(include=['int64', 'float64']).columns 
               if df[col].dropna().nunique() == 2]

# Treat these as categorical
cat_cols = df.select_dtypes(include=['object']).columns.tolist() + binary_cols
num_cols = [col for col in df.select_dtypes(include=['int64', 'float64']).columns 
            if col not in binary_cols]
   

with tab3:
    st.subheader("Numerical Feature Statistics")
    num_stats = df[num_cols].describe().T
    st.dataframe(num_stats, use_container_width=True)

    st.subheader("Categorical Feature Distributions")
    if len(cat_cols):
        for col in cat_cols:
            st.markdown(f"**{col}**")
            st.write(df[col].value_counts())
    else:
        st.info("No categorical columns found.")


with tab4:
    st.subheader("Feature Correlation to Target")

    # Choose a target
    numeric_cols = num_cols
    st.session_state["numercal_columns"] = numeric_cols
    target = st.selectbox("Select a numerical target column", numeric_cols)

    if target:
        st.session_state["target_variable"] = target
        correlation = df.corr(numeric_only=True)[target].drop(target).sort_values(ascending=False)

        rank_df = pd.DataFrame({
            "Feature": correlation.index,
            "Correlation": correlation.values
        })

        st.markdown("### Correlation Table")
        st.dataframe(rank_df, use_container_width=True)

        st.markdown("### Correlation Bar Chart")
        st.bar_chart(rank_df.set_index("Feature"))

