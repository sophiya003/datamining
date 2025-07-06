import streamlit as st
import pandas as pd
import numpy as np


st.set_page_config(page_title="DATA MEANING", layout="wide")

st.markdown("""
<style>
    /* Main background gradient */
    .stApp {
         background: linear-gradient(135deg, #4b0082 0%, #000000 100%); /* Purple to Black Gradient */
    font-family: 'Inter', sans-serif;
    min-height: 100vh;
    }
    
    /* Main container styling */
    .main .block-container {
        background: transparent;
        padding-top: 2rem;
    }
    
    .main-header {
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.2);
        padding: 2rem 1rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        color: white;
        text-align: center;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
    }
    
    /* Transparent card styling */
    .metric-card {
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.2);
        padding: 1.5rem;
        border-radius: 12px;
        color: white;
        transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275);
        cursor: pointer;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.2);
    }

    .metric-card:hover {
        transform: translateY(-8px) scale(1.02);
        background: rgba(255, 255, 255, 0.2);
        box-shadow: 0 15px 40px rgba(0, 0, 0, 0.4);
        border: 1px solid rgba(255, 255, 255, 0.3);
    }

    .success-card {
        background: rgba(76, 175, 80, 0.2);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(76, 175, 80, 0.3);
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
        color: white;
        transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275);
        cursor: pointer;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.2);
    }

    .success-card:hover {
        transform: translateY(-8px) scale(1.02);
        background: rgba(76, 175, 80, 0.3);
        box-shadow: 0 15px 40px rgba(0, 0, 0, 0.4);
        border: 1px solid rgba(76, 175, 80, 0.5);
    }

    .warning-card {
        background: rgba(255, 193, 7, 0.2);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 193, 7, 0.3);
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
        color: white;
        transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275);
        cursor: pointer;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.2);
    }

    .warning-card:hover {
        transform: translateY(-8px) scale(1.02);
        background: rgba(255, 193, 7, 0.3);
        box-shadow: 0 15px 40px rgba(0, 0, 0, 0.4);
        border: 1px solid rgba(255, 193, 7, 0.5);
    }

    .info-card {
        background: rgba(23, 162, 184, 0.2);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(23, 162, 184, 0.3);
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
        color: white;
        transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275);
        cursor: pointer;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.2);
    }

    .info-card:hover {
        transform: translateY(-8px) scale(1.02);
        background: rgba(23, 162, 184, 0.3);
        box-shadow: 0 15px 40px rgba(0, 0, 0, 0.4);
        border: 1px solid rgba(23, 162, 184, 0.5);
    }

    /* Button styling */
    .stButton > button {
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.2);
        color: white;
        border-radius: 10px;
        padding: 0.75rem 1.5rem;
        font-weight: 500;
        transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275);
    }

    .stButton > button:hover {
        transform: translateY(-4px);
        background: rgba(255, 255, 255, 0.2);
        box-shadow: 0 8px 25px rgba(0, 0, 0, 0.3);
        border: 1px solid rgba(255, 255, 255, 0.3);
    }

    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
        border-radius: 10px;
        padding: 0.5rem;
        border: 1px solid rgba(255, 255, 255, 0.2);
    }

    .stTabs [data-baseweb="tab"] {
        background: transparent;
        color: white;
        border-radius: 8px;
        transition: all 0.3s ease;
    }

    .stTabs [data-baseweb="tab"]:hover {
        background: rgba(255, 255, 255, 0.1);
        transform: translateY(-2px);
    }

    .stTabs [aria-selected="true"] {
        background: rgba(255, 255, 255, 0.2) !important;
        color: white !important;
    }

    /* DataFrame styling */
    .stDataFrame {
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
        border-radius: 10px;
        border: 1px solid rgba(255, 255, 255, 0.2);
        overflow: hidden;
    }

    /* Text styling */
    .stMarkdown, .stText, h1, h2, h3, h4, h5, h6, p, li {
        color: white !important;
    }

    /* Selectbox styling */
    .stSelectbox > div > div {
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.2);
        color: white;
        border-radius: 8px;
    }

    /* Section divider */
    .section-divider {
        border-top: 2px solid rgba(255, 255, 255, 0.3);
        margin: 2rem 0;
        opacity: 0.6;
    }

    /* Success, warning, info message styling */
    .stSuccess, .stWarning, .stInfo, .stError {
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.2);
        border-radius: 10px;
        color: white;
    }
</style>
""", unsafe_allow_html=True)


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

if "Unnamed: 0" in df.columns:
    df.drop(columns=["Unnamed: 0"], inplace=True)


tab1, tab2, tab3, tab4 = st.tabs([
    "Data Table", 
    "Data Summary", 
    "Feature Statistics", 
    "Rank Features"
])

with tab1:
    st.subheader("Full Dataset")
    st.dataframe(df, use_container_width=True)

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

cat_cols = df.select_dtypes(include=['object']).columns.tolist() + binary_cols
num_cols = [col for col in df.select_dtypes(include=['int64', 'float64']).columns 
            if col not in binary_cols]

numeric_cols = num_cols
categorical_cols = cat_cols
st.session_state["categorical_columns"]= categorical_cols
st.session_state["numerical_columns"] = numeric_cols
   

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