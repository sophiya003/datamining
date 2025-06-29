import streamlit as st
import pandas as pd
from streamlit_lottie import st_lottie
import requests


st.set_page_config(page_title="Climate Dashboard", layout="wide")

def load_lottie_url(url):
    try:
        r = requests.get(url)
        if r.status_code != 200:
            return None
        return r.json()
    except:
        return None

def show_banner():
    st.markdown("""
    <div style='background-color:#001F3F;padding:1.5rem;border-radius:12px;margin-bottom:1.5rem'>
        <h1 style='color:#00ffcc;text-align:center;'>Climate Data Dashboard</h1>
        <p style='color:#cceeff;text-align:center;font-size:16px;'>Upload, explore, and visualize climate data with clarity and ease.</p>
    </div>
    """, unsafe_allow_html=True)

show_banner()


st.sidebar.title("Navigation")
st.sidebar.markdown("Use the pages in the sidebar to explore temperature, rainfall, CO₂ levels, and more.")


st.subheader("Upload Dataset (CSV format)")
uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])


if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.success("Custom data loaded successfully.")
else:
    df = pd.read_csv("data/nepal_gis_dailydata.csv")
    st.info("Using default climate dataset.")


st.session_state['data'] = df


st.subheader("Data Preview")
st.dataframe(df.head(10), use_container_width=True)


st.subheader("Quick Statistics")
numeric_cols = df.select_dtypes(include=['float', 'int'])

if not numeric_cols.empty:
    col1, col2, col3 = st.columns(3)
    col1.metric("Number of Records", f"{len(df)}")
    col2.metric("Numeric Columns", f"{len(numeric_cols.columns)}")
    col3.metric("Missing Values", f"{df.isnull().sum().sum()}")
else:
    st.warning("No numeric columns found in the dataset.")
