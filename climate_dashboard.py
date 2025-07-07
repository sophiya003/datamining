import streamlit as st
import pandas as pd
from streamlit_lottie import st_lottie
import requests

st.set_page_config(page_title="NEPAL DISTRICT DATA ANALYSIS AND VISUALIZATION", layout="wide")

st.markdown("""
<style>
    .stApp {
         background: linear-gradient(135deg, #4b0082 0%, #000000 100%); /* Purple to Black Gradient */
    font-family: 'Inter', sans-serif;
    min-height: 100vh;
    }
    
    .css-1d391kg {
        background: linear-gradient(180deg, 
            rgba(131, 56, 236, 0.9) 0%,
            rgba(0, 0, 0, 0.9) 100%);
    }
    
    .main .block-container {
        background: transparent;
        border-radius: 15px;
        padding: 2rem;
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    .stMarkdown, .stText, p, span, div {
        color: #ffffff !important;
    }
    
    h1, h2, h3, h4, h5, h6 {
        color: #ffffff !important;
        text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.5);
    }
    
    .metric-container {
        background: transparent;
        border-radius: 10px;
        padding: 1rem;
        border: 1px solid rgba(255, 255, 255, 0.2);
    }
    
    .stButton > button {
        background: linear-gradient(45deg, #8338ec, #240046);
        color: white;
        border: none;
        border-radius: 20px;
        padding: 0.5rem 1rem;
        font-weight: bold;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        background: linear-gradient(45deg, #240046, #8338ec);
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(131, 56, 236, 0.3);
    }
    
    .stFileUploader {
        background: transparent;
        border-radius: 10px;
        padding: 1rem;
        border: 2px dashed rgba(255, 255, 255, 0.3);
    }
    
    .stSuccess {
        background: rgba(0, 255, 0, 0.1);
        border: 1px solid rgba(0, 255, 0, 0.3);
        color: #00ff00 !important;
    }
    
    .stInfo {
        background: rgba(0, 255, 204, 0.1);
        border: 1px solid rgba(0, 255, 204, 0.3);
        color: #00ffcc !important;
    }
    
    .stWarning {
        background: rgba(255, 165, 0, 0.1);
        border: 1px solid rgba(255, 165, 0, 0.3);
        color: #ffa500 !important;
    }
    
    .dataframe {
        background: transparent;
        border-radius: 10px;
        border: 1px solid rgba(255, 255, 255, 0.2);
    }
    
    .css-1d391kg h1 {
        color: #ffffff !important;
        text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.8);
    }
    
    .gradient-banner {
        background: transparent;
        padding: 2rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        border: 2px solid rgba(255, 255, 255, 0.2);
        box-shadow: 0 8px 32px rgba(131, 56, 236, 0.2);
    }
    
    .gradient-banner h1 {
        color: #ffffff !important;
        text-align: center;
        font-size: 2.5rem;
        margin-bottom: 0.5rem;
        text-shadow: 3px 3px 6px rgba(0, 0, 0, 0.7);
    }
    
    .gradient-banner p {
        color: #ffffff !important;
        text-align: center;
        font-size: 1.1rem;
        margin: 0;
        text-shadow: 1px 1px 2px rgba(0, 0, 0, 0.5);
    }
</style>
""", unsafe_allow_html=True)

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
    <div class='gradient-banner'>
        <h1>Nepals district wise data visualization</h1>
        <p>Upload, explore, and visualize district data with clarity and ease.</p>
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
    
    with col1:
        st.markdown("""
        <div class='metric-container'>
            <h3 style='color: #ffffff; margin-bottom: 0.5rem;'>Records</h3>
            <p style='font-size: 2rem; font-weight: bold; color: #ffffff; margin: 0;'>{}</p>
        </div>
        """.format(f"{len(df):,}"), unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class='metric-container'>
            <h3 style='color: #ffffff; margin-bottom: 0.5rem;'>Numeric Columns</h3>
            <p style='font-size: 2rem; font-weight: bold; color: #ffffff; margin: 0;'>{}</p>
        </div>
        """.format(len(numeric_cols.columns)), unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class='metric-container'>
            <h3 style='color: #ffffff; margin-bottom: 0.5rem;'>Missing Values</h3>
            <p style='font-size: 2rem; font-weight: bold; color: #ffffff; margin: 0;'>{}</p>
        </div>
        """.format(f"{df.isnull().sum().sum():,}"), unsafe_allow_html=True)
else:
    st.warning("No numeric columns found in the dataset.")