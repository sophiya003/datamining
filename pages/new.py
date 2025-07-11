import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import requests
import geopandas as gpd
import json
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from scipy.ndimage import gaussian_filter


st.set_page_config(
    page_title="Nepal Data Vizualization", 
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://www.extremelycoolapp.com/help',
        'Report a bug': "https://www.extremelycoolapp.com/bug",
        'About': "# District based data vizualization\nBuilt with Streamlit and Plotly"
    }
)


st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
    
    /* Global Styles with Enhanced Gradients */
    .stApp {
        background: linear-gradient(135deg, #4b0082 0%, #000000 100%); /* Purple to Black Gradient */
    font-family: 'Inter', sans-serif;
    min-height: 100vh;
    }
    
    /* Animated Background */
    .stApp::before {
        content: '';
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background: 
            radial-gradient(circle at 20% 50%, rgba(96, 165, 250, 0.1) 0%, transparent 50%),
            radial-gradient(circle at 80% 20%, rgba(52, 211, 153, 0.1) 0%, transparent 50%),
            radial-gradient(circle at 40% 80%, rgba(168, 85, 247, 0.1) 0%, transparent 50%);
        pointer-events: none;
        z-index: -1;
        animation: float 20s ease-in-out infinite;
    }
    
    @keyframes float {
        0%, 100% { transform: translateY(0px) rotate(0deg); }
        50% { transform: translateY(-20px) rotate(1deg); }
    }
    
    /* Enhanced Header with Glassmorphism */
    .main-header {
        background: rgba(30, 41, 59, 0.3);
        backdrop-filter: blur(20px);
        -webkit-backdrop-filter: blur(20px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        padding: 3rem 2rem;
        border-radius: 24px;
        margin-bottom: 2rem;
        box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.25);
        position: relative;
        overflow: hidden;
    }
    
    .main-header::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 1px;
        background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.4), transparent);
    }
    
    .main-title {
        font-size: 3.5rem;
        font-weight: 800;
        background: linear-gradient(135deg, #60a5fa 0%, #34d399 50%, #a78bfa 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        text-align: center;
        margin-bottom: 1rem;
        text-shadow: 0 0 40px rgba(96, 165, 250, 0.3);
        animation: glow 3s ease-in-out infinite alternate;
    }
    
    @keyframes glow {
        from { filter: drop-shadow(0 0 20px rgba(96, 165, 250, 0.3)); }
        to { filter: drop_shadow(0 0 30px rgba(52, 211, 153, 0.4)); }
    }
    
    .subtitle {
        text-align: center;
        color: #cbd5e1;
        font-size: 1.3rem;
        font-weight: 400;
        opacity: 0.9;
    }
    
    /* Enhanced Metric Cards */
    .metric-card {
        background: rgba(30, 41, 59, 0.4);
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
        padding: 2rem 1.5rem;
        border-radius: 20px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        text-align: center;
        height: 140px;
        display: flex;
        flex-direction: column;
        justify-content: center;
        transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
        position: relative;
        overflow: hidden;
    }
    
    .metric-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.1), transparent);
        transition: left 0.5s;
    }
    
    .metric-card:hover::before {
        left: 100%;
    }
    
    .metric-card:hover {
        transform: translateY(-8px) scale(1.02);
        border-color: rgba(96, 165, 250, 0.6);
        box_shadow: 0 25px 50px -12px rgba(96, 165, 250, 0.2);
        background: rgba(30, 41, 59, 0.6);
    }
    
    .metric-value {
        font-size: 2.8rem;
        font-weight: 800;
        background: linear-gradient(135deg, #ffffff 0%, #60a5fa 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    
    .metric-label {
        font-size: 0.95rem;
        color: #cbd5e1;
        font-weight: 500;
        opacity: 0.8;
    }
    
    /* Enhanced Chart Containers */
    .chart-container {
        background: rgba(30, 41, 59, 0.3);
        backdrop-filter: blur(20px);
        -webkit-backdrop-filter: blur(20px);
        border-radius: 24px;
        padding: 2.5rem;
        margin-bottom: 2rem;
        border: 1px solid rgba(255, 255, 255, 0.1);
        box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.25);
        position: relative;
        overflow: hidden;
    }
    
    .chart-container::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 1px;
        background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.3), transparent);
    }
    
    .chart-title {
        font-size: 1.6rem;
        font-weight: 700;
        background: linear-gradient(135deg, #ffffff 0%, #60a5fa 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 1.5rem;
        text-align: center;
    }
    
    /* Enhanced Sidebar */
    .css-1d391kg {
        background: rgba(15, 23, 42, 0.8);
        backdrop-filter: blur(20px);
        -webkit-backdrop-filter: blur(20px);
    }
    
    .filter-section {
        background: rgba(30, 41, 59, 0.4);
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
        padding: 2rem 1.5rem;
        border-radius: 20px;
        margin-bottom: 1.5rem;
        border: 1px solid rgba(255, 255, 255, 0.1);
        box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.1);
    }
    
    .filter-title {
        color: #60a5fa;
        font-size: 1.3rem;
        font-weight: 700;
        margin-bottom: 1.5rem;
        text-align: center;
        text_shadow: 0 0 10px rgba(96, 165, 250, 0.3);
    }
    
    /* Enhanced Controls */
    .stSelectbox > div > div {
        background: rgba(30, 41, 59, 0.6);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.2);
        border-radius: 12px;
        color: white;
    }
    
    .stMultiSelect > div > div {
        background: rgba(30, 41, 59, 0.6);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.2);
        border-radius: 12px;
    }
    
    /* Enhanced Animations */
    .fade-in {
        animation: fadeInUp 0.8s cubic-bezier(0.4, 0, 0.2, 1);
    }
    
    @keyframes fadeInUp {
        from { 
            opacity: 0; 
            transform: translateY(40px);
        }
        to { 
            opacity: 1; 
            transform: translateY(0);
        }
    }
    
    .slide-in {
        animation: slideInLeft 0.8s cubic-bezier(0.4, 0, 0.2, 1);
    }
    
    @keyframes slideInLeft {
        from { 
            opacity: 0; 
            transform: translateX(-60px);
        }
        to { 
            opacity: 1; 
            transform: translateX(0);
        }
    }
    
    /* Enhanced Tab Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: rgba(30, 41, 59, 0.3);
        padding: 8px;
        border-radius: 16px;
        backdrop-filter: blur(10px);
    }
    
    .stTabs [data-baseweb="tab"] {
        background: rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        padding: 12px 24px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        transition: all 0.3s ease;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #60a5fa 0%, #34d399 100%);
        color: white;
        box_shadow: 0 4px 15px rgba(96, 165, 250, 0.3);
    }
    
    /* Custom Scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: rgba(30, 41, 59, 0.3);
        border-radius: 4px;
    }
    
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(135deg, #60a5fa, #34d399);
        border-radius: 4px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: linear-gradient(135deg, #3b82f6, #10b981);
    }
    
    /* Hide Streamlit Elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .stDeployButton {visibility: hidden;}
    
    /* New Visualization Containers */
    .viz-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
        gap: 2rem;
        margin: 2rem 0;
    }
    
    .viz-card {
        background: rgba(30, 41, 59, 0.4);
        backdrop-filter: blur(16px);
        border-radius: 20px;
        padding: 2rem;
        border: 1px solid rgba(255, 255, 255, 0.1);
        transition: all 0.3s ease;
    }
    
    .viz-card:hover {
        transform: translateY(-5px);
        box_shadow: 0 20px 40px rgba(0, 0, 0, 0.2);
    }
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_data():
    try:
        return pd.read_csv("data/nepal_gis_dailydata.csv")
    except FileNotFoundError:
        return None

if "preprocessed_data" in st.session_state:
    df = st.session_state["preprocessed_data"]
    st.success("Using preprocessed data.")
elif "data" in st.session_state:
    df = st.session_state["data"]
    st.warning("Preprocessed data not found. Using raw uploaded data.")
    if 'Unnamed: 0' in df.columns:
        df.drop(columns=['Unnamed: 0'], inplace=True)
else:
    df = load_data()
    if df is not None:
        st.warning("No session data found. Using fallback dataset.")
    else:
        st.error("No dataset found. Please upload or preprocess data first.")
        st.stop()


df.columns = df.columns.str.strip()
if 'Unnamed: 0' in df.columns:
    df.drop(columns=['Unnamed: 0'], inplace=True)

all_columns = df.columns.tolist()

binary_cols = [col for col in df.select_dtypes(include=['int64', 'float64']).columns 
               if df[col].dropna().nunique() == 2]
non_numeric_columns = cat_cols = df.select_dtypes(include=['object']).columns.tolist() + binary_cols
numeric_columns = list(df.select_dtypes(include=['float', 'int']).columns)


st.markdown("""
<div class="main-header fade-in">
    <h1 class="main-title"> Nepals district data visualizations</h1>
    <p class="subtitle">Advanced data Analysis Platform for Nepal</p>
</div>
""", unsafe_allow_html=True)


with st.sidebar:
    st.markdown("""
    <div class="filter-section">
        <h2 class="filter-title">Control Panel</h2>
    </div>
    """, unsafe_allow_html=True)
    
    
    date_col = "Date"
    region_col = "District"
    
    if date_col in df.columns:
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
        min_date, max_date = df[date_col].min(), df[date_col].max()
        
        st.markdown("**Date Range**")
        start_date, end_date = st.date_input(
            "Select Date Range", 
            [min_date, max_date],
            min_value=min_date,
            max_value=max_date,
            label_visibility="collapsed"
        )
    else:
        start_date = end_date = None
    
   
    if region_col in df.columns:
        st.markdown("**Regions**")
        selected_regions = st.multiselect(
            "Select Regions", 
            options=sorted(df[region_col].dropna().unique()), 
            default=None,
            label_visibility="collapsed"
        )
    else:
        selected_regions = None
    
  
    st.markdown("** Primary Metric**")
    primary_metric = st.selectbox(
        "Select Primary Metric",
        options=numeric_columns,
        label_visibility="collapsed"
    )


filtered_df = df.copy()
if start_date and end_date:
    filtered_df = filtered_df[(filtered_df[date_col] >= pd.to_datetime(start_date)) & 
                             (filtered_df[date_col] <= pd.to_datetime(end_date))]
if selected_regions:
    filtered_df = filtered_df[filtered_df[region_col].isin(selected_regions)]


df_sample = filtered_df.sample(n=min(len(filtered_df), 5000), random_state=1)


st.markdown('<div class="fade-in">', unsafe_allow_html=True)
col1, col2, col3, col4 = st.columns(4)

metrics_data = [
    {"value": f"{len(filtered_df):,}", "label": "Total Records"},
    {"value": f"{len(numeric_columns)}", "label": "Numeric columns"},
    {"value": f"{filtered_df[region_col].nunique() if region_col in filtered_df.columns else 0}", "label": " Districts"},
    {"value": f"{(filtered_df[date_col].max() - filtered_df[date_col].min()).days if date_col in filtered_df.columns else 0}", "label": " Days Span"}
]

for i, (col, metric) in enumerate(zip([col1, col2, col3, col4], metrics_data)):
    with col:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{metric['value']}</div>
            <div class="metric-label">{metric['label']}</div>
        </div>
        """, unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)


st.markdown('<div class="chart-container fade-in">', unsafe_allow_html=True)


col1, = st.columns(1)

with col1:
    st.markdown("### Interactive Geographic Visualization")

@st.cache_data
def load_geojson():
    try:
        url = "https://raw.githubusercontent.com/Acesmndr/nepal-geojson/master/generated-geojson/nepal-with-districts-acesmndr.geojson"
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        return gpd.read_file(url)
    except Exception as e:
        st.error(f"Failed to load Nepal GeoJSON: {e}")
        return None
try:
    gdf = load_geojson()
    
    if gdf is not None:
        col_map1, col_map2 = st.columns([2, 1])
        
        with col_map2:
            geo_col = st.selectbox("Climate Feature", primary_metric, key="geo_col")
            district_col_data = st.selectbox(
    "District Column (Data)",
    options=non_numeric_columns,
    index=non_numeric_columns.index("District") if "District" in non_numeric_columns else 0,
    key="geo_data"
)
            district_col_geo = st.selectbox(" District Column (Map)", options=list(gdf.columns), index=list(gdf.columns).index('DISTRICT') if 'DISTRICT' in gdf.columns else 0, key="geo_geo")
    
            df_districts = set(df[district_col_data].dropna().astype(str).str.upper().str.strip().unique())
            gdf_districts = set(gdf[district_col_geo].dropna().astype(str).str.upper().str.strip().unique())
            missing_in_map = df_districts - gdf_districts
            missing_in_data = gdf_districts - df_districts
            
            st.write(f"Districts in data: {len(df_districts)}")
            st.write(f"Districts in map: {len(gdf_districts)}")
            st.write(f"Districts in data but missing in map: {sorted(missing_in_map)}")
            st.write(f"Districts in map but missing in data: {sorted(missing_in_data)}")

        df_clean = df.copy()
        gdf_clean = gdf.copy()
        
       
        df_clean[district_col_data] = df_clean[district_col_data].astype(str).str.upper().str.strip()
        gdf_clean[district_col_geo] = gdf_clean[district_col_geo].astype(str).str.upper().str.strip()
        
      
        df_clean = df_clean[df_clean[district_col_data] != 'NAN']
        gdf_clean = gdf_clean[gdf_clean[district_col_geo] != 'NAN']

        agg_data = df_clean[[district_col_data, geo_col]].dropna().groupby(district_col_data)[geo_col].mean().reset_index()
        
        merged = gdf_clean.merge(
            agg_data,
            left_on=district_col_geo, 
            right_on=district_col_data,
            how='left'
        )

        with col_map1:
            if len(merged) > 0 and geo_col in merged.columns:
                fig_map = px.choropleth_mapbox(
                    merged,
                    geojson=json.loads(merged.to_json()),
                    locations=district_col_geo,
                    color=geo_col,
                    featureidkey=f"properties.{district_col_geo}",
                    color_continuous_scale='RdYlBu_r',
                    hover_name=district_col_geo,
                    mapbox_style="carto-positron",
                    center={"lat": 28.3949, "lon": 84.1240},
                    zoom=5,
                    title=f"{geo_col} Distribution in Nepal",
                    labels={geo_col: f"{geo_col} Value"}
                )
                fig_map.update_geos(
                    showframe=False,
                    showcoastlines=False,
                    fitbounds="locations",
                    visible=False
                )
                fig_map.update_layout(
                    margin={"r":0,"t":60,"l":0,"b":0},
                    coloraxis_colorbar=dict(
                        title=f"{geo_col}",
                        title_font=dict(color="#1579de", size=14),
                        tickfont=dict(color='#1579de'),
                        thickness=15,
                        len=0.7
                    ),
                    title_font=dict(color="#1579de", size=18),
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    height=550
                )
                st.plotly_chart(fig_map, use_container_width=True)
            else:
                st.warning("No data available for mapping. Please check district name matching.")
                
                
                if len(agg_data) > 0:
                    fig_bar = px.bar(
                        agg_data.head(15), 
                        x=district_col_data, 
                        y=geo_col,
                        title=f"Top 15 Districts - {geo_col}",
                        color=geo_col,
                        color_continuous_scale='RdYlBu_r'
                    )
                    fig_bar.update_layout(
                        xaxis_tickangle=-45,
                        plot_bgcolor='rgba(0,0,0,0)',
                        paper_bgcolor='rgba(0,0,0,0)',
                        title_font=dict(color="#1579de", size=16)
                    )
                    st.plotly_chart(fig_bar, use_container_width=True)
    else:
        st.error("Could not load Nepal geographic data. Showing alternative visualization.")
        
        if len(numeric_columns) > 0:
            col_alt1, col_alt2 = st.columns(2)
            with col_alt1:
                agg_col = st.selectbox("Select Feature", numeric_columns, key="alt_geo")
                
            district_summary = df.groupby(region_col)[agg_col].agg(['mean', 'std', 'count']).reset_index()
            district_summary = district_summary.sort_values('mean', ascending=False).head(20)
            
            fig_alt = px.bar(
                district_summary, 
                x=region_col, 
                y='mean',
                error_y='std',
                title=f" Top 20 Districts by {agg_col}",
                color='mean',
                color_continuous_scale='Viridis'
            )
            fig_alt.update_layout(
                xaxis_tickangle=-45,
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                title_font=dict(color='#2c3e50', size=16),
                height=500
            )
            st.plotly_chart(fig_alt, use_container_width=True)
        
except Exception as e:
    st.error(f"Error generating geographic visualization: {e}")
    st.info("Tip: Check if your data has a district/region column that matches the map boundaries.")
 
st.markdown('</div>', unsafe_allow_html=True)

st.markdown('<div class="chart-container slide-in">', unsafe_allow_html=True)
st.markdown('<h3 class="chart-title"> Category-wise Comparison</h3>', unsafe_allow_html=True)

MAX_CATEGORIES = 10
TOP_N = 10

def get_few_category_columns(df, max_unique=MAX_CATEGORIES):
   
    categorical_cols = []
    for col in df.columns:
        if df[col].dtype == 'object' and df[col].nunique() <= max_unique:
            categorical_cols.append(col)
        elif df[col].dtype in ['int64', 'float64'] and df[col].nunique() <= max_unique:
          
            categorical_cols.append(col)
    return categorical_cols

few_category_columns = get_few_category_columns(df_sample)
tabs = st.tabs(["Pie", "Sunburst", "Stacked Bar"])

with tabs[0]:
    if len(few_category_columns) == 0:
        st.warning("No categorical columns with fewer than 10 unique values found for pie chart.")
    else:
        pie_col = st.selectbox("Pie Chart Column", few_category_columns, key="pie_col")
        if pie_col in df_sample.columns:
            pie_data = df_sample[pie_col].value_counts().nlargest(TOP_N)
            fig_pie = px.pie(
                names=pie_data.index,
                values=pie_data.values,
                template=None,
                color_discrete_sequence=px.colors.qualitative.Set3
            )
            fig_pie.update_traces(textinfo='percent+label')
            fig_pie.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='black', family='Inter'),
                height=400
            )
            st.plotly_chart(fig_pie, use_container_width=True)

with tabs[1]:  
    if len(few_category_columns) < 2:
        st.warning("Need at least 2 categorical columns for sunburst chart.")
    else:
        col1, col2 = st.columns(2)
        
        with col1:
            outer_col = st.selectbox("Outer Ring ", few_category_columns, key="sunburst_outer")
        
        with col2:
            inner_options = [col for col in few_category_columns if col != outer_col]
            inner_col = st.selectbox("Inner Ring ", inner_options, key="sunburst_inner")
        
        if outer_col in df_sample.columns and inner_col in df_sample.columns:
            sunburst_data = df_sample.groupby([outer_col, inner_col]).size().reset_index(name='count')
            sunburst_data = sunburst_data.nlargest(TOP_N, 'count')
            
            fig_sunburst = px.sunburst(
                sunburst_data,
                path=[outer_col, inner_col],
                values='count',
                color='count',
                color_continuous_scale='Viridis',
                title=f"Sunburst Chart: {outer_col} → {inner_col}"
            )
            
            fig_sunburst.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='black', family='Inter'),
                height=500
            )
            st.plotly_chart(fig_sunburst, use_container_width=True)

with tabs[2]: 
    if len(few_category_columns) == 0:
        st.warning("No categorical columns with fewer than 10 unique values found for stacked bar chart.")
    else:
        col1, col2, col3 = st.columns(3)
        
        with col1:
            x_col = st.selectbox(
                "X-axis (Categories)", 
                few_category_columns, 
                key="stacked_x_col"
            )
        
        with col2:
            color_options = [col for col in few_category_columns if col != x_col]
            if len(color_options) == 0:
                st.warning("Need at least 2 categorical columns for stacked bar chart.")
                color_col = None
            else:
                color_col = st.selectbox(
                    "Stack By (Color)", 
                    color_options, 
                    key="stacked_color_col"
                )
        
        with col3:
            numeric_columns = df_sample.select_dtypes(include=[np.number]).columns.tolist()
            if len(numeric_columns) == 0:
                st.warning("No numeric columns found for Y-axis.")
                y_col = None
            else:
                y_col = st.selectbox(
                    "Y-axis (Numeric Values)", 
                    numeric_columns, 
                    key="stacked_y_col"
                )
        
        if x_col in df_sample.columns and color_col is not None and y_col is not None:
            stacked_data = df_sample.groupby([x_col, color_col])[y_col].mean().reset_index()
            top_x_categories = df_sample[x_col].value_counts().nlargest(TOP_N).index
            stacked_data = stacked_data[stacked_data[x_col].isin(top_x_categories)]
            fig_stacked = px.bar(
                stacked_data,
                x=x_col,
                y=y_col,
                color=color_col,
                title=f"Stacked Bar Chart: {y_col} by {x_col} (Stacked by {color_col})",
                color_discrete_sequence=px.colors.qualitative.Set3
            )
            fig_stacked.update_layout(
                template=None,
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='white', family='Inter'),
                height=400,
                xaxis=dict(
                    showgrid=True, 
                    gridcolor='rgba(0,0,0,0.1)',
                    title=x_col
                ),
                yaxis=dict(
                    showgrid=True,
                    gridcolor='rgba(0,0,0,0.1)',
                    title=f"Average {y_col}"
                ),
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                )
            )
            
            st.plotly_chart(fig_stacked, use_container_width=True)
           


st.markdown('<div class="chart-container slide-in">', unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    st.markdown('<h3 class="chart-title">Temporal Trend Analysis</h3>', unsafe_allow_html=True)
    
    if date_col in df_sample.columns and primary_metric in df_sample.columns:
       
        df_trend = df_sample.dropna(subset=[date_col, primary_metric])
        df_trend['YearMonth'] = df_trend[date_col].dt.to_period('M')
        trend_data = df_trend.groupby('YearMonth')[primary_metric].mean().reset_index()
        trend_data['YearMonth'] = trend_data['YearMonth'].astype(str)
        
        fig_trend = px.line(
            trend_data, 
            x='YearMonth', 
            y=primary_metric,
            template="plotly_dark"
        )
        
        fig_trend.update_traces(
            line=dict(color='#60a5fa', width=3),
            mode='lines+markers',
            marker=dict(size=8, color='#34d399', line=dict(width=2, color='#60a5fa'))
        )
        
        fig_trend.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white', family='Inter'),
            height=400,
            showlegend=False,
            xaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.1)'),
            yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.1)')
        )
        
        st.plotly_chart(fig_trend, use_container_width=True)

with col2:
    st.markdown('<h3 class="chart-title"> Regional Performance</h3>', unsafe_allow_html=True)
    
    if region_col in df_sample.columns and primary_metric in df_sample.columns:
        regional_data = df_sample.groupby(region_col)[primary_metric].mean().sort_values(ascending=False).head(10)
        
        fig_bar = px.bar(
            x=regional_data.values,
            y=regional_data.index,
            orientation='h',
            template="plotly_dark",
            color=regional_data.values,
            color_continuous_scale='Viridis'
        )
        
        fig_bar.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white', family='Inter'),
            height=400,
            showlegend=False,
            xaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.1)'),
            yaxis=dict(showgrid=False)
        )
        
        st.plotly_chart(fig_bar, use_container_width=True)

st.markdown('</div>', unsafe_allow_html=True)


st.markdown('<div class="chart-container slide-in">', unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    st.markdown('<h3 class="chart-title"> Distribution Analysis</h3>', unsafe_allow_html=True)
    
    fig_hist = px.histogram(
        df_sample.dropna(subset=[primary_metric]), 
        x=primary_metric,
        nbins=30,
        template="plotly_dark",
        color_discrete_sequence=['#60a5fa']
    )
    
    fig_hist.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white', family='Inter'),
        height=350,
        showlegend=False,
        xaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.1)'),
        yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.1)')
    )
    
    st.plotly_chart(fig_hist, use_container_width=True)

with col2:
    st.markdown('<h3 class="chart-title"> Correlation Matrix</h3>', unsafe_allow_html=True)
    
   
    corr_cols = numeric_columns[:5]
    if len(corr_cols) > 1:
        corr_matrix = df_sample[corr_cols].corr()
        
        fig_corr = px.imshow(
            corr_matrix,
            text_auto=True,
            aspect="auto",
            color_continuous_scale='RdBu',
            template="plotly_dark"
        )
        
        fig_corr.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white', family='Inter'),
            height=350
        )
        
        st.plotly_chart(fig_corr, use_container_width=True)

tabs = st.tabs(["3D Surface Plot"])

def smooth_z_matrix(Z, sigma=1.2):
    
    Z_clean = np.nan_to_num(Z, nan=0.0)
    
    Z_smooth = gaussian_filter(Z_clean, sigma=sigma)
    
    return Z_smooth

def apply_temporal_averaging(df, date_col, value_col, region_col, window_days=7):
    df_smooth = df.copy()
    smoothed_data = []
    
    for region in df[region_col].unique():
        region_data = df[df[region_col] == region].copy()
        region_data = region_data.sort_values(date_col)
        
       
        region_data[f'{value_col}_smooth'] = region_data[value_col].rolling(
            window=min(window_days, len(region_data)), 
            center=True, 
            min_periods=1
        ).mean()
        
        smoothed_data.append(region_data)
    
    return pd.concat(smoothed_data, ignore_index=True)

with tabs[0]:
    st.markdown("### 3D Surface Visualization")
    
    if len(numeric_columns) >= 3:
        surface_var = st.selectbox("Z-axis Variable", numeric_columns, key="surface_var")
        
        try:
            df_surface = df_sample.dropna(subset=[date_col, region_col, surface_var])
            
            if len(df_surface) > 0:
                
                df_surface = apply_temporal_averaging(
                    df_surface, date_col, surface_var, region_col, window_days=7
                )
                plot_var = f'{surface_var}_smooth'
                
               
                all_districts = sorted(df_surface[region_col].unique())
                
               
                col1, col2 = st.columns([3, 1])
                
                with col1:
                    selected_districts = st.multiselect(
                        "Select Districts (up to 10)",
                        options=all_districts,
                        default=all_districts[:5] if len(all_districts) >= 5 else all_districts,
                        max_selections=10,
                        key="district_selector"
                    )
                
                with col2:
                   
                    if st.button("Select All", key="select_all_districts"):
                        selected_districts = all_districts[:10]
                    if st.button("Clear All", key="clear_all_districts"):
                        selected_districts = []
                
               

                
                if selected_districts:
                    
                    all_dates = sorted(df_surface[date_col].unique())
                    
                   
                    try:
                        if df_surface[date_col].dtype == 'datetime64[ns]' or pd.api.types.is_datetime64_any_dtype(df_surface[date_col]):
                            dates_sorted = [d for d in all_dates if d.year >= 2000 and d.year <= 2024]
                        else:
                            
                            dates_sorted = [d for d in all_dates if int(str(d)[:4]) >= 2000 and int(str(d)[:4]) <= 2024]
                    except:
                       
                        dates_sorted = all_dates
                    
                    
                    
                    
                  
                    district_map = {d: i for i, d in enumerate(selected_districts)}
                    date_map = {d: i for i, d in enumerate(dates_sorted)}
                  
                    df_surface_filtered = df_surface[
                        df_surface[region_col].isin(selected_districts) & 
                        df_surface[date_col].isin(dates_sorted)
                    ]
              
                    df_surface_filtered['District_idx'] = df_surface_filtered[region_col].map(district_map)
                    df_surface_filtered['Date_idx'] = df_surface_filtered[date_col].map(date_map)
                   
                    Z = np.full((len(selected_districts), len(dates_sorted)), np.nan)
                    for _, row in df_surface_filtered.iterrows():
                        Z[row['District_idx'], row['Date_idx']] = row[plot_var]
                    
                    
                    Z_filled = np.nan_to_num(Z, nan=np.nanmean(Z))
                    
                
                    Z_smooth = smooth_z_matrix(Z_filled, sigma=1.2)
                  
                    
                   
                    fig_3d = go.Figure(data=[go.Surface(
                        z=Z_smooth,
                        x=list(date_map.values()),
                        y=list(district_map.values()),
                        colorscale='Viridis',
                        hovertemplate='<b>District:</b> %{customdata[0]}<br>' +
                                    '<b>Time:</b> %{customdata[1]}<br>' +
                                    '<b>Value:</b> %{z:.2f}<extra></extra>',
                        customdata=[[[selected_districts[i], str(dates_sorted[j])[:10]] 
                                   for j in range(len(dates_sorted))] 
                                   for i in range(len(selected_districts))],
                        showscale=True,
                        colorbar=dict(
                            title=f"{surface_var} (Smoothed)",
                            tickfont=dict(color='white')
                        ),
                       
                        lighting=dict(
                            ambient=0.3,
                            diffuse=0.9,
                            specular=0.4,
                            roughness=0.05,
                            fresnel=0.2
                        ),
                       
                        contours=dict(
                            z=dict(show=True, usecolormap=True, project_z=True, width=1, color='rgba(255,255,255,0.3)'),
                            x=dict(show=False),
                            y=dict(show=False)
                        )
                    )])
                    
                    fig_3d.update_layout(
                        title=dict(
                            text=f'3D Surface: {surface_var} across {len(selected_districts)} Districts (Optimally Smoothed)',
                            font=dict(color='white', size=16, family='Inter')
                        ),
                        scene=dict(
                            xaxis=dict(
                                title=dict(text='Time (2000-2024)', font=dict(color='white')),
                                tickvals=list(range(0, len(dates_sorted), max(1, len(dates_sorted)//12))),
                                ticktext=[str(dates_sorted[i])[:10] for i in range(0, len(dates_sorted), max(1, len(dates_sorted)//12))],
                                tickfont=dict(color='white', size=10),
                                showgrid=True,
                                gridcolor='rgba(255,255,255,0.2)',
                                tickangle=45
                            ),
                            yaxis=dict(
                                title=dict(text='Districts', font=dict(color='white')),
                                tickvals=list(range(len(selected_districts))),
                                ticktext=[dist[:15] + '...' if len(dist) > 15 else dist for dist in selected_districts],
                                tickfont=dict(color='white', size=10),
                                showgrid=True,
                                gridcolor='rgba(255,255,255,0.2)',
                                tickangle=45
                            ),
                            zaxis=dict(
                                title=dict(text=surface_var, font=dict(color='white')),
                                tickfont=dict(color='white', size=10),
                                showgrid=True,
                                gridcolor='rgba(255,255,255,0.2)'
                            ),
                            bgcolor='rgba(0,0,0,0)',
                            camera=dict(
                                eye=dict(x=1.5, y=1.5, z=1.2)
                            ),
                            aspectmode='cube'
                        ),
                        template=None,
                        font=dict(color='white', family='Inter'),
                        paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(0,0,0,0)',
                        height=600,
                        margin=dict(l=0, r=0, t=50, b=0)
                    )
                    
                    st.plotly_chart(fig_3d, use_container_width=True)
                    
            
                        
                else:
                    st.warning("Please select at least one district to visualize.")
                    
        except Exception as e:
            st.error(f"3D Surface Plot Error: {e}")
            st.error("Please check your data format and try again.")
    else:
        st.warning("Not enough numeric columns for 3D surface plot.")
col1, = st.columns(1)

with col1:
    st.markdown('<h3 class="chart-title"> Boxplot Visualization</h3>', unsafe_allow_html=True)

   
    y_column = st.selectbox("Select Numeric Column", options=numeric_columns, key="box_y")

   
    color_by = st.selectbox("Group By (Optional)", options=[None] + non_numeric_columns, key="box_color")

   
    subset_cols = [y_column] + ([color_by] if color_by else [])
    filtered_df = df_sample.dropna(subset=subset_cols)

   
    fig_box = px.box(
        filtered_df,
        y=y_column,
        color=color_by if color_by else None,
        points="all", 
        template="plotly_dark",
        color_discrete_sequence=px.colors.qualitative.Pastel,
        title=f"Distribution of {y_column}" + (f" grouped by {color_by}" if color_by else ""),
        labels={y_column: y_column, color_by: color_by} if color_by else {y_column: y_column},
        hover_data=filtered_df.columns,
        height=400,
    )

    fig_box.update_layout(
        font=dict(family="Inter", size=14, color="white"),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(30,30,30,0.85)',
        margin=dict(t=50, b=40, l=40, r=40),
        boxmode="group" if color_by else None,
    )

    st.plotly_chart(fig_box, use_container_width=True)


st.markdown('<div class="chart-container fade-in">', unsafe_allow_html=True)
st.markdown('<h3 class="chart-title">Interactive Correlational Analysis</h3>', unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs(["Scatter Plot", "2D Histogram", "Contour Plot"])


with tab1:
    col_scatter1, col_scatter2 = st.columns([4, 1])

    with col_scatter2:
        st.markdown("**Chart Configuration**")
        x1 = st.selectbox("X Axis", numeric_columns, key="x1_new")


        available_y_columns = [col for col in numeric_columns if col != x1]


        y1 = st.selectbox("Y Axis", available_y_columns, key="y1_new")
        color1 = st.selectbox("Color By", [None] + non_numeric_columns, key="c1_new")
        point_size = st.slider("Point Size", 1, 10, 4, key="ps_new")
        alpha = st.slider("Opacity", 0.0, 1.0, 0.4, key="alpha_new")
        show_regression = st.checkbox("Show Regression Line (Linear)", value=True, key="reg_new")
        selected_districts = st.multiselect(
            "Filter by District (optional)",
            options=df_sample["District"].unique(),
            default=[],
            key="dist_new"
        )

    with col_scatter1:
        filtered_df_scatter = df_sample.dropna(subset=[x1, y1])

        if selected_districts:
            filtered_df_scatter = filtered_df_scatter[filtered_df_scatter["District"].isin(selected_districts)]

        if len(filtered_df_scatter) > 5000:
            filtered_df_scatter = filtered_df_scatter.sample(5000, random_state=42)

        fig1 = px.scatter(
            filtered_df_scatter,
            x=x1,
            y=y1,
            color=color1 if color1 else None,
            opacity=alpha,
            trendline="ols" if show_regression else None,
            template="plotly_dark",
            color_discrete_sequence=px.colors.qualitative.Set3,
            height=450
        )

        fig1.update_traces(marker=dict(size=point_size))
        fig1.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white', family='Poppins'),
        )

        st.plotly_chart(fig1, use_container_width=True)


with tab2:
    col_hist2d1, col_hist2d2 = st.columns([4, 1])

    with col_hist2d2:
        st.markdown("**Chart Configuration**")
        x2 = st.selectbox("X Axis", numeric_columns, key="x2_new")
        y2 = st.selectbox("Y Axis", numeric_columns, key="y2_new")
        color2 = st.selectbox("Color By (Not fully supported)", [None] + non_numeric_columns, key="c2_new")
        nbinsx_2d = st.slider("Number of Bins (X)", 10, 100, 30, key="nbinsx_new")
        nbinsy_2d = st.slider("Number of Bins (Y)", 10, 100, 30, key="nbinsy_new")

    with col_hist2d1:
        filtered_df_hist2d = df_sample.dropna(subset=[x2, y2])
        fig2 = go.Figure(go.Histogram2d(
            x=filtered_df_hist2d[x2],
            y=filtered_df_hist2d[y2],
            nbinsx=nbinsx_2d,
            nbinsy=nbinsy_2d,
            colorscale='Viridis'
        ))
        fig2.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white', family='Poppins'),
            height=450,
            xaxis_title=x2,
            yaxis_title=y2,
            title=f"2D Histogram: {y2} vs {x2}",
            xaxis=dict(gridcolor='rgba(128,128,128,0.2)'),
            yaxis=dict(gridcolor='rgba(128,128,128,0.2)')
        )
        st.plotly_chart(fig2, use_container_width=True)

with tab3:
    st.markdown('<div class="chart-container fade-in">', unsafe_allow_html=True)
    st.markdown('<h3 class="chart-title">Interactive Multi-Dimensional Analysis: Contour Plot</h3>', unsafe_allow_html=True)

    col_contour1, col_contour2 = st.columns([4, 1])

    with col_contour2:
        st.markdown("**Chart Configuration**")
        x3 = st.selectbox("X Axis", numeric_columns, key="x3_new")
        y3 = st.selectbox("Y Axis", numeric_columns, key="y3_new")
        show_points = st.checkbox("Show Sample Points", value=True, key="points_contour_new")
        ncontours = st.slider("Number of Contours", 5, 50, 20, key="ncontours_new")

    with col_contour1:
        filtered_df_contour = df_sample.dropna(subset=[x3, y3])
        x_vals = filtered_df_contour[x3]
        y_vals = filtered_df_contour[y3]

        fig3 = go.Figure()

        fig3.add_trace(go.Histogram2dContour(
            x=x_vals,
            y=y_vals,
            colorscale='Turbo',
            contours=dict(showlabels=True, coloring='fill'),
            showscale=True,
            ncontours=ncontours,
            opacity=0.75,
            colorbar=dict(
                title=dict(text="Density Level", font=dict(size=14, color="white")),
                tickfont=dict(color="white")
            )
        ))

        if show_points:
            sample_points = filtered_df_contour.sample(n=min(1000, len(filtered_df_contour)))
            fig3.add_trace(go.Scatter(
                x=sample_points[x3],
                y=sample_points[y3],
                mode='markers',
                marker=dict(size=3, color='white', opacity=0.4),
                name="Sample Points",
                showlegend=False
            ))

        fig3.update_layout(
            template="plotly_dark",
            height=550,
            margin=dict(l=40, r=20, t=50, b=40),
            plot_bgcolor='rgba(0,0,20,0.95)',
            paper_bgcolor='rgba(10,10,30,1)',
            font=dict(color='white', family='Poppins'),
            xaxis=dict(
                title=dict(text=x3, font=dict(size=14, color='white')),
                gridcolor='rgba(255,255,255,0.15)'
            ),
            yaxis=dict(
                title=dict(text=y3, font=dict(size=14, color='white')),
                gridcolor='rgba(255,255,255,0.15)'
            )
        )
        st.plotly_chart(fig3, use_container_width=True)

    st.markdown('</div>', unsafe_allow_html=True)

st.header("Advanced analysis")

tabs = st.tabs([
    "Correlation Matrix", 
    "Anomaly Detection", 
    "Economic Risk Scatter", 
])


with tabs[0]:
    st.subheader("Correlation Matrix")
    with st.expander("Select Columns for Correlation"):
        corr_columns = st.multiselect("Select columns for correlation matrix:", numeric_columns,
                                      default=numeric_columns[:10] if len(numeric_columns) > 10 else numeric_columns)
    if corr_columns:
        sampled = df.sample(min(len(df), 10000), random_state=42)
        corr = sampled[corr_columns].corr()
        fig2 = px.imshow(corr, text_auto=True, color_continuous_scale='RdBu')
        st.plotly_chart(fig2, use_container_width=True)


with tabs[1]:
    st.subheader("Time Series Anomaly Detection")
    with st.expander("Configure Anomaly Plot Columns"):
        col1, col2, col3 = st.columns(3)
        with col1:
            st.write("Date Column: **Date**")
            date_col_anomaly = "Date"
        with col2:
            floods_col = st.selectbox("Column:", numeric_columns, key="anomaly_floods")
        with col3:
            available_drought_cols = [col for col in numeric_columns if col != floods_col]
            drought_col = st.selectbox("Column:", available_drought_cols, key="anomaly_drought")

    if date_col_anomaly in df.columns and floods_col in df.columns and drought_col in df.columns:
        data_agg = df.groupby(date_col_anomaly)[[floods_col, drought_col]].sum().reset_index()
        fig3 = go.Figure()
        fig3.add_trace(go.Scatter(x=data_agg[date_col_anomaly], y=data_agg[floods_col], mode='lines', name=floods_col))
        fig3.add_trace(go.Scatter(x=data_agg[date_col_anomaly], y=data_agg[drought_col], mode='lines', name=drought_col))
        st.plotly_chart(fig3, use_container_width=True)


with tabs[2]:
    st.subheader("Economic Loss vs Population Exposure")
    with st.expander("Configure Risk Map Columns"):
        col1, col2, col3 = st.columns(3)
        with col1:
            st.write("District Column: **District**")
            district_col_risk = "District"
        with col2:
            economic_col = st.selectbox("Column:", numeric_columns, key="risk_economic")
        with col3:
            available_population_cols = [col for col in numeric_columns if col != economic_col]
            population_col = st.selectbox("Column:", available_population_cols, key="risk_population")

    if district_col_risk in df.columns and economic_col in df.columns and population_col in df.columns:
        district_agg = df.groupby(district_col_risk)[[economic_col, population_col]].mean().reset_index()
        fig4 = px.scatter(district_agg, x=economic_col, y=population_col, text=district_col_risk,
                          size=population_col, color=economic_col)
        st.plotly_chart(fig4, use_container_width=True)


st.markdown("""
<div style="text-align: center; margin-top: 3rem; padding: 2rem; border-top: 1px solid rgba(255,255,255,0.1);">
    <h3 style="color: #60a5fa; margin-bottom: 1rem;">Nepal data visualization</h3>
    <p style="color: #94a3b8; font-size: 1.1rem; margin-bottom: 0.5rem;">Data Mining mini project</p>
    <p style="color: #64748b; font-size: 0.9rem;">Built using Streamlit, Plotly By Sophiya and Puja</p>
</div>
""", unsafe_allow_html=True)