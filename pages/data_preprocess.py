import streamlit as st
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(
    page_title="Nepal GIS Data Preprocessing",
    layout="wide",
    initial_sidebar_state="expanded"
)


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
        transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275);
        cursor: pointer;
    }
    
    .main-header:hover {
        transform: translateY(-6px) scale(1.01);
        background: rgba(255, 255, 255, 0.15);
        box-shadow: 0 12px 40px rgba(0, 0, 0, 0.4);
        border: 1px solid rgba(255, 255, 255, 0.3);
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
        border-left: 4px solid rgba(102, 126, 234, 0.8);
    }

    .metric-card:hover {
        transform: translateY(-8px) scale(1.02);
        background: rgba(255, 255, 255, 0.2);
        box-shadow: 0 15px 40px rgba(0, 0, 0, 0.4);
        border: 1px solid rgba(255, 255, 255, 0.3);
        border-left: 4px solid rgba(102, 126, 234, 1);
    }

    .success-card {
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(195, 230, 203, 0.3);
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
        background: rgba(255, 255, 255, 0.2);
        box-shadow: 0 15px 40px rgba(0, 0, 0, 0.4);
        border: 1px solid rgba(195, 230, 203, 0.5);
    }

    .warning-card {
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 234, 167, 0.3);
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
        background: rgba(255, 255, 255, 0.2);
        box-shadow: 0 15px 40px rgba(0, 0, 0, 0.4);
        border: 1px solid rgba(255, 234, 167, 0.5);
    }

    .info-card {
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(190, 229, 235, 0.3);
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
        background: rgba(255, 255, 255, 0.2);
        box-shadow: 0 15px 40px rgba(0, 0, 0, 0.4);
        border: 1px solid rgba(190, 229, 235, 0.5);
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
        border-top: 2px solid rgba(102, 126, 234, 0.6);
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
    data = st.session_state["data"]
    st.success("Using uploaded or session data.")
else:
    try:
        data = pd.read_csv("data/nepal_gis_dailydata.csv")
        st.info("No uploaded data found. Using default dataset.")
    except FileNotFoundError:
        st.error("Default dataset not found.")
        st.stop()


data.columns = (
    data.columns
    .str.replace(r'[^a-zA-Z0-9\s]', '', regex=True) 
    .str.replace(r'\s+', '_', regex=True)            
                               
)

if 'date' in data.columns:
    data['date'] = pd.to_datetime(data['date'], errors='coerce')
else:
    print(" date column not found. Available columns:", data.columns.tolist())    

def classify_columns(data):
    """Simplified column classification"""
    binary_cols = [col for col in data.select_dtypes(include=['int64', 'float64']).columns 
                   if data[col].dropna().nunique() == 2]
    cat_cols = data.select_dtypes(include=['object']).columns.tolist() + binary_cols
    num_cols = [col for col in data.select_dtypes(include=['int64', 'float64']).columns 
                if col not in binary_cols]
    return cat_cols, num_cols

def create_distribution_chart(data, column):
    """Create distribution chart for a column"""
    if data[column].dtype in ['int64', 'float64']:
        fig = px.histogram(data, x=column, nbins=30, title=f"Distribution of {column}")
        fig.update_layout(showlegend=False)
    else:
        value_counts = data[column].value_counts().head(10)
        fig = px.bar(x=value_counts.index, y=value_counts.values, title=f"Top 10 values in {column}")
        fig.update_layout(showlegend=False)
    return fig


st.markdown("""
<div class="main-header">
    <h1> Nepal GIS Data Preprocessing</h1>
    <p>Clean, transform, and prepare your data for analysis and modeling</p>
</div>
""", unsafe_allow_html=True)


with st.sidebar:
    st.markdown("### Navigation")
    
  
    progress_steps = ["Load Data", "Column Classification", "Data Cleaning", "Preprocessing", "Export"]
    current_step = 0
    
    for i, step in enumerate(progress_steps):
        if i <= current_step:
            st.markdown(f"{step}")
        else:
            st.markdown(f"{step}")
    
    st.markdown("---")
    
   
    st.markdown("### Dataset Info")
    stats_placeholder = st.empty()


data_loaded = True


if not data_loaded:
    st.markdown("""
    <div class="warning-card">
        <h4> No Data Available</h4>
        <p>Please either run the EDA step first or upload a CSV file to continue.</p>
    </div>
    """, unsafe_allow_html=True)
    st.stop()


st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
categorical_cols, numerical_cols = classify_columns(data)
st.session_state["categorical_columns"] = categorical_cols
st.session_state["numerical_columns"] = numerical_cols


with stats_placeholder:
    st.metric("Rows", f"{data.shape[0]:,}")
    st.metric("Columns", data.shape[1])
    st.metric("Categorical", len(categorical_cols))
    st.metric("Numerical", len(numerical_cols))


st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
st.markdown("## Data Overview")

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Total Rows", f"{data.shape[0]:,}")
with col2:
    st.metric("Total Columns", data.shape[1])
with col3:
    st.metric("Data Types", len(data.dtypes.unique()))
with col4:
    missing_count = data.isnull().sum().sum()
    st.metric("Missing Values", f"{missing_count:,}")


st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
st.markdown("## Data Cleaning")
st.markdown("###  Duplicate Column Detection")

def find_duplicate_columns(data):
    duplicate_cols = {}
    for i in range(data.shape[1]):
        col1_name = data.columns[i]
        for j in range(i + 1, data.shape[1]):
            col2_name = data.columns[j]
            if data[col1_name].equals(data[col2_name]):
                if col1_name not in duplicate_cols:
                    duplicate_cols[col1_name] = []
                duplicate_cols[col1_name].append(col2_name)
    return duplicate_cols

with st.spinner("Checking for duplicate columns..."):
    duplicate_cols_found = find_duplicate_columns(data)

if duplicate_cols_found:
    st.markdown("""
    <div class="warning-card">
        <h4> Duplicate Columns Found</h4>
        <p>These columns have identical content and can be safely removed:</p>
    </div>
    """, unsafe_allow_html=True)
    
    cols_to_drop = []
    for original, duplicates in duplicate_cols_found.items():
        st.write(f"**{original}** duplicated by: {', '.join(duplicates)}")
        cols_to_drop.extend(duplicates)
    
    if st.button(f" Remove {len(cols_to_drop)} Duplicate Columns"):
        data.drop(columns=cols_to_drop, inplace=True)
        categorical_cols, numerical_cols = classify_columns(data)
        st.session_state["categorical_columns"] = categorical_cols
        st.session_state["numerical_columns"] = numerical_cols
        st.success(f"Removed {len(cols_to_drop)} duplicate columns")
        st.rerun()
else:
    st.markdown("""
    <div class="success-card">
        <h4>No Duplicate Columns</h4>
        <p>All columns contain unique data.</p>
    </div>
    """, unsafe_allow_html=True)

st.markdown("### Missing Values Analysis")

missing_data = data.isnull().sum()
missing_summary = missing_data[missing_data > 0]

if not missing_summary.empty:
    col1, col2 = st.columns([2, 1])
    
    with col1:
        fig = px.bar(
            x=missing_summary.index,
            y=missing_summary.values,
            title="Missing Values by Column",
            labels={'x': 'Columns', 'y': 'Missing Count'}
        )
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("**Missing Values:**")
        for col, count in missing_summary.items():
            percentage = (count / len(data)) * 100
            st.write(f"**{col}**: {count:,} ({percentage:.1f}%)")
    
  
    st.markdown("###  Handle Missing Values")
    
    strategy = st.selectbox(
        "Choose strategy:",
        ['Drop Rows', 'Fill with Median (Numerical)', 'Fill with Mode (Categorical)'],
        key='missing_strategy'
    )
    
    if st.button(" Apply Missing Value Strategy"):
        if strategy == ' Drop Rows':
            initial_rows = data.shape[0]
            data.dropna(inplace=True)
            st.success(f"Removed {initial_rows - data.shape[0]:,} rows with missing values")
        
        elif strategy == 'Fill with Median (Numerical)':
            for col in numerical_cols:
                if col in data.columns and data[col].isnull().any():
                    median_val = data[col].median()
                    data[col].fillna(median_val, inplace=True)
                    st.success(f" Filled '{col}' with median: {median_val:.2f}")
        
        elif strategy == 'Fill with Mode (Categorical)':
            for col in categorical_cols:
                if col in data.columns and data[col].isnull().any():
                    mode_val = data[col].mode()
                    if len(mode_val) > 0:
                        data[col].fillna(mode_val[0], inplace=True)
                        st.success(f"Filled '{col}' with mode: '{mode_val[0]}'")
        
        st.rerun()
else:
    st.markdown("""
    <div class="success-card">
        <h4>No Missing Values</h4>
        <p>Your dataset is complete with no missing values.</p>
    </div>
    """, unsafe_allow_html=True)


st.markdown("### Outlier Detection")

if numerical_cols:
    selected_col = st.selectbox("Select column for outlier analysis:", numerical_cols)
    
    if selected_col:
        col1, col2 = st.columns([2, 1])
        
        with col1:
            fig = create_distribution_chart(data, selected_col)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            stats = data[selected_col].describe()
            st.markdown("**Statistics:**")
            for stat, value in stats.items():
                st.write(f" **{stat}**: {value:.2f}")
            

            Q1 = data[selected_col].quantile(0.25)
            Q3 = data[selected_col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outliers = data[(data[selected_col] < lower_bound) | (data[selected_col] > upper_bound)]
            st.metric(" Outliers Found", len(outliers))
            
            if len(outliers) > 0 and st.button(" Remove Outliers"):
                data = data[(data[selected_col] >= lower_bound) & (data[selected_col] <= upper_bound)]
                st.success(f" Removed {len(outliers)} outliers")
                st.rerun()


st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
st.markdown("## Data Preprocessing")
st.markdown("### Feature Scaling")

if numerical_cols:
    col1, col2 = st.columns([1, 1])
    
    with col1:
        scaling_method = st.selectbox(
            "Scaling method:",
            ['None', 'Min-Max Normalization', 'Z-Score Standardization']
        )
    
    with col2:
        cols_to_scale = st.multiselect(
            "Select columns to scale:",
            numerical_cols,
            default=numerical_cols
        )
    
    if scaling_method != 'None' and cols_to_scale:
        if st.button("Apply Scaling"):
            if scaling_method == 'Min-Max Normalization':
                scaler = MinMaxScaler()
                data[cols_to_scale] = scaler.fit_transform(data[cols_to_scale])
                st.success("pplied Min-Max Normalization")
            elif scaling_method == 'Z-Score Standardization':
                scaler = StandardScaler()
                data[cols_to_scale] = scaler.fit_transform(data[cols_to_scale])
                st.success("Applied Z-Score Standardization")


st.markdown("### Data Sampling")

current_size = data.shape[0]
st.metric(" Current Dataset Size", f"{current_size:,} rows")

col1, col2 = st.columns([1, 1])

with col1:
    sampling_method = st.selectbox(
        "Sampling method:",
        ['None', ' Random Sampling', 'Stratified Sampling']
    )

with col2:
    if sampling_method != 'None':
        sample_size = st.slider(
            "Sample size:",
            min_value=100,
            max_value=min(current_size, 10000),
            value=min(1000, current_size)
        )

if sampling_method != 'None' and st.button(" Apply Sampling"):
    if sampling_method == 'Random Sampling':
        data = data.sample(n=sample_size, random_state=42)
        st.success(f"applied random sampling: {data.shape[0]:,} rows")
    
    elif sampling_method == 'Stratified Sampling' and categorical_cols:
        stratify_col = st.selectbox("Stratify by:", categorical_cols)
        proportions = data[stratify_col].value_counts(normalize=True)
        sampled_dfs = []
        
        for category, proportion in proportions.items():
            category_data = data[data[stratify_col] == category]
            category_sample_size = int(sample_size * proportion)
            if category_sample_size > 0:
                category_sample = category_data.sample(
                    n=min(category_sample_size, len(category_data)), 
                    random_state=42
                )
                sampled_dfs.append(category_sample)
        
        data = pd.concat(sampled_dfs, ignore_index=True)
        st.success(f" Applied stratified sampling: {data.shape[0]:,} rows")
    
        
st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
st.markdown("## Export & Save")

st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
st.markdown("## Final Processed Data")

st.dataframe(data.head(100), use_container_width=True)

st.markdown("Showing the first 100 rows. You can download the full dataset above.")


st.markdown("### Export Options")

col1, col2 = st.columns(2)

with col1:
    if st.button("Save to Session"):
        st.session_state["preprocessed_data"] = data
        st.success("Saved to session state")

with col2:
    csv = data.to_csv(index=False)
    st.download_button(
        label=" Download CSV",
        data=csv,
        file_name="nepal_gis_processed.csv",
        mime="text/csv"
    )




st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
st.markdown("""
<div class="success-card">
    <h2> Preprocessing Complete!</h2>
    <p>Your data has been successfully cleaned and preprocessed. </p>
</div>
""", unsafe_allow_html=True)