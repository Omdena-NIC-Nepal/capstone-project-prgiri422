import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import plotly.express as px
from config import USE_GOOGLE_DRIVE
from utils import load_file, load_model, save_model

# Initialize session state
if 'df' not in st.session_state:
    st.session_state.df = None
if 'featured_df' not in st.session_state:
    st.session_state.featured_df = None
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False

# Set page config
st.set_page_config(
    page_title="Nepal Climate Analysis",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for sidebar
st.markdown("""
<style>
    [data-testid="stSidebar"] {
        background: linear-gradient(135deg, #1e5799 0%,#207cca 100%);
        color: white;
    }
    .sidebar .sidebar-content {
        width: 300px;
    }
    .sidebar-title {
        font-size: 24px !important;
        font-weight: bold !important;
        margin-bottom: 20px !important;
    }
    .sidebar-section {
        margin-bottom: 30px !important;
    }
    .variable-selector {
        margin-top: 20px !important;
    }
</style>
""", unsafe_allow_html=True)

# Load data with caching
@st.cache_data
def load_app_data():
    try:
        df = load_file("data_processed", "clean_data")
        if df is not None and 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'])
        return df
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None

# Initialize data on first run
if st.session_state.df is None and not st.session_state.data_loaded:
    st.session_state.df = load_app_data()
    st.session_state.data_loaded = True

# Sidebar - Fixed navigation
with st.sidebar:
    st.markdown('<div class="sidebar-title">🌦️ Nepal Climate Analysis</div>', unsafe_allow_html=True)
    
    # System status
    st.markdown(f"""
    <div class="sidebar-section">
        🖥️ System Status<br>
        Mode: {'Google Drive' if USE_GOOGLE_DRIVE else 'Local Files'}
        {'' if st.session_state.data_loaded else '<br>⚠️ Data not loaded'}
    </div>
    """, unsafe_allow_html=True)
    
    # Navigation
    st.markdown('<div class="sidebar-section">📌 Navigation</div>', unsafe_allow_html=True)
    page = st.radio(
        "Go to:",
        ["Home", "Data Exploration", "EDA", "Feature Engineering", "Model Training", "Prediction", "About"],
        label_visibility="collapsed"
    )
    
    # Variable selector (only show if data is loaded)
    if st.session_state.df is not None and page in ["Data Exploration", "EDA", "Prediction"]:
        st.markdown('<div class="sidebar-section variable-selector">📊 Variable Selection</div>', unsafe_allow_html=True)
        
        numeric_cols = st.session_state.df.select_dtypes(include=['float64', 'int64']).columns.tolist()
        selected_var = st.selectbox(
            "Select primary variable:",
            numeric_cols,
            index=numeric_cols.index('Temp_2m') if 'Temp_2m' in numeric_cols else 0
        )
        
        if page in ["Data Exploration", "EDA"]:
            secondary_var = st.selectbox(
                "Select secondary variable:",
                [None] + numeric_cols,
                index=0
            )
        
        if 'District' in st.session_state.df.columns:
            districts = ['All'] + sorted(st.session_state.df['District'].unique().tolist())
            selected_district = st.selectbox(
                "Filter by district:",
                districts,
                index=0
            )

def home_page():
    st.title("🌦️ Climate Change Impact Assessment for Nepal")
    st.markdown("""
    ### Welcome to the Climate Analysis Dashboard
    **Key Features:**
    - 📊 Data Exploration
    - 🔍 Exploratory Data Analysis (EDA)
    - ⚙️ Feature Engineering
    - 🤖 Model Training
    - 🔮 Prediction
    """)
    
    if st.session_state.df is not None:
        st.success("✅ Data loaded successfully!")
        st.metric("Total Records", len(st.session_state.df))
       
    else:
        st.error("❌ Failed to load data. See error details above.")

def data_exploration_page():
    st.title("📊 Data Exploration")
    if st.session_state.df is None:
        return st.warning("Please load data first from the Home page")
    
    df = st.session_state.featured_df if st.session_state.featured_df is not None else st.session_state.df
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Date Range", f"{df['Date'].min().date()} to {df['Date'].max().date()}")
    with col2:
        st.metric("Districts", df['District'].nunique())
    
    with st.expander("🔍 View raw data"):
        st.dataframe(df.head())
    
    st.subheader(f"📈 {selected_var} Distribution")
    plot_df = df[df['District'] == selected_district] if selected_district != 'All' else df
    
    fig = px.line(plot_df, x='Date', y=selected_var)
    st.plotly_chart(fig, use_container_width=True)
    
    if secondary_var:
        st.subheader(f"🔗 {selected_var} vs {secondary_var}")
        fig = px.scatter(plot_df, x=selected_var, y=secondary_var, 
                        color='District' if 'District' in df.columns else None)
        st.plotly_chart(fig, use_container_width=True)

def eda_page():
    st.title("🔍 Exploratory Data Analysis")
    if st.session_state.df is None:
        return st.warning("Please load data first")
    
    df = st.session_state.featured_df if st.session_state.featured_df is not None else st.session_state.df
    
    st.subheader("📊 Distribution Plots")
    col1, col2 = st.columns(2)
    with col1:
        fig = px.histogram(df, x=selected_var, nbins=30, title="Temperature Distribution")
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        fig = px.histogram(df, x='Precip', nbins=30, title="Precipitation Distribution")
        st.plotly_chart(fig, use_container_width=True)
    
    st.subheader("📈 Correlation Matrix")
    numeric_df = df.select_dtypes(include=['float64', 'int64'])
    if len(numeric_df.columns) > 1:
        fig = px.imshow(numeric_df.corr(), text_auto=True, aspect="auto")
        st.plotly_chart(fig, use_container_width=True)

def feature_engineering_page():
    st.title("⚙️ Feature Engineering")
    if st.session_state.df is None:
        return st.warning("Please load data first")
    
    df = st.session_state.df.copy()
    
    with st.form("feature_form"):
        st.write("Create new features:")
        create_month = st.checkbox("Extract Month from Date", True)
        create_season = st.checkbox("Create Season Feature", True)
        create_temp_diff = st.checkbox("Create Temperature Difference (Max-Min)", True)
        submitted = st.form_submit_button("Create Features")
    
    if submitted:
        with st.spinner("Creating features..."):
            try:
                if create_month and 'Date' in df.columns:
                    df['Month'] = df['Date'].dt.month
                
                if create_season and 'Month' in df.columns:
                    seasons = {
                        1: 'Winter', 2: 'Winter', 3: 'Spring', 
                        4: 'Spring', 5: 'Spring', 6: 'Summer',
                        7: 'Summer', 8: 'Summer', 9: 'Autumn',
                        10: 'Autumn', 11: 'Autumn', 12: 'Winter'
                    }
                    df['Season'] = df['Month'].map(seasons)
                
                if create_temp_diff and all(c in df.columns for c in ['Temp_max', 'Temp_min']):
                    df['Temp_diff'] = df['Temp_max'] - df['Temp_min']
                
                st.session_state.featured_df = df
                st.success("✅ Features created successfully!")
                
                if st.button("💾 Save Engineered Data"):
                    if save_model(df, "data_processed", "featured_data"):
                        st.success("Data saved successfully!")
            except Exception as e:
                st.error(f"Error creating features: {str(e)}")

def model_training_page():
    st.title("🤖 Model Training")
    if st.session_state.df is None:
        return st.warning("Please load data first")
    
    df = st.session_state.featured_df if st.session_state.featured_df is not None else st.session_state.df
    
    model_type = st.selectbox(
        "Choose model to train:",
        ["Temperature", "Climate Zone", "Vulnerability"]
    )
    
    test_size = st.slider("Test Size (%)", 10, 40, 20)
    
    if model_type == "Temperature":
        st.subheader("Temperature Prediction Model")
        features = st.multiselect(
            "Select features:",
            df.select_dtypes(include=['float64', 'int64']).columns.tolist(),
            default=['Precip', 'Humidity_2m', 'WindSpeed_10m']
        )
        
        if st.button("Train Temperature Model"):
            with st.spinner("Training model..."):
                try:
                    X = df[features]
                    y = df['Temp_2m']
                    X_train, X_test, y_train, y_test = train_test_split(
                        X, y, test_size=test_size/100, random_state=42
                    )
                    
                    model = Ridge()
                    model.fit(X_train, y_train)
                    
                    y_pred = model.predict(X_test)
                    mse = mean_squared_error(y_test, y_pred)
                    
                    st.success(f"✅ Model trained! MSE: {mse:.4f}")
                    if save_model(model, "models", "temp_model"):
                        st.success("Model saved successfully!")
                except Exception as e:
                    st.error(f"Error: {str(e)}")

def prediction_page():
    st.title("🔮 Prediction Dashboard")
    
    model_option = st.selectbox(
        "Select model to use:",
        ["Temperature", "Climate Zone", "Vulnerability"]
    )
    
    pred_date = st.date_input(
        "Select prediction date:",
        min_value=datetime(2020, 1, 1),
        max_value=datetime(2030, 12, 31)
    )
    
    col1, col2 = st.columns(2)
    with col1:
        temp = st.number_input("Temperature (°C)", min_value=-20.0, max_value=50.0, value=25.0)
        precip = st.number_input("Precipitation (mm)", min_value=0.0, max_value=500.0, value=0.5)
    with col2:
        humidity = st.number_input("Humidity (%)", min_value=0.0, max_value=100.0, value=65.0)
        wind_speed = st.number_input("Wind Speed (m/s)", min_value=0.0, max_value=50.0, value=5.0)
    
    if st.button("Run Prediction"):
        with st.spinner("Generating prediction..."):
            try:
                model = load_model(model_option.lower().replace(" ", "_"))
                if model is None:
                    raise ValueError("Model not loaded")
                
                input_features = {
                    'temperature': [temp, precip, humidity],
                    'climate_zone': [temp, precip, humidity, wind_speed],
                    'vulnerability': [temp, precip, humidity, wind_speed]
                }[model_option.lower().replace(" ", "_")]
                
                prediction = model.predict([input_features])[0]
                
                st.success(f"Predicted {model_option}: {prediction:.2f}" if isinstance(prediction, float) else prediction)
                
                st.json({
                    "model": model_option,
                    "prediction": prediction,
                    "features_used": model.feature_names_in_.tolist(),
                    "date": str(pred_date)
                })
            except Exception as e:
                st.error(f"Prediction failed: {str(e)}")

# Page routing
if page == "Home":
    home_page()
elif page == "Data Exploration":
    data_exploration_page()
elif page == "EDA":
    eda_page()
elif page == "Feature Engineering":
    feature_engineering_page()
elif page == "Model Training":
    model_training_page()
elif page == "Prediction":
    prediction_page()
elif page == "About":
    st.title("📘 About This Project")
    st.markdown("""
    ### Climate Change Impact Assessment for Nepal
    **Developed by:** Prabin Giri  
    **Version:** 1.0 (2025)
    """)

st.markdown("---")
st.markdown(f"© 2025 Nepal Climate Analysis Project |")