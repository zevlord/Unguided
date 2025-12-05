import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from statsmodels.tsa.statespace.sarimax import SARIMAX
import warnings

warnings.filterwarnings("ignore")

# Set page layout
st.set_page_config(page_title="Liquor Sales Forecasting", layout="wide")

st.title("🍷 Liquor Sales Analysis & Forecasting")
st.caption("App Version: 3.0 (Dynamic Columns)") # VISUAL CONFIRMATION

# --- Function: Load and Clean Data using Mapped Columns ---
@st.cache_data
def load_and_clean_data(uploaded_file, date_col_name, val_col_name, size_col_name):
    """
    Loads data using the specific column names selected by the user.
    """
    # 1. Read Data
    # Reset file pointer to start to ensure we read from the beginning
    uploaded_file.seek(0)
    
    # Read CSV
    # We read everything at once here. For 500k+ rows, this is okay for Streamlit Cloud.
    sales = pd.read_csv(uploaded_file)
    
    # Clean header whitespace immediately
    sales.columns = sales.columns.str.strip()

    # 2. Date Conversion
    # We use the column name provided by the user (date_col_name)
    try:
        sales[date_col_name] = pd.to_datetime(sales[date_col_name], errors='coerce')
        # Drop rows where date conversion failed
        sales = sales.dropna(subset=[date_col_name])
    except Exception as e:
        st.error(f"Error converting column '{date_col_name}' to datetime. Details: {e}")
        st.stop()

    # 3. Clean Size Column
    # We use the column name provided by the user (size_col_name)
    if size_col_name in sales.columns:
        # Convert to string, lowercase, remove quotes
        cleansize = sales[size_col_name].astype(str).str.lower().str.replace('"', '').str.strip()

        # Extract numeric size
        stdSize = cleansize.str.extract(r'(\d+\.?\d*)')[0].astype(float).fillna(0)
        
        # Extract pack size
        stdPack = cleansize.str.extract(r'(\d+)\s*pk')[0].astype(float).fillna(1)

        # Logic for ML/L/OZ conversion
        multiplier = np.where(cleansize.str.contains('ml'), 1, 
                     np.where(cleansize.str.contains('l') & ~cleansize.str.contains('ml'), 1000,
                     np.where(cleansize.str.contains('oz'), 30, 0)))

        sales['True Size/ml'] = stdSize * multiplier * stdPack
        
        # Filter 0 values
        sales = sales[sales['True Size/ml'] > 0]

    return sales

# --- Sidebar: Configuration ---
st.sidebar.header("1. Upload Data")
uploaded_file = st.sidebar.file_uploader("Upload CSV", type=['csv'])

if uploaded_file is not None:
    # --- STEP A: Read Headers to Setup Dropdowns ---
    # We read just the first row to get column names
    header_df = pd.read_csv(uploaded_file, nrows=0)
    # Strip whitespace from headers so they match our logic later
    header_df.columns = header_df.columns.str.strip()
    all_cols = list(header_df.columns)

    st.sidebar.header("2. Map Columns")
    st.sidebar.info("Please identify the columns in your file:")

    # Helper to guess index
    def get_index(options, search_strings):
        for s in search_strings:
            for i, opt in enumerate(options):
                if s.lower() in opt.lower():
                    return i
        return 0

    # Date Selection
    date_col = st.sidebar.selectbox(
        "Which column contains the Date?", 
        all_cols, 
        index=get_index(all_cols, ['date', 'time'])
    )

    # Value Selection
    val_col = st.sidebar.selectbox(
        "Which column contains the Sales Value ($)?", 
        all_cols, 
        index=get_index(all_cols, ['dollar', 'amount', 'price', 'total', 'sale'])
    )

    # Size Selection
    size_col = st.sidebar.selectbox(
        "Which column contains the Size/Volume?", 
        all_cols, 
        index=get_index(all_cols, ['size', 'desc', 'volume', 'bottle'])
    )

    # --- STEP B: Process Data ---
    if st.sidebar.button("Process Data"):
        with st.spinner('Processing...'):
            # Pass the USER SELECTED column names to the function
            sales_df = load_and_clean_data(uploaded_file, date_col, val_col, size_col)
        
        st.success(f"Successfully processed {len(sales_df):,} rows.")

        # --- STEP C: Aggregate Time Series ---
        # Group by the User Selected Date Column
        salesing = sales_df.groupby(date_col)[val_col].mean()
        salesing = salesing.asfreq('D').fillna(method='ffill')

        # --- Visuals ---
        st.subheader("Data Preview")
        st.dataframe(sales_df.head())
        
        st.subheader("Sales Trends")
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(salesing.index, salesing.values)
        ax.set_title(f"Daily Trends ({val_col})")
        st.pyplot(fig)

        # --- Forecasting ---
        st.subheader("Forecasting (SARIMAX)")
        
        # Simple forecasting logic for demo
        train_size = int(len(salesing) * 0.9)
        train, test = salesing[0:train_size], salesing[train_size:]
        
        try:
            model = SARIMAX(train, order=(1, 1, 1), seasonal_order=(1, 1, 1, 7),
                            enforce_stationarity=False, enforce_invertibility=False)
            results = model.fit(disp=False)
            pred = results.get_forecast(steps=len(test)).predicted_mean
            
            r2 = r2_score(test, pred)
            st.metric("Model Accuracy (R2 Score)", f"{r2:.4f}")
            
            fig2, ax2 = plt.subplots(figsize=(10, 4))
            ax2.plot(test.index, test, label='Actual')
            ax2.plot(test.index, pred, label='Forecast', color='red')
            ax2.legend()
            st.pyplot(fig2)
            
        except Exception as e:
            st.warning(f"Could not fit model on this data: {e}")

else:
    st.info("Please upload a CSV file to proceed.")
