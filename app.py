import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import r2_score, mean_squared_error
from statsmodels.tsa.statespace.sarimax import SARIMAX
import warnings

warnings.filterwarnings("ignore")

# Set page layout
st.set_page_config(page_title="Liquor Sales Forecasting", layout="wide")

st.title("🍷 Liquor Sales Analysis & Forecasting")

# --- Function: Load and Clean Data with Dynamic Columns ---
@st.cache_data
def load_and_clean_data(uploaded_file, date_col, size_col, value_col):
    """
    Loads data using user-defined column names.
    """
    chunk_size = 500000
    chunks = []
    
    # 1. Read Data
    # Reset file pointer to beginning since we read headers earlier
    uploaded_file.seek(0)
    for chunk in pd.read_csv(uploaded_file, chunksize=chunk_size):
        chunks.append(chunk)
    sales = pd.concat(chunks)
    
    # Clean header whitespace
    sales.columns = sales.columns.str.strip()

    # 2. Date Conversion (Using selected column)
    try:
        sales[date_col] = pd.to_datetime(sales[date_col])
    except Exception as e:
        st.error(f"Error converting '{date_col}' to datetime: {e}")
        st.stop()

    # 3. Clean Size Column (Using selected column)
    if size_col in sales.columns:
        # Convert to string, lowercase, remove quotes
        cleansize = sales[size_col].astype(str).str.lower().str.replace('"', '').str.strip()

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
uploaded_file = st.sidebar.file_uploader("Upload Sales CSV", type=['csv'])

if uploaded_file is not None:
    # --- DYNAMIC COLUMN MAPPING ---
    # Read just the header to populate dropdowns
    header_df = pd.read_csv(uploaded_file, nrows=0)
    header_df.columns = header_df.columns.str.strip() # Strip whitespace immediately
    all_cols = list(header_df.columns)

    st.sidebar.header("2. Map Columns")
    st.sidebar.info("Select the correct columns from your file.")

    # Try to auto-select based on common names, otherwise default to index 0
    def get_index(options, search_strings):
        for s in search_strings:
            for i, opt in enumerate(options):
                if s.lower() in opt.lower():
                    return i
        return 0

    # Date Column Selector
    date_col = st.sidebar.selectbox(
        "Date Column", 
        all_cols, 
        index=get_index(all_cols, ['date', 'time'])
    )

    # Sales Value Selector
    val_col = st.sidebar.selectbox(
        "Sales Value ($) Column", 
        all_cols, 
        index=get_index(all_cols, ['dollar', 'amount', 'price', 'total'])
    )

    # Size Selector
    size_col = st.sidebar.selectbox(
        "Size/Description Column", 
        all_cols, 
        index=get_index(all_cols, ['size', 'desc', 'volume'])
    )

    # --- Process Data ---
    with st.spinner('Processing Data...'):
        sales_df = load_and_clean_data(uploaded_file, date_col, size_col, val_col)
    
    st.success(f"Loaded {len(sales_df):,} rows successfully!")

    # --- Create Time Series ---
    # Group by the selected Date column and Sum/Mean the selected Value column
    salesing = sales_df.groupby(date_col)[val_col].mean()
    salesing = salesing.asfreq('D').fillna(method='ffill')

    # --- Visuals & Modeling ---
    tab1, tab2, tab3 = st.tabs(["Data Preview", "Model Validation", "Forecast"])

    with tab1:
        st.subheader("Raw Data Preview")
        st.dataframe(sales_df.head())
        
        st.subheader("Time Series Visualization")
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(salesing.index, salesing.values)
        ax.set_title(f"Daily Average of {val_col}")
        st.pyplot(fig)

    with tab2:
        st.subheader("SARIMAX Model Training")
        
        c1, c2 = st.columns(2)
        with c1:
            p = st.number_input("p", 0, 5, 1)
            d = st.number_input("d", 0, 2, 1)
            q = st.number_input("q", 0, 5, 1)
        with c2:
            s = st.number_input("Seasonality (Days)", 1, 365, 7)

        if st.button("Train Model"):
            # Simple Train/Test Split
            train_size = int(len(salesing) * 0.8)
            train, test = salesing[0:train_size], salesing[train_size:]
            
            with st.spinner("Training..."):
                try:
                    model = SARIMAX(train, order=(p, d, q), seasonal_order=(1, 1, 1, s),
                                    enforce_stationarity=False, enforce_invertibility=False)
                    results = model.fit(disp=False)
                    
                    forecast = results.get_forecast(steps=len(test))
                    pred = forecast.predicted_mean
                    pred.index = test.index # Align index for plotting

                    r2 = r2_score(test, pred)
                    
                    fig2, ax2 = plt.subplots(figsize=(12, 6))
                    ax2.plot(train.index, train, label='Train')
                    ax2.plot(test.index, test, label='Test')
                    ax2.plot(pred.index, pred, label='Forecast', color='red')
                    ax2.set_title(f"SARIMAX Forecast (R2: {r2:.3f})")
                    ax2.legend()
                    st.pyplot(fig2)
                    
                except Exception as e:
                    st.error(f"Modeling Error: {e}")

    with tab3:
        st.write("Predict Future (Full Dataset)")
        days_future = st.slider("Days into future", 7, 90, 30)
        
        if st.button("Forecast Future"):
            try:
                full_model = SARIMAX(salesing, order=(p, d, q), seasonal_order=(1, 1, 1, s),
                                     enforce_stationarity=False, enforce_invertibility=False)
                res_full = full_model.fit(disp=False)
                
                future = res_full.get_forecast(steps=days_future)
                fut_pred = future.predicted_mean
                
                fig3, ax3 = plt.subplots(figsize=(12, 6))
                ax3.plot(salesing.index, salesing, label='Historical')
                ax3.plot(fut_pred.index, fut_pred, label='Future', color='green')
                ax3.legend()
                st.pyplot(fig3)
                
                st.dataframe(fut_pred.head())
            except Exception as e:
                st.error(f"Error: {e}")

else:
    st.info("Awaiting CSV upload...")
