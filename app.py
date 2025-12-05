import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import r2_score, mean_squared_error
from statsmodels.tsa.statespace.sarimax import SARIMAX
import re

# Set page layout
st.set_page_config(page_title="Liquor Sales Forecasting", layout="wide")

st.title("🍷 Liquor Sales Analysis & SARIMAX Forecasting")
st.markdown("""
This application transforms the raw liquor sales data into a time-series format and uses 
Seasonal ARIMA (SARIMAX) to forecast daily sales averages.
""")

# --- 1. Data Loading & Cleaning ---

@st.cache_data
def load_and_clean_data(uploaded_file):
    """
    Loads data and performs the specific cleaning steps defined in the notebook.
    """
    # Load Data (Notebook Cells 1-2)
    # Using iterator for large files if necessary, but concatenating immediately for processing
    chunk_size = 500000
    chunks = []
    
    # Read CSV
    # Using pandas read_csv. If file is massive, this might take a moment.
    # In a production app, we might limit rows, but here we process all to match the notebook.
    for chunk in pd.read_csv(uploaded_file, chunksize=chunk_size):
        chunks.append(chunk)
    sales = pd.concat(chunks)

    # Date Conversion (Notebook Cell 3)
    sales["SalesDate"] = pd.to_datetime(sales["SalesDate"])

    # Drop Classification (Notebook Cell 4)
    if "Classification" in sales.columns:
        sales = sales.drop("Classification", axis=1)

    # Clean Size Column (Notebook Cell 5)
    # Lowercase and remove specific characters
    cleansize = sales['Size'].str.lower().str.replace('"', '').str.strip()

    # Feature Engineering: Extract Numeric Size (Notebook Cell 8)
    stdSize = cleansize.str.extract(r'(\d+\.?\d*)')[0].astype(float).fillna(0)

    # Feature Engineering: Extract Pack Size (Notebook Cell 9)
    stdPack = cleansize.str.extract(r'(\d+)\s*pk')[0].astype(float).fillna(1)

    # Feature Engineering: Calculate Multiplier (Notebook Cell 10)
    multiplier = np.where(cleansize.str.contains('ml'), 1, 
                 np.where(cleansize.str.contains('l') & ~cleansize.str.contains('ml'), 1000,
                 np.where(cleansize.str.contains('oz'), 30, 0)))

    # Calculate True Size (Notebook Cell 10)
    sales['True Size/ml'] = stdSize * multiplier * stdPack

    # Filter invalid sizes (Notebook Cell 11)
    sales = sales[sales['True Size/ml'] > 0]

    return sales

# Sidebar for File Upload
st.sidebar.header("Data Upload")
uploaded_file = st.sidebar.file_uploader("Upload 'SalesFINAL12312016.csv'", type=['csv'])

if uploaded_file is not None:
    with st.status("Processing Data...", expanded=True) as status:
        st.write("Reading CSV file...")
        sales_df = load_and_clean_data(uploaded_file)
        st.write("Data cleaning complete.")
        
        # Create Time Series (Notebook Cell 13 logic fix)
        # Aggregating SalesDollars by Date (mean)
        st.write("Aggregating time series...")
        salesing = sales_df.groupby('SalesDate')['SalesDollars'].mean()
        salesing = salesing.asfreq('D')
        
        # Fill missing dates (common in sales data) using forward fill to prevent model errors
        salesing = salesing.fillna(method='ffill')
        
        status.update(label="Data Ready!", state="complete", expanded=False)

    # Tabs for different sections of the analysis
    tab1, tab2, tab3 = st.tabs(["Data Preview", "Model Validation", "Future Forecast"])

    with tab1:
        st.subheader("Cleaned Data Snapshot")
        st.dataframe(sales_df.head(10))
        
        st.subheader("Daily Sales Time Series")
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(salesing.index, salesing.values, label='Avg Daily Sales')
        ax.set_title("Average Daily Sales ($)")
        ax.set_ylabel("Sales Dollars")
        st.pyplot(fig)

    with tab2:
        st.subheader("SARIMAX Model Validation")
        
        col1, col2 = st.columns([1, 3])
        
        with col1:
            st.markdown("### Model Config")
            # Parameters from Notebook (1, 1, 1) x (1, 1, 1, 7)
            p = st.number_input("p (AR)", 0, 5, 1)
            d = st.number_input("d (I)", 0, 2, 1)
            q = st.number_input("q (MA)", 0, 5, 1)
            
            st.markdown("### Seasonal Config")
            P = st.number_input("P (Seasonal AR)", 0, 5, 1)
            D = st.number_input("D (Seasonal I)", 0, 2, 1)
            Q = st.number_input("Q (Seasonal MA)", 0, 5, 1)
            s = st.number_input("s (Seasonality)", 1, 365, 7)
            
            split_date_input = st.date_input("Split Date", value=pd.to_datetime("2016-10-01"))

        with col2:
            if st.button("Train & Validate"):
                split_date = pd.to_datetime(split_date_input)
                
                # Split Data (Notebook Cell 44)
                train = salesing[:split_date]
                test = salesing[split_date:]
                
                with st.spinner("Fitting Model..."):
                    try:
                        # Model Definition
                        model_sari = SARIMAX(train, 
                                            order=(p, d, q), 
                                            seasonal_order=(P, D, Q, s), 
                                            enforce_stationarity=False, # Relaxed for app stability
                                            enforce_invertibility=False)
                        results_sari = model_sari.fit(disp=False)
                        
                        # Forecast
                        forecast_res = results_sari.get_forecast(steps=len(test))
                        pred_sari = forecast_res.predicted_mean
                        pred_sari.index = test.index # Align index

                        # Metrics
                        r2 = r2_score(test, pred_sari)
                        mse = mean_squared_error(test, pred_sari)

                        # Plotting (Notebook Cell 45)
                        fig2, ax2 = plt.subplots(figsize=(12, 6))
                        ax2.plot(train.index, train, label='History (Train)')
                        ax2.plot(test.index, test, label='Actual (Test)', alpha=0.7)
                        ax2.plot(pred_sari.index, pred_sari, label='Forecast', color='red')
                        ax2.set_title(f'Daily Sales Forecast (SARIMA) | R²: {r2:.4f}')
                        ax2.legend()
                        st.pyplot(fig2)
                        
                        st.success(f"Model R² Score: {r2:.4f}")
                        
                    except Exception as e:
                        st.error(f"An error occurred during modeling: {e}")

    with tab3:
        st.subheader("Future Forecasting")
        st.markdown("Train on the **entire** 2016 dataset and predict 2017.")
        
        forecast_days = st.slider("Days to Predict", 30, 365, 60)
        
        if st.button("Generate Future Forecast"):
            with st.spinner("Training on full dataset..."):
                try:
                    # Train on full dataset
                    full_model = SARIMAX(salesing, 
                                        order=(p, d, q), 
                                        seasonal_order=(P, D, Q, s), 
                                        enforce_stationarity=False, 
                                        enforce_invertibility=False)
                    full_results = full_model.fit(disp=False)
                    
                    # Predict Future
                    future_forecast = full_results.get_forecast(steps=forecast_days)
                    pred_future = future_forecast.predicted_mean
                    
                    # Plotting
                    fig3, ax3 = plt.subplots(figsize=(12, 6))
                    ax3.plot(salesing.index, salesing, label='Historical Data (2016)')
                    ax3.plot(pred_future.index, pred_future, label='Future Prediction', color='red')
                    ax3.set_title(f'Sales Forecast for Next {forecast_days} Days')
                    ax3.legend()
                    st.pyplot(fig3)
                    
                    # Show raw data
                    st.write("Forecast Values:")
                    st.dataframe(pred_future.head())
                    
                except Exception as e:
                    st.error(f"An error occurred: {e}")

else:
    st.info("Please upload the 'SalesFINAL12312016.csv' file from the sidebar to begin.")