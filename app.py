import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import r2_score
from statsmodels.tsa.statespace.sarimax import SARIMAX
import io

# --- Page Config ---
st.set_page_config(page_title="Inventory & Sales Forecasting", layout="wide")

st.title("📊 Inventory Analysis & SARIMA Forecasting")
st.markdown("""
This application loads sales and inventory data, performs data cleaning, 
and forecasts daily sales using a SARIMA model.
""")

# --- Data Loading & Cleaning ---
@st.cache_data
def load_data():
    # Helper function to clean the malformed CSV rows (double quotes issue)
    def clean_csv_content(filename):
        try:
            with open(filename, 'r') as f:
                lines = f.readlines()
            
            cleaned_lines = []
            for line in lines:
                line = line.strip()
                # Remove outer quotes and fix escaped double quotes
                if line.startswith('"') and line.endswith('"'):
                    line = line[1:-1].replace('""', '"')
                cleaned_lines.append(line)
            
            return pd.read_csv(io.StringIO("\n".join(cleaned_lines)))
        except FileNotFoundError:
            return None

    # Load datasets
    # Note: Using the clean function for Sales and Purchases as per previous context
    sales_df = clean_csv_content('SalesFINAL12312016.csv')
    
    # If cleaned loader fails (e.g. file not found), try standard load or return None
    if sales_df is None: 
        st.error("File 'SalesFINAL12312016.csv' not found. Please make sure it is in the same directory.")
        return None, None, None

    # Purchases might also need cleaning
    purchases_df = clean_csv_content('PurchasesFINAL12312016.csv')
    
    # EndInv usually loads fine
    try:
        end_inv_df = pd.read_csv('EndInvFINAL12312016.csv')
    except FileNotFoundError:
        end_inv_df = None

    return sales_df, purchases_df, end_inv_df

with st.spinner('Loading and cleaning data... (This may take a moment)'):
    sales, purchases, end_inv = load_data()

if sales is not None:
    # --- Data Preprocessing ---
    st.header("1. Data Cleaning & Preparation")
    
    # Date conversion
    sales["SalesDate"] = pd.to_datetime(sales["SalesDate"])
    
    # Drop Classification if exists
    if "Classification" in sales.columns:
        sales = sales.drop("Classification", axis=1)

    # Clean Size
    sales['Size'] = sales['Size'].astype(str)
    cleansize = sales['Size'].str.lower().str.replace('"', '').str.strip()
    
    # Standardize Size
    stdSize = cleansize.str.extract(r'(\d+\.?\d*)')[0].astype(float).fillna(0)
    stdPack = cleansize.str.extract(r'(\d+)\s*pk')[0].astype(float).fillna(1)
    
    multiplier = np.where(cleansize.str.contains('ml'), 1, 
                 np.where(cleansize.str.contains('l') & ~cleansize.str.contains('ml'), 1000,
                 np.where(cleansize.str.contains('oz'), 30, 0)))

    sales['True Size/ml'] = stdSize * multiplier * stdPack
    
    # Filter valid sizes
    sales = sales[sales['True Size/ml'] > 0]
    
    # Display Data info
    col1, col2 = st.columns(2)
    with col1:
        st.write("**Sales Data Preview:**")
        st.dataframe(sales.head())
    with col2:
        st.write("**Missing Values:**")
        st.write(sales.isnull().sum())

    # Label Encoding for categorical features
    st.subheader("Feature Encoding")
    le = LabelEncoder()
    cols_to_encode = ['Store', 'Brand', 'VendorNo', 'Size']
    encoded_data = sales.copy()
    for col in cols_to_encode:
        if col in encoded_data.columns:
            encoded_data[f'{col}_Enc'] = le.fit_transform(encoded_data[col].astype(str))
    
    st.write("Data encoded successfully for Machine Learning.")

    # --- SARIMAX Modeling ---
    st.header("2. SARIMAX Time Series Forecasting")
    
    # Aggregating daily sales
    # Use the cleaned sales data (fixing the 'sales_clean' NameError from notebook)
    daily_sales = sales.groupby('SalesDate')['SalesDollars'].mean()
    daily_sales = daily_sales.asfreq('D')
    
    # Fill missing dates with 0 or interpolate if necessary (SARIMA hates NaNs)
    daily_sales = daily_sales.fillna(0)

    st.write("**Daily Average Sales Trend:**")
    st.line_chart(daily_sales)

    # Model Configuration
    col_a, col_b = st.columns(2)
    with col_a:
        train_size = st.slider("Training Data Split Date (Month)", 1, 11, 9)
    
    # Define Train/Test Split based on notebook logic (Jan-Sep train, Oct-Nov test)
    # Hardcoding dates based on user notebook, but can be made dynamic
    train_end_str = f'2016-0{train_size}-30'
    test_start_str = f'2016-{train_size+1:02d}-01'
    
    Sari_train = daily_sales.loc[:train_end_str]
    Sari_test = daily_sales.loc[test_start_str:'2016-11-30']

    if st.button("Train SARIMAX Model"):
        with st.spinner("Training Model..."):
            # SARIMAX Parameters (1,1,1) x (1,1,1,7)
            model_sari = SARIMAX(Sari_train, 
                                order=(1, 1, 1), 
                                seasonal_order=(1, 1, 1, 7), 
                                enforce_stationarity=True, 
                                enforce_invertibility=True)
            results_sari = model_sari.fit()
            
            # Forecast
            forecast_steps = len(Sari_test)
            if forecast_steps > 0:
                forecast_sari = results_sari.get_forecast(steps=forecast_steps)
                pred_sari = forecast_sari.predicted_mean
                conf_int = forecast_sari.conf_int()

                # Visualization
                fig, ax = plt.subplots(figsize=(12, 6))
                ax.plot(Sari_train.index, Sari_train, label='Training Data')
                ax.plot(Sari_test.index, Sari_test, label='Actual Sales (Test)')
                ax.plot(pred_sari.index, pred_sari, label='SARIMA Forecast', color='red', linestyle='--')
                ax.fill_between(pred_sari.index, conf_int.iloc[:, 0], conf_int.iloc[:, 1], color='pink', alpha=0.3)
                
                ax.set_title("Daily Sales Forecast vs Actual")
                ax.legend()
                st.pyplot(fig)

                # Metrics
                # Align data to avoid index mismatch issues
                df_r2 = pd.DataFrame({'actual': Sari_test, 'predicted': pred_sari}).dropna()
                if not df_r2.empty:
                    score = r2_score(df_r2['actual'], df_r2['predicted'])
                    st.success(f"Model R² Score: {score:.4f}")
                else:
                    st.warning("Not enough data points to calculate R² score.")
            else:
                st.error("Test set is empty. Please adjust the split date.")

    # --- Future Forecasting ---
    st.header("3. Future Forecasting (Full 2017)")
    
    if st.button("Forecast 2017"):
        # Train on full available data
        full_model = SARIMAX(daily_sales, 
                            order=(1, 1, 1), 
                            seasonal_order=(1, 1, 1, 7), 
                            enforce_stationarity=True, 
                            enforce_invertibility=True)
        full_results = full_model.fit()
        
        # Forecast for 2017
        pred_2017 = full_results.get_prediction(start='2016-03-30', end='2017-10-30')
        pred_mean = pred_2017.predicted_mean

        fig2, ax2 = plt.subplots(figsize=(12, 6))
        ax2.plot(daily_sales.index, daily_sales, label='Historical Data (2016)')
        ax2.plot(pred_mean.index, pred_mean, label='2017 Forecast', color='green')
        ax2.set_title("2017 Sales Forecast")
        ax2.legend()
        st.pyplot(fig2)
        
        st.info("Note: Prediction is based on patterns in 2016 data. Seasonal patterns are clearly visible.")

else:
    st.warning("Please ensure the dataset files are in the directory.")
