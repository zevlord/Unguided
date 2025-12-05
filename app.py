import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from prediction import predict_sales, get_unique_values, load_artifacts

# Page Config
st.set_page_config(page_title="Inventory Analytics App", layout="wide")

st.title('Inventory Analysis & Forecasting')
st.markdown("""
This application uses a **Random Forest Regressor** to predict sales revenue based on store inventory data.
It follows the CRISP-DM methodology for data mining.
""")

# Load Data for Dropdowns
stores, brands, vendors = get_unique_values()

if len(stores) == 0:
    st.error("Model artifacts not found! Please run `model.py` first to train the model and generate .sav files.")
else:
    # Tabs for different functionalities
    tab1, tab2 = st.tabs(["🔮 Sales Prediction", "mbatch Data Dashboard"])

    # --- TAB 1: PREDICTION ---
    with tab1:
        st.header("Predict Product Sales")
        st.markdown("Adjust parameters to forecast the `SalesDollars` for a specific product configuration.")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.subheader("Store Details")
            # Limit list length for performance if needed, or use full list
            selected_store = st.selectbox('Select Store ID', stores)
            selected_vendor = st.selectbox('Select Vendor', vendors)

        with col2:
            st.subheader("Product Details")
            # Using a simplified list or search is better for huge datasets
            # For this demo, we assume the user picks from the known brands
            selected_brand = st.selectbox('Select Brand', brands)
        
        with col3:
            st.subheader("Pricing Economics")
            price = st.number_input('Unit Sales Price ($)', min_value=0.0, value=15.0, step=0.5)
            tax = st.number_input('Excise Tax ($)', min_value=0.0, value=0.5, step=0.1)

        st.markdown("---")
        
        # Prediction Button
        if st.button("Predict Sales Revenue", type="primary"):
            with st.spinner('Calculating Forecast...'):
                result = predict_sales(selected_store, selected_brand, selected_vendor, price, tax)
                
                if isinstance(result, str): # Error message
                    st.error(result)
                else:
                    st.success(f"💰 Predicted Sales Revenue: ${result:,.2f}")
                    
                    # Context metric
                    st.info(f"At a price of ${price}, this implies selling approx. {int(result/price)} units.")

    # --- TAB 2: VISUALIZATIONS ---
    with tab2:
        st.header("Inventory Operational Dashboard")
        st.markdown("Key metrics from the historical 2016 dataset.")
        
        # We need to load a bit of data for the visuals
        # In a real app, this would be pre-calculated, but we'll do a quick load here
        @st.cache_data
        def load_viz_data():
            try:
                # Reuse the robust loader logic or just load a summary if available
                # For the demo, we try to load the CSV directly assuming it exists
                # We replicate the cleaning logic briefly
                with open('SalesFINAL12312016.csv', 'r') as f:
                    lines = f.readlines()
                cleaned_lines = []
                for line in lines:
                    line = line.strip()
                    if line.startswith('"') and line.endswith('"'):
                        line = line[1:-1].replace('""', '"')
                    cleaned_lines.append(line)
                from io import StringIO
                df = pd.read_csv(StringIO("\n".join(cleaned_lines)))
                df['SalesDollars'] = pd.to_numeric(df['SalesDollars'], errors='coerce')
                return df.sample(10000) # Sample for speed in visualization
            except:
                return pd.DataFrame()

        df_viz = load_viz_data()

        if not df_viz.empty:
            col_v1, col_v2 = st.columns(2)
            
            with col_v1:
                st.subheader("Top 10 Stores by Revenue")
                top_stores = df_viz.groupby('Store')['SalesDollars'].sum().sort_values(ascending=False).head(10)
                fig1, ax1 = plt.subplots()
                sns.barplot(x=top_stores.values, y=top_stores.index.astype(str), ax=ax1, palette='viridis')
                ax1.set_xlabel("Revenue ($)")
                st.pyplot(fig1)

            with col_v2:
                st.subheader("Sales Price Distribution")
                fig2, ax2 = plt.subplots()
                sns.histplot(df_viz['SalesPrice'], kde=True, ax=ax2, color='orange')
                ax2.set_xlim(0, 100)
                st.pyplot(fig2)
                
            st.subheader("Correlation Matrix")
            corr_cols = ['SalesQuantity', 'SalesDollars', 'SalesPrice', 'ExciseTax']
            fig3, ax3 = plt.subplots(figsize=(10, 4))
            sns.heatmap(df_viz[corr_cols].corr(), annot=True, cmap='coolwarm', ax=ax3)
            st.pyplot(fig3)
            
        else:
            st.warning("Could not load dataset for visualizations. Please ensure 'SalesFINAL12312016.csv' is in the directory.")