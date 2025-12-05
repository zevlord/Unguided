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
a random forest regressor model to predict sales revenue based on store inventory data.
""")

# Load Data for Dropdowns
stores, brands, vendors = get_unique_values()

if len(stores) == 0:
    st.error("Model artifacts not found! Please run `model.py` first to train the model and generate .sav files.")
else:
    # Tabs for different functionalities
    tab1, = st.tabs(["Sales Prediction"])

    with tab1:
        st.header("Predict Product Sales")
        st.markdown("insert value")

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
                    st.success(f"Predicted Sales Revenue: ${result:,.2f}")
                    st.info(f"At a price of ${price}, this implies selling approximately {int(result/price)} units.")


