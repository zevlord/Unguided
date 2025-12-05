import joblib
import numpy as np
import pandas as pd

# loading the artifacts
def load_artifacts():
    try:
        model = joblib.load("rf_inventory_model.sav")
        encoders = joblib.load("feature_encoders.sav")
        return model, encoders
    except FileNotFoundError:
        return None, None

#to predict sales based on user input
def predict_sales(store_id, brand_name, vendor_name, price, tax):
    #predict according to user input
    model, encoders = load_artifacts()
    
    if not model:
        return "Error: Model not found. Please run model.py first."

    try:
        # Encode inputs using the saved encoders
        # Handle cases where input might not be in the training set (fallback to 0 or similar)
        
        try:
            store_enc = encoders['store'].transform([str(store_id)])[0]
        except ValueError:
            # Fallback if store not found (use most common or 0)
            store_enc = 0 
            
        try:
            brand_enc = encoders['brand'].transform([str(brand_name)])[0]
        except ValueError:
            brand_enc = 0
            
        try:
            vendor_enc = encoders['vendor'].transform([str(vendor_name)])[0]
        except ValueError:
            vendor_enc = 0

        # Create feature array
        features = np.array([[store_enc, brand_enc, vendor_enc, price, tax]])
        
        # Predict
        prediction = model.predict(features)
        return prediction[0]
        
    except Exception as e:
        return f"Error during prediction: {str(e)}"

def get_unique_values():
    # Get unique stores, brands, and vendors from encoders
    _, encoders = load_artifacts()
    if not encoders:
        return [], [], []
    
    stores = encoders['store'].classes_
    brands = encoders['brand'].classes_
    vendors = encoders['vendor'].classes_
    
    return stores, brands, vendors