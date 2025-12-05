import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import joblib
import io

def robust_load(filename):
    """
    Parses CSV files where entire rows are quoted (malformed), 
    which standard pd.read_csv fails on.
    """
    try:
        # Try normal load
        df = pd.read_csv(filename)
        if len(df.columns) == 1 and ',' in df.columns[0]:
            raise ValueError("Malformed CSV")
        return df
    except:
        print(f"Detected malformed CSV for {filename}. Parsing manually...")
        with open(filename, 'r') as f:
            lines = f.readlines()
        
        cleaned_lines = []
        for line in lines:
            line = line.strip()
            # Remove outer quotes if present
            if line.startswith('"') and line.endswith('"'):
                line = line[1:-1]
                line = line.replace('""', '"')
            cleaned_lines.append(line)
        
        from io import StringIO
        data_str = "\n".join(cleaned_lines)
        return pd.read_csv(StringIO(data_str))

def train_model():
    print("Loading data...")
    # Load Sales data
    # Note: Using a sample for speed in this demo, remove .sample() for full training
    df = robust_load('SalesFINAL12312016.csv')
    
    # Preprocessing
    print("Preprocessing data...")
    df['SalesPrice'] = pd.to_numeric(df['SalesPrice'], errors='coerce')
    df['ExciseTax'] = pd.to_numeric(df['ExciseTax'], errors='coerce')
    df['SalesDollars'] = pd.to_numeric(df['SalesDollars'], errors='coerce')
    df.dropna(subset=['SalesPrice', 'ExciseTax', 'SalesDollars', 'Brand', 'Store', 'VendorName'], inplace=True)
    
    # We will use a subset for training to ensure the file sizes stay manageable for the demo
    # In production, use the full dataset
    df_sample = df.sample(n=50000, random_state=42)

    # Encoders
    # We need to save these to handle user input in the app
    le_brand = LabelEncoder()
    le_vendor = LabelEncoder()
    le_store = LabelEncoder()

    df_sample['Brand_Enc'] = le_brand.fit_transform(df_sample['Brand'].astype(str))
    df_sample['Vendor_Enc'] = le_vendor.fit_transform(df_sample['VendorName'].astype(str))
    df_sample['Store_Enc'] = le_store.fit_transform(df_sample['Store'].astype(str))

    # Features and Target
    X = df_sample[['Store_Enc', 'Brand_Enc', 'Vendor_Enc', 'SalesPrice', 'ExciseTax']]
    y = df_sample['SalesDollars']

    # Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Model: Random Forest Regressor (Best performing model from analysis)
    print("Training Random Forest Model...")
    rf = RandomForestRegressor(n_estimators=100, max_depth=15, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)

    print(f"Model Training Complete. Score: {rf.score(X_test, y_test):.4f}")

    # Save Model and Encoders
    print("Saving artifacts...")
    joblib.dump(rf, "rf_inventory_model.sav")
    
    # Save encoders in a dictionary for easy loading
    encoders = {
        'brand': le_brand,
        'vendor': le_vendor,
        'store': le_store
    }
    joblib.dump(encoders, "feature_encoders.sav")
    print("Done! Run app.py now.")

if __name__ == "__main__":
    train_model()