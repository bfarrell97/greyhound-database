
import sys
import os
import pandas as pd
from datetime import datetime

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.models.ml_model import GreyhoundMLModel

def force_retrain():
    print("Force Retraining Model...")
    model = GreyhoundMLModel()
    
    # 1. Extract
    print("Extracting data...")
    df = model.extract_training_data(start_date='2024-01-01', end_date='2025-12-31')
    if df is None or len(df) == 0:
        print("No training data found!")
        return
        
    # 2. Prepare
    X, y, df_clean = model.prepare_features(df)
    
    # Split
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    
    # 3. Train
    model.train_model(X_train, y_train, X_test, y_test)
    
    # 4. Calibrate
    model.calibrate_predictions(X_test, y_test)
    
    # 5. Save
    save_path = os.path.join(os.path.dirname(__file__), '..', 'src', 'models', 'greyhound_model.pkl')
    # Ensure dir exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    model.save_model(save_path)
    print(f"Model saved to {save_path}")

if __name__ == "__main__":
    force_retrain()
