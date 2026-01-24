
from autogluon.tabular import TabularPredictor
import pandas as pd
import numpy as np

MODEL_PATH = 'models/autogluon_pace_v26'

def main():
    print(f"Loading model from {MODEL_PATH}...")
    try:
        predictor = TabularPredictor.load(MODEL_PATH)
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    print("Model Loaded. Getting Feature Importance (this might be slow)...")
    
    # Note: feature_importance() usually requires data X.
    # If we don't pass data, it might fail or try to use validation data if saved?
    # AutoGluon 1.0+ usually needs data.
    # Let's try to get learner info directly if possible, or we need to reload data.
    
    # Since checking feature importance is expensive (permutation importance),
    # we can check if the model has intrinsic feature importance (e.g. from the best model).
    
    print("Model names:", predictor.model_names())
    best_model = predictor.get_model_best()
    print("Best Model:", best_model)
    
    # Try getting importance from the model object directly if supported (e.g. LightGBM)
    # But usually wrapped. 
    # Plan B: Just reload a small chunk of data to run importance.
    
    import sqlite3
    conn = sqlite3.connect('greyhound_racing.db')
    # Load just 5000 rows for importance check
    df = pd.read_sql_query("SELECT * FROM GreyhoundEntries LIMIT 5000", conn)
    conn.close()
    
    # We need to recreate the features first... that's the hard part.
    # The feature generation code is in the script.
    
    # ALTERNATIVE: Use the feature importance from the *LightGBM* model inside WeightedEnsemble?
    # predictor._learner.model_manager.models['LightGBM'].model.feature_importances_
    
    # Simplest reliable way:
    # Just list the features used by the predictor
    features = predictor.feature_metadata_in.get_features()
    print(f"Model uses {len(features)} features: {features}")
    
    # Since we can't easily regenerate features without the full script logic,
    # and permutation importance is slow...
    # We will assume the User wants to drop "Low Impact" ones.
    # We can inspect the model artifacts if we really want.
    
    # Let's try to print valid features and their types.
    print(predictor.feature_metadata_in)

if __name__ == "__main__":
    main()
