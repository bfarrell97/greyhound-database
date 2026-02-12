import joblib
import pandas as pd
import numpy as np
import xgboost as xgb
import sys
import os

sys.path.append(os.getcwd())
from src.features.feature_engineering_v41 import FeatureEngineerV41

def debug_predict():
    print("Loading V41 Prod Model...")
    try:
        model = joblib.load('models/xgb_v41_final_prod.pkl')
        print(f"Model Type: {type(model)}")
        
        # Create Dummy Data from FE
        fe = FeatureEngineerV41()
        feats = fe.get_feature_list()
        print(f"Expected Features: {len(feats)}")
        
        dummy_df = pd.DataFrame(np.random.rand(5, len(feats)), columns=feats)
        
        # Test 1: DataFrame + predict_proba
        print("\nTest 1: predict_proba(DataFrame)")
        try:
            probs = model.predict_proba(dummy_df)
            print("Success!")
            print(f"Shape: {probs.shape}")
            print(f"Values: {probs[:, 1]}")
        except Exception as e:
            print(f"Failed: {e}")
            
        # Test 2: DMatrix + predict
        print("\nTest 2: predict(DMatrix)")
        try:
            dtest = xgb.DMatrix(dummy_df)
            preds = model.predict(dtest)
            print("Success!")
            print(f"Shape: {preds.shape}")
            print(f"Values: {preds}")
            
            if np.all(np.isin(preds, [0, 1])):
                print(">> RESULT: Output is CLASS LABELS (0/1)")
            else:
                print(">> RESULT: Output is PROBABILITIES")
                
        except Exception as e:
            print(f"Failed: {e}")
            
    except Exception as e:
        print(f"Critical Error: {e}")

if __name__ == "__main__":
    debug_predict()
