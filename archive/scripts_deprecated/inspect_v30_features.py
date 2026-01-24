# inspect_v30_features.py
import sys
from autogluon.tabular import TabularPredictor

model_path = 'models/autogluon_v30_bsp'
try:
    predictor = TabularPredictor.load(model_path)
    features = predictor.feature_metadata.get_features()
    print('V30 model features count:', len(features))
    for f in features:
        print(f)
except Exception as e:
    print('Error loading V30 model:', e)
    sys.exit(1)
