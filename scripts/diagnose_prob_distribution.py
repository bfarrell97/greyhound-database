import sys, os
sys.path.append(os.getcwd())
from scripts.predict_v44_prod import MarketAlphaEngine
import pandas as pd

engine = MarketAlphaEngine()
df_hist = engine._history_cache.copy()
print(f"History cache rows: {len(df_hist)}")

# Use the pre-computed history cache to compute Steam_Prob distribution
# (avoid re-running feature_engineer on a random sample which can miss group context)
df_sample = df_hist.sort_values('MeetingDate').tail(5000).copy()
X_v44 = df_sample[engine.v44_features]
df_sample['Steam_Prob'] = engine.model_v44.predict_proba(X_v44)[:, 1]
probs = df_sample['Steam_Prob']

# Attach some meta so top rows are informative
df_pred = df_sample.copy()
print('Steam_Prob stats:')
print(f"  min: {probs.min():.4f}")
print(f"  10%: {probs.quantile(0.1):.4f}")
print(f"  25%: {probs.quantile(0.25):.4f}")
print(f"  50%: {probs.quantile(0.5):.4f}")
print(f"  75%: {probs.quantile(0.75):.4f}")
print(f"  90%: {probs.quantile(0.9):.4f}")
print(f"  max: {probs.max():.4f}")
print(f"  mean: {probs.mean():.4f}")

for thr in [0.35, 0.45, 0.6, 0.8]:
    cnt = (probs >= thr).sum()
    print(f"  >={thr:.2f}: {cnt} ({cnt/len(probs):.2%})")

# Top examples
print('\nTop 10 by Steam_Prob:')
print(df_pred.sort_values('Steam_Prob', ascending=False).head(10)[['MeetingDate','TrackName','Dog','Price5Min','Steam_Prob']].to_string(index=False))
