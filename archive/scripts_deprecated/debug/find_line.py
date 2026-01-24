
filename = r'src/models/ml_model.py'
with open(filename, 'r') as f:
    for i, line in enumerate(f, 1):
        if 'def _extract_greyhound_features(self' in line:
            print(f"{i}: {line.strip()}")
