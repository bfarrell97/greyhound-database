
filename = r'src/models/ml_model.py'
with open(filename, 'r') as f:
    for i, line in enumerate(f, 1):
        if 'def predict_race_winners_v2' in line:
            print(f"{i}: {line.strip()}")
