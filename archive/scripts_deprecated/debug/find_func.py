
filename = r'src/gui/app.py'
with open(filename, 'r') as f:
    for i, line in enumerate(f, 1):
        if 'def load_daily_tips' in line:
            print(f"{i}: {line.strip()}")
