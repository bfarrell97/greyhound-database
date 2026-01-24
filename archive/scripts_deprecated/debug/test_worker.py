
import sys
import os

# Ensure src in path
sys.path.append(os.getcwd())

from scripts.import_bsp import process_file

file_path = r'data/bsp/BASIC/2025/Apr/1/34175726/1.241667158.bz2'

print(f"Testing process_file on {file_path}")
try:
    result = process_file(file_path)
    print(f"Result: {result}")
    
    if result:
        print(f"Extracted {len(result)} runners.")
    else:
        print("Result is None.")
except Exception as e:
    print(f"Crash: {e}")
