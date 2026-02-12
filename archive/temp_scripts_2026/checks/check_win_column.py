import pandas as pd
import sqlite3
import sys
import os

sys.path.append(os.getcwd())
try:
    from src.features.feature_engineering_v41 import FeatureEngineerV41
except ImportError:
    sys.path.append(os.path.join(os.getcwd(), '..'))
    from src.features.feature_engineering_v41 import FeatureEngineerV41

def check_win():
    conn = sqlite3.connect('greyhound_racing.db')
    query = "SELECT Position FROM GreyhoundEntries WHERE Price5Min IS NOT NULL"
    df = pd.read_sql_query(query, conn)
    conn.close()
    
    print(f"Total Rows: {len(df)}")
    print(f"Position Value Counts:\n{df['Position'].value_counts().head()}")
    
    fe = FeatureEngineerV41()
    # Mock DF with just Position
    df_mock = pd.DataFrame({'Position': [1, '1', 2, '2', None, 5]})
    df_mock = fe.calculate_features(df_mock)
    print(f"Mock Win Column: {df_mock['win'].tolist()}")

if __name__ == "__main__":
    check_win()
