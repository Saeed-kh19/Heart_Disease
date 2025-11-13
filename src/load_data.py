import os
import pandas as pd

def load_data(csv_path):
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"CSV not found at: {csv_path}")
    
    df = pd.read_csv(csv_path)
    
    df_raw = df.copy()
    
    return df_raw