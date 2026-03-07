
import pandas as pd
import numpy as np

def load_dataset(path):
    data = pd.read_csv(path)
    data.columns = data.columns.str.strip().str.replace('Â','')
    data.replace([np.inf,-np.inf],np.nan,inplace=True)
    data.fillna(data.median(numeric_only=True),inplace=True)
    return data
