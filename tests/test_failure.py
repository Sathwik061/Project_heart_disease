import pytest
import pandas as pd
data_path=r"D:\MLOPS\DSA\PROJECT_HEART\heart_disease\heart_cleaned1.csv"
def load_data(path):
    return pd.read_csv(path)

def test_failure_handling():
    with pytest.raises(Exception):
        load_data("D:\MLOPS\DSA\PROJECT_HEART\heart_disease\heart-disease.names")