import pandas as pd
data_path=r"D:\MLOPS\DSA\PROJECT_HEART\heart_disease\heart_cleaned1.csv"
def load_data(path):
    return pd.read_csv(path)
def test_schema():
    df=load_data(data_path)
    expected_columns=14
    assert df.shape[1]==expected_columns
