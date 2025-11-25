from src.preprocessing import build_pipeline
import pandas as pd

def test_build_pipeline_runs():
    df = pd.DataFrame({'SalePrice':[100,200], 'A':[1,2], 'B':[3,4]})
    X, y, p = build_pipeline(df)
    assert X.shape[0] == 2
    assert 'imputer' in p and 'scaler' in p
