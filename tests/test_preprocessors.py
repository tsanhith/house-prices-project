def test_preprocessors_present():
    import joblib
    m = joblib.load('models/baseline.pkl')
    assert m.get('imputer') is not None, 'imputer missing in saved model'
    assert m.get('scaler') is not None, 'scaler missing in saved model'
