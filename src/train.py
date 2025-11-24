import os, json, joblib, pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from src.preprocessing import build_pipeline  # works because package __init__.py exists

def train_and_save(df_path='data/train.csv', model_path='models/baseline.pkl'):
    df = pd.read_csv(df_path)
    X, y, preprocessors = build_pipeline(df)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    model = Ridge(alpha=1.0)
    model.fit(X_train, y_train)
    preds = model.predict(X_val)
    mse = mean_squared_error(y_val, preds)
    rmse = float(mse ** 0.5)
    r2 = r2_score(y_val, preds)


    os.makedirs('models', exist_ok=True)
    # save model + preprocessors together
    joblib.dump({'model': model, **preprocessors}, model_path)

    metrics = {'rmse': float(rmse), 'r2': float(r2)}
    os.makedirs('reports', exist_ok=True)
    with open('reports/metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)

    print('Saved model and metrics:', metrics)
    return metrics

if __name__ == '__main__':
    train_and_save()
