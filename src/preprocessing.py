import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
import joblib

def build_pipeline(df, target='SalePrice'):
    X = df.drop(columns=[target])
    y = df[target].copy()
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    X_num = X[numeric_cols].copy()

    imputer = SimpleImputer(strategy='median')
    scaler = StandardScaler()

    X_num_imputed = imputer.fit_transform(X_num)
    X_num_scaled = scaler.fit_transform(X_num_imputed)

    X_processed = pd.DataFrame(X_num_scaled, columns=numeric_cols, index=X.index)
    return X_processed, y, {'imputer': imputer, 'scaler': scaler, 'numeric_cols': numeric_cols}

def save_preprocessors(preproc_dict, path='models/preprocessors.joblib'):
    joblib.dump(preproc_dict, path)
