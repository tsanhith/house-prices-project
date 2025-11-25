# House Prices — EDA → Model → API

Short: End-to-end pipeline using Kaggle 'House Prices — Advanced Regression Techniques'.  
- EDA: src/eda.py, notebook 
otebooks/01-eda.ipynb  
- Train: python -m src.train (saves models/baseline.pkl and eports/metrics.json)  
- API: uvicorn app.main:app --reload --port 8000 -> POST /predict

## Quick start
1. Clone: git clone https://github.com/tsanhith/house-prices-project.git
2. cd house-prices-project
3. python -m venv .venv; .\.venv\Scripts\Activate.ps1
4. pip install -r requirements.txt
5. Place 	rain.csv in data/ then python -m src.train
6. uvicorn app.main:app --reload --port 8000 and test /predict.
## Explainability: /explain endpoint (SHAP)

Get feature contributions for a prediction using SHAP.

**Endpoint**
POST /explain
Content-Type: application/json

**Request Body**
{
  "features": { "LotArea": 8000, "OverallQual": 7 },
  "top_k": 5
}

**Response Example**
{
  "prediction_explanation": [
    { "feature": "YrSold", "contribution": 1049689.32 },
    { "feature": "YearBuilt", "contribution": -515825.44 }
  ]
}
