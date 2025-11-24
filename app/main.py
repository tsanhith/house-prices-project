import logging
from pathlib import Path
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np

log = logging.getLogger("houseprices")
logging.basicConfig(level=logging.INFO)

app = FastAPI(title="House Prices API")

class Item(BaseModel):
    features: dict

model_bundle = None
MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "baseline.pkl"

@app.on_event("startup")
def load_model():
    global model_bundle
    try:
        if not MODEL_PATH.exists():
            log.error("Model file not found at %s", MODEL_PATH)
            model_bundle = None
            return
        model_bundle = joblib.load(str(MODEL_PATH))
        log.info("Loaded model from %s", MODEL_PATH)
    except Exception as e:
        model_bundle = None
        log.exception("Failed to load model: %s", e)

@app.get("/")
def health():
    return {"status": "ok", "model_loaded": bool(model_bundle)}

@app.post("/predict")
def predict(item: Item):
    if model_bundle is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Train model or check server logs.")
    model = model_bundle.get("model")
    numeric_cols = model_bundle.get("numeric_cols", [])
    # Build feature vector (missing columns default to 0)
    x = [item.features.get(c, 0) for c in numeric_cols]
    x = np.array(x).reshape(1, -1)
    # Apply preprocessors if present
    if "imputer" in model_bundle:
        x = model_bundle["imputer"].transform(x)
    if "scaler" in model_bundle:
        x = model_bundle["scaler"].transform(x)
    pred = model.predict(x)[0]
    return {"prediction": float(pred)}
