from fastapi.testclient import TestClient
import app.main as main
from types import SimpleNamespace

# tiny no-op transformer with .transform that returns input unchanged
class NoOpTransformer:
    def transform(self, X):
        return X

def setup_module():
    fake_model = SimpleNamespace(predict=lambda X: [12345.0])
    main.model_bundle = {
        "model": fake_model,
        "numeric_cols": ["LotArea", "OverallQual"],
        # provide no-op transformers so API code calling .transform works
        "imputer": NoOpTransformer(),
        "scaler": NoOpTransformer(),
    }

client = TestClient(main.app)

def test_health_ok():
    r = client.get("/")
    assert r.status_code == 200
    assert r.json().get("model_loaded") is True

def test_predict_success():
    payload = {"features": {"LotArea": 8000, "OverallQual": 7}}
    r = client.post("/predict", json=payload)
    assert r.status_code == 200
    assert "prediction" in r.json()
    assert isinstance(r.json()["prediction"], float)
