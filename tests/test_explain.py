from fastapi.testclient import TestClient
import app.main as main
from types import SimpleNamespace

# tiny no-op transformer to keep test simple
class NoOp:
    def transform(self, X):
        return X

def setup_module():
    fake_model = SimpleNamespace(
        predict=lambda X: [12345.0]
    )
    main.model_bundle = {
        "model": fake_model,
        "numeric_cols": ["LotArea", "OverallQual"],
        "imputer": NoOp(),
        "scaler": NoOp(),
    }

client = TestClient(main.app)

def test_explain_success():
    payload = {
        "features": {"LotArea": 8000, "OverallQual": 7},
        "top_k": 3
    }
    r = client.post("/explain", json=payload)

    assert r.status_code == 200
    data = r.json()
    assert "prediction_explanation" in data
    assert isinstance(data["prediction_explanation"], list)
    assert len(data["prediction_explanation"]) > 0
