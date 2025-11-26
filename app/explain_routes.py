from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import logging
import numpy as np
import pandas as pd
from pathlib import Path
import joblib

router = APIRouter()
log = logging.getLogger("houseprices")


class ExplainRequest(BaseModel):
    features: dict
    top_k: int = 5


def _get_model_bundle():
    import app.main as main
    return getattr(main, "model_bundle", None)


def _load_saved_bundle():
    """Try reloading the saved baseline.pkl from disk (safe fallback)."""
    try:
        base = Path(__file__).resolve().parent.parent
        p = base / "models" / "baseline.pkl"
        if p.exists():
            data = joblib.load(str(p))
            log.info("Reloaded saved bundle from %s for fallback", p)
            return data
        log.warning("Saved model file not found at %s", p)
    except Exception as e:
        log.exception("Error reloading saved bundle: %s", e)
    return None


@router.post("/explain")
def explain(req: ExplainRequest):
    model_bundle = _get_model_bundle()
    if model_bundle is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    numeric_cols = list(model_bundle.get("numeric_cols", []))
    model = model_bundle.get("model")
    imputer = model_bundle.get("imputer", None)
    scaler = model_bundle.get("scaler", None)

    # Build DataFrame with exact column ordering
    row = {c: req.features.get(c, 0) for c in numeric_cols}
    X_df = pd.DataFrame([row], columns=numeric_cols)

    try:
        X_imp = imputer.transform(X_df) if imputer is not None else X_df.values
        X_scaled = scaler.transform(X_imp) if scaler is not None else X_imp

        pred = float(model.predict(X_scaled)[0])

        explanation = []

        # 1) Try SHAP (best-effort)
        try:
            import shap
            try:
                background = np.zeros((1, X_scaled.shape[1]))
                explainer = shap.Explainer(model.predict, background)
                shap_vals = explainer(X_scaled)
                vals = np.array(getattr(shap_vals, "values", shap_vals)).flatten()
                explanation = [
                    {"feature": c, "contribution": float(v)}
                    for c, v in zip(numeric_cols, vals.tolist())
                ]
                log.info("SHAP succeeded, produced %d contributions", len(explanation))
            except Exception as e_sh:
                log.warning("SHAP explainer attempt failed: %s", e_sh)
                explanation = []
        except Exception as e_shimp:
            log.info("SHAP import/usage unavailable: %s", e_shimp)
            explanation = []

        # 2) Fallback: try to compute contributions from coefficients or importances
        if not explanation:
            # prefer inspecting the live model
            coef = None
            try:
                if hasattr(model, "coef_"):
                    coef = np.array(model.coef_).flatten()
                elif hasattr(model, "coef"):
                    coef = np.array(model.coef).flatten()
            except Exception as ex:
                log.warning("Accessing live model coef raised: %s", ex)
                coef = None

            # if live model lacks coef, try reloading saved bundle from disk
            if (coef is None or coef.size == 0):
                saved = _load_saved_bundle()
                if saved is not None and "model" in saved:
                    try:
                        saved_model = saved["model"]
                        if hasattr(saved_model, "coef_"):
                            coef = np.array(saved_model.coef_).flatten()
                            log.info("Using coef_ from on-disk saved model (len=%d)", coef.size)
                        elif hasattr(saved_model, "coef"):
                            coef = np.array(saved_model.coef).flatten()
                            log.info("Using coef from on-disk saved model (len=%d)", coef.size)
                        elif hasattr(saved_model, "feature_importances_"):
                            # treat importances as contributions proxy
                            imp = np.array(saved_model.feature_importances_).flatten()
                            L = min(len(imp), X_scaled.shape[1])
                            explanation = [
                                {"feature": c, "contribution": float(v)}
                                for c, v in zip(numeric_cols[:L], imp[:L].tolist())
                            ]
                            log.info("Using feature_importances_ from saved model")
                    except Exception as ex:
                        log.warning("Failed to extract coef/importances from saved model: %s", ex)

            # compute contributions if coef is available
            if coef is not None and coef.size > 0:
                n_features = X_scaled.shape[1]
                L = min(coef.size, n_features)
                vals = np.asarray(X_scaled).flatten()
                if vals.size < L:
                    vals = np.pad(vals, (0, L - vals.size), constant_values=0)
                elif vals.size > L:
                    vals = vals[:L]
                contribs = (coef[:L] * vals).tolist()
                cols_used = numeric_cols[:L]
                explanation = [
                    {"feature": c, "contribution": float(v)}
                    for c, v in zip(cols_used, contribs)
                ]
                log.info("Coef fallback used: coef_len=%d, used_features=%d", coef.size, L)

            # if still empty, try feature_importances_ on live model
            if not explanation:
                try:
                    if hasattr(model, "feature_importances_"):
                        imp = np.array(model.feature_importances_).flatten()
                        L = min(len(imp), X_scaled.shape[1])
                        explanation = [
                            {"feature": c, "contribution": float(v)}
                            for c, v in zip(numeric_cols[:L], imp[:L].tolist())
                        ]
                        log.info("feature_importances_ fallback used (live model)")
                except Exception as ex:
                    log.warning("Failed reading feature_importances_ from live model: %s", ex)

        # 3) Sort and take top_k
        explanation = sorted(explanation, key=lambda x: abs(x["contribution"]), reverse=True)
        explanation = explanation[: max(0, int(req.top_k))]

        log.error("RETURNING %d ITEMS (top: %s)", len(explanation), (explanation[0] if explanation else None))
        return {"prediction_explanation": explanation, "prediction": pred}

    except Exception as e:
        log.exception("Explain error: %s", e)
        raise HTTPException(500, f"Explain failed: {e}")
