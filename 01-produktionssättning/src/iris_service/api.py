"""
Steg 2: Exponera modellen som en webbtjänst med FastAPI.

Starta:   uv run uvicorn iris_service.api:app --reload
Testa:    öppna http://127.0.0.1:8000/docs  (Swagger UI genereras automatiskt)

Nyckelidéer:
- Modellen laddas EN gång vid uppstart (lifespan), inte per request.
- Pydantic-scheman definierar kontraktet: vad som får skickas in, vad som
  kommer ut. Ogiltig input avvisas automatiskt med HTTP 422.
- Vi varnar om sklearn-versionen skiljer sig från den modellen tränades med.
"""

import json
import warnings
from contextlib import asynccontextmanager

import joblib
import numpy as np
import sklearn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from iris_service.config import META_PATH, MODEL_PATH

# Här hamnar modellen när den laddats. Ett enkelt "state"-objekt.
state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Körs vid uppstart (före yield) och vid nedstängning (efter yield)."""
    if not MODEL_PATH.exists():
        raise RuntimeError(
            f"Hittar inte {MODEL_PATH}. Kör 'uv run train' först."
        )
    state["model"] = joblib.load(MODEL_PATH)
    state["meta"] = json.loads(META_PATH.read_text())

    trained_with = state["meta"]["sklearn_version"]
    if trained_with != sklearn.__version__:
        warnings.warn(
            f"Modellen tränades med scikit-learn {trained_with}, "
            f"men {sklearn.__version__} är installerat. Pinna versioner!"
        )
    yield
    state.clear()


app = FastAPI(title="Iris API", version="1.0.0", lifespan=lifespan)


# ---------- Pydantic-scheman: API:ets kontrakt ----------

class IrisFeatures(BaseModel):
    """Request-schema. Field(gt=0) ger validering gratis."""
    sepal_length: float = Field(gt=0, description="cm", examples=[5.1])
    sepal_width: float = Field(gt=0, description="cm", examples=[3.5])
    petal_length: float = Field(gt=0, description="cm", examples=[1.4])
    petal_width: float = Field(gt=0, description="cm", examples=[0.2])

    def to_array(self) -> np.ndarray:
        # Ordningen MÅSTE matcha den ordning modellen tränades på.
        return np.array(
            [[self.sepal_length, self.sepal_width,
              self.petal_length, self.petal_width]]
        )


class Prediction(BaseModel):
    """Response-schema."""
    species: str
    class_index: int
    probabilities: dict[str, float]


# ---------- Endpoints ----------

@app.get("/health")
def health() -> dict:
    """Enkel kontroll att tjänsten är uppe och modellen laddad."""
    return {
        "status": "ok",
        "model_loaded": "model" in state,
        "sklearn_version": sklearn.__version__,
    }


@app.post("/predict", response_model=Prediction)
def predict(features: IrisFeatures) -> Prediction:
    model = state.get("model")
    if model is None:
        raise HTTPException(status_code=503, detail="Modellen är inte laddad")

    X = features.to_array()
    class_index = int(model.predict(X)[0])
    probas = model.predict_proba(X)[0]
    target_names = state["meta"]["target_names"]

    return Prediction(
        species=target_names[class_index],
        class_index=class_index,
        probabilities={
            name: round(float(p), 4) for name, p in zip(target_names, probas)
        },
    )
