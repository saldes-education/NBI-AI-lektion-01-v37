"""
Steg 2: Exponera modellen som en webbtjänst med FastAPI.

Starta:   uv run uvicorn iris_mlflow.api:app --reload
Testa:    http://127.0.0.1:8000/docs  (Swagger UI genereras automatiskt)

Skillnad mot att ladda en joblib-fil från disk:
- Vi laddar "models:/iris-classifier@production" från MLflow-registret.
  Vilken version som är "production" bestäms i registret, inte i koden.
- Artefakten bär med sig sin signatur (feature-namn i rätt ordning) och
  sklearn-versionen den tränades med. Vi läser dem därifrån istället för
  att hårdkoda dem här.
"""

import warnings
from contextlib import asynccontextmanager

import mlflow
import mlflow.sklearn
import pandas as pd
import sklearn
from fastapi import FastAPI, HTTPException
from mlflow.models import get_model_info
from pydantic import BaseModel, Field

from iris_mlflow.config import ALIAS, MODEL_NAME, MODEL_URI, TRACKING_URI

# Här hamnar modellen när den laddats. Ett enkelt "state"-objekt.
state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Körs vid uppstart (före yield) och vid nedstängning (efter yield)."""
    mlflow.set_tracking_uri(TRACKING_URI)
    try:
        info = get_model_info(MODEL_URI)
    except Exception as exc:
        raise RuntimeError(
            f"Hittar ingen modell på {MODEL_URI}. Kör 'uv run train' först."
        ) from exc

    state["model"] = mlflow.sklearn.load_model(MODEL_URI)
    state["feature_names"] = [col["name"] for col in info.signature.inputs.to_dict()]
    state["target_names"] = info.metadata["target_names"]
    # Vilken version aliaset pekar på just nu - hämtas från registret
    client = mlflow.MlflowClient()
    state["model_version"] = client.get_model_version_by_alias(MODEL_NAME, ALIAS).version

    trained_with = info.flavors["sklearn"]["sklearn_version"]
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
    sepal_length: float = Field(gt=0, description="cm", examples=[5.1])
    sepal_width: float = Field(gt=0, description="cm", examples=[3.5])
    petal_length: float = Field(gt=0, description="cm", examples=[1.4])
    petal_width: float = Field(gt=0, description="cm", examples=[0.2])

    def to_frame(self, feature_names: list[str]) -> pd.DataFrame:
        """Bygger en DataFrame med kolumnerna i den ordning artefakten kräver."""
        values = {
            "sepal length (cm)": self.sepal_length,
            "sepal width (cm)": self.sepal_width,
            "petal length (cm)": self.petal_length,
            "petal width (cm)": self.petal_width,
        }
        return pd.DataFrame([[values[name] for name in feature_names]], columns=feature_names)


class Prediction(BaseModel):
    species: str
    class_index: int
    probabilities: dict[str, float]
    model_version: str


# ---------- Endpoints ----------

@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "model_loaded": "model" in state,
        "model_version": state.get("model_version"),
        "sklearn_version": sklearn.__version__,
    }


@app.post("/predict", response_model=Prediction)
def predict(features: IrisFeatures) -> Prediction:
    model = state.get("model")
    if model is None:
        raise HTTPException(status_code=503, detail="Modellen är inte laddad")

    X = features.to_frame(state["feature_names"])
    class_index = int(model.predict(X)[0])
    probas = model.predict_proba(X)[0]
    target_names = state["target_names"]

    return Prediction(
        species=target_names[class_index],
        class_index=class_index,
        probabilities={
            name: round(float(p), 4) for name, p in zip(target_names, probas)
        },
        model_version=str(state["model_version"]),
    )
