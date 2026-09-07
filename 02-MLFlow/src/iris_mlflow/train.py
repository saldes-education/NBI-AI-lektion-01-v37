"""
Steg 1: Träna en modell och registrera den i MLflow.

Vad MLflow gör åt oss (jämfört med en hemmagjord metadata.json):
- Varje körning loggas som en "run": parametrar, mätvärden, tidsstämpel.
- Artefakten sparas tillsammans med en signatur (förväntad in/utdata) och
  exakta biblioteksversioner - fråga 2 och 3 inbyggda i artefakten.
- Modellen registreras under ett namn och får ett versionsnummer.
- Aliaset "production" pekar på den version API:et ska ladda.

Kör:  uv run train

Allt sparas lokalt: metadata i mlflow.db (SQLite), artefakter i mlruns/.
Webbgränssnitt:  uv run mlflow ui --backend-store-uri sqlite:///mlflow.db
"""

import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from iris_mlflow.config import (
    ALIAS,
    ARTIFACT_ROOT,
    EXPERIMENT_NAME,
    MODEL_NAME,
    TRACKING_URI,
)


def set_up_experiment() -> None:
    """Pekar MLflow på projektets register och ser till att experimentet finns.

    Experimentet skapas med en artefaktmapp under projektroten. Utan det
    hamnar mlruns/ i den mapp man råkar stå i när man kör kommandot.
    """
    mlflow.set_tracking_uri(TRACKING_URI)
    if mlflow.get_experiment_by_name(EXPERIMENT_NAME) is None:
        mlflow.create_experiment(EXPERIMENT_NAME, artifact_location=ARTIFACT_ROOT.as_uri())
    mlflow.set_experiment(EXPERIMENT_NAME)


def main() -> None:
    set_up_experiment()

    # 1. Data
    iris = load_iris(as_frame=True)
    X, y = iris.data, iris.target
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 2. Pipeline = preprocessing + modell i ett objekt
    params = {"model": "LogisticRegression", "max_iter": 1000, "scaler": "StandardScaler"}
    pipeline = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=params["max_iter"])),
        ]
    )

    with mlflow.start_run() as run:
        pipeline.fit(X_train, y_train)
        acc = accuracy_score(y_test, pipeline.predict(X_test))
        print(f"Test accuracy: {acc:.3f}")

        # 3. Logga körningen
        mlflow.log_params(params)
        mlflow.log_metric("test_accuracy", acc)

        # 4. Logga artefakten med signatur och registrera den
        signature = infer_signature(X_train, pipeline.predict(X_train))
        info = mlflow.sklearn.log_model(
            sk_model=pipeline,
            name="model",
            signature=signature,
            input_example=X_train.head(2),
            registered_model_name=MODEL_NAME,
            metadata={"target_names": [str(n) for n in iris.target_names]},
        )

        # 5. Peka aliaset "production" på den nya versionen
        client = mlflow.MlflowClient()
        version = info.registered_model_version
        client.set_registered_model_alias(MODEL_NAME, ALIAS, version)

        print(f"Run id: {run.info.run_id}")
        print(f"Registrerade {MODEL_NAME} version {version} med alias '{ALIAS}'")


if __name__ == "__main__":
    main()
