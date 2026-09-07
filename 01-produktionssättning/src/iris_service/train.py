"""
Steg 1: Träna en modell och spara den till disk.

Kör:  uv run train

Poängen med detta skript:
- Preprocessing (StandardScaler) och modell paketeras i ETT Pipeline-objekt.
  Då följer skalningen alltid med modellen. Sparar man bara modellen
  (utan scalern) blir prediktionerna fel i produktion - klassisk fallgrop.
- Vi sparar även metadata (sklearn-version, feature-namn) bredvid modellen,
  så att API:et kan kontrollera att det laddar rätt sak.
"""

import json

import joblib
import sklearn
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from iris_service.config import META_PATH, MODEL_DIR, MODEL_PATH


def main() -> None:
    # 1. Data (samma flöde som i kapitel 2-4: train / val / test)
    iris = load_iris()
    X, y = iris.data, iris.target
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 2. Pipeline = preprocessing + modell i ett objekt
    pipeline = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            # ..., Alla steg uton det sista måste kunna köra `fit` and `transform`
            # ...,
            # ...,
            # ...,
            # ...,
            ("clf", LogisticRegression(max_iter=1000)),
            # Sista måste kunna köra `fit` and `prediction`
        ]
    )
    # scalerna lär sig medelvärdet och std på X_train
    # Klassifieringen tränas på det omvandlade resultet.
    pipeline.fit(X_train, y_train)

    # 3. Utvärdera
    # Scalern omvandalar X med talen den redan lärt sig
    # Klassifieceraren predikerar
    acc = accuracy_score(y_test, pipeline.predict(X_test))
    print(f"Test accuracy: {acc:.3f}")

    # 4. Spara modell + metadata
    MODEL_DIR.mkdir(exist_ok=True)
    joblib.dump(pipeline, MODEL_PATH)
    META_PATH.write_text(
        json.dumps(
            {
                "sklearn_version": sklearn.__version__,
                "feature_names": list(iris.feature_names),
                "target_names": list(iris.target_names),
                "test_accuracy": round(acc, 4),
            },
            indent=2,
        )
    )
    print(f"Sparade modell till {MODEL_PATH}")
    print(f"Sparade metadata till {META_PATH}")


if __name__ == "__main__":
    main()
