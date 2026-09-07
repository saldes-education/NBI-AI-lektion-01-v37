"""
Steg 3: Testa API:et med TestClient (kräver httpx).

Kör:  uv run pytest -v

Vad är rimligt att testa i ett ML-projekt?
- Att tjänsten startar och modellen laddas.
- Att kontraktet hålls: giltig input -> rätt form på svaret.
- Att ogiltig input avvisas (422) - vi litar på Pydantic men verifierar.
- Ett "sanity check" på en känd datapunkt. INTE exakt accuracy - det är
  ett träningsresultat, inte något API:et ansvarar för.
"""

import pytest
from fastapi.testclient import TestClient

from iris_service.api import app

# En typisk setosa-blomma (klass 0) från Iris-datasetet
SETOSA = {
    "sepal_length": 5.1,
    "sepal_width": 3.5,
    "petal_length": 1.4,
    "petal_width": 0.2,
}


@pytest.fixture(scope="module")
def client():
    # 'with' gör att lifespan körs, dvs modellen laddas.
    with TestClient(app) as c:
        yield c


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["model_loaded"] is True


def test_predict_returns_valid_contract(client):
    r = client.post("/predict", json=SETOSA)
    assert r.status_code == 200
    body = r.json()
    assert set(body) == {"species", "class_index", "probabilities"}
    assert body["class_index"] in (0, 1, 2)
    assert abs(sum(body["probabilities"].values()) - 1.0) < 1e-3


def test_predict_known_sample(client):
    r = client.post("/predict", json=SETOSA)
    assert r.json()["species"] == "setosa"


def test_predict_rejects_missing_field(client):
    bad = {k: v for k, v in SETOSA.items() if k != "petal_width"}
    r = client.post("/predict", json=bad)
    assert r.status_code == 422


def test_predict_rejects_negative_value(client):
    bad = {**SETOSA, "sepal_length": -1.0}
    r = client.post("/predict", json=bad)
    assert r.status_code == 422
