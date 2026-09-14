import numpy as np
import pytest


def test_health_lists_baseline_twins(client):
    body = client.get("/health").json()

    assert body["status"] == "operational"
    assert body["artifact_version"] == "baseline"
    assert sorted(body["available_twins"]) == ["italy", "japan", "sweden", "united states"]


def test_countries_exposes_base_year_and_variables(client):
    body = client.get("/countries").json()

    assert body["base_year"] == 2020
    assert "Fertility" in body["variables"]


def test_simulate_applies_transition_matrix(client, baseline_payload):
    response = client.post("/simulate", json={"country": "sweden", "steps_ahead": 15})

    assert response.status_code == 200
    params = baseline_payload["sweden"]
    A = np.array(params["transition_matrix_A"])
    expected = np.linalg.matrix_power(A, 15) @ np.array(params["latent_state_2020"])
    body = response.json()
    assert body["target_year"] == 2035
    assert body["latent_factor_1_modernization"] == pytest.approx(expected[0])
    assert body["latent_factor_2_systemic_strain"] == pytest.approx(expected[1])


def test_predict_decodes_scaled_forecast(client, baseline_payload):
    response = client.post("/predict", json={"country": "Sweden", "years_ahead": 10})

    assert response.status_code == 200
    body = response.json()
    params = baseline_payload["sweden"]
    expected_tfr = (
        body["predictions_scaled"]["Fertility"] * params["normalization_stds"]["Fertility"]
        + params["normalization_means"]["Fertility"]
    )
    assert body["country"] == "SWEDEN"
    assert body["target_year"] == 2030
    assert body["predictions_real_world"]["total_fertility_rate"] == pytest.approx(
        expected_tfr, abs=5e-4
    )
    assert 0.5 < body["predictions_real_world"]["total_fertility_rate"] < 4


@pytest.mark.parametrize("path", ["/simulate", "/predict"])
def test_unknown_country_is_404(client, path):
    assert client.post(path, json={"country": "atlantis"}).status_code == 404


@pytest.mark.parametrize(
    ("path", "payload"),
    [
        ("/simulate", {"country": "sweden", "steps_ahead": 0}),
        ("/predict", {"country": "sweden", "years_ahead": 31}),
    ],
)
def test_out_of_range_horizon_is_422(client, path, payload):
    assert client.post(path, json=payload).status_code == 422


def test_dashboard_renders_country_options(client):
    html = client.get("/").text

    assert '<option value="united states">UNITED STATES</option>' in html
