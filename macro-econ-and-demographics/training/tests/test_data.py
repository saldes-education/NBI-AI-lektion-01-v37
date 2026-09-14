import json
from pathlib import Path

import pytest

from twin_core import VARIABLES
from twin_training.data import TWIN_COUNTRIES, build_inputs, country_frame

REPO_ROOT = Path(__file__).resolve().parents[2]
BASELINE_PAYLOAD = REPO_ROOT / "artifacts" / "baseline" / "cybernetic_twins_api_payload.json"


@pytest.fixture(scope="module")
def inputs():
    return build_inputs(REPO_ROOT / "data" / "raw")


@pytest.fixture(scope="module")
def baseline():
    return json.loads(BASELINE_PAYLOAD.read_text())


@pytest.mark.parametrize("country", TWIN_COUNTRIES)
def test_country_frame_reproduces_notebook_normalisation(inputs, baseline, country):
    frame = country_frame(inputs, country)
    params = baseline[country.lower()]

    assert list(frame.columns) == list(VARIABLES)
    assert not frame.isna().any().any()
    assert frame.mean().to_dict() == pytest.approx(params["normalization_means"])
    assert frame.std().to_dict() == pytest.approx(params["normalization_stds"])
