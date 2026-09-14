import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from twin_api.main import create_app
from twin_api.settings import Settings

BASELINE_DIR = Path(__file__).resolve().parents[3] / "artifacts" / "baseline"


@pytest.fixture(scope="session")
def client():
    app = create_app(Settings(artifact_dir=BASELINE_DIR))
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture(scope="session")
def baseline_payload():
    return json.loads((BASELINE_DIR / "cybernetic_twins_api_payload.json").read_text())
