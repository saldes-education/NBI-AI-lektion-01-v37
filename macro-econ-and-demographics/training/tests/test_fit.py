import json
from pathlib import Path

import numpy as np
import pytest

from twin_core import load_manifest, load_payload
from twin_training.data import build_inputs, country_frame
from twin_training.export import export_artifacts
from twin_training.fit import fit_twin

REPO_ROOT = Path(__file__).resolve().parents[2]
BASELINE_PAYLOAD = REPO_ROOT / "artifacts" / "baseline" / "cybernetic_twins_api_payload.json"


@pytest.mark.slow
def test_fit_and_export_round_trip(tmp_path):
    frame = country_frame(build_inputs(REPO_ROOT / "data" / "raw"), "Sweden")
    out_dir = export_artifacts({"Sweden": fit_twin(frame)}, tmp_path / "v1", artifact_version="v1")

    manifest = load_manifest(out_dir)
    assert manifest.models == {"sweden": "twin_model_sweden.pkl"}
    assert (out_dir / "twin_model_sweden.pkl").is_file()

    params = load_payload(out_dir, manifest)["sweden"]
    baseline = json.loads(BASELINE_PAYLOAD.read_text())["sweden"]
    np.testing.assert_allclose(
        params.transition_matrix_A, baseline["transition_matrix_A"], atol=1e-3
    )
