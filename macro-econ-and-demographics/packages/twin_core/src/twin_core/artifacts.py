"""Layout of a trained artifact directory.

artifacts/<version>/
    manifest.json                        ArtifactManifest
    cybernetic_twins_api_payload.json    {country_key: TwinParams}
    twin_model_<country>.pkl             pickled statsmodels results, one per country
"""

import json
from datetime import datetime
from pathlib import Path

from pydantic import BaseModel, Field

from twin_core.variables import BASE_YEAR

MANIFEST_FILE = "manifest.json"
PAYLOAD_FILE = "cybernetic_twins_api_payload.json"
SCHEMA_VERSION = 1


def country_key(country: str) -> str:
    """Registry key for a country name: 'United States' -> 'united states'."""
    return country.strip().lower()


def model_filename(country: str) -> str:
    return f"twin_model_{country_key(country).replace(' ', '_')}.pkl"


class TwinParams(BaseModel):
    transition_matrix_A: list[list[float]]
    latent_state_2020: list[float]
    latent_attractor_2050: list[float]
    normalization_means: dict[str, float]
    normalization_stds: dict[str, float]


class ArtifactManifest(BaseModel):
    schema_version: int = SCHEMA_VERSION
    artifact_version: str
    created_at: datetime
    base_year: int = BASE_YEAR
    training_window: tuple[int, int] = (1950, BASE_YEAR)
    # Pickled models only load reliably with the versions they were written with.
    library_versions: dict[str, str] = Field(default_factory=dict)
    payload_file: str = PAYLOAD_FILE
    models: dict[str, str]
    diagnostics: dict[str, dict[str, float | int | bool]] = Field(default_factory=dict)
    notes: str | None = None


def load_manifest(artifact_dir: Path) -> ArtifactManifest:
    return ArtifactManifest.model_validate_json((Path(artifact_dir) / MANIFEST_FILE).read_text())


def load_payload(artifact_dir: Path, manifest: ArtifactManifest) -> dict[str, TwinParams]:
    raw = json.loads((Path(artifact_dir) / manifest.payload_file).read_text())
    return {key: TwinParams.model_validate(params) for key, params in raw.items()}
