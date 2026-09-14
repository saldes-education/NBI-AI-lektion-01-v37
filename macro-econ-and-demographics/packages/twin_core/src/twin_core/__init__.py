"""Artifact contract shared by twin training (writer) and serving (reader)."""

from twin_core.artifacts import (
    MANIFEST_FILE,
    PAYLOAD_FILE,
    ArtifactManifest,
    TwinParams,
    country_key,
    load_manifest,
    load_payload,
    model_filename,
)
from twin_core.variables import BASE_YEAR, VARIABLES, destandardize

__all__ = [
    "BASE_YEAR",
    "MANIFEST_FILE",
    "PAYLOAD_FILE",
    "VARIABLES",
    "ArtifactManifest",
    "TwinParams",
    "country_key",
    "destandardize",
    "load_manifest",
    "load_payload",
    "model_filename",
]
