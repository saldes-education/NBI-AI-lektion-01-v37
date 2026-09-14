"""Loads one artifact directory at startup and keeps every twin in memory."""

import logging
import pickle
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

from twin_core import ArtifactManifest, TwinParams, country_key, load_manifest, load_payload

log = logging.getLogger("uvicorn.error")


@dataclass(frozen=True)
class Twin:
    params: TwinParams
    model: Any  # statsmodels DynamicFactorResultsWrapper


@dataclass(frozen=True)
class TwinRegistry:
    manifest: ArtifactManifest
    twins: dict[str, Twin]

    @classmethod
    def load(cls, artifact_dir: Path) -> "TwinRegistry":
        artifact_dir = Path(artifact_dir)
        manifest = load_manifest(artifact_dir)
        _warn_on_version_drift(manifest)
        payload = load_payload(artifact_dir, manifest)

        mismatched = sorted(set(payload) ^ set(manifest.models))
        if mismatched:
            raise ValueError(f"Payload and manifest disagree on countries: {mismatched}")

        twins = {}
        for key, params in payload.items():
            # Unpickling runs code: only point TWIN_ARTIFACT_DIR at artifacts you produced.
            with (artifact_dir / manifest.models[key]).open("rb") as f:
                twins[key] = Twin(params=params, model=pickle.load(f))
        return cls(manifest=manifest, twins=twins)

    @property
    def countries(self) -> list[str]:
        return list(self.twins)

    def get(self, country: str) -> Twin | None:
        return self.twins.get(country_key(country))


def _warn_on_version_drift(manifest: ArtifactManifest) -> None:
    for package, trained_with in manifest.library_versions.items():
        try:
            installed = version(package)
        except PackageNotFoundError:
            installed = None
        if installed != trained_with:
            log.warning(
                "Artifact %s was written with %s %s but %s is installed",
                manifest.artifact_version,
                package,
                trained_with,
                installed,
            )
