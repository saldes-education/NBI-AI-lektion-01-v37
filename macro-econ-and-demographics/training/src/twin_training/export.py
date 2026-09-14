import json
import pickle
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path

from twin_core import (
    MANIFEST_FILE,
    PAYLOAD_FILE,
    ArtifactManifest,
    country_key,
    model_filename,
)
from twin_training.data import TRAINING_WINDOW
from twin_training.fit import FittedTwin

TRACKED_LIBRARIES = ("statsmodels", "pandas", "numpy", "scipy")


def export_artifacts(
    fits: dict[str, FittedTwin],
    out_dir: Path,
    artifact_version: str,
    training_window: tuple[int, int] = TRAINING_WINDOW,
) -> Path:
    """Write pickles, the JSON payload and a manifest into a new artifact directory."""
    out_dir = Path(out_dir)
    # Artifacts are immutable once written; a rerun gets a new version directory.
    out_dir.mkdir(parents=True, exist_ok=False)

    models = {}
    for country, fitted in fits.items():
        filename = model_filename(country)
        with (out_dir / filename).open("wb") as f:
            pickle.dump(fitted.results, f)
        models[country_key(country)] = filename

    payload = {country_key(country): fitted.params.model_dump() for country, fitted in fits.items()}
    (out_dir / PAYLOAD_FILE).write_text(json.dumps(payload, indent=4))

    manifest = ArtifactManifest(
        artifact_version=artifact_version,
        created_at=datetime.now(UTC),
        training_window=training_window,
        library_versions={lib: version(lib) for lib in TRACKED_LIBRARIES},
        models=models,
        diagnostics={country_key(country): fitted.diagnostics for country, fitted in fits.items()},
    )
    (out_dir / MANIFEST_FILE).write_text(manifest.model_dump_json(indent=2))
    return out_dir
