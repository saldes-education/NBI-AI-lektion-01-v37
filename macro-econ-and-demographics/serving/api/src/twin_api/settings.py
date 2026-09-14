import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    artifact_dir: Path
    cors_origins: tuple[str, ...] = ()

    @classmethod
    def from_env(cls) -> "Settings":
        origins = os.environ.get("TWIN_CORS_ORIGINS", "")
        return cls(
            artifact_dir=Path(os.environ.get("TWIN_ARTIFACT_DIR", "artifacts/baseline")),
            cors_origins=tuple(o.strip() for o in origins.split(",") if o.strip()),
        )
