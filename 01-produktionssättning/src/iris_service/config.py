"""
Gemensamma sökvägar för hela paketet.

Sökvägarna utgår från projektroten (mappen med pyproject.toml), inte från
den mapp man råkar stå i när man kör kommandot. Det gör att `uv run train`,
`uv run uvicorn ...` och `uv run pytest` hittar samma modellfil oavsett
varifrån de startas.
"""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = PROJECT_ROOT / "model"
MODEL_PATH = MODEL_DIR / "iris_pipeline.joblib"
META_PATH = MODEL_DIR / "metadata.json"
