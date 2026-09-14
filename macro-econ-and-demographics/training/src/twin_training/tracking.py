"""Optional MLflow journal for ``twin-train`` runs.

The artifact directory written by ``export_artifacts`` is the source of truth; MLflow only
records each run so runs can be compared. Tracking never changes or fails training: with
``--no-tracking``, without mlflow installed, or when the backend errors, the run carries on and
writes the same artifact.

Run structure mirrors the CLI: one parent run per invocation, named after the artifact version,
with one nested run per country.
"""

import logging
import os
import subprocess
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from importlib.metadata import version
from pathlib import Path
from types import ModuleType
from typing import Any

from twin_core import MANIFEST_FILE, PAYLOAD_FILE
from twin_training.export import TRACKED_LIBRARIES

log = logging.getLogger("twin_training")

DEFAULT_EXPERIMENT = "twin-training"
# MLflow 3.16 refuses the file store (./mlruns) unless MLFLOW_ALLOW_FILE_STORE=true, so the
# local default is SQLite. Logged artifacts still land in ./mlruns.
DEFAULT_TRACKING_URI = "sqlite:///mlflow.db"


def resolve_tracking_uri(cli_value: str | None) -> str:
    return cli_value or os.environ.get("MLFLOW_TRACKING_URI") or DEFAULT_TRACKING_URI


def run_params(
    *,
    version: str,
    countries: Sequence[str],
    maxiter: int,
    backtest_split: int,
    backtest_enabled: bool,
    training_window: tuple[int, int],
    data_dir: Path | str,
) -> dict[str, str]:
    return {
        "version": version,
        "countries": ",".join(countries),
        "maxiter": str(maxiter),
        "backtest_split": str(backtest_split),
        "backtest_enabled": str(backtest_enabled).lower(),
        "training_window": f"{training_window[0]}-{training_window[1]}",
        "data_dir": str(data_dir),
    }


def git_tags() -> dict[str, str]:
    """Commit and dirty flag of the working tree; empty without git or outside a repo."""
    try:
        commit = _git("rev-parse", "HEAD")
        dirty = _git("status", "--porcelain") != ""
    except (OSError, subprocess.SubprocessError):
        return {}
    return {"git_commit": commit, "git_dirty": str(dirty).lower()}


def _git(*args: str) -> str:
    result = subprocess.run(["git", *args], capture_output=True, text=True, check=True, timeout=10)
    return result.stdout.strip()


def run_tags() -> dict[str, str]:
    return {**git_tags(), **{f"library.{lib}": version(lib) for lib in TRACKED_LIBRARIES}}


def country_metrics(diagnostics: Mapping[str, float | int | bool]) -> dict[str, float]:
    """Every numeric diagnostic, fit and backtest alike. Booleans are tags, not metrics."""
    return {
        name: float(value)
        for name, value in diagnostics.items()
        if isinstance(value, int | float) and not isinstance(value, bool)
    }


def country_tags(diagnostics: Mapping[str, float | int | bool]) -> dict[str, str]:
    return {
        name: str(value).lower() for name, value in diagnostics.items() if isinstance(value, bool)
    }


class Recorder:
    """Tracking off: every hook is a no-op."""

    @contextmanager
    def country(self, name: str) -> Iterator[None]:
        yield

    def log_country(self, diagnostics: Mapping[str, float | int | bool]) -> None:
        pass

    def log_artifacts(self, artifact_dir: Path) -> None:
        pass


class MlflowRecorder(Recorder):
    """Records into the active parent run.

    The first tracking error logs a warning and stops further logging, so an unreachable backend
    doesn't slow down or flood the rest of the run. Runs that were started are still ended.
    """

    def __init__(self, mlflow: ModuleType) -> None:
        self._mlflow = mlflow
        self._broken = False
        self._in_country = False

    def _call(
        self,
        action: str,
        fn: Callable[..., Any],
        *args: Any,
        even_if_broken: bool = False,
        **kwargs,
    ) -> bool:
        if self._broken and not even_if_broken:
            return False
        try:
            fn(*args, **kwargs)
        except Exception as exc:
            log.warning("MLflow tracking failed while %s; continuing without it: %s", action, exc)
            self._broken = True
            return False
        return True

    @contextmanager
    def country(self, name: str) -> Iterator[None]:
        started = self._call(
            f"starting the {name} run", self._mlflow.start_run, run_name=name, nested=True
        )
        self._in_country = started
        status = "FAILED"
        try:
            yield
            status = "FINISHED"
        finally:
            self._in_country = False
            if started:
                self._call(
                    f"ending the {name} run",
                    self._mlflow.end_run,
                    status=status,
                    even_if_broken=True,
                )

    def log_country(self, diagnostics: Mapping[str, float | int | bool]) -> None:
        # Without a child run these would land on the parent, mixing countries together.
        if not self._in_country:
            return
        self._call(
            "logging country metrics", self._mlflow.log_metrics, country_metrics(diagnostics)
        )
        self._call("tagging the country run", self._mlflow.set_tags, country_tags(diagnostics))

    def log_artifacts(self, artifact_dir: Path) -> None:
        # Manifest and payload only: the pickles live in artifacts/ and are a trust boundary.
        for filename in (MANIFEST_FILE, PAYLOAD_FILE):
            path = Path(artifact_dir) / filename
            self._call(f"logging {filename}", self._mlflow.log_artifact, str(path))


def _import_mlflow() -> ModuleType | None:
    try:
        import mlflow
    except ImportError:
        return None
    return mlflow


def _start_parent_run(mlflow: ModuleType, uri: str, experiment: str, run_name: str) -> None:
    mlflow.set_tracking_uri(uri)
    mlflow.set_experiment(experiment)
    mlflow.start_run(run_name=run_name)


@contextmanager
def tracking_session(
    *,
    enabled: bool,
    tracking_uri: str | None,
    experiment: str,
    run_name: str,
    params: Mapping[str, str],
) -> Iterator[Recorder]:
    """Parent run around a whole ``twin-train`` invocation; yields the recorder to use inside."""
    if not enabled:
        log.info("MLflow tracking off (--no-tracking)")
        yield Recorder()
        return

    mlflow = _import_mlflow()
    if mlflow is None:
        log.info("MLflow tracking off: mlflow is not installed (install the 'tracking' extra)")
        yield Recorder()
        return

    uri = resolve_tracking_uri(tracking_uri)
    recorder = MlflowRecorder(mlflow)
    if not recorder._call(
        "starting the parent run", _start_parent_run, mlflow, uri, experiment, run_name
    ):
        yield Recorder()
        return

    log.info("MLflow tracking on: %s, experiment %r, run %r", uri, experiment, run_name)
    recorder._call("logging run parameters", mlflow.log_params, dict(params))
    recorder._call("tagging the run", lambda: mlflow.set_tags(run_tags()))

    status = "FAILED"
    try:
        yield recorder
        status = "FINISHED"
    finally:
        recorder._call("ending the parent run", mlflow.end_run, status=status, even_if_broken=True)
