"""Tracking tests. None of them import mlflow or write mlruns/: a fake module stands in for it."""

import logging
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

from twin_core import MANIFEST_FILE, PAYLOAD_FILE, TwinParams
from twin_training import cli, tracking
from twin_training.fit import FittedTwin
from twin_training.tracking import (
    MlflowRecorder,
    Recorder,
    country_metrics,
    country_tags,
    git_tags,
    run_params,
    tracking_session,
)

DIAGNOSTICS = {
    "converged": False,
    "n_obs": 71,
    "llf": 181.3,
    "aic": -314.7,
    "backtest_train_window_end": 2000,
    "backtest_n_eval_years": 20,
    "backtest_rmse_Fertility": 2.5,
    "backtest_baseline_rmse_Fertility": 0.28,
    "backtest_skill_Fertility": -8.0,
}
PARAMS = {"version": "v1", "countries": "Sweden"}
TAGS = {"git_commit": "abc123", "git_dirty": "false"}


def fitted_twin(diagnostics=DIAGNOSTICS) -> FittedTwin:
    params = TwinParams(
        transition_matrix_A=[[1.0]],
        latent_state_2020=[0.0],
        latent_attractor_2050=[0.0],
        normalization_means={},
        normalization_stds={},
    )
    return FittedTwin(results=None, params=params, diagnostics=dict(diagnostics))


class FakeMlflow:
    """Stands in for the mlflow module: records every call and raises on the named ones."""

    def __init__(self, fail_on=()):
        self.calls = []
        self.fail_on = set(fail_on)

    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)

        def method(*args, **kwargs):
            self.calls.append((name, args, kwargs))
            if name in self.fail_on:
                raise RuntimeError(f"{name} is broken")

        return method

    def names(self):
        return [name for name, _, _ in self.calls]


@pytest.fixture
def install_fake_mlflow(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    monkeypatch.setattr(tracking, "run_tags", lambda: dict(TAGS))

    def install(fail_on=()):
        fake = FakeMlflow(fail_on)
        monkeypatch.setitem(sys.modules, "mlflow", fake)
        return fake

    return install


def session(**overrides):
    kwargs = dict(
        enabled=True, tracking_uri=None, experiment="twin-training", run_name="v1", params=PARAMS
    )
    return tracking_session(**{**kwargs, **overrides})


def test_disabled_session_is_a_noop(install_fake_mlflow, caplog):
    fake = install_fake_mlflow()
    caplog.set_level(logging.INFO, logger="twin_training")

    with session(enabled=False) as recorder:
        with recorder.country("Sweden"):
            recorder.log_country(DIAGNOSTICS)
        recorder.log_artifacts(Path("artifacts/v1"))

    assert type(recorder) is Recorder
    assert fake.calls == []
    assert [r.getMessage() for r in caplog.records] == ["MLflow tracking off (--no-tracking)"]


def test_missing_mlflow_turns_tracking_off(monkeypatch, caplog):
    monkeypatch.setitem(sys.modules, "mlflow", None)  # makes `import mlflow` raise ImportError
    caplog.set_level(logging.INFO, logger="twin_training")

    with session() as recorder:
        with recorder.country("Sweden"):
            recorder.log_country(DIAGNOSTICS)

    assert type(recorder) is Recorder
    assert len(caplog.records) == 1
    assert "mlflow is not installed" in caplog.records[0].getMessage()


def test_run_params_are_strings_for_every_cli_setting():
    params = run_params(
        version="v1",
        countries=["Sweden", "United States"],
        maxiter=250,
        backtest_split=2000,
        backtest_enabled=True,
        training_window=(1950, 2020),
        data_dir=Path("data/raw"),
    )

    assert params == {
        "version": "v1",
        "countries": "Sweden,United States",
        "maxiter": "250",
        "backtest_split": "2000",
        "backtest_enabled": "true",
        "training_window": "1950-2020",
        "data_dir": "data/raw",
    }


def test_country_metrics_and_tags_come_from_fitted_twin_diagnostics():
    diagnostics = fitted_twin().diagnostics

    assert country_metrics(diagnostics) == {
        "n_obs": 71.0,
        "llf": 181.3,
        "aic": -314.7,
        "backtest_train_window_end": 2000.0,
        "backtest_n_eval_years": 20.0,
        "backtest_rmse_Fertility": 2.5,
        "backtest_baseline_rmse_Fertility": 0.28,
        "backtest_skill_Fertility": -8.0,
    }
    assert country_tags(diagnostics) == {"converged": "false"}


def test_baseline_style_empty_diagnostics_log_nothing():
    assert country_metrics({}) == {}
    assert country_tags({}) == {}


@pytest.mark.parametrize(
    "error", [FileNotFoundError("git"), subprocess.CalledProcessError(128, "git")]
)
def test_git_tags_tolerate_missing_git_or_repo(monkeypatch, error):
    def fail(*args, **kwargs):
        raise error

    monkeypatch.setattr(tracking.subprocess, "run", fail)

    assert git_tags() == {}


def test_session_records_parent_run_country_runs_and_artifacts(install_fake_mlflow, tmp_path):
    fake = install_fake_mlflow()

    with session() as recorder:
        assert isinstance(recorder, MlflowRecorder)
        with recorder.country("Sweden"):
            recorder.log_country(fitted_twin().diagnostics)
        recorder.log_artifacts(tmp_path / "v1")

    assert fake.calls == [
        ("set_tracking_uri", ("sqlite:///mlflow.db",), {}),
        ("set_experiment", ("twin-training",), {}),
        ("start_run", (), {"run_name": "v1"}),
        ("log_params", (PARAMS,), {}),
        ("set_tags", (TAGS,), {}),
        ("start_run", (), {"run_name": "Sweden", "nested": True}),
        ("log_metrics", (country_metrics(DIAGNOSTICS),), {}),
        ("set_tags", ({"converged": "false"},), {}),
        ("end_run", (), {"status": "FINISHED"}),
        ("log_artifact", (str(tmp_path / "v1" / MANIFEST_FILE),), {}),
        ("log_artifact", (str(tmp_path / "v1" / PAYLOAD_FILE),), {}),
        ("end_run", (), {"status": "FINISHED"}),
    ]


def test_tracking_uri_comes_from_flag_then_environment(install_fake_mlflow, monkeypatch):
    fake = install_fake_mlflow()
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://tracking.example:5000")

    with session():
        pass
    with session(tracking_uri="file:/tmp/elsewhere"):
        pass

    uris = [args[0] for name, args, _ in fake.calls if name == "set_tracking_uri"]
    assert uris == ["http://tracking.example:5000", "file:/tmp/elsewhere"]


def test_tracking_error_warns_once_and_does_not_propagate(install_fake_mlflow, caplog, tmp_path):
    fake = install_fake_mlflow(fail_on={"log_metrics"})
    caplog.set_level(logging.INFO, logger="twin_training")

    with session() as recorder:
        for country in ("Sweden", "Japan"):
            with recorder.country(country):
                recorder.log_country(DIAGNOSTICS)
        recorder.log_artifacts(tmp_path / "v1")

    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1
    assert "log_metrics is broken" in warnings[0].getMessage()
    # Logging stops after the failure, but every run that was started is ended.
    assert fake.names() == [
        "set_tracking_uri",
        "set_experiment",
        "start_run",
        "log_params",
        "set_tags",
        "start_run",
        "log_metrics",
        "end_run",
        "end_run",
    ]


def test_backend_failure_at_start_leaves_training_untracked(install_fake_mlflow, caplog):
    fake = install_fake_mlflow(fail_on={"set_experiment"})
    caplog.set_level(logging.INFO, logger="twin_training")

    with session() as recorder:
        with recorder.country("Sweden"):
            recorder.log_country(DIAGNOSTICS)

    assert type(recorder) is Recorder
    assert fake.names() == ["set_tracking_uri", "set_experiment"]
    assert any(r.levelno == logging.WARNING for r in caplog.records)


def test_training_error_propagates_and_marks_runs_failed(install_fake_mlflow):
    fake = install_fake_mlflow()

    with pytest.raises(ValueError, match="fit blew up"):
        with session() as recorder:
            with recorder.country("Sweden"):
                raise ValueError("fit blew up")

    statuses = [kwargs["status"] for name, _, kwargs in fake.calls if name == "end_run"]
    assert statuses == ["FAILED", "FAILED"]


@pytest.fixture
def stub_training(monkeypatch):
    """Replaces data loading, fitting and export in the CLI so it runs without fitting."""
    exported = []
    monkeypatch.setattr(cli, "build_inputs", lambda data_dir: None)
    monkeypatch.setattr(
        cli, "country_frame", lambda inputs, country: pd.DataFrame(index=range(1950, 2021))
    )
    monkeypatch.setattr(cli, "fit_twin", lambda frame, maxiter: fitted_twin())

    def export(fits, out_dir, artifact_version):
        exported.append({country: fitted.diagnostics for country, fitted in fits.items()})
        return out_dir

    monkeypatch.setattr(cli, "export_artifacts", export)
    return exported


CLI_ARGS = ["--version", "v1", "--countries", "Sweden", "Japan", "--no-backtest"]


def test_cli_exports_the_same_fits_with_and_without_tracking(install_fake_mlflow, stub_training):
    fake = install_fake_mlflow()

    cli.main([*CLI_ARGS, "--no-tracking"])
    assert fake.calls == []

    cli.main(CLI_ARGS)
    assert stub_training[0] == stub_training[1]

    child_runs = [kwargs["run_name"] for name, _, kwargs in fake.calls if kwargs.get("nested")]
    assert child_runs == ["Sweden", "Japan"]
    params = next(args[0] for name, args, _ in fake.calls if name == "log_params")
    assert params["backtest_enabled"] == "false"
    assert params["countries"] == "Sweden,Japan"


def test_cli_trains_when_the_tracking_backend_is_down(install_fake_mlflow, stub_training):
    install_fake_mlflow(fail_on={"set_tracking_uri"})

    cli.main(CLI_ARGS)

    assert list(stub_training[0]) == ["Sweden", "Japan"]
