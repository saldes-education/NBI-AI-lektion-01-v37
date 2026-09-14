import math
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from twin_core import VARIABLES, ArtifactManifest
from twin_training import evaluate
from twin_training.data import build_inputs, country_frame
from twin_training.evaluate import (
    BacktestResult,
    persistence_forecast,
    rmse,
    run_backtest,
    skill,
    split_frame,
)

REPO_ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def frame():
    years = range(1990, 2000)
    data = {name: np.arange(10, dtype=float) * (i + 1) for i, name in enumerate(VARIABLES)}
    return pd.DataFrame(data, index=years)


def test_split_frame_puts_split_year_in_training(frame):
    train, test = split_frame(frame, 1995)

    assert train.index.max() == 1995
    assert test.index.min() == 1996
    assert len(train) + len(test) == len(frame)


@pytest.mark.parametrize("split_year", [1985, 1999])
def test_split_frame_rejects_empty_side(frame, split_year):
    with pytest.raises(ValueError, match="leaves no"):
        split_frame(frame, split_year)


def test_split_frame_rejects_missing_years(frame):
    with pytest.raises(ValueError, match="consecutive"):
        split_frame(frame.drop(index=1993), 1995)


def test_persistence_holds_last_training_value_flat(frame):
    train, test = split_frame(frame, 1995)

    forecast = persistence_forecast(train, test.index)

    assert list(forecast.index) == list(test.index)
    assert list(forecast.columns) == list(VARIABLES)
    for name in VARIABLES:
        assert (forecast[name] == train.loc[1995, name]).all()


def test_rmse_is_per_variable_in_input_units():
    index = pd.Index([2001, 2002])
    actual = pd.DataFrame({"a": [3.0, 4.0], "b": [1.0, 1.0]}, index=index)
    forecast = pd.DataFrame({"a": [0.0, 0.0], "b": [1.0, 1.0]}, index=index)

    assert rmse(forecast, actual) == pytest.approx({"a": math.sqrt(12.5), "b": 0.0})


def test_rmse_aligns_on_year_not_position():
    actual = pd.DataFrame({"a": [1.0, 2.0]}, index=[2001, 2002])
    forecast = pd.DataFrame({"a": [2.0, 1.0]}, index=[2002, 2001])

    assert rmse(forecast, actual) == {"a": 0.0}


def test_skill_is_relative_improvement_over_baseline():
    result = skill(
        {"a": 1.0, "b": 3.0, "c": 4.0, "d": 1.0}, {"a": 2.0, "b": 3.0, "c": 2.0, "d": 0.0}
    )

    # "d" is undefined when persistence is exact and must not produce inf/NaN.
    assert result == pytest.approx({"a": 0.5, "b": 0.0, "c": -1.0})


def test_persistence_scores_zero_skill_against_itself(frame):
    train, test = split_frame(frame, 1995)
    baseline = rmse(persistence_forecast(train, test.index), test)

    assert skill(baseline, baseline) == pytest.approx(dict.fromkeys(VARIABLES, 0.0))


def test_model_forecast_standardises_on_training_window_only(frame, monkeypatch):
    seen = {}

    class ZeroForecast:
        def forecast(self, steps):
            return pd.DataFrame(np.zeros((steps, len(VARIABLES))), columns=list(VARIABLES))

    def fake_fit(scaled, maxiter):
        seen["scaled"] = scaled
        return ZeroForecast()

    monkeypatch.setattr(evaluate, "fit_dynamic_factor", fake_fit)
    train, test = split_frame(frame, 1995)

    forecast = evaluate.model_forecast(train, test.index)

    # The fit sees only training rows, centred and scaled on those rows.
    assert list(seen["scaled"].index) == list(train.index)
    np.testing.assert_allclose(seen["scaled"].mean(), 0.0, atol=1e-12)
    np.testing.assert_allclose(seen["scaled"].std(), 1.0)
    # A zero z-score forecast decodes to the training mean, not the full-frame mean.
    for name in VARIABLES:
        np.testing.assert_allclose(forecast[name], train[name].mean())
        assert train[name].mean() != pytest.approx(frame[name].mean())


def test_backtest_diagnostics_fit_the_manifest_contract():
    result = BacktestResult(
        train_window_end=2000,
        n_eval_years=20,
        rmse_model={"Fertility": 0.1},
        rmse_baseline={"Fertility": 0.2},
        skill={"Fertility": 0.5},
    )

    manifest = ArtifactManifest(
        artifact_version="test",
        created_at=datetime.now(UTC),
        models={"sweden": "twin_model_sweden.pkl"},
        diagnostics={"sweden": {"converged": True, **result.diagnostics()}},
    )

    assert manifest.diagnostics["sweden"] == {
        "converged": True,
        "backtest_train_window_end": 2000,
        "backtest_n_eval_years": 20,
        "backtest_rmse_Fertility": 0.1,
        "backtest_baseline_rmse_Fertility": 0.2,
        "backtest_skill_Fertility": 0.5,
    }


@pytest.mark.slow
def test_run_backtest_on_sweden():
    frame = country_frame(build_inputs(REPO_ROOT / "data" / "raw"), "Sweden")

    result = run_backtest(frame, split_year=2000)

    assert result.train_window_end == 2000
    assert result.n_eval_years == 20
    assert set(result.rmse_model) == set(result.rmse_baseline) == set(VARIABLES)
    assert all(math.isfinite(v) for v in result.diagnostics().values())
