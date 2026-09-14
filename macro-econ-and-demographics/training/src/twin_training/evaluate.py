"""Out-of-sample backtest of a country twin against a persistence baseline.

The twin is refit on the years up to the split, standardised with that window's own mean and
std so nothing from the evaluation years leaks into the model, then forecast over the rest of
the frame. Forecasts are de-standardised and scored by RMSE per variable in real-world units.
The persistence baseline holds the last training observation flat over the same years, and
skill is ``1 - rmse_model / rmse_baseline`` (positive means the model beat persistence).

Debt caveat: ``TwinInputs.debt`` is one cross-country series shared by every twin, so the
Debt scores of all countries measure the same series four times over. They are not
per-country evidence.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from twin_core import VARIABLES, destandardize
from twin_training.fit import fit_dynamic_factor

DEFAULT_SPLIT_YEAR = 2000


@dataclass(frozen=True)
class BacktestResult:
    train_window_end: int
    n_eval_years: int
    rmse_model: dict[str, float]
    rmse_baseline: dict[str, float]
    skill: dict[str, float]

    def diagnostics(self) -> dict[str, float | int]:
        """Flat keys that fit ArtifactManifest.diagnostics (one flat dict per country)."""
        flat: dict[str, float | int] = {
            "backtest_train_window_end": self.train_window_end,
            "backtest_n_eval_years": self.n_eval_years,
        }
        for name, value in self.rmse_model.items():
            flat[f"backtest_rmse_{name}"] = value
        for name, value in self.rmse_baseline.items():
            flat[f"backtest_baseline_rmse_{name}"] = value
        for name, value in self.skill.items():
            flat[f"backtest_skill_{name}"] = value
        return flat


def split_frame(frame: pd.DataFrame, split_year: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Training rows are years <= split_year; evaluation rows are the years after it."""
    years = frame.index.to_numpy()
    if len(years) and not np.array_equal(years, np.arange(years[0], years[0] + len(years))):
        raise ValueError("Backtest needs one row per consecutive year")
    train, test = frame.loc[:split_year], frame.loc[split_year + 1 :]
    if train.empty or test.empty:
        raise ValueError(
            f"Split year {split_year} leaves no training or evaluation years "
            f"in {years.min()}-{years.max()}"
        )
    return train, test


def persistence_forecast(train: pd.DataFrame, index: pd.Index) -> pd.DataFrame:
    """Last observed training value of each variable, held flat over ``index``."""
    last = train.iloc[-1].to_numpy()
    return pd.DataFrame(np.tile(last, (len(index), 1)), index=index, columns=train.columns)


def model_forecast(train: pd.DataFrame, index: pd.Index, maxiter: int = 250) -> pd.DataFrame:
    """Fit on ``train`` alone and forecast the years in ``index``, in real-world units."""
    # Normalisation comes from the training window only; the full-window fit's must not be used.
    means, stds = train.mean().to_dict(), train.std().to_dict()
    scaled = (train - pd.Series(means)) / pd.Series(stds)
    results = fit_dynamic_factor(scaled, maxiter=maxiter)

    last_year = int(train.index.max())
    steps = int(index.max()) - last_year
    forecast = pd.DataFrame(results.forecast(steps=steps)).set_axis(
        range(last_year + 1, last_year + steps + 1)
    )
    rows = [destandardize(row, means, stds) for row in forecast.loc[index].to_dict("records")]
    return pd.DataFrame(rows, index=index, columns=list(VARIABLES))


def rmse(forecast: pd.DataFrame, actual: pd.DataFrame) -> dict[str, float]:
    errors = forecast.loc[actual.index, actual.columns] - actual
    return {name: float(np.sqrt((errors[name] ** 2).mean())) for name in actual.columns}


def skill(rmse_model: dict[str, float], rmse_baseline: dict[str, float]) -> dict[str, float]:
    """``1 - model / baseline`` per variable.

    Undefined when persistence is exact (baseline RMSE 0); those variables are left out
    because the manifest cannot hold inf or NaN.
    """
    return {
        name: 1 - rmse_model[name] / baseline
        for name, baseline in rmse_baseline.items()
        if baseline > 0
    }


def run_backtest(
    frame: pd.DataFrame, split_year: int = DEFAULT_SPLIT_YEAR, maxiter: int = 250
) -> BacktestResult:
    train, test = split_frame(frame, split_year)
    rmse_model = rmse(model_forecast(train, test.index, maxiter=maxiter), test)
    rmse_baseline = rmse(persistence_forecast(train, test.index), test)
    return BacktestResult(
        train_window_end=int(train.index.max()),
        n_eval_years=len(test),
        rmse_model=rmse_model,
        rmse_baseline=rmse_baseline,
        skill=skill(rmse_model, rmse_baseline),
    )
