"""Per-country Dynamic Factor Model fit, as exported by notebook cell 40."""

from dataclasses import dataclass
from typing import Any

import pandas as pd
import statsmodels.api as sm

from twin_core import TwinParams

# Years the latent state is rolled forward for the stored attractor (2020 -> 2050).
ATTRACTOR_HORIZON = 30


@dataclass(frozen=True)
class FittedTwin:
    results: Any  # statsmodels DynamicFactorResultsWrapper
    params: TwinParams
    diagnostics: dict[str, float | int | bool]


def fit_dynamic_factor(scaled: pd.DataFrame, maxiter: int = 250) -> Any:
    """The twin's model specification, shared by the served fit and the backtest."""
    model = sm.tsa.DynamicFactor(scaled, k_factors=2, factor_order=1, error_order=1)
    return model.fit(disp=False, maxiter=maxiter)


def fit_twin(frame: pd.DataFrame, maxiter: int = 250) -> FittedTwin:
    """Fit a Dynamic Factor Model (k=2) on one country's standardised series."""
    means, stds = frame.mean(), frame.std()
    scaled = (frame - means) / stds
    results = fit_dynamic_factor(scaled, maxiter=maxiter)

    transition = results.filter_results.transition[:, :, 0]
    state = results.smoothed_state[:, -1]
    attractor = state.copy()
    for _ in range(ATTRACTOR_HORIZON):
        attractor = transition @ attractor

    params = TwinParams(
        transition_matrix_A=transition.tolist(),
        latent_state_2020=state.tolist(),
        latent_attractor_2050=attractor.tolist(),
        normalization_means=means.to_dict(),
        normalization_stds=stds.to_dict(),
    )
    diagnostics = {
        "converged": bool(results.mle_retvals.get("converged", False)),
        "n_obs": int(results.nobs),
        "llf": float(results.llf),
        "aic": float(results.aic),
    }
    return FittedTwin(results=results, params=params, diagnostics=diagnostics)
