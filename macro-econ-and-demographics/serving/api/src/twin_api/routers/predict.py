"""Forecasts in real-world units from the pickled statsmodels twins."""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from twin_api.deps import RegistryDep
from twin_core import country_key, destandardize

router = APIRouter(tags=["forecast"])


class PredictionRequest(BaseModel):
    country: str = Field(..., examples=["sweden"])
    years_ahead: int = Field(default=10, ge=1, le=30, examples=[15])


class RealWorldMetrics(BaseModel):
    total_fertility_rate: float
    debt_to_gdp_pct: float
    urbanization_share_pct: float
    years_of_schooling: float
    child_mortality_rate_pct: float


class PredictionResponse(BaseModel):
    country: str
    target_year: int
    forecast_horizon_years: int
    predictions_scaled: dict[str, float]
    predictions_real_world: RealWorldMetrics


@router.post("/predict", response_model=PredictionResponse)
def predict_real_world_metrics(
    request: PredictionRequest, registry: RegistryDep
) -> PredictionResponse:
    twin = registry.get(request.country)
    if twin is None:
        raise HTTPException(
            status_code=404, detail=f"Country template '{request.country}' unknown."
        )

    forecast = twin.model.forecast(steps=request.years_ahead)
    scaled = {name: float(value) for name, value in forecast.iloc[-1].items()}
    real = destandardize(scaled, twin.params.normalization_means, twin.params.normalization_stds)

    return PredictionResponse(
        country=country_key(request.country).upper(),
        target_year=registry.manifest.base_year + request.years_ahead,
        forecast_horizon_years=request.years_ahead,
        predictions_scaled=scaled,
        predictions_real_world=RealWorldMetrics(
            total_fertility_rate=round(real["Fertility"], 3),
            debt_to_gdp_pct=round(real["Debt"], 2),
            urbanization_share_pct=round(real["Urbanization"], 2),
            years_of_schooling=round(real["Education"], 2),
            child_mortality_rate_pct=round(real["Child_Mortality"], 3),
        ),
    )
