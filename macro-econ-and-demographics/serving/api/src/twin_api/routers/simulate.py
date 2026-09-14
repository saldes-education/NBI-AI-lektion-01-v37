"""Latent-state trajectories from the exported transition matrices."""

import numpy as np
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from twin_api.deps import RegistryDep
from twin_core import country_key

router = APIRouter(tags=["simulate"])


class SimulationRequest(BaseModel):
    country: str = Field(..., examples=["sweden"], description="Target nation framework to query.")
    steps_ahead: int = Field(
        default=10, ge=1, le=100, examples=[15], description="Years forward to slide the system."
    )


class SimulationResponse(BaseModel):
    country: str
    target_year: int
    steps_simulated: int
    latent_factor_1_modernization: float
    latent_factor_2_systemic_strain: float


@router.post("/simulate", response_model=SimulationResponse)
def simulate_trajectory(request: SimulationRequest, registry: RegistryDep) -> SimulationResponse:
    """Slides the country's latent state vector recursively through the transition matrix."""
    twin = registry.get(request.country)
    if twin is None:
        raise HTTPException(
            status_code=404,
            detail=f"Twin for country '{request.country}' not found. "
            f"Available matrices: {registry.countries}",
        )

    A = np.array(twin.params.transition_matrix_A)
    x = np.array(twin.params.latent_state_2020)
    # X_(t+1) = A * X_t
    for _ in range(request.steps_ahead):
        x = A @ x

    return SimulationResponse(
        country=country_key(request.country).upper(),
        target_year=registry.manifest.base_year + request.steps_ahead,
        steps_simulated=request.steps_ahead,
        latent_factor_1_modernization=float(x[0]),
        latent_factor_2_systemic_strain=float(x[1]),
    )
