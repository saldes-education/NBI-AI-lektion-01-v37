from fastapi import APIRouter
from pydantic import BaseModel

from twin_api.deps import RegistryDep
from twin_core import VARIABLES

router = APIRouter(tags=["meta"])


class HealthResponse(BaseModel):
    status: str
    artifact_version: str
    available_twins: list[str]
    engine: str


class CountriesResponse(BaseModel):
    base_year: int
    countries: list[str]
    variables: list[str]


@router.get("/health", response_model=HealthResponse)
def health(registry: RegistryDep) -> HealthResponse:
    return HealthResponse(
        status="operational",
        artifact_version=registry.manifest.artifact_version,
        available_twins=registry.countries,
        engine="Dynamic Factor State-Space VAR(1)",
    )


@router.get("/countries", response_model=CountriesResponse)
def countries(registry: RegistryDep) -> CountriesResponse:
    return CountriesResponse(
        base_year=registry.manifest.base_year,
        countries=registry.countries,
        variables=list(VARIABLES),
    )
