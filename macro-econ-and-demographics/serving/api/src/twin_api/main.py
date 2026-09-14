import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from twin_api import __version__
from twin_api.registry import TwinRegistry
from twin_api.routers import dashboard, health, predict, simulate
from twin_api.settings import Settings

log = logging.getLogger("uvicorn.error")


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or Settings.from_env()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        registry = TwinRegistry.load(settings.artifact_dir)
        log.info(
            "Loaded artifact %s from %s: %s",
            registry.manifest.artifact_version,
            settings.artifact_dir,
            registry.countries,
        )
        app.state.registry = registry
        yield

    app = FastAPI(
        title="Cybernetic Digital Twin API",
        description="State-space twins of fertility, debt, urbanisation, schooling and "
        "child mortality per country.",
        version=__version__,
        lifespan=lifespan,
    )
    if settings.cors_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=list(settings.cors_origins),
            allow_methods=["GET", "POST"],
            allow_headers=["Content-Type"],
        )
    for module in (health, simulate, predict, dashboard):
        app.include_router(module.router)
    return app


app = create_app()
