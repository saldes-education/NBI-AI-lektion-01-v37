from typing import Annotated

from fastapi import Depends, Request

from twin_api.registry import TwinRegistry


def get_registry(request: Request) -> TwinRegistry:
    return request.app.state.registry


RegistryDep = Annotated[TwinRegistry, Depends(get_registry)]
