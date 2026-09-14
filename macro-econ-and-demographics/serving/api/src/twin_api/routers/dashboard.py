"""Interim control panel for /predict, until the app in frontend/ exists."""

from html import escape
from importlib.resources import files

from fastapi import APIRouter
from fastapi.responses import HTMLResponse

from twin_api.deps import RegistryDep

router = APIRouter(include_in_schema=False)

_TEMPLATE = files("twin_api").joinpath("static/dashboard.html").read_text()


@router.get("/", response_class=HTMLResponse)
def render_dashboard(registry: RegistryDep) -> str:
    options = "".join(
        f'<option value="{escape(c)}">{escape(c.upper())}</option>' for c in registry.countries
    )
    return _TEMPLATE.replace("<!--COUNTRY_OPTIONS-->", options)
