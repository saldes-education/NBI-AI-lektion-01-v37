"""Twin input series, lifted from training/notebooks/analysis.ipynb.

Each step mirrors the notebook cell that produced it, so exported twins stay comparable
with the analysis: cells 2/7 (HFD fertility), 16 (JST debt), 21 (social spending and the
debt window), 25 (fertility window), 28 (urbanisation, schooling, child mortality).
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pycountry

TWIN_COUNTRIES = ("Sweden", "Japan", "United States", "Italy")
TRAINING_WINDOW = (1950, 2020)

# Countries the notebook pivots on; only TWIN_COUNTRIES are fitted.
TARGET_COUNTRIES = [
    "United Kingdom",
    "United States",
    "Germany",
    "France",
    "Japan",
    "Sweden",
    "Italy",
]


@dataclass(frozen=True)
class TwinInputs:
    """Annual, gap-filled series indexed by year (country columns unless noted)."""

    fertility: pd.DataFrame
    # Cross-country mean of public + private debt/GDP, one series shared by all countries.
    debt: pd.Series
    urbanization: pd.DataFrame
    education: pd.DataFrame
    child_mortality: pd.DataFrame


def _country_name(code: str) -> str | None:
    try:
        return pycountry.countries.get(alpha_3=code).name
    except (AttributeError, KeyError):
        return None


def _smooth(series: pd.DataFrame | pd.Series, start: int, end: int, method: str):
    years = np.arange(start, end + 1)
    return series.loc[start:end].reindex(years).interpolate(method=method)


def load_fertility(data_dir: Path) -> pd.DataFrame:
    df = pd.read_csv(data_dir / "tfr" / "tfrRR.txt", sep=r"\s+", skiprows=2)
    df["Country_name"] = df["Code"].apply(_country_name)
    return df[df["Country_name"].isin(TARGET_COUNTRIES)].pivot(
        index="Year", columns="Country_name", values="TFR"
    )


def load_global_debt(data_dir: Path) -> pd.Series:
    jst = pd.read_excel(data_dir / "debt" / "JSTdatasetR6.xlsx")
    jst["private_debt_gdp"] = (jst["tloans"] / jst["gdp"]) * 100
    jst["total_debt_gdp"] = jst["debtgdp"] + jst["private_debt_gdp"]
    return jst.groupby("year")["total_debt_gdp"].mean().dropna()


def load_owid(path: Path, value_name: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = ["Country", "Code", "Year", value_name]
    return df[df["Country"].isin(TARGET_COUNTRIES)].pivot(
        index="Year", columns="Country", values=value_name
    )


def build_inputs(data_dir: Path) -> TwinInputs:
    data_dir = Path(data_dir)

    debt_trend = load_global_debt(data_dir)
    social = load_owid(
        data_dir / "social-spending-oecd-longrun" / "social-spending-oecd-longrun.csv",
        "Social_Spending_Pct_GDP",
    )
    start = int(max(social.index.min(), debt_trend.index.min()))
    end = int(min(social.index.max(), debt_trend.index.max()))
    social = _smooth(social, start, end, "linear")
    debt = _smooth(debt_trend, start, end, "linear")

    fertility = load_fertility(data_dir)
    start = int(max(social.index.min(), fertility.index.min()))
    end = int(min(social.index.max(), fertility.index.max()))
    fertility = _smooth(fertility, start, end, "linear")

    modernization = (
        load_owid(data_dir / "share-of-population-urban.csv", "Urban_Share"),
        load_owid(data_dir / "mean-years-of-schooling-long-run.csv", "Years_of_Schooling"),
        load_owid(data_dir / "child-mortality-igme.csv", "Child_Mortality_Rate"),
    )
    start = int(max(frame.index.min() for frame in modernization))
    end = int(min(frame.index.max() for frame in modernization))
    urbanization, education, child_mortality = (
        _smooth(frame, start, end, "quadratic") for frame in modernization
    )

    return TwinInputs(fertility, debt, urbanization, education, child_mortality)


def country_frame(
    inputs: TwinInputs, country: str, window: tuple[int, int] = TRAINING_WINDOW
) -> pd.DataFrame:
    """Aligned observations for one twin, columns in twin_core.VARIABLES order."""
    start, end = window
    return pd.DataFrame(
        {
            "Fertility": inputs.fertility.loc[start:end, country],
            "Debt": inputs.debt.loc[start:end],
            "Urbanization": inputs.urbanization.loc[start:end, country],
            "Education": inputs.education.loc[start:end, country],
            "Child_Mortality": inputs.child_mortality.loc[start:end, country],
        }
    ).dropna()
