from collections.abc import Mapping

# Observed series of every twin, in model column order.
VARIABLES = ("Fertility", "Debt", "Urbanization", "Education", "Child_Mortality")

# Last year of the training window; forecasts step forward from here.
BASE_YEAR = 2020


def destandardize(
    scaled: Mapping[str, float],
    means: Mapping[str, float],
    stds: Mapping[str, float],
) -> dict[str, float]:
    """Map z-scores back to real-world units using the training normalisation."""
    return {name: scaled[name] * stds[name] + means[name] for name in VARIABLES}
