from pathlib import Path

import gdsfactory as gf
from gdsfactory.generic_tech import LAYER

from gplugins.lumerical.recipes.fdtd_recipe import FdtdRecipe


def test_fdtd_recipe():
    c = gf.components.straight(width=0.6, length=1.0, layer=LAYER.WG)

    recipe = FdtdRecipe(c, dirpath=Path(__file__).resolve().parent / "test_runs")
    recipe.eval()

    # Change the component and check whether recipe is still fresh
    comp = gf.components.taper()
    recipe.cell = comp
    if recipe.is_fresh():
        raise AssertionError(
            f"Expected: False | Got: {recipe.is_fresh()}. Recipe should not be fresh after component is changed."
            + "The recipe should be re-eval'ed before it is fresh again."
        )
