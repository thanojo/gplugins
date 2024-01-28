import gdsfactory as gf
import time
from gdsfactory.components import straight
from gplugins.lumerical.FdtdDesignRecipe import FdtdDesignRecipe
from gdsfactory.generic_tech import LAYER, LAYER_STACK



if __name__ == "__main__":
    c = gf.Component("two_references")
    wr1 = c << gf.components.straight(width=0.6, layer=LAYER.WG)
    wr2 = c << gf.components.straight(width=0.6, layer=LAYER.WG)
    wr2.movey(10)
    c.add_ports(wr1.get_ports_list(), prefix="bot_")
    c.add_ports(wr2.get_ports_list(), prefix="top_")

    recipe = FdtdDesignRecipe(c)
    recipe.eval()
