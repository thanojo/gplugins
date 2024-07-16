from pathlib import Path
from gplugins.lumerical.recipes.taper_design_recipe import RoutingTaperDesignRecipe, RoutingTaperDesignIntent
from functools import partial
import gdsfactory as gf
from gplugins.lumerical.simulation_settings import SimulationSettingsLumericalEme, SimulationSettingsLumericalFdtd
from gplugins.lumerical.convergence_settings import ConvergenceSettingsLumericalEme, ConvergenceSettingsLumericalFdtd
from gdsfactory.config import logger
from gdsfactory.components.taper_cross_section import taper_cross_section

### 0. DEFINE WHERE FILES ARE SAVED
dirpath = Path("../recipes/recipe_runs")
dirpath.mkdir(parents=True, exist_ok=True)

### 1. DEFINE DESIGN INTENT
design_intent = RoutingTaperDesignIntent(
    narrow_waveguide_routing_loss_per_cm=3,  # dB/cm
)

narrow_waveguide_cross_section = partial(
    gf.cross_section.cross_section,
    layer=(1, 0),
    width=0.5,
)
wide_waveguide_cross_section = partial(
    gf.cross_section.cross_section,
    layer=(1, 0),
    width=3.0,
)

### 2. DEFINE LAYER STACK
from gdsfactory.technology.layer_stack import LayerLevel, LayerStack

layerstack_lumerical = LayerStack(
    layers={
        "clad": LayerLevel(
            layer=(99999, 0),
            thickness=3.0,
            zmin=0.0,
            material="sio2",
            sidewall_angle=0.0,
            mesh_order=9,
            layer_type="background",
        ),
        "box": LayerLevel(
            layer=(99999, 0),
            thickness=3.0,
            zmin=-3.0,
            material="sio2",
            sidewall_angle=0.0,
            mesh_order=9,
            layer_type="background",
        ),
        "core": LayerLevel(
            layer=(1, 0),
            thickness=0.22,
            zmin=0.0,
            material="si",
            sidewall_angle=2.0,
            width_to_z=0.5,
            mesh_order=2,
            layer_type="grow",
            info={"active": True},
        ),
    }
)

### 3. DEFINE SIMULATION AND CONVERGENCE SETTINGS
eme_convergence_setup = ConvergenceSettingsLumericalEme(
    sparam_diff=1 - 10 ** (-0.005 / 10)
)
eme_simulation_setup = SimulationSettingsLumericalEme()

fdtd_convergence_setup = ConvergenceSettingsLumericalFdtd(
    sparam_diff=0.01
)
fdtd_simulation_setup = SimulationSettingsLumericalFdtd(
    mesh_accuracy=2, port_translation=1.0, port_field_intensity_threshold=1e-6,
)

taper_recipe = RoutingTaperDesignRecipe(cell=taper_cross_section,
                                        cross_section1=narrow_waveguide_cross_section,
                                        cross_section2=wide_waveguide_cross_section,
                                        design_intent=design_intent,
                                        eme_simulation_setup=eme_simulation_setup,
                                        eme_convergence_setup=eme_convergence_setup,
                                        fdtd_simulation_setup=fdtd_simulation_setup,
                                        fdtd_convergence_setup=fdtd_convergence_setup,
                                        layer_stack=layerstack_lumerical,
                                        dirpath=dirpath)
taper_recipe.override_recipe = False
success = taper_recipe.eval()
if success:
    logger.info("Completed taper design recipe.")
else:
    logger.info("Incomplete run of taper design recipe.")