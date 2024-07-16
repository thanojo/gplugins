from gplugins.lumerical.simulation_settings import LUMERICAL_MODE_SIMULATION_SETTINGS
from gplugins.lumerical.mode import LumericalModeSimulation
from gplugins.lumerical.config import DEBUG_LUMERICAL
from pathlib import Path

def test_mode():
    import gdsfactory as gf
    c = gf.components.straight()

    ### TODO: Update generic PDK with dopants in layer_stack
    from gdsfactory.generic_tech.layer_map import LAYER
    from gdsfactory.pdk import get_layer_stack
    from gdsfactory.technology.layer_stack import LayerLevel

    layer_stack = get_layer_stack()
    layer_stack.layers["substrate"].layer_type = "background"
    layer_stack.layers["substrate"].background_doping_ion = None
    layer_stack.layers["substrate"].background_doping_concentration = None
    layer_stack.layers["box"].layer_type = "background"
    layer_stack.layers["clad"].layer_type = "background"
    layer_stack.layers["core"].sidewall_angle = 0
    layer_stack.layers["slab90"].sidewall_angle = 0
    layer_stack.layers["via_contact"].sidewall_angle = 0
    layer_stack.layers["N"] = LayerLevel(
        layer=LAYER.N,
        thickness=0.22,
        zmin=0,
        material="si",
        mesh_order=4,
        background_doping_concentration=5e17,
        background_doping_ion="n",
        orientation="100",
        layer_type="doping",
    )
    layer_stack.layers["P"] = LayerLevel(
        layer=LAYER.P,
        thickness=0.22,
        zmin=0,
        material="si",
        mesh_order=4,
        background_doping_concentration=7e17,
        background_doping_ion="p",
        orientation="100",
        layer_type="doping",
    )
    layer_stack.layers["NP"] = LayerLevel(
        layer=LAYER.NP,
        thickness=0.09,
        zmin=0,
        material="si",
        mesh_order=4,
        background_doping_concentration=3e18,
        background_doping_ion="n",
        orientation="100",
        layer_type="doping",
    )
    layer_stack.layers["PP"] = LayerLevel(
        layer=LAYER.PP,
        thickness=0.09,
        zmin=0,
        material="si",
        mesh_order=4,
        background_doping_concentration=2e18,
        background_doping_ion="p",
        orientation="100",
        layer_type="doping",
    )
    layer_stack.layers["NPP"] = LayerLevel(
        layer=LAYER.NPP,
        thickness=0.09,
        zmin=0,
        material="si",
        mesh_order=4,
        background_doping_concentration=1e19,
        background_doping_ion="n",
        orientation="100",
        layer_type="doping",
    )
    layer_stack.layers["PPP"] = LayerLevel(
        layer=LAYER.PPP,
        thickness=0.09,
        zmin=0,
        material="si",
        mesh_order=4,
        background_doping_concentration=1e19,
        background_doping_ion="p",
        orientation="100",
        layer_type="doping",
    )

    LUMERICAL_MODE_SIMULATION_SETTINGS.x = 2
    LUMERICAL_MODE_SIMULATION_SETTINGS.y = 0
    LUMERICAL_MODE_SIMULATION_SETTINGS.z = 0.11
    LUMERICAL_MODE_SIMULATION_SETTINGS.xspan = 2
    LUMERICAL_MODE_SIMULATION_SETTINGS.zspan = 1
    LUMERICAL_MODE_SIMULATION_SETTINGS.injection_axis = "2D X normal"
    LUMERICAL_MODE_SIMULATION_SETTINGS.mesh_cells_per_wavl = 60
    sim = LumericalModeSimulation(component=c,
                                  layerstack=layer_stack,
                                  simulation_settings=LUMERICAL_MODE_SIMULATION_SETTINGS,
                                  run_mesh_convergence=True,
                                  run_port_convergence=True,
                                  override_convergence=False,
                                  dirpath=Path(__file__).resolve().parent / "test_runs",
                                  hide=not DEBUG_LUMERICAL)
    sim.plot_index()
    sim.plot_neff_vs_wavelength()
    sim.plot_ng_vs_wavelength()
    sim.plot_dispersion_vs_wavelength()