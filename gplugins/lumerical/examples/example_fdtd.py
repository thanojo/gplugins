from functools import partial

import gdsfactory as gf
from gdsfactory.components.taper_cross_section import taper_cross_section
from gdsfactory.technology.layer_stack import LayerLevel, LayerStack

from gplugins.lumerical.convergence_settings import LUMERICAL_FDTD_CONVERGENCE_SETTINGS
from gplugins.lumerical.fdtd import LumericalFdtdSimulation
from gplugins.lumerical.simulation_settings import SIMULATION_SETTINGS_LUMERICAL_FDTD


xs_wg = partial(
    gf.cross_section.cross_section,
    layer=(1, 0),
    width=0.5,
)

xs_wg_wide = partial(
    gf.cross_section.cross_section,
    layer=(1, 0),
    width=2.0,
)

taper = taper_cross_section(
    cross_section1=xs_wg,
    cross_section2=xs_wg_wide,
    length=5,
    width_type="parabolic",
)

layerstack_lumerical2021 = LayerStack(
    layers={
        "clad": LayerLevel(
            name=None,
            layer=(99999, 0),
            thickness=3.0,
            thickness_tolerance=None,
            zmin=0.0,
            zmin_tolerance=None,
            material="sio2",
            sidewall_angle=0.0,
            sidewall_angle_tolerance=None,
            width_to_z=0.0,
            z_to_bias=None,
            mesh_order=9,
            layer_type="background",
            mode=None,
            into=None,
            resistivity=None,
            bias=None,
            derived_layer=None,
            info={},
            background_doping_concentration=None,
            background_doping_ion=None,
            orientation="100",
        ),
        "box": LayerLevel(
            name=None,
            layer=(99999, 0),
            thickness=3.0,
            thickness_tolerance=None,
            zmin=-3.0,
            zmin_tolerance=None,
            material="sio2",
            sidewall_angle=0.0,
            sidewall_angle_tolerance=None,
            width_to_z=0.0,
            z_to_bias=None,
            mesh_order=9,
            layer_type="background",
            mode=None,
            into=None,
            resistivity=None,
            bias=None,
            derived_layer=None,
            info={},
            background_doping_concentration=None,
            background_doping_ion=None,
            orientation="100",
        ),
        "core": LayerLevel(
            name=None,
            layer=(1, 0),
            thickness=0.22,
            thickness_tolerance=None,
            zmin=0.0,
            zmin_tolerance=None,
            material="si",
            sidewall_angle=10.0,
            sidewall_angle_tolerance=None,
            width_to_z=0.5,
            z_to_bias=None,
            mesh_order=2,
            layer_type="grow",
            mode=None,
            into=None,
            resistivity=None,
            bias=None,
            derived_layer=None,
            info={"active": True},
            background_doping_concentration=100000000000000.0,
            background_doping_ion="Boron",
            orientation="100",
        ),
        # KNOWN ISSUE: Lumerical 2021 version of Layer Builder does not support dopants in process file
    }
)
SIMULATION_SETTINGS_LUMERICAL_FDTD.port_translation = 1.0
sim = LumericalFdtdSimulation(
    component=taper,
    layerstack=layerstack_lumerical2021,
    convergence_settings=LUMERICAL_FDTD_CONVERGENCE_SETTINGS,
    simulation_settings=SIMULATION_SETTINGS_LUMERICAL_FDTD,
    hide=False,
    run_port_convergence=False,
    run_mesh_convergence=False,
)

sp = sim.write_sparameters(overwrite=True)
print(sp)
