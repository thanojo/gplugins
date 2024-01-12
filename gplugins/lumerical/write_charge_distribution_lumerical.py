"""Write charge distribution with Lumerical CHARGE."""
from __future__ import annotations

import shutil
import time
from typing import TYPE_CHECKING

import gdsfactory as gf
import numpy as np
import yaml
from gdsfactory.config import __version__, logger
from gdsfactory.generic_tech.simulation_settings import (
    SIMULATION_SETTINGS_LUMERICAL_FDTD,
    SimulationSettingsLumericalFdtd,
)
from gdsfactory.pdk import get_layer_stack
from gdsfactory.technology import LayerStack

from gplugins.common.utils.get_sparameters_path import (
    get_sparameters_path_lumerical as get_sparameters_path,
)

from gplugins.lumerical.write_sparameters_lumerical import set_material

if TYPE_CHECKING:
    from gdsfactory.typings import ComponentSpec, MaterialSpec, PathType

run_false_warning = """
You have passed run=False to debug the simulation

run=False returns the simulation session for you to debug and make sure it is correct

To compute the Sparameters you need to pass run=True
"""

# It seems like CHARGE doesn't start with any materials defined, so we define them here
def add_charge_material(s:object, name: str, em:Optional[str], ct:Optional[str]):
   s.addmodelmaterial();
   s.set("name",name)
   #s.set("color",[0.85, 0, 0, 1]) # red
   if em:
       s.addmaterialproperties("EM",em)  # importing from optical material database

   if ct:
       s.select(f"materials::{name}")
       s.addmaterialproperties("CT",ct)  # importing from electrical material database

def add_charge_materials(s: object):
    add_charge_material(s,"Si (Silicon) - Palik","Si (Silicon) - Palik","Si (Silicon)")
    add_charge_material(s,"SiO2 (Glass) - Palik","SiO2 (Glass) - Palik", "SiO2 (Glass) - Sze")
    logger.warning("TODO find palik glass characterization, using Sze for now") # just because the mapping says palik
    add_charge_material(s,"Si3N4 (Silicon Nitride) - Phillip","Si3N4 (Silicon Nitride) - Phillip", "Si3N4 (Silicon Nitride) - Sze")
    add_charge_material(s, "W (tungsten) - Palik",   "W (tungsten) - Palik", None)
    add_charge_material(s, "Cu (copper) - CRC", None, "Cu (copper) - CRC")
    add_charge_material(s, "Air", None, "Air")



def write_charge_distribution_lumerical(
    component: ComponentSpec,
    session: object | None = None,
    run: bool = True,
    overwrite: bool = False,
    dirpath: PathType | None = None,
    layer_stack: LayerStack | None = None,
    simulation_settings: SimulationSettingsLumericalFdtd = SIMULATION_SETTINGS_LUMERICAL_FDTD,
    material_name_to_lumerical: dict[str, MaterialSpec] | None = None,
    delete_fsp_files: bool = True,
    xmargin: float = 0,
    ymargin: float = 0,
    xmargin_left: float = 0,
    xmargin_right: float = 0,
    ymargin_top: float = 0,
    ymargin_bot: float = 0,
    zmargin: float = 1.0,
    **settings,
) -> np.ndarray:
    r"""Finds and returns the charge distribution for a component and biased contacts Lumerical DEVICE/CHARGE.

    If simulation exists it returns the charge profile (3d n and p carrier concentrations TODO verify that's what it is)  directly unless overwrite=True
    which forces a re-run of the simulation

    Writes charge in TODO some format

    For your Fab technology you can overwrite

    - simulation_settings
    - dirpath
    - layerStack

    converts gdsfactory units (um) to Lumerical units (m)

    Disclaimer: This is probably wrong in some way, not sure tho

    Args:
        component: Component to simulate.
        session: you can pass a session=lumapi.DEVICE() or it will create one.
        run: True runs Lumerical, False only draws simulation.
        overwrite: run even if simulation results already exists.
        dirpath: directory to store sparameters in npz.
            Defaults to active Pdk.sparameters_path.
        layer_stack: contains layer to thickness, zmin and material.
            Defaults to active pdk.layer_stack.
        simulation_settings: dataclass with all simulation_settings.
        material_name_to_lumerical: alias to lumerical material's database name
            or refractive index.
            translate material name in LayerStack to lumerical's database name.
        delete_fsp_files: deletes lumerical fsp files after simulation.
        xmargin: left/right distance from component to PML.
        xmargin_left: left distance from component to PML.
        xmargin_right: right distance from component to PML.
        ymargin: left/right distance from component to PML.
        ymargin_top: top distance from component to PML.
        ymargin_bot: bottom distance from component to PML.
        zmargin: thickness for cladding above and below core.

    Keyword Args:
        background_material: for the background.
        port_margin: on both sides of the port width (um).
        port_height: port height (um).
        port_extension: port extension (um).
        mesh_accuracy: 2 (1: coarse, 2: fine, 3: superfine).
        wavelength_start: 1.2 (um).
        wavelength_stop: 1.6 (um).
        wavelength_points: 500.
        simulation_time: (s) related to max path length 3e8/2.4*10e-12*1e6 = 1.25mm.
        simulation_temperature: in kelvin (default = 300).
        frequency_dependent_profile: computes mode profiles for different wavelengths.
        field_profile_samples: number of wavelengths to compute field profile.


    .. code::

         top view
              ________________________________
             |                               |
             | xmargin                       | port_extension
             |<------>          port_margin ||<-->
          o2_|___________          _________||_o3
             |           \        /          |
             |            \      /           |
             |             ======            |
             |            /      \           |
          o1_|___________/        \__________|_o4
             |   |                           |
             |   |ymargin                    |
             |   |                           |
             |___|___________________________|

        side view
              ________________________________
             |                               |
             |                               |
             |                               |
             |ymargin                        |
             |<---> _____         _____      |
             |     |     |       |     |     |
             |     |     |       |     |     |
             |     |_____|       |_____|     |
             |       |                       |
             |       |                       |
             |       |zmargin                |
             |       |                       |
             |_______|_______________________|



    Return:
        Sparameters np.ndarray (wavelengths, o1@0,o1@0, o1@0,o2@0 ...)
            suffix `a` for angle in radians and `m` for module.

    """
    component = gf.get_component(component)
    sim_settings = dict(simulation_settings)

    layer_stack = layer_stack or get_layer_stack()

    layer_to_thickness = layer_stack.get_layer_to_thickness()
    layer_to_zmin = layer_stack.get_layer_to_zmin()
    layer_to_material = layer_stack.get_layer_to_material()

    if hasattr(component.info, "simulation_settings"):
        sim_settings |= component.info.simulation_settings
        logger.info(
            f"Updating {component.name!r} sim settings {component.simulation_settings}"
        )
    for setting in settings:
        if setting not in sim_settings:
            raise ValueError(
                f"Invalid setting {setting!r} not in ({list(sim_settings.keys())})"
            )

    sim_settings.update(**settings)
    ss = SimulationSettingsLumericalFdtd(**sim_settings)

    component_with_booleans = layer_stack.get_component_with_derived_layers(component)
    component_with_padding = gf.add_padding_container(
        component_with_booleans,
        default=0,
        top=ymargin or ymargin_top,
        bottom=ymargin or ymargin_bot,
        left=xmargin or xmargin_left,
        right=xmargin or xmargin_right,
    )

    component_extended = gf.components.extend_ports(
        component_with_padding, length=ss.distance_monitors_to_pml
    )

    ports = component.get_ports_list(port_type="optical")

    component_extended_beyond_pml = gf.components.extension.extend_ports(
        component=component_extended, length=ss.port_extension
    )
    component_extended_beyond_pml.name = "top"
    gdspath = component_extended_beyond_pml.write_gds()

    filepath_npz = get_sparameters_path(
        component=component,
        dirpath=dirpath,
        layer_stack=layer_stack,
        **settings,
    )
    filepath = filepath_npz.with_suffix(".dat")
    filepath_sim_settings = filepath.with_suffix(".yml")
    filepath_fsp = filepath.with_suffix(".fsp")
    fspdir = filepath.parent / f"{filepath.stem}_s-parametersweep"

    if run and filepath_npz.exists() and not overwrite:
        logger.info(f"Reading Sparameters from {filepath_npz.absolute()!r}")
        return np.load(filepath_npz)

    if not run and session is None:
        print(run_false_warning)

    logger.info(f"Writing Sparameters to {filepath_npz.absolute()!r}")
    x_min = (component_extended.xmin - xmargin) * 1e-6
    x_max = (component_extended.xmax + xmargin) * 1e-6
    y_min = (component_extended.ymin - ymargin) * 1e-6
    y_max = (component_extended.ymax + ymargin) * 1e-6

    layers_thickness = [
        layer_to_thickness[layer]
        for layer in component_with_booleans.get_layers()
        if layer in layer_to_thickness
    ]
    if not layers_thickness:
        raise ValueError(
            f"no layers for component {component.get_layers()}"
            f"in layer stack {layer_stack}"
        )
    layers_zmin = [
        layer_to_zmin[layer]
        for layer in component_with_booleans.get_layers()
        if layer in layer_to_zmin
    ]
    component_thickness = max(layers_thickness)
    component_zmin = min(layers_zmin)

    z = (component_zmin + component_thickness) / 2 * 1e-6
    z_span = (2 * zmargin + component_thickness) * 1e-6

    x_span = x_max - x_min
    y_span = y_max - y_min

    sim_settings.update(dict(layer_stack=layer_stack.to_dict()))

    sim_settings = dict(
        simulation_settings=sim_settings,
        component=component.to_dict(),
        version=__version__,
    )

    logger.info(
        f"Simulation size = {x_span*1e6:.3f}, {y_span*1e6:.3f}, {z_span*1e6:.3f} um"
    )

    # from pprint import pprint
    # filepath_sim_settings.write_text(yaml.dump(sim_settings))
    # print(filepath_sim_settings)
    # pprint(sim_settings)
    # return

    try:
        import lumapi
    except ModuleNotFoundError as e:
        print(
            "Cannot import lumapi (Python Lumerical API). "
            "You can add set the PYTHONPATH variable or add it with `sys.path.append()`"
        )
        raise e
    except OSError as e:
        raise e

    start = time.time()
    s = session or lumapi.DEVICE(hide=False)
    s.newproject()
    #s.deleteall()
    s.clear()
    add_charge_materials(s)

    # add si and 
    print( component_with_booleans.get_layers())

    print("adding cladding...")
    s.addrect(
        x_min=x_min,
        x_max=x_max,
        y_min=y_min,
        y_max=y_max,
        z=z,
        mesh_order=3,
        z_span=z_span,
        name="clad",
    )
    s.set("alpha", .1)

    material_name_to_lumerical_new = material_name_to_lumerical or {}
    material_name_to_lumerical = ss.material_name_to_lumerical.copy()
    material_name_to_lumerical.update(**material_name_to_lumerical_new)

    # TODO propogate this to gdsfactory/generic_tech/simulation_settings.py, it's not process specific
    material_name_to_lumerical["tungsten"] = "W (tungsten) - Palik"
    material_name_to_lumerical["cu"] = "Cu (copper) - CRC"
    material_name_to_lumerical["air"] = "Air"
    print(material_name_to_lumerical)

    material = material_name_to_lumerical[ss.background_material]

    set_material(session=s, structure="clad", material=material)

    component_layers = component_with_booleans.get_layers()

    print(component_layers)
    for layer, thickness in layer_to_thickness.items():
        logger.info(f"handling layer {layer}");
        if layer not in component_layers:
            logger.info(f"skipping layer {layer} because it isn't present in the component")
            continue

        if layer not in layer_to_material:
            raise ValueError(f"{layer} not in {layer_to_material.keys()}")

        material_name = layer_to_material[layer]
        if material_name not in material_name_to_lumerical:
            raise ValueError(
                f"{material_name!r} not in {list(material_name_to_lumerical.keys())}"
            )
        material = material_name_to_lumerical[material_name]

        if layer not in layer_to_zmin:
            raise ValueError(f"{layer} not in {list(layer_to_zmin.keys())}")

        zmin = layer_to_zmin[layer]
        zmax = zmin + thickness
        z = (zmax + zmin) / 2

        s.gdsimport(str(gdspath), "top", f"{layer[0]}:{layer[1]}")
        layername = f"GDS_LAYER_{layer[0]}:{layer[1]}"
        s.setnamed(layername, "z", z * 1e-6)
        s.setnamed(layername, "z span", thickness * 1e-6)
        set_material(session=s, structure=layername, material=material)
        logger.info(f"adding {layer}, thickness = {thickness} um, zmin = {zmin} um ")


    if run:
        s.save(str(filepath_fsp))
        print("TODO elam figure add sweep + run for CHARGE")


    filepath_sim_settings.write_text(yaml.dump(sim_settings))
    return s

