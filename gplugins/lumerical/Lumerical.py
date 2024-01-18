import time
import shutil
import yaml

import lumapi 
import numpy as np
import gdsfactory as gf

from gdsfactory.pdk import ComponentSpec, get_layer_stack
from gplugins.common.utils.get_sparameters_path import (
    get_sparameters_path_lumerical as get_sparameters_path,
)
from gdsfactory.typings import Component, LayerStack, MaterialSpec
from gdsfactory.config import PathType, __version__, logger
from gdsfactory.generic_tech.simulation_settings import * # TODO wildcard imports are bad

from gplugins.lumerical.write_charge_distribution_lumerical import add_charge_materials, add_doping, write_charge_distribution_lumerical
from gplugins.lumerical.write_sparameters_lumerical import set_material

run_false_warning = """
You have passed run=False to debug the simulation

run=False returns the simulation session for you to debug and make sure it is correct

To compute the Sparameters you need to pass run=True
"""


class Lumerical:
    component = Component()
    material_name_to_lumerical: dict[str, str]  = {}
    layer_stack = get_layer_stack()
    device_session = None
    fdtd_session = None
    device_component = None
    fdtd_component = None

    def __init__(self,
        component: Component,
        layer_stack: LayerStack,
        material_name_to_lumerical: dict[str, str] = {}):
        self.material_name_to_lumerical = material_name_to_lumerical_default
        self.material_name_to_lumerical.update(**material_name_to_lumerical)
        # TODO this is upstreamed, update gdsfactory version
        self.material_name_to_lumerical["tungsten"] = "W (Tungsten) - Palik"
        self.material_name_to_lumerical["cu"] = "Cu (Copper) - CRC"
        self.material_name_to_lumerical["air"] = "Air"

        self.layer_stack = layer_stack
        self.component = component

    def draw_geometry(self,
        session: object | None,
        sim_settings: SimulationSettingsLumerical,
        engine="CHARGE",
        xmargin: float = 0,
        ymargin: float = 0,
        xmargin_left: float = 0,
        xmargin_right: float = 0,
        ymargin_top: float = 0,
        ymargin_bot: float = 0,
        zmargin: float = 0):

        if engine == "FDTD":
            if (not session):
                session = lumapi.FDTD()
        elif engine == "CHARGE":
            if (not session):
                session = lumapi.DEVICE()
            add_charge_materials(session) # DEVICE needs materials to be defined manually for some reason


        component_with_booleans = self.layer_stack.get_component_with_derived_layers(self.component)
        component_layers = component_with_booleans.get_layers()
        layer_to_thickness = self.layer_stack.get_layer_to_thickness()
        layer_to_zmin = self.layer_stack.get_layer_to_zmin()
        layer_to_material = self.layer_stack.get_layer_to_material()



        # pad boundaries so the device doesn't interact with lumerical boundary conditions
        component_with_padding = gf.add_padding_container(
            component_with_booleans,
            default=0,
            top=ymargin or ymargin_top,
            bottom=ymargin or ymargin_bot,
            left=xmargin or xmargin_left,
            right=xmargin or xmargin_right,
        )

        component_extended = gf.components.extend_ports(
            component_with_padding, length=sim_settings.distance_monitors_to_pml
        )

        port_type = "optical" if engine == "FDTD"else "electrical"
        ports = self.component.get_ports_list(port_type=port_type)
        if not ports:
            pass
            # raise ValueError(f"{self.component.name!r} does not have any {port_type} ports")

        component_extended_beyond_pml = gf.components.extension.extend_ports(
            component=component_extended, length=sim_settings.port_extension
        )
        component_extended_beyond_pml.name = "top"
        gdspath = component_extended_beyond_pml.write_gds()

        layers_zmin = [
            layer_to_zmin[layer]
            for layer in component_with_booleans.get_layers()
            if layer in layer_to_zmin
        ]
        layers_thickness = [
            layer_to_thickness[layer]
            for layer in component_with_booleans.get_layers()
            if layer in layer_to_thickness
        ]

        last_layer_idx = np.argmax(layers_zmin)
        component_thickness = layers_zmin[last_layer_idx] + \
            layers_thickness[last_layer_idx] - min(layers_zmin)

        component_zmin = min(layers_zmin)
        z = (component_zmin - zmargin) * 1e-6
        z = (component_zmin + component_thickness/2) * 1e-6
        z_span = (2 * zmargin + component_thickness) * 1e-6

        x_min = (component_extended.xmin - xmargin) * 1e-6
        x_max = (component_extended.xmax + xmargin) * 1e-6
        y_min = (component_extended.ymin - ymargin) * 1e-6
        y_max = (component_extended.ymax + ymargin) * 1e-6

        x_span = x_max - x_min
        y_span = y_max - y_min

        # Add simulation region
        if engine == "FDTD":
            # TODO make this idempotent
            session.addfdtd(
                dimension="3D",
                x_min=x_min,
                x_max=x_max,
                y_min=y_min,
                y_max=y_max,
                z=z,
                z_span=z_span,
                mesh_accuracy=sim_settings.mesh_accuracy,
                use_early_shutoff=True,
                simulation_time=sim_settings.simulation_time,
                simulation_temperature=sim_settings.simulation_temperature,
            )
        else:
            session.addchargesolver()

            session.setnamed("simulation region", "name", "CHARGE simulation region")
            session.setnamed("CHARGE simulation region", "dimension", 4)
            session.setnamed("CHARGE simulation region", "x",0)
            session.setnamed("CHARGE simulation region", "y",0)
            session.setnamed("CHARGE simulation region", "z",z)
            session.setnamed("CHARGE simulation region", "x span", x_span)
            session.setnamed("CHARGE simulation region", "y span", y_span)
            session.setnamed("CHARGE simulation region", "z span",z_span)

            session.setnamed("CHARGE","simulation region", "CHARGE simulation region");

        # draw device by importing layer GDSes into lumerical
        for name, layer_obj in self.layer_stack.layers.items():# layer, thickness in layer_to_thickness.items():
            layer = layer_obj.layer
            thickness = layer_obj.thickness

            #layer = layer_obj.
            logger.info(f"handling layer {layer}");
            if layer not in component_layers:
                logger.info(f"skipping layer {layer} because it isn't present in the component")
                continue

            if layer not in layer_to_material:
                raise ValueError(f"{layer} not in {layer_to_material.keys()}")

            material_name = layer_to_material[layer]
            if material_name not in self.material_name_to_lumerical:
                raise ValueError(
                    f"{material_name!r} not in {list(self.material_name_to_lumerical.keys())}"
                )
            material = self.material_name_to_lumerical[material_name]

            if layer not in layer_to_zmin:
                raise ValueError(f"{layer} not in {list(layer_to_zmin.keys())}")

            zmin = layer_to_zmin[layer]
            zmax = zmin + thickness
            z = (zmax + zmin) / 2

            if (layer_obj.background_doping_concentration is not None):
                logger.info(f"adding doping {layer}, thickness = {thickness} um, zmin = {zmin} um ")
                add_doping(session, str(gdspath), name, layer_obj)
            else:
                print("material ", material)
                if (material == "Air" and engine != "DEVICE"):
                    logger.info("Skipping {material} because this is not a DEVICE simulation")
                    continue  # Air is only used in DEVICE sims

                session.gdsimport(
                        str(gdspath),
                        "top", f"{layer[0]}:{layer[1]}")
                layername = f"GDS_LAYER_{layer[0]}:{layer[1]}"
                session.setnamed(layername, "z", z * 1e-6)
                session.setnamed(layername, "z span", thickness * 1e-6)
                set_material(session=session, structure=layername, material=material)
                logger.info(f"adding {layer}, thickness = {thickness} um, zmin = {zmin} um ")

    def add_ports(self, session: object, engine: str, simulation_settings: SimulationSettingsLumerical):
        port_type = "optical" if engine == "FDTD"else "electrical"
        for i, port in enumerate(self.component.get_ports_list(port_type=port_type)):
            zmin = self.layer_stack.get_layer_to_zmin()[port.layer]
            thickness = self.layer_stack.get_layer_to_thickness()[port.layer]
            z = (zmin + thickness) / 2
            zspan = 2 * simulation_settings.port_margin + thickness

            session.addport()
            p = f"FDTD::ports::port {i+1}"
            session.setnamed(p, "x", port.x * 1e-6)
            session.setnamed(p, "y", port.y * 1e-6)
            session.setnamed(p, "z", z * 1e-6)
            session.setnamed(p, "z span", zspan * 1e-6)
            session.setnamed(p, "frequency dependent profile",
                       simulation_settings.frequency_dependent_profile)
            session.setnamed(p, "number of field profile samples",
                       simulation_settings.field_profile_samples)

            deg = int(port.orientation)
            # if port.orientation not in [0, 90, 180, 270]:
            #     raise ValueError(f"{port.orientation} needs to be [0, 90, 180, 270]")

            if -45 <= deg <= 45:
                direction = "Backward"
                injection_axis = "x-axis"
                dxp = 0
                dyp = 2 * simulation_settings.port_margin + port.width
            elif 45 < deg < 90 + 45:
                direction = "Backward"
                injection_axis = "y-axis"
                dxp = 2 * simulation_settings.port_margin + port.width
                dyp = 0
            elif 90 + 45 < deg < 180 + 45:
                direction = "Forward"
                injection_axis = "x-axis"
                dxp = 0
                dyp = 2 * simulation_settings.port_margin + port.width
            elif 180 + 45 < deg < 180 + 45 + 90:
                direction = "Forward"
                injection_axis = "y-axis"
                dxp = 2 * simulation_settings.port_margin + port.width
                dyp = 0

            else:
                raise ValueError(
                    f"port {port.name!r} orientation {port.orientation} is not valid"
                )

            session.setnamed(p, "direction", direction)
            session.setnamed(p, "injection axis", injection_axis)
            session.setnamed(p, "y span", dyp * 1e-6)
            session.setnamed(p, "x span", dxp * 1e-6)
            # s.setnamed(p, "theta", deg)
            session.setnamed(p, "name", port.name)
            # s.setnamed(p, "name", f"o{i+1}")

            logger.info(
                f"port {p} {port.name!r}: at ({port.x}, {port.y}, 0)"
                f"size = ({dxp}, {dyp}, {zspan})"
            )
    def write_sparameters_lumerical(self,
        session: object | None = None,
        overwrite: bool = False,
        run: bool = True,
        dirpath: PathType | None = None,
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
        start_time:w = time.time()

        session = session or lumapi.FDTD(hide=False)

        sim_setting_overrides = dict(simulation_settings)
        sim_setting_overrides.update(**settings)
        sim_setting_overrides.update(dict(layer_stack=self.layer_stack.to_dict()))
        sim_settings = SimulationSettingsLumericalFdtd(**sim_setting_overrides)

        self.draw_geometry(session=session,
                           engine="FDTD",
                           xmargin=xmargin,
                           ymargin=ymargin,
                           xmargin_left=xmargin_left,
                           xmargin_right=xmargin_right,
                           ymargin_top=ymargin_top,
                           ymargin_bot=ymargin_bot,
                           zmargin=zmargin,
                           sim_settings=sim_settings,
                           **settings)
        self.add_ports(session, "FDTD", sim_settings)

        filepath_npz = get_sparameters_path(
            component=self.component,
            dirpath=dirpath,
            layer_stack=self.layer_stack,
            **settings,
        )

        # break early if sparams already exists
        if run and filepath_npz.exists() and not overwrite:
            logger.info(f"Reading Sparameters from {filepath_npz.absolute()!r}")
            return np.load(filepath_npz)

        if not run and session is None:
            print(run_false_warning)

        if filepath_npz.exists() and not overwrite:
            logger.info(f"Reading Sparameters from {filepath_npz.absolute()!r}")
            return np.load(filepath_npz)

        filepath = filepath_npz.with_suffix(".dat")
        filepath_sim_settings = filepath.with_suffix(".yml")
        filepath_fsp = filepath.with_suffix(".fsp")
        fspdir = filepath.parent / f"{filepath.stem}_s-parametersweep"

        session.setglobalsource("wavelength stop", sim_settings.wavelength_stop * 1e-6)
        session.setglobalsource("wavelength start", sim_settings.wavelength_start * 1e-6)
        session.setnamed("FDTD::ports", "monitor frequency points", sim_settings.wavelength_points)

        if run:
            session.save(str(filepath_fsp))
            session.deletesweep("s-parameter sweep")

            session.addsweep(3)
            session.setsweep("s-parameter sweep", "Excite all ports", 0)
            session.setsweep("S sweep", "auto symmetry", True)
            session.runsweep("s-parameter sweep")
            sp = session.getsweepresult("s-parameter sweep", "S parameters")
            session.exportsweep("s-parameter sweep", str(filepath))
            logger.info(f"wrote sparameters to {str(filepath)!r}")

            sp["wavelengths"] = sp.pop("lambda").flatten() * 1e6
            np.savez_compressed(filepath, **sp)

            end = time.time()
            sim_settings.update(compute_time_seconds=end - start_time)
            sim_settings.update(compute_time_minutes=(end - start_time) / 60)
            filepath_sim_settings.write_text(yaml.dump(sim_settings))
            if delete_fsp_files and fspdir.exists():
                shutil.rmtree(fspdir)
                logger.info(
                    f"deleting simulation files in {str(fspdir)!r}. "
                    "To keep them, use delete_fsp_files=False flag"
                )
            logger.info(f"Writing Sparameters to {filepath_npz.absolute()!r}")

            return sp
        return session
    def run_charge(self,
        bias_port: str,
        constant_ports: str,
        bias_voltages: list[float],
        constant_voltage: float =0):
        pass
        #write_charge_distribution_lumerical(self.component,

