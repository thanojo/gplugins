import xml.etree.ElementTree as ET
from pathlib import Path
from xml.dom import minidom
from xml.etree.ElementTree import Element, SubElement

from gdsfactory.config import logger
from gdsfactory.pdk import get_layer_stack
from gdsfactory.technology import LayerStack
from gdsfactory.typings import PathType

from gplugins.lumerical.config import ENABLE_DOPING

um = 1e-6


def layerstack_to_lbr(
    material_map: dict[str, str],
    layerstack: LayerStack | None = None,
    dirpath: PathType | None = "",
) -> None:
    """
    Generate an XML file representing a Lumerical Layer Builder process file based on provided material map.

    Args:
        material_map: A dictionary mapping materials used in the layer stack to Lumerical materials.
        layerstack: Layer stack that has info on layer names, layer numbers, thicknesses, etc.
        dirpath: Directory to save process file (process.lbr)

    Returns:
        Process file path

    Notes:
        This function generates an XML file representing a Layer Builder file for Lumerical, based on the provided the active PDK
        and material map. It creates 'process.lbr' in the current working directory, containing layer information like name,
        material, thickness, sidewall angle, and other properties specified in the layer stack.
    """
    layerstack = layerstack or get_layer_stack()

    layer_builder = Element("layer_builder")

    process_name = SubElement(layer_builder, "process_name")
    process_name.text = "process"

    layers = SubElement(layer_builder, "layers")
    doping_layers = SubElement(layer_builder, "doping_layers")
    for layer_name, layer_info in layerstack.to_dict().items():
        if layer_info["layer_type"] == "grow":
            process = "Grow"
        elif layer_info["layer_type"] == "background":
            process = "Background"
        elif layer_info["layer_type"] == "doping":
            process = "Implant"
        else:
            logger.warning(
                f'"{layer_info["layer_type"]}" layer type not supported for "{layer_name}" in Lumerical. Skipping in LBR process file generation.'
            )
            process = "Grow"

        ### Set optical and metal layers
        if process == "Grow" or process == "Background":
            layer = SubElement(layers, "layer")

            # Default params
            layer_params = {
                "enabled": "1",
                "pattern_alpha": "0.8",
                "start_position_auto": "0",
                "background_alpha": "0.3",
                "pattern_material_index": "0",
                "material_index": "0",
                "name": layer_name,
                "layer_name": f'{layer_info["layer"][0]}:{layer_info["layer"][1]}',
                "start_position": f'{layer_info["zmin"] * um}',
                "thickness": f'{layer_info["thickness"] * um}',
                "process": f"{process}",
                "sidewall_angle": f'{90 - layer_info["sidewall_angle"]}',
                "pattern_growth_delta": f"{layer_info['bias'] * um}"
                if layer_info["bias"]
                else "0",
            }

            for param, val in layer_params.items():
                layer.set(param, val)

            if process == "Grow":
                layer.set(
                    "pattern_material",
                    f'{material_map.get(layer_info["material"], "")}',
                )
            elif process == "Background":
                layer.set("material", f'{material_map.get(layer_info["material"], "")}')

        if (process == "Implant" or process == "Background") and ENABLE_DOPING:
            ### Set doping layers
            # KNOWN ISSUE: If a metal or optical layer has the same name as a doping layer, Layer Builder will not compile
            # the process file correctly and the doping layer will not appear. Therefore, doping layer names MUST be unique.
            # FIX: Appending "_doping" to name

            # KNOWN ISSUE: If the 'process' is not 'Background' or 'Implant' for dopants, this will crash CHARGE upon importing process file.
            # FIX: Ensure process is Background or Implant before proceeding to create entry

            # KNOWN ISSUE: Dopant must be either 'p' or 'n'. Anything else will cause CHARGE to crash upon importing process file.
            # FIX: Raise ValueErrorr when dopant is specified incorrectly
            if layer_info.get(
                "background_doping_concentration", False
            ) and layer_info.get("background_doping_ion", False):
                doping_layer = SubElement(doping_layers, "layer")
                doping_params = {
                    "z_surface_positions": f'{layer_info["zmin"] * um}',
                    "distribution_function": "Gaussian",
                    "phi": "0",
                    "lateral_scatter": "2e-08",
                    "range": f"{layer_info['thickness'] * um}",
                    "theta": "0",
                    "mask_layer_number": f'{layer_info["layer"][0]}:{layer_info["layer"][1]}',
                    "kurtosis": "0",
                    "process": f"{process}",
                    "skewness": "0",
                    "straggle": "4.9999999999999998e-08",
                    "concentration": f"{layer_info['background_doping_concentration']}",
                    "enabled": "1",
                    "name": f"{layer_name}_doping",
                }
                for param, val in doping_params.items():
                    doping_layer.set(param, val)

                if (
                    layer_info["background_doping_ion"] == "n"
                    or layer_info["background_doping_ion"] == "p"
                ):
                    doping_layer.set("dopant", layer_info["background_doping_ion"])
                else:
                    raise ValueError(
                        f'Dopant must be "p" or "n". Got {layer_info["background_doping_ion"]}.'
                    )

    # If no doping layers exist, delete element
    if len(doping_layers) == 0:
        layer_builder.remove(doping_layers)

    # Prettify XML
    rough_string = ET.tostring(layer_builder, "utf-8")
    reparsed = minidom.parseString(rough_string)
    xml_str = reparsed.toprettyxml(indent="  ")

    if dirpath:
        process_file_path = Path(str(dirpath)) / "process.lbr"
    else:
        process_file_path = Path(__file__).resolve().parent / "process.lbr"
    with open(str(process_file_path), "w") as f:
        f.write(xml_str)

    return process_file_path


def draw_geometry(
    session: object,
    gdspath: PathType,
    process_file_path: PathType,
) -> None:
    """
    Draw geometry in Lumerical simulator

    Parameters:
        session: Lumerical session
        gdspath: GDS path
        process_file_path: Process file path
    """
    s = session
    s.addlayerbuilder()
    s.set("x", 0)
    s.set("y", 0)
    s.set("z", 0)
    s.loadgdsfile(str(gdspath))
    try:
        s.loadprocessfile(str(process_file_path))
    except Exception as err:
        raise Exception(
            f"{err}\nProcess file cannot be imported. Likely causes are dopants in the process file or syntax errors."
        ) from err
