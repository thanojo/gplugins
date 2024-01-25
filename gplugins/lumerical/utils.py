from gdsfactory.technology import LayerStack
from gdsfactory.typings import PathType
from gdsfactory.pdk import get_layer_stack
from gdsfactory.config import logger
from xml.etree.ElementTree import Element, SubElement
from xml.dom import minidom
import xml.etree.ElementTree as ET
from pathlib import Path

um = 1e-6


def to_lbr(
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
        None

    Notes:
        This function generates an XML file representing a Layer Builder file for Lumerical, based on the provided the active PDK
        and material map. It creates 'process.lbr' in the current working directory, containing layer information like name,
        material, thickness, sidewall angle, and other properties specified in the layer stack. It skips layers that are not
        of type 'grow' or 'background' and logs a warning for each skipped layer.
    """
    layerstack = layerstack or get_layer_stack()

    layer_builder = Element("layer_builder")

    process_name = SubElement(layer_builder, "process_name")
    process_name.text = "process"

    layers = SubElement(layer_builder, "layers")
    for layer_name, layer_info in layerstack.to_dict().items():
        if layer_info["layer_type"] == "grow":
            process = "Grow"
        elif layer_info["layer_type"] == "background":
            process = "Background"
        else:
            logger.warning(
                f'"{layer_info["layer_type"]}" layer type not supported for "{layer_name}" in Lumerical. Skipping in LBR process file generation.'
            )
            continue

        layer = SubElement(layers, "layer")

        # Default params
        layer.set("enabled", "1")
        layer.set("pattern_alpha", "0.8")
        layer.set("start_position_auto", "0")
        layer.set("background_alpha", "0.3")
        layer.set("pattern_material_index", "0")
        layer.set("material_index", "0")

        # Layer specific params
        layer.set("name", layer_name)
        layer.set("layer_name", f'{layer_info["layer"][0]}:{layer_info["layer"][1]}')
        layer.set("start_position", f'{layer_info["zmin"] * um}')
        layer.set("thickness", f'{layer_info["thickness"] * um}')
        layer.set("start_position", f'{layer_info["zmin"] * um}')
        layer.set("pattern_material", f'{material_map.get(layer_info["material"], "")}')
        layer.set("process", f"{process}")
        layer.set("sidewall_angle", f'{90-layer_info["sidewall_angle"]}')
        if layer_info["bias"]:
            layer.set("pattern_growth_delta", f"{layer_info['bias'] * um}")
        else:
            layer.set("pattern_growth_delta", "0")

    # Prettify XML
    rough_string = ET.tostring(layer_builder, "utf-8")
    reparsed = minidom.parseString(rough_string)
    xml_str = reparsed.toprettyxml(indent="  ")

    process_file_path = Path(dirpath) / "process.lbr"
    with open(process_file_path, "w") as f:
        f.write(xml_str)
