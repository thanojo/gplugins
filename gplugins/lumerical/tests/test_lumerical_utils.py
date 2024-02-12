from gplugins.lumerical.utils import to_lbr
from gdsfactory.technology.layer_stack import LayerStack, LayerLevel
from gdsfactory.config import logger
from gplugins.lumerical.config import DEBUG_LUMERICAL


def test_to_lbr():
    # Inputs
    layer_map = {
        "si": "Si (Silicon) - Palik",
        "sio2": "SiO2 (Glass) - Palik",
        "sin": "Si3N4 (Silicon Nitride) - Phillip",
        "TiN": "TiN - Palik",
        "Aluminum": "Al (Aluminium) Palik",
    }

    # Create layerstack
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

    layerstack_lumerical2023 = LayerStack(
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
            "substrate": LayerLevel(
                name=None,
                layer=(99999, 0),
                thickness=10.0,
                thickness_tolerance=None,
                zmin=-13.0,
                zmin_tolerance=None,
                material="si",
                sidewall_angle=0.0,
                sidewall_angle_tolerance=None,
                width_to_z=0.0,
                z_to_bias=None,
                mesh_order=101,
                layer_type="background",
                mode=None,
                into=None,
                resistivity=None,
                bias=None,
                derived_layer=None,
                info={},
                background_doping_concentration=100000000000000.0,
                background_doping_ion="p",
                orientation="100",
            ),
            "n++": LayerLevel(
                name="n++",
                layer=(5, 0),
                thickness=0.22,
                zmin=0,
                layer_type="doping",
                background_doping_concentration=1e17,
                background_doping_ion="n",
            ),
        }
    )

    # Create LBR process file
    process_file_lumerical2021 = to_lbr(layer_map, layerstack=layerstack_lumerical2021)

    # Check process file in Lumerical MODE
    try:
        import lumapi
    except Exception as err:
        assert (
            False
        ), f"{err}\nUnable to import lumapi. Check sys.path for location to lumapi.py."

    success = False
    message = ""
    try:
        mode = lumapi.MODE(hide=not DEBUG_LUMERICAL)
        mode.addlayerbuilder()
        mode.loadprocessfile(str(process_file_lumerical2021))
        success = success or True
        message += "\nSUCCESS: Passives only process file successfully loaded."
    except Exception as err:
        success = success or False
        message += f"\n{err}\nWARNING: Passives only process file unsucessfully loaded."
    mode.close()

    # Create LBR process file
    process_file_lumerical2023 = to_lbr(layer_map, layerstack=layerstack_lumerical2023)

    try:
        mode = lumapi.MODE(hide=not DEBUG_LUMERICAL)
        mode.addlayerbuilder()
        mode.loadprocessfile(str(process_file_lumerical2023))
        success = success or True
        message += "\nSUCCESS: Passives and dopants process file successfully loaded."
    except Exception as err:
        success = success or False
        message += f"\nWARNING: {err}\nPassives and dopants process file unsucessfully loaded. Lumerical 2021 version does not support dopants."
    mode.close()

    if success:
        message += (
            "\nTest passes if one of passives only or passives + dopants process files work."
            + "\nThis is used to capture Lumerical versions 2021 that only work with passive only process files "
            + "and subsequent versions of Lumerical that support both passives and dopants."
        )
    logger.info(message)
    assert success, message
