from gplugins.lumerical.utils import to_lbr


def test_to_lbr():
    # Inputs
    layer_map = {
        "si": "Si (Silicon) - Palik",
        "sio2": "SiO2 (Glass) - Palik",
        "sin": "Si3N4 (Silicon Nitride) - Phillip",
        "TiN": "TiN - Palik",
        "Aluminum": "Al (Aluminium) Palik",
    }

    # Create LBR process file
    to_lbr(layer_map)

    # Check process file in Lumerical MODE
    try:
        import lumapi
    except Exception as err:
        assert (
            False
        ), f"{err}\nUnable to import lumapi. Check sys.path for location to lumapi.py."
    try:
        mode = lumapi.MODE(hide=False)
        mode.addlayerbuilder()
        mode.loadprocessfile("process.lbr")
        mode.close()
        assert True
    except:
        assert (
            False
        ), f"Unable to import process file (.lbr) into Lumerical. Check process file formatting."
