from gplugins.lumerical.interconnect import create_compact_model
from gplugins.lumerical.compact_models import WAVEGUIDE_COMPACT_MODEL
from pathlib import Path

def test_create_compact_model():
    dirpath = Path(__file__).resolve().parent / "test_runs" / "model_library"
    dirpath.mkdir(parents=True, exist_ok=True)
    WAVEGUIDE_COMPACT_MODEL.settings["name"] = "WAVEGUIDE_TEST_MODEL"
    create_compact_model(model=WAVEGUIDE_COMPACT_MODEL, dirpath=dirpath)

    # Check if the waveguide model is available on file
    model_path = dirpath / f"{WAVEGUIDE_COMPACT_MODEL.settings['name']}.ice"
    if not model_path.resolve().is_file():
        raise FileNotFoundError(f"Could not find {WAVEGUIDE_COMPACT_MODEL.settings['name']}.ice file.")