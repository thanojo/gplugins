from pydantic import BaseModel
from typing import Literal
from collections import OrderedDict

MODE_WAVEGUIDE_SETTINGS = OrderedDict({
    "name": "WAVEGUIDE",
    "ldf filename": "",
    "length": 1e-6,  # meters
    "loss": 0,  # dB/m
    "excess loss": 0,  # dB
    "load mode profile": True,
})

STRAIGHT_WAVEGUIDE_SETTINGS = OrderedDict({
    "name": "WAVEGUIDE",
    "frequency": 1.93414e14,  # Hz
    "label 1": "TE",
    "effective index 1": 2.4,
    "loss 1": 0,  # dB/m
    "group index 1": 3.4,
    "dispersion 1": 0,  # s/m/m
    "dispersion slope 1": 0,  # s/m^2/m
    "label 2": "TM",
    "effective index 2": 2.4,
    "loss 2": 0,  # dB/m
    "group index 2": 3.4,
    "dispersion 2": 0,  # s/m/m
    "dispersion slope 2": 0,  # s/m^2/m
})

OPTICAL_MODULATOR_MEASURED_SETTINGS = OrderedDict({
    "name": "PHASESHIFTER",
    "frequency": 1.93414e14, # Hz,
    "length": 1e-6, # m
    "input parameter": "table",
    "load from file": False,
    "measurement type": "effective index",
    "measurement": None,
})


OPTICAL_MODULATOR_LOOKUP_TABLE_SETTINGS = OrderedDict({
    "name": "PHASESHIFTER",
    "load from file": True,
    "filename": "",
    "table": None,
})


OPTICAL_NPORT_SPARAM_SETTINGS = OrderedDict({
    "name": "SPARAM",
    "load from file": True,
    "s parameters filename": "",
    "passivity": "ignore",
    "reciprocity": "ignore",
})

class LumericalCompactModel(BaseModel):
    model: Literal["Straight Waveguide",
                   "MODE Waveguide",
                   "Optical Modulator Measured",
                   "Optical Modulator Lookup Table",
                   "Optical N Port S-Parameter",
                    ] = "MODE Waveguide"
    settings: dict = MODE_WAVEGUIDE_SETTINGS

    class Config:
        arbitrary_types_allowed = True

WAVEGUIDE_COMPACT_MODEL = LumericalCompactModel(model="MODE Waveguide",
                                                settings=MODE_WAVEGUIDE_SETTINGS)

PHASESHIFTER_COMPACT_MODEL = LumericalCompactModel(model="Optical Modulator Measured",
                                                   settings=OPTICAL_MODULATOR_MEASURED_SETTINGS)

SPARAM_COMPACT_MODEL = LumericalCompactModel(model="Optical N Port S-Parameter",
                                             settings=OPTICAL_NPORT_SPARAM_SETTINGS)