import numpy as np

# Used to display Lumerical GUIs, log status, and plot results
DEBUG_LUMERICAL = False

# Enable dopant generation in lbr process file.
# Set this to False if using Lumerical versions 2021 or prior (Layer Builder
# has issues with dopants for these versions of Lumerical).
ENABLE_DOPING = True

# Opacity in DEVICE
OPACITY = 0.4
# Format for colors (R, G, B, opacity) for DEVICE
MATERIAL_COLORS = [
    np.array([1, 0, 0, OPACITY]),
    np.array([1, 0.5, 0, OPACITY]),
    np.array([1, 1, 0, OPACITY]),
    np.array([0.5, 1, 0, OPACITY]),
    np.array([0, 1, 1, OPACITY]),
    np.array([0, 0.5, 1, OPACITY]),
    np.array([0, 0, 1, OPACITY]),
    np.array([0.5, 0, 1, OPACITY]),
    np.array([1, 0, 1, OPACITY]),
] * 10

um = 1e-6
cm = 1e-2
marker_list = [
    "o",
    "v",
    "^",
    "<",
    ">",
    "1",
    "2",
    "3",
    "4",
    "s",
    "p",
    "P",
    "*",
    "h",
    "+",
    "X",
    "D",
] * 10
