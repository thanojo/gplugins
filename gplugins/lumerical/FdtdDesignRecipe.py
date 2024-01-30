import numpy as np

from typing import List, Optional

import gdsfactory as gf
from gdsfactory import Component
from gdsfactory.path import hashlib
from gdsfactory.generic_tech.simulation_settings import SIMULATION_SETTINGS_LUMERICAL_FDTD, SimulationSettingsLumericalFdtd
from gplugins.design_recipe import DesignRecipe
from gplugins.lumerical.Lumerical import DEBUG_LUMERICAL, Engine, draw_geometry
from gdsfactory.config import PathType, __version__, logger
from gdsfactory.pdk import ComponentSpec, Layer, LayerStack, get_layer_stack

import lumapi
from lumapi import Lumerical


class FdtdDesignRecipe(DesignRecipe):
    sim_settings: SimulationSettingsLumericalFdtd
    session: Lumerical

    # The s-parameters for the optical ports of `self.component`
    sp: np.ndarray

    # TODO specify subset of ports to run sparam sweep on?
    def __init__(self,
                 component: Component,
                 dependencies: List[DesignRecipe] = [],
                 sim_settings: SimulationSettingsLumericalFdtd =
                 SIMULATION_SETTINGS_LUMERICAL_FDTD,
                 layer_stack: LayerStack = get_layer_stack(),
                 session: Optional[Lumerical] = None):
        super().__init__(component, dependencies)
        self.sim_settings = sim_settings
        if session is None:
            self.session = lumapi.FDTD(hide=not DEBUG_LUMERICAL)
        self.layer_stack = layer_stack
        self.sp = np.zeros(0)

    def __hash__(self) -> int:
        h = hashlib.sha1()
        h.update(str(super().__hash__()).encode('utf-8'))
        h.update(str(self.sim_settings).encode('utf-8'))
        return int.from_bytes(h.digest(), 'big')

    def eval(self, force_rerun_all=False) -> bool:
        """
        After evaluating dependencies, draw `self.component` in Lumerical FDTD
        and extract the S-parameters of its optical ports.
        S-parameters are stored `self.sp`
        """
        # TODO figure out how we want to write these out to a file/database

        # Evaluate dependencies
        success = super().eval(force_rerun_all)
        if (success):
            # TODO eval fdtd
            success = draw_geometry(
                  self.component,
                  self.session,
                  self.sim_settings,
                  self.layer_stack,
                  Engine.FDTD)
            logger.warning("TODO actually evaluate FDTD")

# TODO find some way to automatically hook into eval()
# and do this at the end? can maybe use decorators?
            self.last_hash = hash(self)

        return success
