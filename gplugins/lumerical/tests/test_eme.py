from pathlib import Path

import gdsfactory as gf

from gplugins.lumerical.config import DEBUG_LUMERICAL
from gplugins.lumerical.eme import LumericalEmeSimulation


def test_lumerical_eme_simulation_setup():
    from functools import partial

    from gdsfactory.components.taper_cross_section import taper_cross_section

    xs_wg = partial(
        gf.cross_section.cross_section,
        layer=(1, 0),
        width=0.5,
    )

    xs_wg_wide = partial(
        gf.cross_section.cross_section,
        layer=(1, 0),
        width=2.0,
    )

    taper = taper_cross_section(
        cross_section1=xs_wg, cross_section2=xs_wg_wide, length=5
    )

    sim = LumericalEmeSimulation(
        taper,
        run_mode_convergence=True,
        run_overall_convergence=True,
        hide=not DEBUG_LUMERICAL,
        dirpath=Path(__file__).resolve().parent / "test_runs",
    )

    sim.plot_length_sweep()
