from typing import Literal

from pydantic import BaseModel


class ConvergenceSettingsLumericalEme(BaseModel):
    """
    Lumerical EME convergence settings

    Parameters:
        passes: Number of passes / simulations sweeping a convergence parameter before checking for convergence
        sparam_diff: Maximum difference in sparams after x passes sweeping a convergence parameter. Used to check for convergence.
    """

    passes: int = 5
    sparam_diff: float = 0.01

    class Config:
        arbitrary_types_allowed = True


LUMERICAL_EME_CONVERGENCE_SETTINGS = ConvergenceSettingsLumericalEme()


class ConvergenceSettingsLumericalFdtd(BaseModel):
    """
    Lumerical FDTD convergence settings

    Parameters:
        sparam_diff: Maximum difference in sparams after x passes sweeping a convergence parameter. Used to check for convergence.
    """

    sparam_diff: float = 0.005

    class Config:
        arbitrary_types_allowed = True


LUMERICAL_FDTD_CONVERGENCE_SETTINGS = ConvergenceSettingsLumericalFdtd()


class ConvergenceSettingsLumericalCharge(BaseModel):
    """
    Lumerical CHARGE convergence settings

    Parameters:
        solver_type: Convergence solver type
        dc_update_mode: Chooses mode for solving system of equations for electrostatic potential, charge density, and heat transport
        global_iteration_limit: Maximum number of iterations in each solver before convergence has failed
        gradient_mixing: Type of gradient mixing to aid in convergence
        convergence_criteria: Criteria for determining simulation convergence
        update_abs_tol: Absolute value used to determine convergence
        update_rel_tol: Relative value used to determine convergence
        residual_abs_tol: Absolute value of residuals used to determine convergence

    """

    solver_type: Literal["gummel", "newton"] = "gummel"
    dc_update_mode: Literal[
        "self consistent", "charge", "electrostatic"
    ] = "self consistent"
    global_iteration_limit: int = 40
    gradient_mixing: Literal["disabled", "fast", "conservative"] = "disabled"
    convergence_criteria: Literal["update", "residual", "any"] = "update"
    update_abs_tol: float = 1e-4
    update_rel_tol: float = 1e-6
    residual_abs_tol: float = 1e-4


LUMERICAL_CHARGE_CONVERGENCE_SETTINGS = ConvergenceSettingsLumericalCharge()


class ConvergenceSettingsLumericalMode(BaseModel):
    """
    Lumerical MODE convergence settings

    Parameters:
        neff_diff: Effective index variation that is considered stable.
        ng_diff: Group index variation that is considered stable.
        pol_diff: Polarization percentage variation that is considered stable.
        mesh_cell_step: Number of mesh cells to increase during mesh convergence sweep.
        mesh_stable_limit: Number of data points that are stable to be considered for mesh convergence.
        field_stable_limit: Number of data points that are stable to be considered for field intensity convergence.
    """
    neff_diff: float = 0.01
    ng_diff: float = 0.01
    pol_diff: float = 0.001  # 0.01 = 1%

    mesh_cell_step: int = 1
    mesh_stable_limit: int = 10

    field_stable_limit: int = 3

    class Config:
        arbitrary_types_allowed = True

LUMERICAL_MODE_CONVERGENCE_SETTINGS = ConvergenceSettingsLumericalMode()