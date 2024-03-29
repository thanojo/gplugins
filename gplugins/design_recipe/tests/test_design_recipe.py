from pathlib import Path

import pandas as pd
from gdsfactory import Component
from gdsfactory.components.straight import straight
from gdsfactory.config import logger
from gdsfactory.pdk import LayerStack, get_layer_stack
from pydantic import BaseModel

from gplugins.design_recipe.DesignRecipe import DesignRecipe, Simulation, eval_decorator


class TestSettings(BaseModel):
    parameter1: int = 1
    parameter2: float = 1.2
    parameter3: list | None = None

    class Config:
        arbitrary_types_allowed = True


def test_design_recipe():
    # Sample BaseModel

    class CustomRecipe1(DesignRecipe):
        def __init__(
            self,
            component: Component | None = None,
            layer_stack: LayerStack | None = None,
            dependencies: list[DesignRecipe] | None = None,
            dirpath: Path | None = None,
        ):
            super().__init__(
                cell=component,
                dependencies=dependencies,
                layer_stack=layer_stack,
                dirpath=dirpath,
            )
            self.recipe_setup.test_setup1 = {"a": [1, 2, 3], "b": [4, 3, 2]}
            self.recipe_setup.test_setup2 = [1, 2, 3]

        @eval_decorator
        def eval(self, run_convergence: bool = True) -> bool:
            self.recipe_results.results1 = [5, 4, 3, 2, 1]
            return True

    class CustomRecipe2(DesignRecipe):
        def __init__(
            self,
            component: Component | None = None,
            layer_stack: LayerStack | None = None,
            dependencies: list[DesignRecipe] | None = None,
            dirpath: Path | None = None,
        ):
            super().__init__(
                cell=component,
                dependencies=dependencies,
                layer_stack=layer_stack,
                dirpath=dirpath,
            )
            self.recipe_setup.test_setup1 = "abcdefg"
            self.recipe_setup.test_setup2 = TestSettings()

        @eval_decorator
        def eval(self, run_convergence: bool = True) -> bool:
            self.recipe_results.results1 = 42
            return True

    class CustomRecipe3(DesignRecipe):
        def __init__(
            self,
            component: Component | None = None,
            layer_stack: LayerStack | None = None,
            dependencies: list[DesignRecipe] | None = None,
            dirpath: Path | None = None,
        ):
            super().__init__(
                cell=component,
                dependencies=dependencies,
                layer_stack=layer_stack,
                dirpath=dirpath,
            )
            self.recipe_setup.test_setup1 = "testing"
            self.recipe_setup.test_setup2 = 1 + 1j

        @eval_decorator
        def eval(self, run_convergence: bool = True) -> bool:
            self.recipe_results.results1 = 1 + 2j
            return True

    # Specify test directory
    dirpath = Path(__file__).resolve().parent / "test_recipe"
    dirpath.mkdir(parents=True, exist_ok=True)

    # Clean up test directory if data is there
    import shutil

    check_dirpath1 = (
        dirpath / "CustomRecipe1_1011179653845525059685567166285757734403866725674"
    )
    check_dirpath2 = (
        dirpath / "CustomRecipe2_611883188232344136464084854113520728566543914331"
    )
    check_dirpath3 = (
        dirpath / "CustomRecipe3_457049578073299597208293822079204251822531831950"
    )
    if check_dirpath1.is_dir():
        shutil.rmtree(str(check_dirpath1))
    if check_dirpath2.is_dir():
        shutil.rmtree(str(check_dirpath2))
    if check_dirpath3.is_dir():
        shutil.rmtree(str(check_dirpath3))

    # Set up recipes
    component1 = straight(width=0.6)
    component2 = straight(width=0.5)
    component3 = straight(width=0.4)
    A = CustomRecipe1(
        component=component1, layer_stack=get_layer_stack(), dirpath=dirpath
    )
    B = CustomRecipe2(
        component=component2, layer_stack=get_layer_stack(), dirpath=dirpath
    )
    C = CustomRecipe3(
        component=component3, layer_stack=get_layer_stack(), dirpath=dirpath
    )

    # Test with no dependencies
    A.eval()

    # Check if recipe directory exists
    check_dirpath = check_dirpath1
    if not check_dirpath.is_dir():
        raise NotADirectoryError("Recipe directory not created with hash.")

    # Check if recipe results exists
    check_results = check_dirpath / "recipe_results.pkl"
    if not check_results.is_file():
        raise FileNotFoundError("Recipe results (recipe_results.pkl) not found.")

    # Check if recipe dependencies exists
    check_dependencies = check_dirpath / "recipe_dependencies.txt"
    if not check_dependencies.is_file():
        raise FileNotFoundError("Recipe results (recipe_dependencies.txt) not found.")

    # Check if recipe dependencies has any characters or values. Empty is expected.
    with open(check_dependencies) as f:
        chars = f.read()
        if len(chars) > 1:
            raise ValueError("Recipe dependencies should be empty.")

    # Test with dependencies
    A.dependencies = [B, C]
    A.eval()

    # Check if recipe dependencies has any characters or values. Empty is expected.
    with open(check_dependencies) as f:
        chars = f.read()
        if not len(chars) > 1:
            raise ValueError("Recipe dependencies should have dependencies listed.")

    # Check if dependent recipes exist
    check_dirpath = check_dirpath2
    check_results = check_dirpath / "recipe_results.pkl"
    check_dependencies = check_dirpath / "recipe_dependencies.txt"
    if not check_dirpath.is_dir():
        raise NotADirectoryError("Recipe directory not created with hash.")
    if not check_results.is_file():
        raise FileNotFoundError("Recipe results (recipe_results.pkl) not found.")
    if not check_dependencies.is_file():
        raise FileNotFoundError("Recipe results (recipe_dependencies.txt) not found.")

    check_dirpath = check_dirpath3
    check_results = check_dirpath / "recipe_results.pkl"
    check_dependencies = check_dirpath / "recipe_dependencies.txt"
    if not check_dirpath.is_dir():
        raise NotADirectoryError("Recipe directory not created with hash.")
    if not check_results.is_file():
        raise FileNotFoundError("Recipe results (recipe_results.pkl) not found.")
    if not check_dependencies.is_file():
        raise FileNotFoundError("Recipe results (recipe_dependencies.txt) not found.")


def test_simulation():
    class TestSimulator(Simulation):
        def __init__(
            self,
            component: Component,
            layerstack: LayerStack | None = None,
            simulation_settings: BaseModel | None = None,
            convergence_settings: BaseModel | None = None,
            dirpath: Path | None = None,
            run_custom_convergence1: bool = False,
            run_custom_convergence2: bool = False,
            override_convergence: bool = False,
            setting1: float = 1.2,
            setting2: dict | None = None,
            setting3: list | None = None,
        ):
            super().__init__(
                component=component,
                layerstack=layerstack,
                simulation_settings=simulation_settings,
                convergence_settings=convergence_settings,
                dirpath=dirpath,
            )

            self.setting1 = setting1
            self.setting2 = setting2
            self.setting3 = setting3

            # If convergence data is already available, update simulation settings
            if (
                self.convergence_is_fresh()
                and self.convergence_results.available()
                and not override_convergence
            ):
                try:
                    self.load_convergence_results()
                    # Check if convergence settings, component, and layerstack are the same.
                    # If the same, use the simulation settings from file.
                    # Else, run convergence testing by overriding convergence results.
                    # This covers any collisions in hashes.
                    if self.is_same_convergence_results():
                        self.convergence_settings = (
                            self.convergence_results.convergence_settings
                        )
                        self.simulation_settings = (
                            self.convergence_results.simulation_settings
                        )
                        # Update hash since settings have changed
                        self.last_hash = hash(self)
                    else:
                        override_convergence = True
                except (AttributeError, FileNotFoundError) as err:
                    logger.warning(f"{err}\nRun convergence.")
                    override_convergence = True

            # Set up simulation
            # . . . . . .
            # . . . . . .
            # . . . . . .

            # Run convergence testing if no convergence results are available
            # Or user wants to override convergence results
            # Or if setup has changed
            if (
                not self.convergence_results.available()
                or override_convergence
                or not self.convergence_is_fresh()
            ):
                if run_custom_convergence1:
                    self.convergence_results.custom_convergence_data1 = (
                        self.run_custom_convergence1()
                    )

                if run_custom_convergence2:
                    self.convergence_results.custom_convergence_data2 = (
                        self.run_custom_convergence2()
                    )

                if run_custom_convergence1 and run_custom_convergence2:
                    # Save setup and results for convergence
                    self.save_convergence_results()
                    logger.info("Saved convergence results.")

        def run_custom_convergence1(self):
            return pd.DataFrame({"a": [4, 3, 2, 1], "b": [5, 4, 3, 2]})

        def run_custom_convergence2(self):
            return pd.DataFrame({"c": [40, 30, 20, 10], "d": [50, 40, 30, 20]})

    # Specify test directory
    dirpath = Path(__file__).resolve().parent / "test_recipe"
    dirpath.mkdir(parents=True, exist_ok=True)

    # Clean up test directory if data is there
    import shutil

    check_dirpath1 = dirpath / "TestSimulator_510428257926875840"
    if check_dirpath1.is_dir():
        shutil.rmtree(str(check_dirpath1))

    sim1 = TestSimulator(
        component=straight(),
        simulation_settings=TestSettings(),
        convergence_settings=TestSettings(),
        dirpath=dirpath,
        run_custom_convergence1=True,
        run_custom_convergence2=True,
        override_convergence=False,
        setting1=1.3,
        setting2={"L": "T", "Y": "P"},
        setting3=[3, 1, 4, 5],
    )

    # Check if dependent recipes exist
    check_dirpath = check_dirpath1
    check_results = check_dirpath1 / "convergence_results.pkl"
    if not check_dirpath.is_dir():
        raise NotADirectoryError("Simulation directory not created with hash.")
    if not check_results.is_file():
        raise FileNotFoundError(
            "Convergence results (convergence_results.pkl) not found."
        )

    sim2 = TestSimulator(
        component=straight(),
        simulation_settings=TestSettings(),
        convergence_settings=TestSettings(),
        dirpath=dirpath,
        run_custom_convergence1=True,
        run_custom_convergence2=True,
        override_convergence=False,
    )

    # Check that convergence results and simulation settings are same
    if not sim1.simulation_settings == sim2.simulation_settings:
        raise Exception("Simulation settings are unequal")
    if not all(
        sim1.convergence_results.custom_convergence_data1
        == sim2.convergence_results.custom_convergence_data1
    ) or not all(
        sim1.convergence_results.custom_convergence_data2
        == sim2.convergence_results.custom_convergence_data2
    ):
        raise Exception("Convergence results are unequal")
