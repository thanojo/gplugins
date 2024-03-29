from __future__ import annotations

import json
import os
import pickle
from collections.abc import Callable
from pathlib import Path, PosixPath, WindowsPath

import pydantic
from gdsfactory import Component
from gdsfactory.config import logger
from gdsfactory.path import hashlib
from gdsfactory.pdk import LayerStack, get_layer_stack
from gdsfactory.typings import ComponentFactory
from pydantic import BaseModel

import gplugins.design_recipe as dr


class Setup:
    """
    A dynamic class to store any setup information

    This class allows designers to arbitrarily add setup information.
    """

    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

    def __eq__(self, other):
        if not isinstance(other, Setup):
            # Don't attempt to compare against unrelated types
            return NotImplemented
        return self.__dict__ == other.__dict__


class DesignRecipe:
    """
    A DesignRecipe represents a flow of operations on GDSFactory components,
    with zero or more dependencies. Note that dependencies are assumed to be independent,
    dependent dependencies should be nested. When `eval()`ed, A DesignRecipe `eval()`s
    its dependencies if they they've become stale,
    and optionally executes some tool-specific functionality.
    For example,an FdtdDesignRecipe might simulate its `component` in
    Lumerial FDTD to extract its s-parameters.


    Attributes:
        dependencies: This `DesignRecipe`s dependencies, assumed to be independent
        last_hash: The hash of the system last time eval() was executed.
                   This is used to determine whether the recipe's configuration
                   has changed, meaning it must be re-eval'ed.
        cell: GDSFactory layout cell or component. This is not necessarily the
              same `component` referred to in the `dependencies` recipes.
        layer_stack: PDK layerstack
        dirpath: Root directory where recipes runs are stored
        recipe_dirpath: Recipe directory where recipe results are stored.
                        This is only created upon eval of the recipe.
        run_convergence: Run convergence if True. Accurate simulations come from
                         simulations that have run convergence.
        override_recipe: Overrides recipe results if True. This runs the design
                         recipe regardless if results are available and overwrites
                         recipe results.
        recipe_results: A dynamic object used to store recipe results and the
                        setup to these results
    """

    dependencies: dr.ConstituentRecipes
    last_hash: int = -1
    cell: ComponentFactory | Component | None = None
    layer_stack: LayerStack | None = None
    dirpath: Path | None = None
    recipe_dirpath: Path | None = None
    run_convergence: bool = True
    override_recipe: bool = False
    recipe_setup: Setup | None = None
    recipe_results: Results | None = None

    def __init__(
        self,
        cell: ComponentFactory | Component,
        dependencies: list[dr.DesignRecipe] | None = None,
        layer_stack: LayerStack = get_layer_stack(),
        dirpath: Path | None = None,
    ):
        self.dependencies = dr.ConstituentRecipes(dependencies)
        self.dirpath = dirpath or Path(".")
        self.cell = cell

        # Initialize recipe setup
        if isinstance(self.cell, Callable):
            cell_hash = self.cell().hash_geometry()
        elif type(self.cell) == Component:
            cell_hash = self.cell.hash_geometry()
        self.recipe_setup = Setup(cell_hash=cell_hash, layer_stack=layer_stack)
        self.recipe_results = Results(prefix="recipe")

    def __hash__(self) -> int:
        """
        Returns a hash of all state this DesignRecipe contains.
        This is used to determine 'freshness' of a recipe (i.e. if it needs to be rerun)

        Hashed items:
        - component / cell geometry
        - layer stack
        """
        h = hashlib.sha1()
        for attr in self.recipe_setup.__dict__.values():
            if (
                isinstance(attr, int)
                or isinstance(attr, float)
                or isinstance(attr, complex)
                or isinstance(attr, str)
            ):
                h.update(str(attr).encode("utf-8"))
            elif isinstance(attr, dict) or isinstance(attr, list):
                h.update(json.dumps(attr, sort_keys=True).encode("utf-8"))
            elif isinstance(attr, BaseModel):
                h.update(attr.model_dump_json().encode("utf-8"))
            else:
                h.update(str(attr).encode("utf-8"))
        h.update(str(self.run_convergence).encode("utf-8"))
        return int.from_bytes(h.digest(), "big")

    def is_fresh(self) -> bool:
        """
        Returns if this DesignRecipe needs to be re-`eval()`ed.
        This could be either caused by this DesignRecipe's
        configuration being changed, or that of one of its dependencies.
        """
        return self.__hash__() == self.last_hash and all(
            recipe.is_fresh() for recipe in self.dependencies
        )

    def eval(self, force_rerun_all=False) -> bool:
        """
        Evaluate this DesignRecipe. This should be overridden by
        subclasses with their specific functionalities
        (e.g. run the fdtd engine).
        Here we only evaluate dependencies,
        since the generic DesignRecipe has no underlying task.

        Parameters:
            force_rerun_all: Forces design recipes to be re-evaluated
        """
        success = self.eval_dependencies(force_rerun_all)

        self.last_hash = hash(self)
        return success

    def eval_dependencies(self, force_rerun_all=False) -> bool:
        """
        Evaluate this `DesignRecipe`'s dependencies.
        Because `dependencies` are assumed to be independent,
        they can be evaluated in any order.

        Parameters:
            force_rerun_all: Forces design recipes to be re-evaluated
        """
        success = True
        for recipe in self.dependencies:
            if force_rerun_all or (not recipe.is_fresh()):
                success = success and recipe.eval(force_rerun_all)
        return success

    def load_recipe_results(self):
        """
        Loads recipe results from pickle file into class attribute
        """
        self.recipe_results = self.recipe_results.get_pickle()

    def save_recipe_results(self):
        """
        Saves recipe_results to pickle file while adding setup information.
        This includes:
        - component hash
        - layerstack
        - convergence_settings
        - simulation_settings

        This is usually done after convergence testing is completed and simulation
        settings are accurate and should be saved for future reference/recall.
        """
        self.recipe_results.recipe_setup = self.recipe_setup
        self.recipe_dirpath.resolve().mkdir(parents=True, exist_ok=True)
        self.recipe_results.save_pickle()

    def is_same_recipe_results(self) -> bool:
        """
        Returns whether recipe results' setup are the same as the current setup
        for the recipe. This is important for preventing hash collisions.
        """
        try:
            return self.recipe_results.recipe_setup == self.recipe_setup
        except AttributeError:
            return False


def eval_decorator(func):
    """
    Design recipe eval decorator

    Parameters:
        func: Design recipe eval method

    Returns:
        Design recipe eval method decorated with dependency execution and hashing
    """

    def design_recipe_eval(*args, **kwargs):
        """
        Evaluates design recipe and its dependencies then hashes the design recipe
        and returns successful execution
        """
        self = args[0]
        # Evaluate independent dependent recipes
        success = self.eval_dependencies()

        if "run_convergence" in kwargs:
            self.run_convergence = kwargs["run_convergence"]
        self.last_hash = self.__hash__()

        # Create directory for recipe results
        self.recipe_dirpath = (
            self.dirpath / f"{self.__class__.__name__}_{self.last_hash}"
        )
        self.recipe_dirpath.mkdir(parents=True, exist_ok=True)
        self.recipe_results.dirpath = self.recipe_dirpath
        self.recipe_results.prefix = "recipe"
        logger.info(f"Hashed Directory: {self.recipe_dirpath}")

        # Add text file outlining dependencies for traceability
        with open(
            str(self.recipe_dirpath.resolve() / "recipe_dependencies.txt"), "w"
        ) as f:
            dependencies = [
                f"{recipe.__class__.__name__}_{recipe.last_hash}"
                for recipe in self.dependencies
            ]
            dependencies = "\n".join(dependencies)
            f.write(dependencies)

        # Check if results already available
        # Results must be stored in directory with the same hash.
        if (
            self.recipe_results.available()
            and not self.override_recipe
            and self.is_fresh()
        ):
            # Load results if available
            self.load_recipe_results()

            if not self.is_same_recipe_results():
                # If the recipe setup is not the same as in the results,
                # eval the design recipe
                success = success and func(*args, **kwargs)
                self.save_recipe_results()
        else:
            # If results not available, recipe config has changed,
            # or user wants to override recipe results,
            # eval the design recipe
            success = success and func(*args, **kwargs)
            self.save_recipe_results()

        # Return successful execution
        return success

    return design_recipe_eval


class Results:
    """
    Results are stored in this dynamic class. Any type of results can be stored.

    This class allows designers to arbitrarily add results. Results are pickled
    to be saved onto working system. Results can be retrieved via unpickling.
    """

    def __init__(self, prefix: str = "", dirpath: Path | None = None, **kwargs):
        if isinstance(dirpath, str):
            dirpath = Path(dirpath)
        self.dirpath = dirpath or Path(".")
        self.prefix = prefix
        for key, value in kwargs.items():
            setattr(self, key, value)

    def save_pickle(self, dirpath: Path | None = None):
        """
        Save results by pickling as *_results.pkl file

        Parameters:
            dirpath: Directory to store pickle file
        """
        if dirpath is None:
            with open(
                str(self.dirpath.resolve() / f"{self.prefix}_results.pkl"), "wb"
            ) as f:
                pickle.dump(self, f)
                logger.info(
                    f"Cached results to {self.dirpath} -> {self.prefix}_results.pkl"
                )
        else:
            with open(str(dirpath.resolve() / f"{self.prefix}_results.pkl"), "wb") as f:
                pickle.dump(self, f)
                logger.info(f"Cached results to {dirpath} -> {self.prefix}_results.pkl")

    def get_pickle(self, dirpath: Path | None = None) -> object:
        """
        Get results from *_results.pkl file

        Parameters:
            dirpath: Directory to get pickle file

        Returns:
            Results as an object with results
        """
        if isinstance(dirpath, str):
            dirpath = Path(dirpath)
        if dirpath is None:
            with open(
                str(self.dirpath.resolve() / f"{self.prefix}_results.pkl"), "rb"
            ) as f:
                unpickler = PathUnpickler(f)
                results = unpickler.load()
                if not results.dirpath == self.dirpath:
                    results.dirpath = self.dirpath
                logger.info(
                    f"Recalled results from {self.dirpath} -> {self.prefix}_results.pkl"
                )
        else:
            with open(str(dirpath.resolve() / f"{self.prefix}_results.pkl"), "rb") as f:
                unpickler = PathUnpickler(f)
                results = unpickler.load()
                if not results.dirpath == dirpath:
                    results.dirpath = dirpath
                logger.info(
                    f"Recalled results from {dirpath} -> {self.prefix}_results.pkl"
                )

        return results

    def available(self, dirpath: Path | None = None) -> bool:
        """
        Check if '*_results.pkl' file exists and results can be loaded

        Parameters:
            dirpath: Directory with pickle file

        Returns:
            True if results exist, False otherwise.
        """
        if isinstance(dirpath, str):
            dirpath = Path(dirpath)
        if dirpath is None:
            results_file = self.dirpath.resolve() / f"{self.prefix}_results.pkl"
        else:
            results_file = dirpath.resolve() / f"{self.prefix}_results.pkl"
        return results_file.is_file()


class Simulation:
    """
    Represents the simulation object used to simulate GDSFactory devices.

    This simulation object's purpose is to reduce time simulating by recalling
    hashed results.
    """

    # the hash of the system last time convergence was executed
    last_hash: int = -1

    # A dynamic object used to store convergence results
    convergence_results: Results

    def __init__(
        self,
        component: Component,
        layerstack: LayerStack | None = None,
        simulation_settings: pydantic.BaseModel | None = None,
        convergence_settings: pydantic.BaseModel | None = None,
        dirpath: Path | None = None,
    ):
        self.dirpath = dirpath or Path(".")
        self.component = component
        self.layerstack = layerstack or get_layer_stack()
        self.simulation_settings = simulation_settings
        self.convergence_settings = convergence_settings

        self.last_hash = hash(self)

        # Create directory for simulation files
        self.simulation_dirpath = (
            self.dirpath / f"{self.__class__.__name__}_{self.last_hash}"
        )
        self.simulation_dirpath.mkdir(parents=True, exist_ok=True)

        # Create attribute for convergence results
        self.convergence_results = Results(
            prefix="convergence", dirpath=self.simulation_dirpath
        )

    def __hash__(self) -> int:
        """
        Returns a hash of all state this Simulation contains
        Subclasses should include functionality-specific state (e.g. convergence info) here.
        This is used to determine simulation convergence (i.e. if it needs to be rerun)

        Hashed items:
        - component
        - layer stack
        - simulation settings
        - convergence settings
        """
        h = hashlib.sha1()
        if self.component is not None:
            h.update(self.component.hash_geometry(precision=1e-4).encode("utf-8"))
        if self.layerstack is not None:
            h.update(self.layerstack.model_dump_json().encode("utf-8"))
        if self.simulation_settings is not None:
            h.update(self.simulation_settings.model_dump_json().encode("utf-8"))
        if self.convergence_settings is not None:
            h.update(self.convergence_settings.model_dump_json().encode("utf-8"))
        return int.from_bytes(h.digest(), "big")

    def convergence_is_fresh(self) -> bool:
        """
        Returns if this simulation needs to be re-run.
        This could be caused by this simulation's
        configuration being changed.
        """
        return hash(self) == self.last_hash

    def load_convergence_results(self):
        """
        Loads convergence results from pickle file into class attribute
        """
        self.convergence_results = self.convergence_results.get_pickle()

    def save_convergence_results(self):
        """
        Saves convergence_results to pickle file while adding setup information
        and resultant accurate simulation settings.
        This includes:
        - component hash
        - layerstack
        - convergence_settings
        - simulation_settings

        This is usually done after convergence testing is completed and simulation
        settings are accurate and should be saved for future reference/recall.
        """
        self.convergence_results.convergence_settings = self.convergence_settings
        self.convergence_results.simulation_settings = self.simulation_settings
        self.convergence_results.component_hash = self.component.hash_geometry()
        self.convergence_results.layerstack = self.layerstack
        self.simulation_dirpath.mkdir(parents=True, exist_ok=True)
        self.convergence_results.save_pickle()

    def is_same_convergence_results(self) -> bool:
        """
        Returns whether convergence results' setup are the same as the current
        setup for the simulation. This is important for preventing hash collisions.
        """
        try:
            return (
                self.convergence_results.convergence_settings
                == self.convergence_settings
                and self.convergence_results.component_hash
                == self.component.hash_geometry()
                and self.convergence_results.layerstack == self.layerstack
            )
        except AttributeError:
            return False


class PathUnpickler(pickle.Unpickler):
    """
    Unpickles objects while handling OS-dependent paths
    """

    def find_class(self, module, name):
        if module == "pathlib" and (name == "PosixPath" or name == "WindowsPath"):
            return WindowsPath if os.name == "nt" else PosixPath
        return super().find_class(module, name)
