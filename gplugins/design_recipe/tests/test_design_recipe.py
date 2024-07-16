from gplugins.design_recipe.DesignRecipe import DesignRecipe, eval_decorator
from gplugins.design_recipe.ConstituentRecipes import ConstituentRecipes
from gdsfactory import Component
from gdsfactory.components.straight import straight
from gdsfactory.config import logger
from gdsfactory.pdk import LayerStack, get_layer_stack
from pathlib import Path
from gplugins.lumerical.convergence_settings import LUMERICAL_FDTD_CONVERGENCE_SETTINGS

def test_design_recipe():
    class CustomRecipe1(DesignRecipe):
        def __init__(self,
                     component: Component | None = None,
                     layer_stack: LayerStack | None = None,
                     dependencies: list[DesignRecipe] | None = None,
                     dirpath: Path | None = None,
                     ):
            super().__init__(cell=component, dependencies=dependencies, layer_stack=layer_stack, dirpath=dirpath)
            self.recipe_setup.test_setup1 = {"a": [1,2,3], "b": [4,3,2]}
            self.recipe_setup.test_setup2 = [1, 2, 3]

        @eval_decorator
        def eval(self, run_convergence: bool = True) -> bool:
            self.recipe_results.results1 = [5,4,3,2,1]
            return True

    class CustomRecipe2(DesignRecipe):
        def __init__(self,
                     component: Component | None = None,
                     layer_stack: LayerStack | None = None,
                     dependencies: list[DesignRecipe] | None = None,
                     dirpath: Path | None = None,
                     ):
            super().__init__(cell=component, dependencies=dependencies, layer_stack=layer_stack, dirpath=dirpath)
            self.recipe_setup.test_setup1 = "abcdefg"
            self.recipe_setup.test_setup2 = LUMERICAL_FDTD_CONVERGENCE_SETTINGS

        @eval_decorator
        def eval(self, run_convergence: bool = True) -> bool:
            self.recipe_results.results1 = 42
            return True

    class CustomRecipe3(DesignRecipe):
        def __init__(self,
                     component: Component | None = None,
                     layer_stack: LayerStack | None = None,
                     dependencies: list[DesignRecipe] | None = None,
                     dirpath: Path | None = None,
                     ):
            super().__init__(cell=component, dependencies=dependencies, layer_stack=layer_stack, dirpath=dirpath)
            self.recipe_setup.test_setup1 = "testing"
            self.recipe_setup.test_setup2 = (1+1j)

        @eval_decorator
        def eval(self, run_convergence: bool = True) -> bool:
            self.recipe_results.results1 = (1+2j)
            return True

    # Specify test directory
    dirpath = Path(__file__).resolve().parent / "test_recipe"
    dirpath.mkdir(parents=True, exist_ok=True)

    # Clean up test directory if data is there
    import shutil
    check_dirpath1 = dirpath / "CustomRecipe1_1011179653845525059685567166285757734403866725674"
    check_dirpath2 = dirpath / "CustomRecipe2_1327931766567412728144722440192793038374486243823"
    check_dirpath3 = dirpath / "CustomRecipe3_457049578073299597208293822079204251822531831950"
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
    A = CustomRecipe1(component=component1, layer_stack=get_layer_stack(), dirpath=dirpath)
    B = CustomRecipe2(component=component2, layer_stack=get_layer_stack(), dirpath=dirpath)
    C = CustomRecipe3(component=component3, layer_stack=get_layer_stack(), dirpath=dirpath)

    # Test with no dependencies
    A.eval()

    # Check if recipe directory exists
    check_dirpath = dirpath / "CustomRecipe1_1011179653845525059685567166285757734403866725674"
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
    with open(check_dependencies, "r") as f:
        chars = f.read()
        if len(chars) > 1:
            raise ValueError("Recipe dependencies should be empty.")

    # Test with dependencies
    A.dependencies = [B, C]
    A.eval()

    # Check if recipe dependencies has any characters or values. Empty is expected.
    with open(check_dependencies, "r") as f:
        chars = f.read()
        if not len(chars) > 1:
            raise ValueError("Recipe dependencies should have dependencies listed.")

    # Check if dependent recipes exist
    check_dirpath = dirpath / "CustomRecipe2_1327931766567412728144722440192793038374486243823"
    check_results = check_dirpath / "recipe_results.pkl"
    check_dependencies = check_dirpath / "recipe_dependencies.txt"
    if not check_dirpath.is_dir():
        raise NotADirectoryError("Recipe directory not created with hash.")
    if not check_results.is_file():
        raise FileNotFoundError("Recipe results (recipe_results.pkl) not found.")
    if not check_dependencies.is_file():
        raise FileNotFoundError("Recipe results (recipe_dependencies.txt) not found.")

    check_dirpath = dirpath / "CustomRecipe3_457049578073299597208293822079204251822531831950"
    check_results = check_dirpath / "recipe_results.pkl"
    check_dependencies = check_dirpath / "recipe_dependencies.txt"
    if not check_dirpath.is_dir():
        raise NotADirectoryError("Recipe directory not created with hash.")
    if not check_results.is_file():
        raise FileNotFoundError("Recipe results (recipe_results.pkl) not found.")
    if not check_dependencies.is_file():
        raise FileNotFoundError("Recipe results (recipe_dependencies.txt) not found.")