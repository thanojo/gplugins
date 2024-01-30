from __future__ import annotations

import threading
from typing import List

from gdsfactory.path import hashlib
import gplugins.design_recipe as dr



# moved into a class in case we want to make this a parallel-dispatch dependency DAG instead of a serial array
class ConstituentRecipes:
    """
    A class to represent a flow of DesignRecipes.
    The execution order is currently striclty serial,
    but this could be extended with a dependency DAG as the underlying
    structure from which DesignRecipes are yielded and evaluated.
    Note that `constituent_recipes` is not assumed to be ordered.
    That is, non-nested dependencies are considered independent.
    """

    constituent_recipes: List[dr.DesignRecipe] = []

    # The next index to yield out of constituent_recipes.
    idx_to_yield: int = 0

    # A lock to serialize iterator accesses,
    # useful when the iterator is accessed by multiple threads.
    iter_lock: threading.Lock

    def __init__(self, recipes: List[dr.DesignRecipe] = []):
        self.constituent_recipes = recipes
        self.lock = threading.Lock()

    def __iter__(self):
        """
        Return an iterable of DesignRecipes, to be evaluated.
        """
        # TODO is it possible we'd want to iterate through this multiple times at once?
        # we can't handle that right now, so try and prevent it
        assert self.idx_to_yield == 0, \
            "you can't concurrently iterate through  multiple instances of this ConstituentRecipes"

        return self

    def __next__(self):
        """
        Return the next DesignRecipe to be (checked for freshness and) evaluated.
        In the future, asynchronous dispatch and enforcement of a dependency graph
        could be impelmented here.
        """
        with self.lock:
            if self.idx_to_yield >= len(self.constituent_recipes):
                self.idx_to_yield = 0  # reset
                raise StopIteration

            self.idx_to_yield += 1
            return self.constituent_recipes[self.idx_to_yield - 1]
