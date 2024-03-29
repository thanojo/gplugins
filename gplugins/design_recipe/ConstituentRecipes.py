from __future__ import annotations

import threading

import gplugins.design_recipe as dr


class ConstituentRecipes:
    """
    A class to represent a flow of DesignRecipes.
    The execution order is currently strictly serial,
    but this could be extended with a dependency DAG as the underlying
    structure from which DesignRecipes are yielded and evaluated.
    Note that `constituent_recipes` is not assumed to be ordered.
    That is, non-nested dependencies are considered independent.
    """

    constituent_recipes: list[dr.DesignRecipe]

    # The next index to yield out of constituent_recipes.
    idx_to_yield: int = 0

    # A lock to serialize iterator accesses,
    # useful when the iterator is accessed by multiple threads.
    iter_lock: threading.Lock

    def __init__(self, recipes: list[dr.DesignRecipe]):
        self.constituent_recipes = recipes or []
        self.lock = threading.Lock()

    def __iter__(self):
        """
        Return an iterable of DesignRecipes, to be evaluated.
        """
        # TODO is it possible we'd want to iterate through this multiple times at once?
        # we can't handle that right now, so try and prevent it
        assert (
            self.idx_to_yield == 0
        ), "you can't concurrently iterate through multiple instances of this ConstituentRecipes"

        return self

    def __next__(self):
        """
        Return the next DesignRecipe to be (checked for freshness and) evaluated.
        In the future, asynchronous dispatch and enforcement of a dependency graph
        could be implemented here.
        """
        with self.lock:
            if self.idx_to_yield >= len(self.constituent_recipes):
                self.idx_to_yield = 0  # reset
                raise StopIteration

            self.idx_to_yield += 1
            return self.constituent_recipes[self.idx_to_yield - 1]
