from jax import Array as JxArray
from abc import ABC, abstractmethod
from typing import Callable, Annotated
from copy import deepcopy

ObjectiveFunction = Callable[[JxArray], JxArray]


class MiniMLOptimizer(ABC):

    def __init__(self, objective: ObjectiveFunction) -> None:
        self._objective = objective
