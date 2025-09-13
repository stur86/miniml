from jax import Array
from miniml.model import MiniMLModel, MiniMLModelList

class Stack(MiniMLModel):
    """A MiniML model that stacks multiple MiniML models sequentially."""
    
    def __init__(self, models: list[MiniMLModel]) -> None:
        if len(models) == 0:
            raise ValueError("Stack must contain at least one model")
        self._model_list = MiniMLModelList(models)
        super().__init__()

    def _predict_kernel(self, X: Array, buffer: Array) -> Array:
        for model in self._model_list.contents:
            X = model._predict_kernel(X, buffer)
        return X