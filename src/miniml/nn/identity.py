from jax import Array
from miniml.model import MiniMLModel

class Identity(MiniMLModel):
    """A MiniML model that returns its input unchanged."""
    
    def __init__(self) -> None:
        super().__init__()

    def _predict_kernel(self, X: Array, buffer: Array) -> Array:
        return X