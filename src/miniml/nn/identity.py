from jax import Array
from miniml.model import MiniMLModel

class Identity(MiniMLModel):
    """A MiniML model that returns its input unchanged."""
    
    def __init__(self, scale: float = 1.0) -> None:
        """Initialize the Identity model.

        Args:
            scale (float, optional): The scale factor to apply to the input. Defaults to 1.0.
        """
        
        self._scale = scale
        super().__init__()

    def _predict_kernel(self, X: Array, buffer: Array) -> Array:
        return X * self._scale