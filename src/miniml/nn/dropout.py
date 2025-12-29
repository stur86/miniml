from jax import Array
from miniml.model import MiniMLModel, PredictMode
from miniml.random import RandomMask


class Dropout(MiniMLModel):
    
    def __init__(self, shape: tuple[int, ...], rate: float = 0.5) -> None:
        """Initialize the Dropout model.

        Args:
            shape (tuple[int, ...]): The shape of the input array to apply dropout to.
            rate (float, optional): The dropout rate (fraction of units to drop). Defaults to 0.5.
        """
        if not (0.0 <= rate < 1.0):
            raise ValueError("Dropout rate must be in the range [0.0, 1.0).")
        
        self._rate = rate
        self._dropout_mask = RandomMask(shape=shape, p=(1-rate))
        super().__init__()
        
    def _predict_kernel(self, X: Array, buffer: Array, rng_key: Array | None = None, mode: PredictMode = PredictMode.INFERENCE) -> Array:
        if mode == PredictMode.INFERENCE or self._rate == 0.0:
            return X
        else:
            if rng_key is None:
                raise ValueError("rng_key must be provided during training mode for Dropout.")
            mask = self._dropout_mask.generate(rng_key)
            X = X * mask / (1.0 - self._rate)
        return X
