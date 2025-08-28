from jax import Array as JXArray
import jax.numpy as jnp
from numpy.typing import DTypeLike
from miniml import MiniMLModel, MiniMLParam
from miniml.loss import LossFunction, squared_error_loss


class Linear(MiniMLModel):

    def __init__(self, n_in: int, n_out: int, loss: LossFunction = squared_error_loss, dtype: DTypeLike = jnp.float32) -> None:

        self._W = MiniMLParam((n_in, n_out), dtype=dtype)
        self._b = MiniMLParam((n_out,), dtype=dtype)

        super().__init__(loss)

    def predict(self, X: JXArray) -> JXArray:
        return X@self._W.value + self._b.value
