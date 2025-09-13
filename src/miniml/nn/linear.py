from jax import Array as JXArray
import jax.numpy as jnp
from numpy.typing import DTypeLike
from miniml import MiniMLModel, MiniMLParam
from miniml.loss import (
    LossFunction,
    squared_error_loss,
    RegLossFunction,
    LNormRegularization,
)


class Linear(MiniMLModel):

    def __init__(
        self,
        n_in: int,
        n_out: int,
        loss: LossFunction = squared_error_loss,
        reg_loss: RegLossFunction = LNormRegularization(2),
        dtype: DTypeLike = jnp.float32,
        apply_bias_reg: bool = False,
    ) -> None:

        self._W = MiniMLParam((n_in, n_out), dtype=dtype, reg_loss=reg_loss)
        bias_reg = reg_loss if apply_bias_reg else None
        self._b = MiniMLParam((n_out,), dtype=dtype, reg_loss=bias_reg)

        super().__init__(loss)

    def predict_kernel(self, X: JXArray, buffer: JXArray) -> JXArray:
        return X @ self._W(buffer) + self._b(buffer)
