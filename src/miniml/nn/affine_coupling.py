import jax.numpy as jnp
from jax import Array as JXArray

from miniml.loss import (
    LossFunction,
    RegLossFunction,
    squared_error_loss,
)
from miniml.model import MiniMLModel, PredictKernelOutput, PredictMode
from miniml.nn.activations import ActivationFunction, relu
from miniml.nn.mlp import MLP
from miniml.param import DTypeLike


class AffineCouplingLayer(MiniMLModel):
    def __init__(
        self,
        dim: int,
        target_indices: slice | set[int] | None = None,
        hidden_size: int | None = None,
        activation: ActivationFunction = relu,
        loss: LossFunction = squared_error_loss,
        reg_loss: RegLossFunction = MLP._REG_LOSS_DEFAULT,
        dtype: DTypeLike = jnp.float32,
    ) -> None:
        self._dim = dim
        if target_indices is None:
            self._target = self._slice_to_set(slice(None, None, 2), dim)
        elif isinstance(target_indices, slice):
            self._target = self._slice_to_set(target_indices, dim)
        else:
            self._target = set(target_indices)
        self._input = set(range(dim)) - self._target
        n_in, n_out = len(self._input), len(self._target)
        if hidden_size is None:
            n_hidden = 2 * n_in
        else:
            n_hidden = hidden_size
        # Generate the MLP
        self._mlp = MLP(
            [n_in, n_hidden, n_out],
            activation=activation,
            loss=loss,
            reg_loss=reg_loss,
            dtype=dtype,
        )

        super().__init__(loss=loss)

    @staticmethod
    def _slice_to_set(input: slice, dim: int) -> set:
        return set(range(dim)[input])

    def _predict_kernel(
            self,
            X: JXArray,
            buffer: JXArray,
            rng_key: JXArray | None = None,
            mode: PredictMode = PredictMode.INFERENCE,
            **predict_kwargs,
        ) -> PredictKernelOutput:
        
