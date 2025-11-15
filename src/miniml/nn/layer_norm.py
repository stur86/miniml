from jax import Array as JxArray, numpy as jnp
from numpy.typing import DTypeLike
from miniml.model import MiniMLModel
from miniml.param import MiniMLParam
from miniml.loss import LossFunction, squared_error_loss


class LayerNorm(MiniMLModel):
    """A MiniML model that implements layer normalization."""

    _normalized_shape: tuple[int, ...]
    _eps: float

    def __init__(
        self,
        normalized_shape: int | tuple[int, ...],
        eps: float = 1e-5,
        loss: LossFunction = squared_error_loss,
        dtype: DTypeLike = jnp.float32,
    ) -> None:
        r"""Initialize the LayerNorm model.

        The layer normalization is computed as:
        $$
        \text{LayerNorm}(x) = \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} \cdot \gamma + \beta
        $$
        where \( \mu \) and \( \sigma^2 \) are the mean and variance computed over the last `normalized_shape` dimensions of the input \( x \),
        \( \gamma \) and \( \beta \) are learnable parameters, and \( \epsilon \) is a small constant to prevent division by zero.

        Args:
            normalized_shape (int | tuple[int, ...]): Input shape from an expected input of size
                `(..., normalized_shape[0], normalized_shape[1], ..., normalized_shape[-1])`
            eps (float, optional): A small value added to the denominator for numerical stability. Defaults to 1e-5.
            loss (LossFunction, optional): Loss function for the model. Defaults to squared_error_loss.
            dtype (DTypeLike, optional): Data type for the model parameters. Defaults to jnp.float32.
        """

        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)
        self._normalized_shape = normalized_shape
        self._eps = eps

        param_shape = normalized_shape
        self._gamma = MiniMLParam(param_shape, dtype=dtype)
        self._beta = MiniMLParam(param_shape, dtype=dtype)

        super().__init__(loss=loss)

    def _predict_kernel(self, X: JxArray, buffer: JxArray) -> JxArray:
        gamma = self._gamma(buffer)
        beta = self._beta(buffer)
        ndim = len(self._normalized_shape)
        mean = jnp.mean(X, axis=tuple(range(-ndim, 0)), keepdims=True)
        var = jnp.var(X, axis=tuple(range(-ndim, 0)), keepdims=True)
        X_norm = (X - mean) / jnp.sqrt(var + self._eps)
        return X_norm * gamma + beta
