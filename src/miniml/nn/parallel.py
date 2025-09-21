from typing import Literal
from miniml.model import MiniMLModel, MiniMLModelList
from jax import Array
import jax.numpy as jnp
from miniml.loss import LossFunction, squared_error_loss


class Parallel(MiniMLModel):
    """A MiniML model that applies multiple MiniML models in parallel and sums or
    concatenates their outputs."""

    def __init__(
        self,
        models: list[MiniMLModel],
        mode: Literal["sum", "concat"],
        concat_axis: int = -1,
        loss: LossFunction = squared_error_loss
    ) -> None:
        """Initialize the Parallel model.

        Args:
            models (list[MiniMLModel]): The models to apply in parallel.
            mode (Literal["sum", "concat"]): The mode of combining outputs.
            concat_axis (int, optional): The axis to concatenate along if mode is "concat". Defaults to -1.
            loss (LossFunction, optional): The loss function to use. Defaults to squared_error_loss.

        Raises:
            ValueError: If models list is empty or mode is invalid.
            ValueError: If mode is not a valid option.
        """
        if len(models) == 0:
            raise ValueError("Parallel must contain at least one model")
        self._model_list = MiniMLModelList(models)
        self._concat_axis = concat_axis
        try:
            self._predict_func = getattr(self, f"_predictf_{mode}")
        except AttributeError:
            raise ValueError(f"Invalid mode '{mode}'. Choose 'sum' or 'concat'.")

        super().__init__(loss=loss)

    def _predictf_sum(self, X: Array, buffer: Array) -> Array:
        outputs = [
            model._predict_kernel(X, buffer) for model in self._model_list.contents
        ]
        return jnp.sum(jnp.stack(outputs), axis=0)

    def _predictf_concat(self, X: Array, buffer: Array) -> Array:
        outputs = [
            model._predict_kernel(X, buffer) for model in self._model_list.contents
        ]
        return jnp.concatenate(outputs, axis=self._concat_axis)

    def _predict_kernel(self, X: Array, buffer: Array) -> Array:
        return self._predict_func(X, buffer)
