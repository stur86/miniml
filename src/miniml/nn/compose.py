from jax import Array
import jax
import jax.numpy as jnp
from typing import Literal
from miniml.model import MiniMLModel, MiniMLModelList, PredictMode, PredictKernelOutput
from miniml.loss import LossFunction


class Identity(MiniMLModel):
    """A MiniML model that returns its input unchanged or scaled."""

    def __init__(self, scale: float = 1.0) -> None:
        """Initialize the Identity model.

        Args:
            scale (float, optional): The scale factor to apply to the input. Defaults to 1.0.
        """

        self._scale = scale
        super().__init__()

    def _predict_kernel(
        self,
        X: Array,
        buffer: Array,
        rng_key: Array | None = None,
        mode: PredictMode = PredictMode.INFERENCE,
        **predict_kwargs,
    ) -> Array:
        return X * self._scale

class Take(MiniMLModel):
    """A MiniML model that takes specific indices from the input array along a given axis."""

    def __init__(self, indices: list[int] | int, axis: int = -1) -> None:
        """Initialize the Take model.

        Args:
            indices (list[int] | int): The indices to take from the input array.
            axis (int, optional): The axis along which to take the indices. Defaults to -1.
        """

        self._indices = jnp.array(indices)
        self._axis = axis
        super().__init__()

    def _predict_kernel(
        self,
        X: Array,
        buffer: Array,
        rng_key: Array | None = None,
        mode: PredictMode = PredictMode.INFERENCE,
        **predict_kwargs,
    ) -> Array:
        return jnp.take(X, indices=self._indices, axis=self._axis)

class Stack(MiniMLModel):
    """A MiniML model that stacks multiple MiniML models sequentially."""

    def __init__(
        self, models: list[MiniMLModel], loss: LossFunction | None = None
    ) -> None:
        """Initialize the Stack model.

        Args:
            models (list[MiniMLModel]): The models to stack.
            loss (LossFunction, optional): The loss function to use. Defaults to None.

        Raises:
            ValueError: If the model list is empty.
        """
        if len(models) == 0:
            raise ValueError("Stack must contain at least one model")
        self._model_list = MiniMLModelList(models)
        super().__init__(loss=loss)

    def _predict_kernel(
        self,
        X: Array,
        buffer: Array,
        rng_key: Array | None = None,
        mode: PredictMode = PredictMode.INFERENCE,
        **predict_kwargs,
    ) -> "Array | PredictKernelOutput":
        total_activity_loss = jnp.zeros(())
        has_any_activity = False
        for model in self._model_list.contents:
            rng_key, subkey = (
                jax.random.split(rng_key) if rng_key is not None else (None, None)
            )
            result = model._predict_kernel(
                X,
                buffer,
                rng_key=subkey,
                mode=mode,
                **predict_kwargs,
            )
            has_child_activity = (
                isinstance(result, PredictKernelOutput)
                and result.activity_loss is not None
            )
            X, child_activity_loss = MiniMLModel._unpack_kernel_output(result)
            total_activity_loss = jax.lax.cond(
                has_child_activity,
                lambda: total_activity_loss + child_activity_loss,
                lambda: total_activity_loss,
            )
            has_any_activity = has_any_activity or has_child_activity
        if has_any_activity:
            return PredictKernelOutput(y_pred=X, activity_loss=total_activity_loss)
        return X


class Parallel(MiniMLModel):
    """A MiniML model that applies multiple MiniML models in parallel and sums or
    concatenates their outputs."""

    def __init__(
        self,
        models: list[MiniMLModel],
        mode: Literal["sum", "concat"],
        concat_axis: int = -1,
        loss: LossFunction | None = None,
    ) -> None:
        """Initialize the Parallel model.

        Args:
            models (list[MiniMLModel]): The models to apply in parallel.
            mode (Literal["sum", "concat"]): The mode of combining outputs.
            concat_axis (int, optional): The axis to concatenate along if mode is "concat". Defaults to -1.
            loss (LossFunction | None, optional): The loss function to use. Defaults to None.

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
        
    def _iter_predictf(
        self,
        X: Array,
        buffer: Array,
        rng_key: Array | None,
        mode: PredictMode,
        **predict_kwargs,
    ) -> "tuple[list[Array], Array, bool]":
        outputs = []
        total_activity_loss = jnp.zeros(())
        has_any_activity = False
        for model in self._model_list.contents:
            rng_key, subkey = (
                jax.random.split(rng_key) if rng_key is not None else (None, None)
            )
            result = model._predict_kernel(
                X,
                buffer,
                rng_key=subkey,
                mode=mode,
                **predict_kwargs,
            )
            has_child_activity = (
                isinstance(result, PredictKernelOutput)
                and result.activity_loss is not None
            )
            y_pred, child_activity_loss = MiniMLModel._unpack_kernel_output(result)
            outputs.append(y_pred)
            total_activity_loss = jax.lax.cond(
                has_child_activity,
                lambda: total_activity_loss + child_activity_loss,
                lambda: total_activity_loss,
            )
            has_any_activity = has_any_activity or has_child_activity
        return outputs, total_activity_loss, has_any_activity

    def _predictf_sum(
        self,
        X: Array,
        buffer: Array,
        rng_key: Array | None,
        mode: PredictMode,
        **predict_kwargs,
    ) -> "Array | PredictKernelOutput":
        outputs, total_activity_loss, has_any_activity = self._iter_predictf(
            X,
            buffer,
            rng_key=rng_key,
            mode=mode,
            **predict_kwargs,
        )
        y_pred = jnp.sum(jnp.stack(outputs), axis=0)
        if has_any_activity:
            return PredictKernelOutput(y_pred=y_pred, activity_loss=total_activity_loss)
        return y_pred

    def _predictf_concat(
        self,
        X: Array,
        buffer: Array,
        rng_key: Array | None,
        mode: PredictMode,
        **predict_kwargs,
    ) -> "Array | PredictKernelOutput":
        outputs, total_activity_loss, has_any_activity = self._iter_predictf(
            X,
            buffer,
            rng_key=rng_key,
            mode=mode,
            **predict_kwargs,
        )
        y_pred = jnp.concatenate(outputs, axis=self._concat_axis)
        if has_any_activity:
            return PredictKernelOutput(y_pred=y_pred, activity_loss=total_activity_loss)
        return y_pred

    def _predict_kernel(
        self,
        X: Array,
        buffer: Array,
        rng_key: Array | None = None,
        mode: PredictMode = PredictMode.INFERENCE,
        **predict_kwargs,
    ) -> "Array | PredictKernelOutput":
        return self._predict_func(
            X,
            buffer,
            rng_key=rng_key,
            mode=mode,
            **predict_kwargs,
        )
