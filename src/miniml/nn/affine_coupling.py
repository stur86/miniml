from typing import Any

import jax
import jax.numpy as jnp
from jax import Array as JXArray

from miniml.loss import (
    LossFunction,
    RegLossFunction,
    squared_error_loss,
)
from miniml.model import (
    InvertibleModel,
    MiniMLModel,
    PredictKernelOutput,
    PredictMode,
)
from miniml.nn.activations import ActivationFunction, relu
from miniml.nn.mlp import MLP
from miniml.param import DTypeLike, MiniMLError


class AffineCouplingLayer(InvertibleModel, MiniMLModel):
    r"""An invertible affine coupling layer.

    The ``dim``-dimensional input is split into two disjoint index sets: the
    *input* set (left unchanged) and the *target* set (transformed).  Two
    separate MLPs, ``_mlp_q`` and ``_mlp_p``, are computed on the input set and
    used to transform the target set as

    $$
    y_t = x_t \odot \exp(q(x_i)) + p(x_i)
    $$

    where :math:`x_i` is the unchanged input set.  Since the input set is left
    untouched, the layer is trivially invertible:

    $$
    x_t = (y_t - p(y_i)) \odot \exp(-q(y_i))
    $$

    The raw output of ``_mlp_q`` is exponentiated so that the scale factor is
    always strictly positive and the inverse is numerically stable.
    """

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
        """Construct an invertible affine coupling layer.

        Args:
            dim (int): Total dimension of the input.
            target_indices (slice | set[int] | None, optional): Indices to
                transform.  Defaults to None, which transforms every other
                index (even positions).
            hidden_size (int | None, optional): Hidden size of the two MLPs.
                Defaults to None, which uses ``2 * n_in``.
            activation (ActivationFunction, optional): Activation function for
                the MLPs.  Defaults to relu.
            loss (LossFunction, optional): Loss function for the model.
                Defaults to squared_error_loss.
            reg_loss (RegLossFunction, optional): Regularization function for
                the MLP layers.  Defaults to LNormRegularization(2).
            dtype (DTypeLike, optional): Data type for the model parameters.
                Defaults to jnp.float32.

        Raises:
            MiniMLError: If both input and target index sets are not
                non-empty, or an index is out of range.
        """
        if dim <= 0:
            raise MiniMLError("dim must be a positive integer")

        if target_indices is None:
            target = self._slice_to_set(slice(None, None, 2), dim)
        elif isinstance(target_indices, slice):
            target = self._slice_to_set(target_indices, dim)
        else:
            target = set(target_indices)
        input = set(range(dim)) - target

        if any(i < 0 or i >= dim for i in target):
            raise MiniMLError(
                f"target_indices contain indices outside [0, {dim}): {sorted(target)}"
            )
        if len(target) == 0 or len(input) == 0:
            raise MiniMLError(
                "AffineCouplingLayer requires at least one input and one "
                f"target index (got input={sorted(input)}, target={sorted(target)})"
            )

        self._input_indices = jnp.array(sorted(input))
        self._target_indices = jnp.array(sorted(target))

        n_in, n_out = len(input), len(target)
        if hidden_size is None:
            n_hidden = 2 * n_in
        else:
            n_hidden = hidden_size

        layer_sizes = [n_in, n_hidden, n_out]
        self._mlp_q = MLP(
            layer_sizes,
            activation=activation,
            loss=loss,
            reg_loss=reg_loss,
            dtype=dtype,
        )
        self._mlp_p = MLP(
            layer_sizes,
            activation=activation,
            loss=loss,
            reg_loss=reg_loss,
            dtype=dtype,
        )

        super().__init__(loss=loss)

    @staticmethod
    def _slice_to_set(input: slice, dim: int) -> set:
        return set(range(dim)[input])

    def _coupling_values(
        self,
        X_in: JXArray,
        buffer: JXArray,
        rng_key: JXArray | None,
        mode: PredictMode,
        predict_kwargs: dict,
    ) -> "tuple[JXArray, JXArray, JXArray | None]":
        """Compute (q, p, activity_loss) for the coupling transformation.

        q is the strictly positive scale, p is the translation, both computed
        from the unchanged input set ``X_in``.  The activity loss is the sum of
        the two MLPs' own losses, or None if neither carries one.
        """
        activity_loss: JXArray | None = None
        values: list[JXArray] = []

        for mlp in (self._mlp_q, self._mlp_p):
            rng_key, subkey = (
                jax.random.split(rng_key) if rng_key is not None else (None, None)
            )
            result = mlp._predict_kernel(
                X_in,
                buffer,
                rng_key=subkey,
                mode=mode,
                **predict_kwargs,
            )
            value, activity_loss = MiniMLModel._unpack_kernel_output(
                result, activity_loss
            )
            values.append(value)

        q, p = values
        return jnp.exp(q), p, activity_loss

    def _predict_kernel(
        self,
        X: JXArray,
        buffer: JXArray,
        rng_key: JXArray | None = None,
        mode: PredictMode = PredictMode.INFERENCE,
        **predict_kwargs,
    ) -> "JXArray | PredictKernelOutput":
        X_t = X[..., self._target_indices]
        X_in = X[..., self._input_indices]
        q, p, activity_loss = self._coupling_values(
            X_in, buffer, rng_key, mode, predict_kwargs
        )
        Y_t = X_t * q + p
        Y = X.at[..., self._target_indices].set(Y_t)
        return MiniMLModel._with_activity_loss(Y, activity_loss)

    def _invert_kernel(
        self,
        Y: JXArray,
        buffer: JXArray,
        rng_key: JXArray | None = None,
        mode: PredictMode = PredictMode.INFERENCE,
        **invert_kwargs,
    ) -> "JXArray | PredictKernelOutput":
        Y_t = Y[..., self._target_indices]
        X_in = Y[..., self._input_indices]
        q, p, activity_loss = self._coupling_values(
            X_in, buffer, rng_key, mode, invert_kwargs
        )
        X_t = (Y_t - p) / q
        X = Y.at[..., self._target_indices].set(X_t)
        return MiniMLModel._with_activity_loss(X, activity_loss)

    def _log_det_jac_kernel(
        self,
        X: JXArray,
        buffer: JXArray,
        rng_key: JXArray | None = None,
        mode: PredictMode = PredictMode.INFERENCE,
        **predict_kwargs,
    ) -> JXArray:
        """Core kernel for :meth:`log_det_jac`.  Only ``_mlp_q`` is evaluated."""
        X_in = X[..., self._input_indices]
        result = self._mlp_q._predict_kernel(
            X_in,
            buffer,
            rng_key=rng_key,
            mode=mode,
            **predict_kwargs,
        )
        q, _ = MiniMLModel._unpack_kernel_output(result)
        return jnp.sum(q, axis=-1)

    def log_det_jac(self, X: JXArray, **predict_kwargs: dict[str, Any]) -> JXArray:
        r"""Log absolute determinant of the Jacobian of the transformation at ``X``.

        The input set passes through unchanged and each target component is
        scaled by its own :math:`\exp(q)`, so the Jacobian is triangular and

        $$
        \log\left|\det\frac{\partial y}{\partial x}\right| = \sum_t q_t(x_i)
        $$

        which is what generative flows add to the log likelihood of the
        transformed sample.  It is always finite, since the scale is an
        exponential and can not vanish.

        Args:
            X (JXArray): Input data, of shape ``(..., dim)``.
            **predict_kwargs: Additional named arguments for prediction.

        Returns:
            JXArray: The log determinant for each sample, of shape ``(...)``.
                Sum it to get the total for a batch.
        """
        if not hasattr(self, "_jit_log_det_jac_kernel"):

            def _inference_log_det_jac(
                X: JXArray, buffer: JXArray, **kwargs: Any
            ) -> JXArray:
                return self._log_det_jac_kernel(
                    X,
                    buffer=buffer,
                    rng_key=None,
                    mode=PredictMode.INFERENCE,
                    **kwargs,
                )

            self._jit_log_det_jac_kernel = jax.jit(_inference_log_det_jac, inline=True)

        return self._jit_log_det_jac_kernel(X, buffer=self._buffer, **predict_kwargs)
