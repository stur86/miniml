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
            self._target = self._slice_to_set(slice(None, None, 2), dim)
        elif isinstance(target_indices, slice):
            self._target = self._slice_to_set(target_indices, dim)
        else:
            self._target = set(target_indices)
        self._input = set(range(dim)) - self._target

        if any(i < 0 or i >= dim for i in self._target):
            raise MiniMLError(
                f"target_indices contain indices outside [0, {dim}): {sorted(self._target)}"
            )
        if len(self._target) == 0 or len(self._input) == 0:
            raise MiniMLError(
                "AffineCouplingLayer requires at least one input and one "
                f"target index (got input={sorted(self._input)}, target={sorted(self._target)})"
            )

        self._input_indices = jnp.array(sorted(self._input))
        self._target_indices = jnp.array(sorted(self._target))

        n_in, n_out = len(self._input), len(self._target)
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
    ) -> "tuple[JXArray, JXArray, JXArray]":
        """Compute (q, p, activity_loss) for the coupling transformation.

        q is the strictly positive scale, p is the translation, both computed
        from the unchanged input set ``X_in``.
        """
        total_activity_loss = jnp.zeros((), dtype=self._dtype)

        rng_key, q_key = (
            jax.random.split(rng_key) if rng_key is not None else (None, None)
        )
        q_result = self._mlp_q._predict_kernel(
            X_in,
            buffer,
            rng_key=q_key,
            mode=mode,
            **predict_kwargs,
        )
        q, activity_loss = MiniMLModel._unpack_kernel_output(q_result)
        total_activity_loss = total_activity_loss + activity_loss

        rng_key, p_key = (
            jax.random.split(rng_key) if rng_key is not None else (None, None)
        )
        p_result = self._mlp_p._predict_kernel(
            X_in,
            buffer,
            rng_key=p_key,
            mode=mode,
            **predict_kwargs,
        )
        p, activity_loss = MiniMLModel._unpack_kernel_output(p_result)
        total_activity_loss = total_activity_loss + activity_loss

        return jnp.exp(q), p, total_activity_loss

    def _predict_kernel(
        self,
        X: JXArray,
        buffer: JXArray,
        rng_key: JXArray | None = None,
        mode: PredictMode = PredictMode.INFERENCE,
        **predict_kwargs,
    ) -> PredictKernelOutput:
        X_t = X[..., self._target_indices]
        X_in = X[..., self._input_indices]
        q, p, activity_loss = self._coupling_values(
            X_in, buffer, rng_key, mode, predict_kwargs
        )
        Y_t = X_t * q + p
        Y = X.at[..., self._target_indices].set(Y_t)
        return PredictKernelOutput(y_pred=Y, activity_loss=activity_loss)

    def _invert_kernel(
        self,
        Y: JXArray,
        buffer: JXArray,
        rng_key: JXArray | None = None,
        mode: PredictMode = PredictMode.INFERENCE,
        **invert_kwargs,
    ) -> PredictKernelOutput:
        Y_t = Y[..., self._target_indices]
        X_in = Y[..., self._input_indices]
        q, p, activity_loss = self._coupling_values(
            X_in, buffer, rng_key, mode, invert_kwargs
        )
        X_t = (Y_t - p) / q
        X = Y.at[..., self._target_indices].set(X_t)
        return PredictKernelOutput(y_pred=X, activity_loss=activity_loss)
