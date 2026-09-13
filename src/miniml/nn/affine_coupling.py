import warnings
from abc import ABC, abstractmethod
from collections import Counter
from collections.abc import Callable, Iterable
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array as JXArray

from miniml.loss import (
    LossFunction,
    RegLossFunction,
    squared_error_loss,
)
from miniml.model import (
    InvertibleModel,
    MiniMLModel,
    MiniMLModelList,
    PredictKernelOutput,
    PredictMode,
)
from miniml.nn.activations import ActivationFunction, relu
from miniml.nn.mlp import MLP
from miniml.param import DTypeLike, MiniMLError

CouplingPartition = Callable[[int, int], "slice | set[int]"]
"""Rule choosing which indices a coupling layer transforms.

Takes the total dimension and the index of the layer in the stack, and returns
the target indices for that layer, in any form ``AffineCouplingLayer`` accepts.
"""


class CouplingPartitionBase(ABC):
    """Base class for partition rules that carry parameters of their own.

    A stack calls a rule once per layer while it is built, and then stores the
    indices it returned, so a rule is never needed again to load a saved model.
    Keeping it a pure function of ``(dim, layer_idx)`` is still worth it, since
    it is what makes building the same stack twice give the same model;
    :class:`RandomPartition` does so by deriving its draw from the layer index.
    """

    @abstractmethod
    def __call__(self, dim: int, layer_idx: int) -> "slice | set[int]":
        """Return the indices transformed by layer ``layer_idx``."""


def alternating_partition(dim: int, layer_idx: int) -> slice:
    """Checkerboard rule: even indices, then odd ones, then even again.

    This is the usual choice for a coupling flow, and guarantees that every
    dimension is transformed by half of the layers.

    Args:
        dim (int): Total dimension of the input.  Unused.
        layer_idx (int): Index of the layer in the stack.

    Returns:
        slice: Even indices on even layers, odd indices on odd ones.
    """
    return slice(layer_idx % 2, None, 2)


class RandomPartition(CouplingPartitionBase):
    """Rule picking a random subset of the indices for each layer.

    The draw is reproducible: it depends only on the seed and the layer index, so
    building the same stack twice gives the same partitions.
    """

    def __init__(self, seed: int = 0, frac: float = 0.5) -> None:
        """Construct a random partition rule.

        Args:
            seed (int, optional): Seed for the draw. Defaults to 0.
            frac (float, optional): Fraction of the dimensions to transform in
                each layer.  Rounded to at least one, and at most ``dim - 1``.
                Defaults to 0.5.

        Raises:
            MiniMLError: If the fraction is not between 0 and 1.
        """
        if not 0.0 < frac < 1.0:
            raise MiniMLError("frac must be strictly between 0 and 1")
        self._seed = int(seed)
        self._frac = float(frac)

    def __call__(self, dim: int, layer_idx: int) -> set[int]:
        """Draw the indices transformed by layer ``layer_idx``.

        Args:
            dim (int): Total dimension of the input.
            layer_idx (int): Index of the layer in the stack.

        Returns:
            set[int]: The drawn target indices.
        """
        key = jax.random.fold_in(jax.random.key(self._seed), layer_idx)
        n_target = min(max(1, round(self._frac * dim)), dim - 1)
        return set(jax.random.permutation(key, dim)[:n_target].tolist())


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
        rnd_scale: float = 1.0,
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
            rnd_scale (float, optional): Scale applied to the randomization of
                both MLPs.  The layer scales its target set by the exponential
                of ``_mlp_q``, so a smaller value starts it closer to the
                identity, which matters when several layers are chained; see
                :class:`AffineCouplingStack`.  Defaults to 1.0.
            dtype (DTypeLike, optional): Data type for the model parameters.
                Defaults to jnp.float32.

        Raises:
            MiniMLError: If both input and target index sets are not
                non-empty, or an index is out of range.
            MiniMLError: If the randomization scale is not positive.
        """
        if rnd_scale <= 0.0:
            raise MiniMLError(
                "rnd_scale must be positive: at zero both MLPs would be left all "
                "zeros, where they have no gradient to train from"
            )
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

        if rnd_scale != 1.0:
            for mlp in (self._mlp_q, self._mlp_p):
                for ref in mlp._get_inner_params():
                    ref.param.rnd_scale = ref.param.rnd_scale * rnd_scale

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


class AffineCouplingStack(InvertibleModel, MiniMLModel):
    """A stack of affine coupling layers, each transforming a different subset.

    A single :class:`AffineCouplingLayer` leaves part of its input untouched, so
    several are chained, with the transformed subset changing from one layer to
    the next, to let every dimension act on, and be acted on by, the others.  Which
    indices each layer transforms can be given directly, one set per layer, or
    left to a rule of the dimension and the layer index; see
    :data:`CouplingPartition`.  Either way the stack keeps the resolved indices.

    The stack is invertible as a whole: ``invert()`` runs the layers backwards,
    each undoing its own transformation.
    """

    def __init__(
        self,
        dim: int,
        n_layers: int | None = None,
        partition: "CouplingPartition | Iterable[slice | set[int]]" = (
            alternating_partition
        ),
        hidden_size: int | None = None,
        activation: ActivationFunction = relu,
        loss: LossFunction = squared_error_loss,
        reg_loss: RegLossFunction = MLP._REG_LOSS_DEFAULT,
        rnd_scale: float | None = None,
        dtype: DTypeLike = jnp.float32,
    ) -> None:
        """Construct a stack of affine coupling layers.

        Args:
            dim (int): Total dimension of the input.
            n_layers (int | None, optional): Number of coupling layers.  Defaults
                to None, which means the length of ``partition`` if it is a
                sequence, and 4 if it is a rule.
            partition (CouplingPartition | Iterable[slice | set[int]], optional):
                Either the target indices of each layer, one entry per layer, or
                a rule returning them, called as ``partition(dim, layer_idx)``.
                Defaults to :func:`alternating_partition`.  A rule is resolved
                into the indices themselves at construction, and it is those that
                get saved, so the rule is never called again.
            hidden_size (int | None, optional): Hidden size of the layers' MLPs.
                Defaults to None, which uses ``2 * n_in`` in each layer.
            activation (ActivationFunction, optional): Activation function for
                the MLPs.  Defaults to relu.
            loss (LossFunction, optional): Loss function for the model.
                Defaults to squared_error_loss.
            reg_loss (RegLossFunction, optional): Regularization function for
                the MLP layers.  Defaults to LNormRegularization(2).
            rnd_scale (float | None, optional): Scale applied to the
                randomization of every layer.  Defaults to None, which picks one
                per layer from how often each dimension is transformed; see
                :meth:`_rnd_scales`.
            dtype (DTypeLike, optional): Data type for the model parameters.
                Defaults to jnp.float32.

        Raises:
            MiniMLError: If the number of layers is not positive, or does not
                match the number of partitions given.
            MiniMLError: If a partition is neither a slice nor a set of indices.
            MiniMLError: If a partition is not a valid split of ``dim``.
        """
        targets = self._resolve_partitions(dim, n_layers, partition)

        untouched = set(range(dim)) - set().union(*targets)
        if untouched:
            warnings.warn(
                f"The partition never transforms dimensions {sorted(untouched)}: "
                "they will pass through the stack unchanged",
                stacklevel=2,
            )

        scales = (
            self._rnd_scales(targets)
            if rnd_scale is None
            else [rnd_scale] * len(targets)
        )

        self._dim = dim
        self._partitions = targets
        self._layer_list = MiniMLModelList(
            [
                AffineCouplingLayer(
                    dim,
                    target_indices=target,
                    hidden_size=hidden_size,
                    activation=activation,
                    loss=loss,
                    reg_loss=reg_loss,
                    rnd_scale=scale,
                    dtype=dtype,
                )
                for target, scale in zip(targets, scales)
            ]
        )
        super().__init__(loss=loss)

        # Save the resolved indices rather than the rule that produced them, so
        # that loading the model back does not depend on calling it again
        self._replace_init_args(
            dim,
            n_layers=len(targets),
            partition=[set(target) for target in targets],
            hidden_size=hidden_size,
            activation=activation,
            loss=loss,
            reg_loss=reg_loss,
            rnd_scale=rnd_scale,
            dtype=dtype,
        )

    @staticmethod
    def _rnd_scales(targets: list[set[int]]) -> list[float]:
        r"""Pick a randomization scale for each layer, from the partitions.

        A dimension transformed by $k$ layers has each of them multiply it by the
        exponential of its own scale network, so the exponents add up and their
        spread grows as $\sqrt{k}$.  Damping each layer by $1/\sqrt{2k}$ keeps
        that total spread around $1/\sqrt{2}$, whatever the depth, which keeps a
        freshly randomized stack in a range where it stays invertible in single
        precision.  Each layer is damped for the most often transformed of its own
        dimensions, so an uneven partition is handled as well as a regular one.

        For the usual checkerboard this is exactly $1/\sqrt{n_{layers}}$, since
        every dimension is then transformed by half of the layers.

        Args:
            targets (list[set[int]]): The target indices of each layer.

        Returns:
            list[float]: The randomization scale of each layer.
        """
        counts = Counter(d for target in targets for d in target)
        return [
            float(1.0 / np.sqrt(2 * max(counts[d] for d in target)))
            for target in targets
        ]

    @staticmethod
    def _resolve_partitions(
        dim: int,
        n_layers: int | None,
        partition: "CouplingPartition | Iterable[slice | set[int]]",
    ) -> list[set[int]]:
        """Reduce a partition rule or sequence to one set of indices per layer.

        Args:
            dim (int): Total dimension of the input.
            n_layers (int | None): Requested number of layers, if any.
            partition (CouplingPartition | Iterable[slice | set[int]]): The rule
                or the sequence of target indices.

        Raises:
            MiniMLError: If the number of layers is not positive, or does not
                match the number of partitions given.
            MiniMLError: If a partition is neither a slice nor a set of indices.

        Returns:
            list[set[int]]: The target indices of each layer.
        """
        if callable(partition):
            n_layers = 4 if n_layers is None else n_layers
            if n_layers <= 0:
                raise MiniMLError("n_layers must be a positive integer")
            raw = [partition(dim, i) for i in range(n_layers)]
        else:
            raw = list(partition)
            if len(raw) == 0:
                raise MiniMLError("n_layers must be a positive integer")
            if n_layers is not None and n_layers != len(raw):
                raise MiniMLError(
                    f"n_layers is {n_layers}, but {len(raw)} partitions were given"
                )

        targets = []
        for i, target in enumerate(raw):
            if isinstance(target, slice):
                targets.append(AffineCouplingLayer._slice_to_set(target, dim))
            else:
                try:
                    targets.append({int(t) for t in target})
                except TypeError:
                    raise MiniMLError(
                        f"Partition {i} is neither a slice nor a set of indices: {target!r}"
                    )
        return targets

    @property
    def dim(self) -> int:
        """Total dimension of the input."""
        return self._dim

    @property
    def partitions(self) -> list[set[int]]:
        """The indices transformed by each layer, in the order they are applied."""
        return [set(target) for target in self._partitions]

    @property
    def layers(self) -> list[MiniMLModel]:
        """The coupling layers, in the order they are applied."""
        return self._layer_list.contents

    def _predict_kernel(
        self,
        X: JXArray,
        buffer: JXArray,
        rng_key: JXArray | None = None,
        mode: PredictMode = PredictMode.INFERENCE,
        **predict_kwargs,
    ) -> "JXArray | PredictKernelOutput":
        activity_loss: JXArray | None = None
        for layer in self.layers:
            rng_key, subkey = (
                jax.random.split(rng_key) if rng_key is not None else (None, None)
            )
            result = layer._predict_kernel(
                X,
                buffer,
                rng_key=subkey,
                mode=mode,
                **predict_kwargs,
            )
            X, activity_loss = MiniMLModel._unpack_kernel_output(result, activity_loss)
        return MiniMLModel._with_activity_loss(X, activity_loss)

    def _invert_kernel(
        self,
        Y: JXArray,
        buffer: JXArray,
        rng_key: JXArray | None = None,
        mode: PredictMode = PredictMode.INFERENCE,
        **invert_kwargs,
    ) -> "JXArray | PredictKernelOutput":
        activity_loss: JXArray | None = None
        for layer in reversed(self.layers):
            rng_key, subkey = (
                jax.random.split(rng_key) if rng_key is not None else (None, None)
            )
            result = layer._invert_kernel(
                Y,
                buffer,
                rng_key=subkey,
                mode=mode,
                **invert_kwargs,
            )
            Y, activity_loss = MiniMLModel._unpack_kernel_output(result, activity_loss)
        return MiniMLModel._with_activity_loss(Y, activity_loss)

    def _log_det_jac_kernel(
        self,
        X: JXArray,
        buffer: JXArray,
        rng_key: JXArray | None = None,
        mode: PredictMode = PredictMode.INFERENCE,
        **predict_kwargs,
    ) -> JXArray:
        """Core kernel for :meth:`log_det_jac`.

        Each layer contributes the log determinant at its own input, so the
        running value has to be transformed as it goes.
        """
        log_det_jac = jnp.zeros((), dtype=self._dtype)
        for layer in self.layers:
            log_det_jac = log_det_jac + layer._log_det_jac_kernel(
                X,
                buffer,
                rng_key=rng_key,
                mode=mode,
                **predict_kwargs,
            )
            result = layer._predict_kernel(
                X,
                buffer,
                rng_key=rng_key,
                mode=mode,
                **predict_kwargs,
            )
            X, _ = MiniMLModel._unpack_kernel_output(result)
        return log_det_jac

    def log_det_jac(self, X: JXArray, **predict_kwargs: dict[str, Any]) -> JXArray:
        r"""Log absolute determinant of the Jacobian of the whole stack at ``X``.

        The layers are applied one after the other, so the determinants multiply
        and their logarithms add up, each evaluated at the input of its own layer.

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
