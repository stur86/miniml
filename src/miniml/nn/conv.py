from typing import Any, Sequence

import numpy as np
from jax import lax, numpy as jnp, Array as JXArray
from numpy.typing import DTypeLike
from miniml.model import MiniMLModel, PredictMode
from miniml.param import MiniMLParam, MiniMLError
from miniml.loss import (
    LossFunction,
    squared_error_loss,
    RegLossFunction,
    LNormRegularization,
)

_IntOrTuple = int | Sequence[int]
_Padding = str | int | Sequence[int | tuple[int, int]]


def _expand_to_ndim(value: _IntOrTuple, n_dim: int, name: str) -> tuple[int, ...]:
    """Expand an int or a sequence of ints into a tuple of length `n_dim`.

    Args:
        value (int | Sequence[int]): The value to expand.
        n_dim (int): Number of spatial dimensions.
        name (str): Name of the argument, used in error messages.

    Raises:
        MiniMLError: If a sequence is passed whose length is not `n_dim`.

    Returns:
        tuple[int, ...]: The expanded tuple.
    """
    if isinstance(value, int):
        return (value,) * n_dim
    expanded = tuple(int(v) for v in value)
    if len(expanded) != n_dim:
        raise MiniMLError(f"{name} must be an int or a sequence of {n_dim} ints")
    return expanded


class Conv(MiniMLModel):
    """A MiniML model that applies an n-dimensional convolution to the input data."""

    _n_dim: int
    _in_channels: int
    _out_channels: int
    _kernel_size: tuple[int, ...]
    _stride: tuple[int, ...]
    _dilation: tuple[int, ...]
    _groups: int
    _padding: str | tuple[tuple[int, int], ...]

    def __init__(
        self,
        n_dim: int,
        in_channels: int,
        out_channels: int,
        kernel_size: _IntOrTuple,
        stride: _IntOrTuple = 1,
        padding: _Padding = "valid",
        dilation: _IntOrTuple = 1,
        groups: int = 1,
        bias: bool = True,
        loss: LossFunction = squared_error_loss,
        reg_loss: RegLossFunction = LNormRegularization(2),
        dtype: DTypeLike = jnp.float32,
        apply_bias_reg: bool = False,
    ) -> None:
        r"""Convolutional layer working on an arbitrary number of spatial dimensions.

        For an input $X$ of shape `(batch, in_channels, *spatial)` the output is

        $$
        \hat{y}_{n,o,\mathbf{i}} = b_o + \sum_{c}\sum_{\mathbf{k}}
            W_{o,c,\mathbf{k}} X_{n,c,\,s\mathbf{i}+d\mathbf{k}}
        $$

        where $\mathbf{i}$ and $\mathbf{k}$ are multi-indices over the spatial and kernel
        dimensions respectively, and $s$ and $d$ are stride and dilation. Inputs are in
        channels-first layout; an unbatched input of shape `(in_channels, *spatial)` is
        also accepted and returns an unbatched output.

        ```py
        # A 2D convolution on 1-channel 8x8 images, producing 4 channels
        conv = Conv(2, 1, 4, kernel_size=3, padding="same")
        conv.randomize()
        y = conv.predict(X)  # X has shape (batch, 1, 8, 8), y has shape (batch, 4, 8, 8)
        ```

        Args:
            n_dim (int): Number of spatial dimensions the convolution runs over.
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            kernel_size (int | Sequence[int]): Size of the convolution kernel. An int is
                used for all spatial dimensions.
            stride (int | Sequence[int], optional): Stride of the convolution. Defaults to 1.
            padding (str | int | Sequence[int | tuple[int, int]], optional): Padding applied to
                the input. Can be `"valid"` (no padding), `"same"` (output keeps the input
                size for unit stride), an int applied symmetrically to all spatial dimensions,
                or a sequence with one int or `(low, high)` pair per spatial dimension.
                Defaults to "valid".
            dilation (int | Sequence[int], optional): Dilation of the kernel. Defaults to 1.
            groups (int, optional): Number of groups the channels are split into. Both
                `in_channels` and `out_channels` must be divisible by it. Defaults to 1.
            bias (bool, optional): Whether to add a per-channel bias term. Defaults to True.
            loss (LossFunction, optional): Loss function for the model. Defaults to squared_error_loss.
            reg_loss (RegLossFunction, optional): Regularization function for the weights.
                Defaults to LNormRegularization(2).
            dtype (DTypeLike, optional): Data type for the model parameters. Defaults to jnp.float32.
            apply_bias_reg (bool, optional): Whether to apply regularization to the bias term.
                Defaults to False.

        Raises:
            MiniMLError: If any of the arguments is invalid or inconsistent with the others.
        """

        if n_dim < 1:
            raise MiniMLError("n_dim must be a positive integer")
        if in_channels < 1 or out_channels < 1:
            raise MiniMLError("in_channels and out_channels must be positive integers")
        if groups < 1:
            raise MiniMLError("groups must be a positive integer")
        if in_channels % groups != 0 or out_channels % groups != 0:
            raise MiniMLError(
                f"in_channels ({in_channels}) and out_channels ({out_channels}) must both be divisible by groups ({groups})"
            )

        self._n_dim = n_dim
        self._in_channels = in_channels
        self._out_channels = out_channels
        self._kernel_size = _expand_to_ndim(kernel_size, n_dim, "kernel_size")
        self._stride = _expand_to_ndim(stride, n_dim, "stride")
        self._dilation = _expand_to_ndim(dilation, n_dim, "dilation")
        self._groups = groups
        self._padding = self._parse_padding(padding)

        if any(k < 1 for k in self._kernel_size):
            raise MiniMLError("kernel_size values must be positive integers")
        if any(s < 1 for s in self._stride):
            raise MiniMLError("stride values must be positive integers")
        if any(d < 1 for d in self._dilation):
            raise MiniMLError("dilation values must be positive integers")

        # Convolution over batch/channel dimensions plus the spatial ones:
        # (batch, channels, *spatial) for input and output, (out, in, *kernel) for weights
        spec = tuple(range(n_dim + 2))
        self._dim_numbers = lax.ConvDimensionNumbers(spec, spec, spec)

        fan_in = (in_channels // groups) * int(np.prod(self._kernel_size))
        self._W = MiniMLParam(
            (out_channels, in_channels // groups) + self._kernel_size,
            dtype=dtype,
            reg_loss=reg_loss,
            rnd_scale=1.0 / np.sqrt(fan_in),
        )
        if bias:
            bias_reg = reg_loss if apply_bias_reg else None
            self._b = MiniMLParam(
                (out_channels,), dtype=dtype, reg_loss=bias_reg, rnd_scale=0.0
            )

        super().__init__(loss)

    def _parse_padding(self, padding: _Padding) -> str | tuple[tuple[int, int], ...]:
        """Normalize the padding argument into a form accepted by `jax.lax`.

        Args:
            padding (str | int | Sequence[int | tuple[int, int]]): The padding specification.

        Raises:
            MiniMLError: If the padding specification is invalid.

        Returns:
            str | tuple[tuple[int, int], ...]: Either "SAME"/"VALID" or one
                `(low, high)` pair per spatial dimension.
        """
        if isinstance(padding, str):
            pad_name = padding.upper()
            if pad_name not in ("SAME", "VALID"):
                raise MiniMLError(
                    f"Invalid padding mode {padding}; must be 'same' or 'valid'"
                )
            return pad_name

        if isinstance(padding, int):
            padding = (padding,) * self._n_dim

        pad_seq = tuple(padding)
        if len(pad_seq) != self._n_dim:
            raise MiniMLError(
                f"padding must be a string, an int, or a sequence of {self._n_dim} elements"
            )

        pads: list[tuple[int, int]] = []
        for p in pad_seq:
            if isinstance(p, int):
                pads.append((p, p))
            else:
                pair = tuple(int(v) for v in p)
                if len(pair) != 2:
                    raise MiniMLError(
                        "Per-dimension padding must be an int or a (low, high) pair"
                    )
                pads.append((pair[0], pair[1]))
        if any(lo < 0 or hi < 0 for lo, hi in pads):
            raise MiniMLError("padding values must be non-negative")
        return tuple(pads)

    @property
    def n_dim(self) -> int:
        """Number of spatial dimensions of the convolution."""
        return self._n_dim

    @property
    def in_channels(self) -> int:
        """Number of input channels."""
        return self._in_channels

    @property
    def out_channels(self) -> int:
        """Number of output channels."""
        return self._out_channels

    @property
    def kernel_size(self) -> tuple[int, ...]:
        """Size of the convolution kernel along each spatial dimension."""
        return self._kernel_size

    @property
    def stride(self) -> tuple[int, ...]:
        """Stride of the convolution along each spatial dimension."""
        return self._stride

    @property
    def dilation(self) -> tuple[int, ...]:
        """Dilation of the kernel along each spatial dimension."""
        return self._dilation

    @property
    def groups(self) -> int:
        """Number of groups the channels are split into."""
        return self._groups

    @property
    def padding(self) -> str | tuple[tuple[int, int], ...]:
        """The normalized padding specification."""
        return self._padding

    @property
    def has_bias(self) -> bool:
        """Whether the layer includes a bias term."""
        return hasattr(self, "_b")

    def _predict_kernel(
        self,
        X: JXArray,
        buffer: JXArray,
        rng_key: JXArray | None = None,
        mode: PredictMode = PredictMode.INFERENCE,
        **predict_kwargs: Any,
    ) -> JXArray:
        unbatched = X.ndim == self._n_dim + 1
        if unbatched:
            X = X[None]
        elif X.ndim != self._n_dim + 2:
            raise MiniMLError(
                f"Input to a {self._n_dim}D convolution must have {self._n_dim + 1} or {self._n_dim + 2} dimensions, found {X.ndim}"
            )
        if X.shape[1] != self._in_channels:
            raise MiniMLError(
                f"Input has {X.shape[1]} channels, expected {self._in_channels}"
            )

        y = lax.conv_general_dilated(
            X,
            self._W(buffer),
            window_strides=self._stride,
            padding=self._padding,
            rhs_dilation=self._dilation,
            dimension_numbers=self._dim_numbers,
            feature_group_count=self._groups,
        )

        if self.has_bias:
            y = y + self._b(buffer).reshape((1, -1) + (1,) * self._n_dim)

        return y[0] if unbatched else y
