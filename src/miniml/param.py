import numpy as np
from dataclasses import dataclass
from jax import Array as JXArray
import jax.numpy as jnp
from typing import Protocol
from numpy.typing import DTypeLike
from miniml.utils import ImmutableBiDict
from miniml.loss import RegLossFunction


class MiniMLError(Exception):
    """Class for errors raised by MiniML"""

    pass


_supported_types = ImmutableBiDict(
    [
        ("float16", jnp.float16),
        ("float32", jnp.float32),
        ("float64", jnp.float64),
        ("complex64", jnp.complex64),
        ("complex128", jnp.complex128),
    ]
)


class BufferContainer(Protocol):
    _buffer: JXArray


class MiniMLParam:
    """MiniML Parameter"""

    _shape: tuple[int, ...]
    _dtype: DTypeLike
    _dtype_name: str
    _size: int
    _reg_loss: RegLossFunction | None = None

    _bufc: BufferContainer | None = None
    _buf_i0: int = -1

    def __init__(
        self,
        shape: tuple[int, ...],
        dtype: DTypeLike = np.float32,
        reg_loss: RegLossFunction | None = None,
        reg_scale: float = 1.0,
    ) -> None:
        """Construct a MiniML Parameter

        Args:
            shape (tuple[int,...]): The shape of the parameter.
            dtype (DTypeLike, optional): The data type of the parameter. Defaults to np.float32.
            reg_loss (RegLossFunction, optional): The regularization loss function. Defaults to None.
            reg_scale (float, optional): The scale factor for the regularization loss. Defaults to 1.0.
        """
        if dtype not in _supported_types.values():
            raise MiniMLError(f"Parameter dtype {dtype} not supported")

        self._shape = shape
        self._dtype = dtype
        self._dtype_name = _supported_types.get_inverse(dtype)  # type: ignore
        self._size = int(np.prod(shape))
        self._reg_loss = reg_loss
        self._reg_scale = reg_scale

    @property
    def shape(self) -> tuple[int, ...]:
        """The shape of the parameter."""
        return self._shape

    @property
    def size(self) -> int:
        """The size of the parameter."""
        return self._size

    @property
    def dtype(self) -> DTypeLike:
        """The data type of the parameter."""
        return self._dtype

    @property
    def dtype_name(self) -> str:
        """The name of the data type of the parameter."""
        return self._dtype_name

    def _validate_buffer(self, i0: int, buf: JXArray) -> None:
        """Validate that a buffer and starting index pair
        is suitable for this parameter.

        Args:
            i0 (int): The starting index in the buffer.
            buf (JXArray): The buffer to validate.

        Raises:
            MiniMLError: If the buffer is not 1-dimensional.
            MiniMLError: If the buffer is too small.
        """
        i1 = i0 + self.size
        if buf.ndim != 1:
            raise MiniMLError("Buffer must be 1-dimensional")
        if i1 > len(buf):
            raise MiniMLError(
                f"Buffer is too small for parameter of shape {self.shape} counting from index {i0}"
            )

    def bind(self, i0: int, bufc: BufferContainer) -> None:
        """Bind the parameter to a buffer container. The
        actual buffer inside may be swapped.

        Args:
            i0 (int): The starting index in the buffer.
            bufc (BufferContainer): The container for the
                buffer to bind to.

        Raises:
            MiniMLError: If the buffer is not 1-dimensional.
            MiniMLError: If the buffer is too small.
            MiniMLError: If the parameter is already bound.
        """

        if self.bound:
            raise MiniMLError("Parameter already bound to buffer")

        self._validate_buffer(i0, bufc._buffer)
        self._bufc = bufc
        self._buf_i0 = i0

    def regularization_loss(self, buffer: JXArray | None = None) -> JXArray:
        """Returns the regularization loss for this parameter

        Args:
            buffer (JXArray | None, optional): The buffer to use. If None, the bound buffer is used.
                Defaults to None.

        Returns:
            JXArray: Regularization loss, should be a scalar
        """
        if self._reg_loss is None:
            return jnp.array(0.0, dtype=jnp.dtype(self._dtype))
        return self._reg_loss(self(buffer)) * self._reg_scale

    def unbind(self) -> None:
        """Unbind this parameter from its buffer container."""
        self._buf_i0 = -1
        self._bufc = None

    @property
    def bound(self) -> bool:
        """Whether the parameter is bound to a buffer."""
        return self._bufc is not None

    def __call__(self, buffer: JXArray | None = None) -> JXArray:
        """Get the value of the parameter. If a buffer is provided,
        it is used instead of the bound buffer.

        Args:
            buffer (JXArray | None, optional): The buffer to use. Defaults to None.

        Raises:
            MiniMLError: If the parameter is not bound and no buffer is provided.

        Returns:
            JXArray: The value of the parameter.
        """

        i0 = self._buf_i0
        i1 = i0 + self.size
        if buffer is None:
            if self._bufc is None:
                raise MiniMLError("Parameter not bound to buffer")
            buffer = self._bufc._buffer
        else:
            self._validate_buffer(i0, buffer)
        return buffer[i0:i1].reshape(self.shape)

    @property
    def value(self) -> JXArray:
        """Get the value of the parameter from the bound buffer.
        Syntactic sugar for `param()`.

        Raises:
            MiniMLError: If the parameter is not bound.
        Returns:
            JXArray: The value of the parameter.
        """
        return self()

    def __repr__(self) -> str:
        return f"MiniMLParam[{self.dtype}] ({self.shape})"

    def _get_inner_params(self) -> list["MiniMLParamRef"]:
        return [MiniMLParamRef("v", self)]


@dataclass(frozen=True)
class MiniMLParamRef:
    path: str
    param: MiniMLParam

    def as_child(self, parent_path: str) -> "MiniMLParamRef":
        return MiniMLParamRef(f"{parent_path}.{self.path}", self.param)


class MiniMLParamList:
    """A list of parameters"""

    _contents: list[MiniMLParam]

    def __init__(self, contents: list[MiniMLParam]) -> None:
        """Initialize the list of parameters.

        Args:
            contents (list[MiniMLParam]): The list of parameters to include.
        """
        self._contents = contents

    @property
    def contents(self) -> list[MiniMLParam]:
        """The list of parameters."""
        return self._contents

    def __getitem__(self, index: int) -> MiniMLParam:
        """Access a parameter by index."""
        return self._contents[index]

    def __len__(self) -> int:
        """Total length of the list."""
        return len(self._contents)

    def _get_inner_params(self) -> list[MiniMLParamRef]:
        return [
            param._get_inner_params()[0].as_child(f"{i}")
            for i, param in enumerate(self._contents)
        ]
