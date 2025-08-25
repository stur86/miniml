import numpy as np
from jax import Array
from typing import Protocol
from numpy.typing import DTypeLike
from miniml.utils import ImmutableBiDict
from miniml.loss import RegLossFunction

class MiniMLError(Exception):
    pass

_supported_types = ImmutableBiDict([
    ("float16", np.float16),
    ("float32", np.float32),
    ("float64", np.float64),
    ("float128", np.float128),
    ("complex64", np.complex64),
    ("complex128", np.complex128),
    ("complex256", np.complex256),
])

class BufferContainer(Protocol):
    _buffer: Array

class MiniMLParam:
    """MiniML Parameter"""


    _shape: tuple[int, ...]
    _dtype: DTypeLike
    _dtype_name: str
    _size: int
    _reg_loss: RegLossFunction | None = None

    _bufc: BufferContainer | None = None
    _buf_i0: int = -1

    def __init__(self, shape: tuple[int, ...], dtype: DTypeLike = np.float32, 
                 reg_loss: RegLossFunction | None = None) -> None:
        """Construct a MiniML Parameter

        Args:
            shape (tuple[int,...]): The shape of the parameter.
            dtype (DTypeLike, optional): The data type of the parameter. Defaults to np.float32.
            reg_loss (RegLossFunction, optional): The regularization loss function. Defaults to None.
        """
        if dtype not in _supported_types.values():
            raise MiniMLError(f"Parameter dtype {dtype} not supported")
        
        self._shape = shape
        self._dtype = dtype
        self._dtype_name = _supported_types.get_inverse(dtype)  # type: ignore
        self._size = int(np.prod(shape))
        self._reg_loss = reg_loss

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

    def bind(self, i0: int, bufc: BufferContainer) -> None:
        """Bind the parameter to a buffer container. The
        actual buffer inside may be swapped.

        Args:
            i0 (int): The starting index in the buffer.
            buf (BufferContainer): The container for the
                buffer to bind to.

        Raises:
            MiniMLError: If the buffer is not 1-dimensional.
            MiniMLError: If the buffer is too small.
            MiniMLError: If the parameter is already bound.
        """

        if self.bound:
            raise MiniMLError("Parameter already bound to buffer")

        i1 = i0 + self.size
        buf = bufc._buffer
        if buf.ndim != 1:
            raise MiniMLError("Buffer must be 1-dimensional")
        if i1 > len(buf):
            raise MiniMLError(
                f"Buffer is too small for parameter of shape {self.shape} counting from index {i0}"
            )
        self._bufc = bufc
        self._buf_i0 = i0

    def regularization_loss(self) -> float:
        if self._reg_loss is None:
            return 0.0
        return self._reg_loss(self.value)

    def unbind(self) -> None:
        self._bufc = None

    @property
    def bound(self) -> bool:
        return self._bufc is not None

    @property
    def value(self) -> Array:
        i0 = self._buf_i0
        i1 = i0 + self.size
        if self._bufc is None:
            raise MiniMLError("Parameter not bound to buffer")
        return self._bufc._buffer[i0:i1].reshape(self.shape)

    def __repr__(self) -> str:
        return f"MiniMLParam[{self.dtype}] ({self.shape})"
