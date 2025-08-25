import numpy as np
from jax import Array as JXArray
import jax.numpy as jnp
from typing import Protocol
from numpy.typing import DTypeLike
from miniml.utils import ImmutableBiDict
from miniml.loss import RegLossFunction

class MiniMLError(Exception):
    pass

_supported_types = ImmutableBiDict([
    ("float16", jnp.float16),
    ("float32", jnp.float32),
    ("float64", jnp.float64),
    ("complex64", jnp.complex64),
    ("complex128", jnp.complex128),
])

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

    def regularization_loss(self) -> JXArray:
        """Returns the regularization loss for this parameter

        Returns:
            JXArray: Regularization loss, should be a scalar
        """
        if self._reg_loss is None:
            return jnp.array(0.0, dtype=jnp.dtype(self._dtype))
        return self._reg_loss(self.value)

    def unbind(self) -> None:
        """Unbind this parameter from its buffer container."""
        self._bufc = None
        
    @property
    def bound(self) -> bool:
        """Whether the parameter is bound to a buffer."""
        return self._bufc is not None

    @property
    def value(self) -> JXArray:
        """Tensor value of the parameter"""
        i0 = self._buf_i0
        i1 = i0 + self.size
        if self._bufc is None:
            raise MiniMLError("Parameter not bound to buffer")
        return self._bufc._buffer[i0:i1].reshape(self.shape)

    def __repr__(self) -> str:
        return f"MiniMLParam[{self.dtype}] ({self.shape})"
