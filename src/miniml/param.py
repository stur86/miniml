import numpy as np
from numpy.typing import DTypeLike, NDArray

class MiniMLError(Exception):
    pass

class MiniMLParam:
    """MiniML Parameter
    """
    
    _shape: tuple[int,...]
    _dtype: DTypeLike
    _size: int

    _buf: NDArray
    
    def __init__(self, shape: tuple[int,...], dtype: DTypeLike = np.float32) -> None:
        """Construct a MiniML Parameter

        Args:
            shape (tuple[int,...]): The shape of the parameter.
            dtype (DTypeLike, optional): The data type of the parameter. Defaults to np.float32.
        """
        self._shape = shape
        self._dtype = dtype
        self._size = int(np.prod(shape))
            
    @property
    def shape(self) -> tuple[int,...]:
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
    
    def bind(self, i0: int, buf: NDArray) -> None:
        """Bind the parameter to a buffer.

        Args:
            i0 (int): The starting index in the buffer.
            buf (NDArray): The buffer to bind to.

        Raises:
            MiniMLError: If the buffer is not 1-dimensional.
            MiniMLError: If the buffer is too small.
            MiniMLError: If the parameter is already bound.
        """

        if self.bound:
            raise MiniMLError("Parameter already bound to buffer")
        
        i1 = i0 + self.size
        if buf.ndim != 1:
            raise MiniMLError("Buffer must be 1-dimensional")
        if i1 > len(buf):
            raise MiniMLError(f"Buffer is too small for parameter of shape {self.shape} counting from index {i0}")
        v = buf[i0:i1].reshape(self.shape)
        v.flags.writeable = False
        self._buf = v
        
    def unbind(self) -> None:
        del self._buf
        
    @property
    def bound(self) -> bool:
        return hasattr(self, "_buf")
    
    @property
    def value(self) -> NDArray:
        if not self.bound:
            raise MiniMLError("Parameter not bound to buffer")
        return self._buf
        
    def __repr__(self) -> str:
        return f"MiniMLParam[{self.dtype}] ({self.shape})"
    