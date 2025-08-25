import pytest
import jax.numpy as jnp
import numpy as np
from miniml.param import MiniMLError, MiniMLParam

class MockContainer:
    _buffer: jnp.ndarray
    
    def __init__(self) -> None:
        self._buffer = jnp.zeros((10,), dtype=jnp.float32)

def test_param():
    p = MiniMLParam((3, 2), np.float32)
    assert p.shape == (3, 2)
    assert p.size == 6
    assert p.dtype == np.float32
    assert p.dtype_name == "float32"
    
    # Try binding
    bufc = MockContainer()
    assert not p.bound
    p.bind(0, bufc)
    assert p.bound
    assert np.array_equal(p.value, bufc._buffer[0:6].reshape((3, 2)))
    p.unbind()

    p.bind(2, bufc)
    assert p.bound
    assert np.array_equal(p.value, bufc._buffer[2:8].reshape((3, 2)))

def test_param_errors():
    
    with pytest.raises(MiniMLError, match="Parameter dtype .* not supported"):
        MiniMLParam((3, 2), np.int32)

    p = MiniMLParam((3, 2), np.float32)
    bufc = MockContainer()
    bufc._buffer = jnp.zeros((2, 5), dtype=jnp.float32)

    with pytest.raises(MiniMLError, match="Buffer must be 1-dimensional"):
        p.bind(0, bufc)
        
    bufc._buffer = jnp.zeros((5,), dtype=jnp.float32)

    with pytest.raises(MiniMLError, match="Buffer is too small for parameter of shape"):
        p.bind(5, bufc)

    bufc._buffer = jnp.zeros((10,), dtype=jnp.float32)
    p.bind(0, bufc)
    with pytest.raises(MiniMLError, match="Parameter already bound to buffer"):
        p.bind(1, bufc)
    p.unbind()
    
    with pytest.raises(MiniMLError, match="Parameter not bound to buffer"):
        _ = p.value