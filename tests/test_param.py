import pytest
import numpy as np
from miniml.param import MiniMLError, MiniMLParam

def test_param():
    p = MiniMLParam((3, 2), np.float32)
    assert p.shape == (3, 2)
    assert p.size == 6
    assert p.dtype == np.float32
    assert p.dtype_name == "float32"
    
    # Try binding
    buf = np.arange(10, dtype=np.float32)
    assert not p.bound
    p.bind(0, buf)
    assert p.bound
    assert np.array_equal(p.value, buf[0:6].reshape((3, 2)))
    p.unbind()    

    p.bind(2, buf)
    assert p.bound
    assert np.array_equal(p.value, buf[2:8].reshape((3, 2)))
    
def test_param_errors():
    
    with pytest.raises(MiniMLError, match="Parameter dtype .* not supported"):
        MiniMLParam((3, 2), np.int32)

    p = MiniMLParam((3, 2), np.float32)
    buf = np.arange(10, dtype=np.float32)

    with pytest.raises(MiniMLError, match="Buffer must be 1-dimensional"):
        p.bind(0, buf.reshape((2, 5)))
        
    with pytest.raises(MiniMLError, match="Buffer is too small for parameter of shape"):
        p.bind(5, buf)
        
    p.bind(0, buf)
    with pytest.raises(MiniMLError, match="Parameter already bound to buffer"):
        p.bind(1, buf)
    p.unbind()
    
    with pytest.raises(MiniMLError, match="Parameter not bound to buffer"):
        _ = p.value