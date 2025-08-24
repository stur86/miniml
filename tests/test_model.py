from pathlib import Path
import numpy as np
from miniml.param import MiniMLParam
from miniml.model import MiniMLModel


def test_model(tmp_path: Path):
    
    class ConstantModel(MiniMLModel):
        def __init__(self):
            self._c = MiniMLParam((1,))
            super().__init__()

    class LinearModel(MiniMLModel):
        
        def __init__(self):
            self._b = MiniMLParam((5,))
            self._M = MiniMLParam((5,5))
            self._c = ConstantModel()
            
            super().__init__()
                        
    m = LinearModel()

    assert len(m._params) == 3
    assert len(m._c._params) == 1
    
    m.bind()
    
    for p in m._params:
        assert p.bound
    
    assert m._params[0] is m._M
    assert m._params[1] is m._b
    assert m._params[2] is m._c._c
        
    m.randomize()
    
    # Save path
    M_val = m._M.value.copy()
    save_path = tmp_path / "model.npz"
    m.save(save_path)
    
    # Re-randomize
    m.randomize()
    
    assert not np.array_equal(m._M.value, M_val)
    
    # Reload
    m.load(save_path)
    
    assert np.array_equal(m._M.value, M_val)
