from pathlib import Path
import numpy as np
import pytest
from miniml.param import MiniMLParam, MiniMLError
from miniml.model import MiniMLModel


def test_model(tmp_path: Path):

    class ConstantModel(MiniMLModel):
        def __init__(self):
            self._c = MiniMLParam((1,))
            super().__init__()

    class LinearModel(MiniMLModel):

        def __init__(self):
            self._b = MiniMLParam((5,))
            self._M = MiniMLParam((5, 5))
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


def test_dtype_mismatch():
    class ModelA(MiniMLModel):
        def __init__(self):
            self.p1 = MiniMLParam((2,), dtype=np.float32)
            self.p2 = MiniMLParam((2,), dtype=np.float64)
            super().__init__()

    with pytest.raises(MiniMLError, match="same dtype"):
        ModelA()


def test_child_model_no_super():
    class BadChild(MiniMLModel):
        def __init__(self):
            self.p = MiniMLParam((1,))
            # Forgot super().__init__()

    class Parent(MiniMLModel):
        def __init__(self):
            self.child = BadChild()
            super().__init__()

    with pytest.raises(MiniMLError, match="was not properly initialized"):
        Parent()


def test_save_before_bind(tmp_path: Path):
    class M(MiniMLModel):
        def __init__(self):
            self.p = MiniMLParam((1,))
            super().__init__()

    m = M()
    with pytest.raises(MiniMLError, match="bound to buffers; can not save"):
        m.save(tmp_path / "fail.npz")


def test_load_before_bind(tmp_path: Path):
    class M(MiniMLModel):
        def __init__(self):
            self.p = MiniMLParam((2,))
            super().__init__()

    m = M()
    m.bind()
    m.randomize()
    save_path = tmp_path / "m.npz"
    m.save(save_path)
    m2 = M()
    # Should bind automatically on load
    m2.load(save_path)
    assert np.allclose(m2._buffer, m._buffer)
