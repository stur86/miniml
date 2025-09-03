from pathlib import Path
import numpy as np
import pytest
import jax.numpy as jnp
from jax import Array as JXArray
from miniml.param import MiniMLParam, MiniMLError
from miniml.model import MiniMLModel, MiniMLModelList
from miniml.loss import squared_error_loss, LNormRegularization


def test_model(tmp_path: Path):

    class ConstantModel(MiniMLModel):
        def __init__(self):
            self._c = MiniMLParam((1,))
            super().__init__()

        def predict(self, X: JXArray) -> JXArray:
            return self._c.value

    class LinearModel(MiniMLModel):

        def __init__(self):
            self._b = MiniMLParam((5,))
            self._M = MiniMLParam((5, 5))
            self._c = ConstantModel()

            super().__init__()

        def predict(self, X: JXArray) -> JXArray:
            return self._M.value @ X + self._b.value[:, None] + self._c.predict(X)

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
    m_loaded = LinearModel.load(save_path)

    assert np.array_equal(m_loaded._M.value, M_val)


def test_dtype_mismatch():
    class ModelA(MiniMLModel):
        def __init__(self):
            self.p1 = MiniMLParam((2,), dtype=np.float32)
            self.p2 = MiniMLParam((2,), dtype=np.float64)
            super().__init__()

        def predict(self, X: JXArray) -> JXArray:
            return super().predict(X)

    with pytest.raises(MiniMLError, match="parameter dtype mismatch"):
        ModelA()


def test_child_model_no_super():
    class BadChild(MiniMLModel):
        def __init__(self):
            self.p = MiniMLParam((1,))
            # Forgot super().__init__()

        def predict(self, X: JXArray) -> JXArray:
            return super().predict(X)

    class Parent(MiniMLModel):
        def __init__(self):
            self.child = BadChild()
            super().__init__()

        def predict(self, X: JXArray) -> JXArray:
            return super().predict(X)

    with pytest.raises(MiniMLError, match="was not properly initialized"):
        Parent()


def test_save_before_bind(tmp_path: Path):
    class M(MiniMLModel):
        def __init__(self):
            self.p = MiniMLParam((1,))
            super().__init__()

        def predict(self, X: JXArray) -> JXArray:
            return super().predict(X)

    m = M()
    with pytest.raises(MiniMLError, match="bound to buffers; can not save"):
        m.save(tmp_path / "fail.npz")


def test_load_before_bind(tmp_path: Path):
    class M(MiniMLModel):
        def __init__(self, n: int):
            self.p = MiniMLParam((n,))
            super().__init__()

        def predict(self, X: JXArray) -> JXArray:
            return super().predict(X)

    m = M(n=2)
    m.bind()
    m.randomize()
    save_path = tmp_path / "m.npz"
    m.save(save_path)
    m2 = M.load(save_path)
    assert np.allclose(m2._buffer, m._buffer)


def test_squared_error_loss():
    y_true = jnp.array([1.0, 2.0, 3.0])
    y_pred = jnp.array([1.0, 2.5, 2.0])
    loss = squared_error_loss(y_true, y_pred)
    expected = 0.0**2 + 0.5**2 + 1.0**2
    assert jnp.isclose(loss, expected)


def test_param_regularization_loss():
    from miniml.param import MiniMLParam

    # L2 regularization on a parameter
    param = MiniMLParam((3,), reg_loss=LNormRegularization(), dtype=jnp.float32)

    class DummyModel:
        def __init__(self, buf):
            self._buffer = buf

    # Bind to dummy buffer
    buf = jnp.array([3.0, 4.0, 0.0], dtype=jnp.float32)
    dummy = DummyModel(buf)
    param.bind(0, dummy)
    reg = param.regularization_loss()
    # L2 norm: (3^2 + 4^2 + 0^2) = 25
    assert jnp.isclose(reg, 25.0)


@pytest.mark.parametrize("method", ["L-BFGS-B", "Nelder-Mead", "trust-krylov"])
def test_linear_model_fit_no_reg(method: str):
    # Fit y = a*x + b, no regularization, analytical solution
    class LinearModel(MiniMLModel):
        def __init__(self):
            self.a = MiniMLParam((1,))
            self.b = MiniMLParam((1,))
            super().__init__()

        def predict(self, X):
            return self.a.value * X + self.b.value

    # Generate data: y = 2x + 1
    X = jnp.linspace(0, 10, 20)
    y = 2 * X + 1
    model = LinearModel()
    model.bind()
    model._buffer = jnp.array([0.0, 0.0], dtype=jnp.float32)  # init to zeros
    res = model.fit(X, y, fit_args={"method": method})
    assert res.success

    a_fit, b_fit = model.a.value[0], model.b.value[0]
    assert jnp.isclose(a_fit, 2.0, atol=1e-2)
    assert jnp.isclose(b_fit, 1.0, atol=1e-2)


@pytest.mark.parametrize("method", ["L-BFGS-B", "Nelder-Mead", "trust-krylov"])
def test_linear_model_fit_with_l2_reg(method: str):
    # Fit y = a*x + b, with L2 regularization on a
    class LinearModel(MiniMLModel):
        def __init__(self):
            self.a = MiniMLParam((1,), reg_loss=LNormRegularization())
            self.b = MiniMLParam((1,))
            super().__init__()

        def predict(self, X):
            return self.a.value * X + self.b.value

    # Data: y = 2x + 1
    X = jnp.linspace(0, 10, 20)
    y = 2 * X + 1
    reg_lambda = 10.0
    model = LinearModel()
    model.randomize(seed=42)
    res = model.fit(X, y, reg_lambda=reg_lambda, fit_args={"method": method})
    assert res.success
    assert jnp.isclose(res.loss, model.total_loss(y, model.predict(X), reg_lambda))

    a_fit, b_fit = model.a.value[0], model.b.value[0]
    # Analytical ridge regression solution for a: a = Sxy / (Sxx + lambda)
    Xc = X - X.mean()
    Sxx = jnp.sum(Xc**2)
    Sxy = jnp.sum(Xc * (y - y.mean()))
    a_analytical = Sxy / (Sxx + reg_lambda)
    b_analytical = y.mean() - a_analytical * X.mean()
    assert jnp.isclose(a_fit, a_analytical, atol=1e-2)
    assert jnp.isclose(b_fit, b_analytical, atol=1e-2)


def test_model_list():
    class M1(MiniMLModel):
        def __init__(self):
            self.p = MiniMLParam((1,))
            super().__init__()

        def predict(self, X: JXArray) -> JXArray:
            return super().predict(X)

    class M2(MiniMLModel):
        def __init__(self):
            self.p = MiniMLParam((2,))
            super().__init__()

        def predict(self, X: JXArray) -> JXArray:
            return super().predict(X)

    mlist = MiniMLModelList([M1(), M2()])
    assert len(mlist._contents) == 2
    assert isinstance(mlist._contents[0], M1)
    assert isinstance(mlist._contents[1], M2)
    assert len(mlist._get_inner_params()) == 2
    assert mlist._get_inner_params()[0] == mlist._contents[0]._params[0]
    assert mlist._get_inner_params()[1] == mlist._contents[1]._params[0]
