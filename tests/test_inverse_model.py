"""Tests for InverseModel, the parameter-sharing inverse view."""

import jax.numpy as jnp
import pytest

from miniml import InverseModel, MiniMLError, MiniMLModel, SharedModel
from miniml.nn.affine_coupling import AffineCouplingLayer
from miniml.nn.compose import Stack
from miniml.nn.linear import Linear


def _fitted_layer(dim: int = 4, seed: int = 0) -> AffineCouplingLayer:
    layer = AffineCouplingLayer(dim)
    layer.bind()
    layer.randomize(seed=seed)
    return layer


# ---------------------------------------------------------------------------
# SharedModel
# ---------------------------------------------------------------------------


def test_shared_model_reports_no_parameters():
    layer = AffineCouplingLayer(4)
    shared = SharedModel(layer)
    assert shared.model is layer
    assert shared._get_inner_params() == []
    assert len(layer._get_inner_params()) > 0


def test_shared_model_member_is_not_counted():
    """A model behind a SharedModel does not make its holder a co-owner."""

    class Holder(MiniMLModel):
        def __init__(self, model):
            self._shared = SharedModel(model)
            super().__init__()

        def _predict_kernel(self, X, buffer, rng_key=None, mode=None, **kw):
            return X

    layer = AffineCouplingLayer(4)
    assert Holder(layer).size == 0

    # Order of assignment relative to super().__init__() does not matter
    class LateHolder(Holder):
        def __init__(self, model):
            super().__init__(model)
            self._other = SharedModel(model)

    assert LateHolder(layer).size == 0


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


def test_inverse_model_rejects_non_invertible():
    with pytest.raises(MiniMLError, match="InvertibleModel"):
        InverseModel(Linear(2, 2))


def test_inverse_model_owns_no_parameters():
    layer = _fitted_layer()
    inv = layer.inverse_model()
    assert isinstance(inv, InverseModel)
    assert inv.size == 0
    assert inv.param_names == []
    assert inv.inverted is layer


def test_inverse_of_inverse_is_the_original():
    layer = _fitted_layer()
    assert layer.inverse_model().inverse_model() is layer


# ---------------------------------------------------------------------------
# Behaviour
# ---------------------------------------------------------------------------


def test_inverse_model_round_trip():
    layer = _fitted_layer()
    inv = layer.inverse_model()
    X = jnp.arange(8.0).reshape(2, 4)

    assert jnp.allclose(inv.predict(layer.predict(X)), X, atol=1e-5)
    assert jnp.allclose(layer.predict(inv.predict(X)), X, atol=1e-5)


def test_inverse_model_swaps_the_two_kernels():
    layer = _fitted_layer()
    inv = layer.inverse_model()
    X = jnp.arange(8.0).reshape(2, 4)

    assert jnp.allclose(inv.predict(X), layer.invert(X))
    assert jnp.allclose(inv.invert(X), layer.predict(X))


def test_inverse_model_shares_the_buffer():
    layer = _fitted_layer()
    inv = layer.inverse_model()
    assert inv.bound
    assert inv._buffer is layer._buffer

    X = jnp.arange(8.0).reshape(2, 4)
    before = inv.predict(X)
    layer.randomize(seed=7)
    # The view reads the new parameter values, without being rebound
    assert not jnp.allclose(inv.predict(X), before)
    assert jnp.allclose(inv.predict(layer.predict(X)), X, atol=1e-5)


def test_inverse_model_follows_a_fit():
    layer = _fitted_layer()
    inv = layer.inverse_model()
    X = jnp.arange(8.0).reshape(2, 4)

    before = inv.predict(X)
    layer.fit(X, jnp.zeros((2, 4)))
    assert not jnp.allclose(inv.predict(X), before)
    assert jnp.allclose(inv.predict(layer.predict(X)), X, atol=1e-5)


def test_inverse_model_binding_follows_the_owner():
    layer = AffineCouplingLayer(4)
    inv = layer.inverse_model()
    assert not inv.bound

    inv.bind()
    # Binding the view binds its owner, and does not give it a buffer of its own
    assert layer.bound
    assert inv._buffer is layer._buffer

    inv.unbind()
    assert not layer.bound
    assert not inv.bound


# ---------------------------------------------------------------------------
# Composition
# ---------------------------------------------------------------------------


def test_stack_with_inverse_is_the_identity():
    layer = AffineCouplingLayer(4)
    stack = Stack([layer, layer.inverse_model()])
    # The shared parameters are counted once, not twice
    assert stack.size == layer.size

    stack.bind()
    stack.randomize(seed=3)
    X = jnp.arange(8.0).reshape(2, 4)
    assert jnp.allclose(stack.predict(X), X, atol=1e-5)


def test_stack_with_inverse_stays_the_identity_after_fit():
    layer = AffineCouplingLayer(4)
    stack = Stack([layer, layer.inverse_model()])
    stack.bind()
    stack.randomize(seed=3)
    X = jnp.arange(8.0).reshape(2, 4)

    res = stack.fit(X, jnp.zeros((2, 4)))
    assert res.success
    assert jnp.allclose(stack.predict(X), X, atol=1e-5)


def test_inverse_model_refuses_to_be_saved(tmp_path):
    """Saving a view would write a file that can not be loaded back."""
    layer = _fitted_layer()
    with pytest.raises(MiniMLError, match="owns no parameters and can not be saved"):
        layer.inverse_model().save(tmp_path / "inv")
    assert not list(tmp_path.iterdir())
