from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from miniml import InvertibleModel
from miniml.model import MiniMLModel
from miniml.nn.affine_coupling import AffineCouplingLayer
from miniml.param import MiniMLError


@pytest.mark.parametrize(
    "dim,input,output",
    [
        (5, slice(None, None), {0, 1, 2, 3, 4}),
        (5, slice(None, 3), {0, 1, 2}),
        (3, slice(None, None, 2), {0, 2}),
        (5, slice(1, None, 2), {1, 3}),
    ],
)
def test_slice_to_set(dim: int, input: slice, output: set):
    ans = AffineCouplingLayer._slice_to_set(input, dim)
    assert ans == output


@pytest.mark.parametrize(
    "dim,target_indices",
    [
        (5, None),
        (5, slice(1, None, 2)),
        (5, slice(None, 3)),
        (6, {0, 1, 2}),
        (6, {2, 4}),
        (6, [1, 3, 5]),
    ],
)
def test_is_invertible_model(dim: int, target_indices):
    layer = AffineCouplingLayer(dim=dim, target_indices=target_indices)
    assert isinstance(layer, InvertibleModel)


@pytest.mark.parametrize(
    "dim,target_indices",
    [
        (5, None),
        (5, slice(1, None, 2)),
        (6, {0, 1, 2}),
        (4, {1, 3}),
    ],
)
def test_affine_coupling_forward(dim: int, target_indices):
    layer = AffineCouplingLayer(dim=dim, target_indices=target_indices)
    layer.randomize(seed=42)

    X = jnp.array(np.random.default_rng(0).normal(size=(7, dim)))
    Y = layer.predict(X)

    # Input part is unchanged
    assert jnp.allclose(Y[..., layer._input_indices], X[..., layer._input_indices])
    # Target part follows y_t = x_t * exp(q(x_i)) + p(x_i)
    X_in, X_t = X[..., layer._input_indices], X[..., layer._target_indices]
    q = jnp.exp(
        MiniMLModel._unpack_kernel_output(
            layer._mlp_q._predict_kernel(X_in, layer._buffer)
        )[0]
    )
    p = MiniMLModel._unpack_kernel_output(
        layer._mlp_p._predict_kernel(X_in, layer._buffer)
    )[0]
    assert jnp.allclose(Y[..., layer._target_indices], X_t * q + p)


@pytest.mark.parametrize(
    "dim,target_indices,shape",
    [
        (5, None, (7, 5)),
        (5, slice(1, None, 2), (3, 5)),
        (6, {0, 1, 2}, (8, 6)),
        (6, {2, 4}, (6,)),
        (4, {1, 3}, (4, 3, 4)),
    ],
)
def test_affine_coupling_invert_roundtrip(dim: int, target_indices, shape: tuple):
    layer = AffineCouplingLayer(dim=dim, target_indices=target_indices)
    layer.randomize(seed=42)

    X = jnp.array(np.random.default_rng(1).normal(size=shape))
    Y = layer.predict(X)

    assert jnp.allclose(layer.invert(Y), X, atol=1e-5)


def test_affine_coupling_invert_is_nontrivial():
    layer = AffineCouplingLayer(dim=4, target_indices={1, 3})
    layer.randomize(seed=0)

    X = jnp.array(np.random.default_rng(2).normal(size=(5, 4)))
    Y = layer.predict(X)

    # The transformed part should actually change the input
    assert not jnp.allclose(
        Y[..., layer._target_indices], X[..., layer._target_indices]
    )


def test_affine_coupling_has_two_mlps():
    layer = AffineCouplingLayer(dim=5, target_indices={1, 3})
    layer.randomize(seed=7)

    n_params = layer._mlp_q.size + layer._mlp_p.size
    assert layer.size == n_params
    assert any(p.startswith("_mlp_q.") for p in layer.param_names)
    assert any(p.startswith("_mlp_p.") for p in layer.param_names)
    assert len(layer.param_names) == len(layer._mlp_q.param_names) + len(
        layer._mlp_p.param_names
    )


def test_affine_coupling_default_target_is_even():
    layer = AffineCouplingLayer(dim=5)
    assert set(layer._target_indices.tolist()) == {0, 2, 4}
    assert set(layer._input_indices.tolist()) == {1, 3}


def test_affine_coupling_invalid_targets():
    # Empty input set
    with pytest.raises(MiniMLError):
        AffineCouplingLayer(dim=5, target_indices={0, 1, 2, 3, 4})
    # Empty target set
    with pytest.raises(MiniMLError):
        AffineCouplingLayer(dim=5, target_indices=set())
    # Out of range indices
    with pytest.raises(MiniMLError):
        AffineCouplingLayer(dim=5, target_indices={1, 5})
    # Non-positive dim
    with pytest.raises(MiniMLError):
        AffineCouplingLayer(dim=0)


def test_affine_coupling_save_and_load(tmp_path: Path):
    layer = AffineCouplingLayer(dim=6, target_indices={1, 3, 5})
    layer.randomize(seed=11)

    model_path = tmp_path / "coupling.npz"
    layer.save(model_path, state_only=True)

    loaded = AffineCouplingLayer(dim=6, target_indices={1, 3, 5})
    loaded.load_state(model_path)

    X = jnp.array(np.random.default_rng(3).normal(size=(4, 6)))
    assert jnp.array_equal(layer.predict(X), loaded.predict(X))
    assert jnp.array_equal(
        layer.invert(layer.predict(X)), loaded.invert(layer.predict(X))
    )


@pytest.mark.parametrize(
    "dim,target_indices",
    [
        (5, None),
        (5, slice(1, None, 2)),
        (6, {0, 1, 2}),
        (4, {1, 3}),
    ],
)
def test_log_det_jac_matches_autodiff(dim: int, target_indices):
    layer = AffineCouplingLayer(dim=dim, target_indices=target_indices)
    layer.randomize(seed=42)

    X = jnp.array(np.random.default_rng(3).normal(size=(4, dim)))
    ldj = layer.log_det_jac(X)

    # Compare against the determinant of the full Jacobian, one sample at a time
    def _single(x):
        return layer.predict(x[None])[0]

    ref = jnp.array(
        [jnp.log(jnp.abs(jnp.linalg.det(jax.jacfwd(_single)(x)))) for x in X]
    )
    assert ldj.shape == (4,)
    assert jnp.allclose(ldj, ref, atol=1e-5)


def test_log_det_jac_is_per_sample():
    """One value per sample, so batches keep their individual likelihoods."""
    layer = AffineCouplingLayer(dim=4)
    layer.randomize(seed=0)

    X = jnp.array(np.random.default_rng(4).normal(size=(3, 2, 4)))
    assert layer.log_det_jac(X).shape == (3, 2)
    # An unbatched sample gives a scalar
    assert layer.log_det_jac(X[0, 0]).shape == ()
    # ... and samples are independent of each other
    assert jnp.allclose(layer.log_det_jac(X)[0], layer.log_det_jac(X[0]))


def test_log_det_jac_is_finite_for_extreme_inputs():
    """The scale is an exponential, so the determinant can never vanish."""
    layer = AffineCouplingLayer(dim=4)
    layer.randomize(seed=1)

    X = jnp.array([[-1e3, 1e3, -1e3, 1e3], [0.0, 0.0, 0.0, 0.0]])
    assert bool(jnp.all(jnp.isfinite(layer.log_det_jac(X))))
