from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from miniml import InvertibleModel
from miniml.model import MiniMLModel
from miniml.nn.affine_coupling import (
    AffineCouplingLayer,
    AffineCouplingStack,
    RandomPartition,
    alternating_partition,
)
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


# ---------------------------------------------------------------------------
# Partition rules
# ---------------------------------------------------------------------------


def test_alternating_partition_alternates():
    assert [sorted(range(5)[alternating_partition(5, i)]) for i in range(4)] == [
        [0, 2, 4],
        [1, 3],
        [0, 2, 4],
        [1, 3],
    ]


def test_random_partition_is_reproducible():
    """Same seed, same partitions: a reloaded stack must rebuild the same model."""
    a = RandomPartition(seed=7)
    b = RandomPartition(seed=7)
    assert [a(8, i) for i in range(4)] == [b(8, i) for i in range(4)]
    # Repeated calls do not advance any state
    assert a(8, 0) == a(8, 0)
    # Different layers get different draws
    assert a(8, 0) != a(8, 1)


@pytest.mark.parametrize("frac,size", [(0.5, 4), (0.25, 2), (0.9, 7)])
def test_random_partition_respects_frac(frac: float, size: int):
    part = RandomPartition(seed=0, frac=frac)
    assert len(part(8, 0)) == size


@pytest.mark.parametrize("dim", [2, 3, 9])
def test_random_partition_always_leaves_a_split(dim: int):
    """Neither side of the split may be empty, whatever the fraction."""
    for frac in (0.01, 0.5, 0.99):
        target = RandomPartition(seed=0, frac=frac)(dim, 0)
        assert 0 < len(target) < dim


@pytest.mark.parametrize("frac", [0.0, 1.0, -0.5, 2.0])
def test_random_partition_invalid_frac(frac: float):
    with pytest.raises(MiniMLError, match="frac must be strictly between 0 and 1"):
        RandomPartition(frac=frac)


# ---------------------------------------------------------------------------
# AffineCouplingStack
# ---------------------------------------------------------------------------


def _randomized_stack(dim: int, n_layers: int, seed: int = 1, **kwargs):
    """A randomized stack.

    No damping is needed: the stack scales down the randomization of its layers
    by itself, so that the exponentials they apply do not compound out of range.
    """
    stack = AffineCouplingStack(dim, n_layers=n_layers, **kwargs)
    stack.bind()
    stack.randomize(seed=seed)
    return stack


def test_stack_is_invertible_model():
    stack = AffineCouplingStack(4, n_layers=2)
    assert isinstance(stack, InvertibleModel)
    assert isinstance(stack, MiniMLModel)


def test_stack_default_partition_alternates():
    stack = AffineCouplingStack(5, n_layers=3)
    targets = [sorted(layer._target_indices.tolist()) for layer in stack.layers]
    assert targets == [[0, 2, 4], [1, 3], [0, 2, 4]]


@pytest.mark.filterwarnings("ignore:The partition never transforms")
def test_stack_passes_dim_and_index_to_the_partition():
    seen = []

    def _record(dim: int, layer_idx: int):
        seen.append((dim, layer_idx))
        return {layer_idx % 3}

    AffineCouplingStack(6, n_layers=3, partition=_record)
    assert seen == [(6, 0), (6, 1), (6, 2)]


def test_stack_invalid_n_layers():
    with pytest.raises(MiniMLError, match="n_layers must be a positive integer"):
        AffineCouplingStack(4, n_layers=0)


def test_stack_warns_about_untouched_dimensions():
    with pytest.warns(UserWarning, match=r"never transforms dimensions \[1, 3\]"):
        AffineCouplingStack(4, n_layers=3, partition=lambda dim, i: {0, 2})


def test_stack_owns_the_parameters_of_all_its_layers():
    stack = AffineCouplingStack(4, n_layers=3)
    assert stack.size == sum(layer.size for layer in stack.layers)
    assert len(stack.layers) == 3
    assert stack.dim == 4


@pytest.mark.parametrize(
    "dim,n_layers,partition",
    [
        (5, 1, None),
        (5, 4, None),
        (6, 3, RandomPartition(seed=1)),
        (6, 6, RandomPartition(seed=2, frac=0.25)),
    ],
)
@pytest.mark.filterwarnings("ignore:The partition never transforms")
def test_stack_invert_roundtrip(dim: int, n_layers: int, partition):
    kwargs = {} if partition is None else {"partition": partition}
    stack = _randomized_stack(dim, n_layers, **kwargs)

    X = jnp.array(np.random.default_rng(1).normal(size=(4, dim)))
    assert jnp.allclose(stack.invert(stack.predict(X)), X, atol=1e-5)


def test_stack_transformation_is_nontrivial():
    stack = _randomized_stack(5, 4)
    X = jnp.array(np.random.default_rng(2).normal(size=(4, 5)))
    # Every dimension is transformed by some layer, so none passes through
    assert not jnp.any(jnp.isclose(stack.predict(X), X))


def test_stack_applies_its_layers_in_order():
    stack = _randomized_stack(5, 3)
    X = jnp.array(np.random.default_rng(3).normal(size=(4, 5)))

    Y = X
    for layer in stack.layers:
        Y = layer._predict_kernel(Y, stack._buffer)
    assert jnp.allclose(stack.predict(X), Y, atol=1e-6)


@pytest.mark.parametrize(
    "dim,n_layers,partition",
    [
        (5, 3, None),
        (6, 4, RandomPartition(seed=1)),
    ],
)
def test_stack_log_det_jac_matches_autodiff(dim: int, n_layers: int, partition):
    kwargs = {} if partition is None else {"partition": partition}
    stack = _randomized_stack(dim, n_layers, **kwargs)

    X = jnp.array(np.random.default_rng(4).normal(size=(4, dim)))

    def _single(x):
        return stack.predict(x[None])[0]

    ref = jnp.array(
        [jnp.log(jnp.abs(jnp.linalg.det(jax.jacfwd(_single)(x)))) for x in X]
    )
    ldj = stack.log_det_jac(X)
    assert ldj.shape == (4,)
    assert jnp.allclose(ldj, ref, atol=1e-5)


def test_stack_log_det_jac_sums_the_layers():
    """Each layer contributes the determinant at its own input, not at X."""
    stack = _randomized_stack(5, 3)
    X = jnp.array(np.random.default_rng(5).normal(size=(4, 5)))

    total = jnp.zeros((4,))
    running = X
    for layer in stack.layers:
        # The layers hold no buffer of their own, so go through the kernels
        total = total + layer._log_det_jac_kernel(running, stack._buffer)
        running = layer._predict_kernel(running, stack._buffer)
    assert jnp.allclose(stack.log_det_jac(X), total, atol=1e-6)


@pytest.mark.parametrize("partition", [None, RandomPartition(seed=7)])
def test_stack_save_and_load(tmp_path: Path, partition):
    """The partition rule has to survive the round trip through the saved args."""
    kwargs = {} if partition is None else {"partition": partition}
    stack = _randomized_stack(5, 3, **kwargs)
    X = jnp.array(np.random.default_rng(6).normal(size=(4, 5)))
    expected = stack.predict(X)

    stack.save(tmp_path / "stack")
    loaded = AffineCouplingStack.load(tmp_path / "stack.npz")

    assert [sorted(layer._target_indices.tolist()) for layer in loaded.layers] == [
        sorted(layer._target_indices.tolist()) for layer in stack.layers
    ]
    assert jnp.allclose(loaded.predict(X), expected)


# ---------------------------------------------------------------------------
# Partitions given directly, as data
# ---------------------------------------------------------------------------


@pytest.mark.filterwarnings("ignore:The partition never transforms")
def test_stack_accepts_explicit_partitions():
    """Slices and index sets can be mixed, one entry per layer."""
    stack = AffineCouplingStack(4, partition=[{0, 1}, slice(1, None, 2), {3}])
    assert stack.partitions == [{0, 1}, {1, 3}, {3}]
    assert len(stack.layers) == 3


def test_stack_resolves_a_rule_into_partitions():
    stack = AffineCouplingStack(5, n_layers=3)
    assert stack.partitions == [{0, 2, 4}, {1, 3}, {0, 2, 4}]


def test_stack_default_number_of_layers():
    assert len(AffineCouplingStack(4).layers) == 4


def test_stack_partitions_are_copies():
    stack = AffineCouplingStack(4, n_layers=2)
    stack.partitions[0].add(99)
    assert 99 not in stack.partitions[0]


def test_stack_n_layers_must_match_the_partitions():
    with pytest.raises(MiniMLError, match="n_layers is 5, but 2 partitions"):
        AffineCouplingStack(4, n_layers=5, partition=[{0}, {1}])


@pytest.mark.parametrize("partition", [[], ()])
def test_stack_rejects_an_empty_partition_list(partition):
    with pytest.raises(MiniMLError, match="n_layers must be a positive integer"):
        AffineCouplingStack(4, partition=partition)


def test_stack_rejects_a_partition_that_is_not_indices():
    with pytest.raises(MiniMLError, match="Partition 1 is neither a slice nor a set"):
        AffineCouplingStack(4, partition=[{0, 1}, 3])


def test_stack_saves_partitions_rather_than_the_rule(tmp_path: Path):
    """A rule that can not be pickled must not stop the model being saved."""
    stack = _randomized_stack(
        4, 2, partition=lambda dim, layer_idx: {layer_idx, 2 + layer_idx}
    )
    X = jnp.array(np.random.default_rng(7).normal(size=(3, 4)))
    expected = stack.predict(X)

    stack.save(tmp_path / "stack")
    loaded = AffineCouplingStack.load(tmp_path / "stack.npz")

    assert loaded.partitions == stack.partitions == [{0, 2}, {1, 3}]
    assert jnp.allclose(loaded.predict(X), expected)


def test_stack_load_does_not_call_the_rule(tmp_path: Path):
    """The saved indices are used as they are, even if the rule has changed."""
    calls = []

    def _rule(dim: int, layer_idx: int):
        calls.append(layer_idx)
        return {layer_idx, 2 + layer_idx}

    stack = _randomized_stack(4, 2, partition=_rule)
    stack.save(tmp_path / "stack")
    calls.clear()

    loaded = AffineCouplingStack.load(tmp_path / "stack.npz")
    assert calls == []
    assert loaded.partitions == [{0, 2}, {1, 3}]


# ---------------------------------------------------------------------------
# Randomization scale
# ---------------------------------------------------------------------------


def _weight_scales(layer: AffineCouplingLayer) -> list[float]:
    """The randomization scales of both MLPs, ignoring the zeroed biases."""
    return [
        ref.param.rnd_scale
        for mlp in (layer._mlp_q, layer._mlp_p)
        for ref in mlp._get_inner_params()
        if ref.param.rnd_scale > 0.0
    ]


def test_layer_rnd_scale_damps_both_mlps():
    """Both branches compound through a stack, so both have to be damped."""
    plain = _weight_scales(AffineCouplingLayer(4))
    damped = _weight_scales(AffineCouplingLayer(4, rnd_scale=0.25))
    assert len(damped) == len(plain) > 0
    assert damped == [pytest.approx(0.25 * s) for s in plain]


def test_layer_rnd_scale_leaves_the_zeroed_biases_alone():
    layer = AffineCouplingLayer(4, rnd_scale=0.25)
    scales = [
        ref.param.rnd_scale
        for mlp in (layer._mlp_q, layer._mlp_p)
        for ref in mlp._get_inner_params()
    ]
    assert 0.0 in scales


@pytest.mark.parametrize("rnd_scale", [0.0, -1.0])
def test_layer_rnd_scale_must_be_positive(rnd_scale: float):
    with pytest.raises(MiniMLError, match="rnd_scale must be positive"):
        AffineCouplingLayer(4, rnd_scale=rnd_scale)


@pytest.mark.filterwarnings("ignore:The partition never transforms")
@pytest.mark.parametrize("n_layers", [1, 2, 8, 32])
def test_stack_rnd_scale_matches_the_checkerboard_rule(n_layers: int):
    """Every dimension is transformed by half of the layers, giving 1/sqrt(n)."""
    stack = AffineCouplingStack(8, n_layers=n_layers)
    expected = 1.0 / np.sqrt(max(n_layers, 2))
    assert stack._rnd_scales(stack.partitions) == [pytest.approx(expected)] * n_layers


def test_stack_rnd_scale_follows_the_counts():
    """A dimension transformed more often damps the layers that transform it."""
    # Dimension 0 is transformed 3 times, dimensions 1 and 2 only once
    stack = AffineCouplingStack(3, partition=[{0}, {0}, {0, 1}, {2}])
    assert stack._rnd_scales(stack.partitions) == [
        pytest.approx(1 / np.sqrt(6)),
        pytest.approx(1 / np.sqrt(6)),
        pytest.approx(1 / np.sqrt(6)),
        pytest.approx(1 / np.sqrt(2)),
    ]


def test_stack_explicit_rnd_scale_overrides_the_rule():
    stack = AffineCouplingStack(4, n_layers=3, rnd_scale=0.5)
    plain = _weight_scales(AffineCouplingLayer(4))
    for layer in stack.layers:
        assert _weight_scales(layer) == [pytest.approx(0.5 * s) for s in plain]


def test_stack_rnd_scale_survives_save_and_load(tmp_path: Path):
    stack = _randomized_stack(4, 3, rnd_scale=0.5)
    stack.save(tmp_path / "stack")
    loaded = AffineCouplingStack.load(tmp_path / "stack.npz")
    assert [_weight_scales(layer) for layer in loaded.layers] == [
        _weight_scales(layer) for layer in stack.layers
    ]


@pytest.mark.parametrize("n_layers", [16, 64])
def test_deep_randomized_stack_stays_invertible(n_layers: int):
    """Without damping, the scales compound and float32 loses the round trip."""
    stack = _randomized_stack(6, n_layers, seed=0)
    X = jnp.array(np.random.default_rng(8).normal(size=(16, 6)))

    Y = stack.predict(X)
    assert jnp.all(jnp.isfinite(Y))
    # The outputs stay in the same range as the inputs, rather than exploding
    assert float(jnp.abs(Y).max()) < 20.0
    assert jnp.allclose(stack.invert(Y), X, atol=1e-4)
