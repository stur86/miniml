import jax.numpy as jnp
import pytest

from miniml.nn.affine_coupling import AffineCouplingLayer


@pytest.mark.parametrize(
    "dim,input,output",
    [
        (5, slice(None, None), {0, 1, 2, 3, 4}),
        (5, slice(None, 3), {0, 1, 2}),
        (3, slice(None, None, 2), {0, 2}),
        (5, slice(1, None, 2), {1, 3}),
    ]
)
def test_slice_to_set(dim: int, input: slice, output: set):
    ans = AffineCouplingLayer._slice_to_set(input, dim)
    assert ans == output
