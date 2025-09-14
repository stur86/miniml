import pytest
import jax.numpy as jnp
from miniml.nn.rbf import (
    RBFunction,
    gaussian_rbf,
    inverse_multiquadric_rbf,
    inverse_quadratic_rbf,
    linear_rbf,
    multiquadric_rbf,
    r_tanh_rbf,
    PolyharmonicRBF,
    bump_rbf
)


@pytest.mark.parametrize("f,input_data,expected_output", [
    (gaussian_rbf, jnp.array([0.0, 1.0, 2.0]), jnp.array([1.0, jnp.exp(-0.5), jnp.exp(-2.0)])),
    (multiquadric_rbf, jnp.array([0.0, 1.0, 2.0]), jnp.array([1.0, jnp.sqrt(2), jnp.sqrt(5)])),
    (inverse_multiquadric_rbf, jnp.array([0.0, 1.0, 2.0]), jnp.array([1.0, 1/jnp.sqrt(2), 1/jnp.sqrt(5)])),
    (inverse_quadratic_rbf, jnp.array([0.0, 1.0, 2.0]), jnp.array([1.0, 0.5, 0.2])),
    (linear_rbf, jnp.array([0.0, 1.0, 2.0]), jnp.array([0.0, 1.0, 2.0])),
    (r_tanh_rbf, jnp.array([0.0, 1.0, 2.0]), jnp.array([0.0, jnp.tanh(1.0), 2.0 * jnp.tanh(2.0)])),
    (PolyharmonicRBF(degree=3), jnp.array([0.0, 1.0, 2.0]), jnp.array([0.0, 1.0, 8.0])),
    (PolyharmonicRBF(degree=4), jnp.array([0.0, 1.0, 2.0]), jnp.array([0.0, 0.0, 16.0*jnp.log(2.0)])),
    (bump_rbf, jnp.array([0.0, 0.5, 1.0, 1.5]), jnp.array([1.0, jnp.exp(1 - 1/(1 - 0.25)), 0.0, 0.0])),
])  
def test_rbf_function(f: RBFunction, input_data, expected_output):
    output = f(input_data)
    assert jnp.allclose(output, expected_output), f"Failed for {f}"