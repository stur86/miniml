from miniml.nn.layer_norm import LayerNorm
from jax import numpy as jnp


def test_single_dim() -> None:
    """Test the LayerNorm model."""
    ln = LayerNorm(normalized_shape=4, eps=1e-5, dtype=jnp.float32)
    ln.bind()
    ln.set_params(
        {
            "_gamma.v": jnp.ones((4,), dtype=jnp.float32),
            "_beta.v": jnp.zeros((4,), dtype=jnp.float32),
        }
    )

    # Test input
    X = jnp.array([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]], dtype=jnp.float32)

    # Forward pass
    output = ln.predict(X)

    # Manually compute expected output
    mean = jnp.mean(X, axis=-1, keepdims=True)
    var = jnp.var(X, axis=-1, keepdims=True)
    gamma = jnp.ones((4,), dtype=jnp.float32)
    beta = jnp.zeros((4,), dtype=jnp.float32)
    expected_output = (X - mean) / jnp.sqrt(var + 1e-5) * gamma + beta

    # Check if output matches expected output
    assert jnp.allclose(
        output, expected_output
    ), "LayerNorm output does not match expected output."


def test_multi_dim() -> None:
    """Test the LayerNorm model with multi-dimensional normalized shape."""
    ln = LayerNorm(normalized_shape=(2, 3), eps=1e-5, dtype=jnp.float32)
    ln.bind()

    gamma = jnp.linspace(1.0, 2.0, num=6, dtype=jnp.float32).reshape((2, 3))
    beta = jnp.linspace(0.0, 1.0, num=6, dtype=jnp.float32).reshape((2, 3))

    ln.set_params({"_gamma.v": gamma, "_beta.v": beta})

    # Test input
    X = jnp.array(
        [[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], [[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]]],
        dtype=jnp.float32,
    )

    # Forward pass
    output = ln.predict(X)

    # Manually compute expected output
    mean = jnp.mean(X, axis=(-2, -1), keepdims=True)
    var = jnp.var(X, axis=(-2, -1), keepdims=True)
    expected_output = (X - mean) / jnp.sqrt(var + 1e-5) * gamma + beta

    # Check if output matches expected output
    assert jnp.allclose(
        output, expected_output
    ), "LayerNorm output does not match expected output."
