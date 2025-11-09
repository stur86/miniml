import jax.numpy as jnp
from miniml.optim.adam import AdamOptimizer
from miniml.optim.base import MiniMLOptimResult


def test_adam_optimizer() -> None:
    # Create a dummy function
    def dummy_objective(x):
        return (x**2).sum()

    tol = 1e-12
    optimizer = AdamOptimizer(alpha=0.1, tol=tol, maxiter=1000)
    x0 = jnp.ones(5, dtype=jnp.float32)
    result = optimizer(dummy_objective, x0)
    assert isinstance(result, MiniMLOptimResult)
    assert jnp.allclose(result.x_opt, jnp.zeros(5), atol=1e-9)
    assert result.objective_value is not None
    assert jnp.allclose(result.objective_value, 0.0, atol=1e-9)
    assert result.n_iterations is not None
    assert result.n_iterations <= optimizer._maxiter
    assert result.success
