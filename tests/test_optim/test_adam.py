import jax.numpy as jnp
from miniml.optim.adam import AdamOptimizer, AdamWOptimizer
from miniml.optim.base import MiniMLOptimResult


def test_adam_optimizer() -> None:
    # Create a dummy function
    def dummy_objective(x, _=None):
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


def test_adamw_optimizer() -> None:
    # Create a dummy function
    def dummy_objective(x, _=None):
        return (x**2).sum()

    tol = 1e-12
    optimizer = AdamWOptimizer(alpha=0.1, tol=tol, maxiter=1000)
    x0 = jnp.ones(5, dtype=jnp.float32)
    result = optimizer(dummy_objective, x0)
    assert isinstance(result, MiniMLOptimResult)
    assert jnp.allclose(result.x_opt, jnp.zeros(5), atol=1e-9)
    assert result.objective_value is not None
    assert jnp.allclose(result.objective_value, 0.0, atol=1e-9)
    assert result.n_iterations is not None
    assert result.n_iterations <= optimizer._maxiter
    assert result.success


def test_adamw_weight_decay_shrinks_parameters_without_gradient() -> None:
    # Constant objective: gradient is zero, so only weight decay acts
    def constant_objective(x, _=None):
        return jnp.array(0.0, dtype=x.dtype)

    weight_decay = 0.1
    optimizer = AdamWOptimizer(alpha=0.1, weight_decay=weight_decay, tol=0.0, maxiter=50)
    x0 = jnp.ones(3, dtype=jnp.float32)
    result = optimizer(constant_objective, x0)
    assert isinstance(result, MiniMLOptimResult)
    # With decoupled weight decay and zero gradient, parameters should decrease in norm
    assert jnp.linalg.norm(result.x_opt) < jnp.linalg.norm(x0)


def test_adam_weight_decay_shrinks_parameters_without_gradient() -> None:
    # Constant objective: gradient is zero, but coupled weight decay
    # is applied inside the gradient, so parameters should also shrink.
    def constant_objective(x, _=None):
        return jnp.array(0.0, dtype=x.dtype)

    weight_decay = 0.1
    optimizer = AdamOptimizer(alpha=0.1, weight_decay=weight_decay, tol=0.0, maxiter=50)
    x0 = jnp.ones(3, dtype=jnp.float32)
    result = optimizer(constant_objective, x0)
    assert isinstance(result, MiniMLOptimResult)
    assert jnp.linalg.norm(result.x_opt) < jnp.linalg.norm(x0)


def test_adam_and_adamw_match_when_no_weight_decay() -> None:
    # For weight_decay = 0, Adam and AdamW share the same update rule.
    def quadratic(x, _=None):
        return (x**2).sum()

    alpha = 0.05
    tol = 1e-10
    maxiter = 500
    x0 = jnp.linspace(-1.0, 1.0, 5, dtype=jnp.float32)

    adam = AdamOptimizer(alpha=alpha, weight_decay=0.0, tol=tol, maxiter=maxiter)
    adamw = AdamWOptimizer(alpha=alpha, weight_decay=0.0, tol=tol, maxiter=maxiter)

    result_adam = adam(quadratic, x0)
    result_adamw = adamw(quadratic, x0)

    assert isinstance(result_adam, MiniMLOptimResult)
    assert isinstance(result_adamw, MiniMLOptimResult)
    assert jnp.allclose(result_adam.x_opt, result_adamw.x_opt, atol=1e-9)
    assert result_adam.success == result_adamw.success
