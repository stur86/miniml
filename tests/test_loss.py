import pytest
import jax.numpy as jnp
import numpy as np
from miniml.loss import (
    LNormRegularization,
    cross_entropy_loss,
    CrossEntropyLogLoss,
    squared_error_loss,
    binary_match_loss,
)


def test_squared_error_loss() -> None:
    y = jnp.ones(10)
    rng = np.random.default_rng(42)
    y_hat = y + rng.choice([-1, 1], size=10) * 2
    loss = squared_error_loss(y, y_hat)
    assert loss == 40


def test_cross_entropy_loss() -> None:
    y = jnp.array([[0, 1], [1, 0], [0.7, 0.3]])
    y_hat = jnp.array([[0.1, 0.9], [0.8, 0.2], [0.6, 0.4]])
    loss = cross_entropy_loss(y, y_hat)
    assert np.isclose(loss, -(y * jnp.log(y_hat)).sum())


def test_cross_entropy_log_loss() -> None:
    y = jnp.array([[0, 1], [1, 0], [0.7, 0.3]])
    y_logits = jnp.array([[4], [-1], [2]])
    y_exp = jnp.exp(y_logits)
    y_hat = jnp.array([y_exp / (1 + y_exp), 1 / (1 + y_exp)]).squeeze().T
    loss = CrossEntropyLogLoss(zero_ref=True)(y, y_logits)
    assert np.isclose(loss, cross_entropy_loss(y, y_hat))


@pytest.mark.parametrize("k", [1, 2, 2.5, 3])
def test_lnorm_regularization(k: int) -> None:
    y = jnp.linspace(0, 1, 10)
    loss = LNormRegularization(k, root=True)(y)
    assert np.isclose(loss, jnp.linalg.norm(y, ord=k))

    # Without root
    loss = LNormRegularization(k, root=False)(y)
    assert np.isclose(loss, jnp.linalg.norm(y, ord=k) ** k)


def test_binary_match_loss() -> None:
    y = jnp.array([[0, 1], [1, 0], [0, 1]])
    y_logits = jnp.array([[-2, 2], [-1, -1], [3, 4]])
    loss = binary_match_loss(y, y_logits)

    # Manually compute expected loss
    y_exp = jnp.exp(y_logits)
    y_norms = jnp.log1p(y_exp)  # log(1 + exp(y_pred))
    expected_loss = jnp.mean(y_norms) - jnp.mean(y * y_logits)

    assert np.isclose(loss, expected_loss)
    
    big_logit = 1e15
    y_exact_logits = jnp.where(y == 1, big_logit, -big_logit)
    
    assert np.isclose(binary_match_loss(y, y_exact_logits), 0.0)
