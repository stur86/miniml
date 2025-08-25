from jax import Array
import jax.numpy as jnp
from functools import partial

from typing import Callable

RegLossFunction = Callable[[Array], float]
LossFunction = Callable[[Array, Array], float]

def squared_error_loss(y_true: Array, y_pred: Array) -> Array:
    """Compute the squared loss between true and predicted values."""
    if y_true.shape != y_pred.shape:
        raise ValueError("Shapes of true and predicted values must match.")
    return jnp.sum((y_true - y_pred) ** 2)

def cross_entropy_loss(y_true: Array, y_pred: Array) -> Array:
    """Compute the cross-entropy loss between true and predicted values."""
    if y_true.shape != y_pred.shape:
        raise ValueError("Shapes of true and predicted values must match.")
    # Avoid log(0) by clipping predictions
    log_y_pred = jnp.log(jnp.clip(y_pred, 1e-15, 1 - 1e-15))
    return -jnp.sum(y_true * log_y_pred)

def norm_regularization(y: Array, p: float) -> Array:
    """Compute the norm regularization term."""
    return jnp.sum(jnp.abs(y) ** p)**(1.0/p)

l2_regularization = partial(norm_regularization, p=2.0)
l1_regularization = partial(norm_regularization, p=1.0)