import numpy as np
import jax.numpy as jnp
from functools import partial
from numbers import Number
from numpy.typing import NDArray
from typing import Callable

RegLossFunction = Callable[[NDArray], float]
LossFunction = Callable[[NDArray, NDArray], float]

def squared_error_loss(y_true: NDArray, y_pred: NDArray) -> jnp.ndarray:
    """Compute the squared loss between true and predicted values."""
    if y_true.shape != y_pred.shape:
        raise ValueError("Shapes of true and predicted values must match.")
    return jnp.sum((y_true - y_pred) ** 2)

def cross_entropy_loss(y_true: NDArray, y_pred: NDArray) -> jnp.ndarray:
    """Compute the cross-entropy loss between true and predicted values."""
    if y_true.shape != y_pred.shape:
        raise ValueError("Shapes of true and predicted values must match.")
    # Avoid log(0) by clipping predictions
    log_y_pred = jnp.log(jnp.clip(y_pred, 1e-15, 1 - 1e-15))
    return -jnp.sum(y_true * log_y_pred)


def norm_regularization(y: NDArray, p: float) -> jnp.ndarray:
    """Compute the norm regularization term."""
    return jnp.sum(jnp.abs(y) ** p)**(1.0/p)

l2_regularization = partial(norm_regularization, p=2.0)
l1_regularization = partial(norm_regularization, p=1.0)