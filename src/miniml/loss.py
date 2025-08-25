from abc import ABC, abstractmethod
from jax.nn import log_softmax
from jax import Array as JXArray
import jax.numpy as jnp

from typing import Callable

RegLossFunction = Callable[[JXArray], JXArray]
LossFunction = Callable[[JXArray, JXArray], JXArray]

class LossFunctionBase(ABC):
    
    @abstractmethod
    def __call__(self, y_true: JXArray, y_pred: JXArray) -> JXArray:
        """Compute the loss between true and predicted values."""
        pass
    
class RegLossFunctionBase(ABC):
    
    @abstractmethod
    def __call__(self, y: JXArray) -> JXArray:
        """Compute the regularization loss for the given values."""
        pass

def squared_error_loss(y_true: JXArray, y_pred: JXArray) -> JXArray:
    """Compute the squared loss between true and predicted values."""
    if y_true.shape != y_pred.shape:
        raise ValueError("Shapes of true and predicted values must match.")
    return jnp.sum((y_true - y_pred) ** 2)

def cross_entropy_loss(y_true: JXArray, y_pred: JXArray) -> JXArray:
    """Compute the cross-entropy loss between true and predicted values."""
    if y_true.shape != y_pred.shape:
        raise ValueError("Shapes of true and predicted values must match.")
    # Avoid log(0) by clipping predictions
    log_y_pred = jnp.log(jnp.clip(y_pred, 1e-15, 1 - 1e-15))
    return -jnp.sum(y_true * log_y_pred)


class CrossEntropyLogLoss(LossFunctionBase):
    
    def __init__(self, zero_ref: bool = False) -> None:
        self.zero_ref = zero_ref
        
    def __call__(self, y_true: JXArray, log_y_pred: JXArray) -> JXArray:
        if self.zero_ref:
            # Add a zero reference category
            zero_shape = log_y_pred.shape[:-1] + (1,)
            log_y_pred = jnp.concatenate([log_y_pred, jnp.zeros(zero_shape)], axis=-1)

        if y_true.shape != log_y_pred.shape:
            raise ValueError("Shapes of true and predicted values must match.")

        log_y_pred = log_softmax(log_y_pred, axis=-1)
        return -jnp.sum(y_true * log_y_pred)

class LNormRegularization(RegLossFunctionBase):
    def __init__(self, k: int = 2, root: bool = False) -> None:
        self.k = k
        self.return_root = root
        
    def __call__(self, y: JXArray) -> JXArray:
        ans = jnp.sum(jnp.abs(y) ** self.k)
        if self.return_root:
            return ans**(1.0/self.k)
        return ans
