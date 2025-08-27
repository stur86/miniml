from jax import Array as JXArray
import jax.numpy as jnp
from typing import Callable
from jax.scipy.special import erf
from jax.nn import softplus as jax_softplus

ActivationFunction = Callable[[JXArray], JXArray]

def relu(x: JXArray) -> JXArray:
    """ReLU activation function."""
    return jnp.maximum(0, x)

class LeakyReLU:
    """Leaky ReLU activation function."""
    
    def __init__(self, alpha: float = 0.01) -> None:
        self.alpha = alpha
        
    def __call__(self, x: JXArray) -> JXArray:
        return jnp.where(x > 0, x, self.alpha * x)

def sigmoid(x: JXArray) -> JXArray:
    """Sigmoid activation function."""
    return 1 / (1 + jnp.exp(-x))

def silu(x: JXArray) -> JXArray:
    """SiLU (Swish) activation function."""
    return x * sigmoid(x)

def gelu(x: JXArray) -> JXArray:
    """GELU activation function."""
    return 0.5 * x * (1 + erf(x / jnp.sqrt(2)))

def tanh(x: JXArray) -> JXArray:
    """Tanh activation function."""
    return jnp.tanh(x)

def softplus(x: JXArray) -> JXArray:
    """Softplus activation function."""
    return jax_softplus(x)