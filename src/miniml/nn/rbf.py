"""Radial Basis Functions for RBF Networks."""

from jax import Array as JXArray
from typing import Callable
import jax.numpy as jnp

RBFunction = Callable[[JXArray], JXArray]


def gaussian_rbf(r: JXArray) -> JXArray:
    r"""Gaussian Radial Basis Function.

    $$
    \phi(r) = e^{-\frac{r^2}{2}}
    $$

    Args:
        r (JXArray): Input array.

    Returns:
        JXArray: Output after applying the Gaussian RBF.
    """
    return jnp.exp(-jnp.square(r) / 2)


def multiquadric_rbf(r: JXArray) -> JXArray:
    r"""Multiquadric Radial Basis Function.

    $$
    \phi(r) = \sqrt{1 + r^2}
    $$

    Args:
        r (JXArray): Input array.

    Returns:
        JXArray: Output after applying the Multiquadric RBF.
    """
    return jnp.sqrt(1 + jnp.square(r))


def inverse_multiquadric_rbf(r: JXArray) -> JXArray:
    r"""Inverse Multiquadric Radial Basis Function.

    $$
    \phi(r) = \frac{1}{\sqrt{1 + r^2}}
    $$

    Args:
        r (JXArray): Input array.

    Returns:
        JXArray: Output after applying the Inverse Multiquadric RBF.
    """
    return 1 / jnp.sqrt(1 + jnp.square(r))


def inverse_quadratic_rbf(r: JXArray) -> JXArray:
    r"""Inverse Quadratic Radial Basis Function.

    $$
    \phi(r) = \frac{1}{1 + r^2}
    $$

    Args:
        r (JXArray): Input array.

    Returns:
        JXArray: Output after applying the Inverse Quadratic RBF.
    """
    return 1 / (1 + jnp.square(r))


def linear_rbf(r: JXArray) -> JXArray:
    r"""Linear Radial Basis Function.

    $$
    \phi(r) = r
    $$

    Args:
        r (JXArray): Input array.

    Returns:
        JXArray: Output after applying the Linear RBF.
    """
    return r


def r_tanh_rbf(r: JXArray) -> JXArray:
    r"""Radius-times-Tanh Radial Basis Function.

    $$
    \phi(r) = r \tanh(r)
    $$

    Args:
        r (JXArray): Input array.

    Returns:
        JXArray: Output after applying the Radius-times-Tanh RBF.
    """
    return r * jnp.tanh(r)


class PolyharmonicRBF:
    r"""Polyharmonic Spline Radial Basis Function.
    
    $$
    \phi(r) = 
    \begin{cases}
    r^k, & \text{if } k \text{ is odd} \\
    r^k \log(r), & \text{if } k \text{ is even}
    \end{cases}
    $$

    Attributes:
        degree (int): Degree of the polynomial harmonic RBF.
    """

    def __init__(self, degree: int, eps: float = 1e-10):
        self.degree = degree
        self.eps = eps

        self._method = self._even_degree if degree % 2 == 0 else self._odd_degree

    def _even_degree(self, r: JXArray) -> JXArray:
        return jnp.power(r, self.degree) * jnp.log(
            r + self.eps
        )  # Adding a small constant to avoid log(0)

    def _odd_degree(self, r: JXArray) -> JXArray:
        return jnp.power(r, self.degree)

    def __call__(self, r: JXArray) -> JXArray:
        """Apply the Polynomial Harmonic RBF.

        Args:
            r (JXArray): Input array.

        Returns:
            JXArray: Output after applying the Polynomial Harmonic RBF.
        """
        return self._method(r)

    def __repr__(self) -> str:
        return f"PolyharmonicRBF(degree={self.degree})"


def bump_rbf(r: JXArray) -> JXArray:
    r"""Bump Radial Basis Function.
    
    $$
    \phi(r) = 
    \begin{cases}
    e^{1 - \frac{1}{1 - r^2}}, & \text{if } |r| < 1 \\
    0, & \text{otherwise}
    \end{cases}
    $$

    Args:
        r (JXArray): Input array.

    Returns:
        JXArray: Output after applying the Bump RBF.
    """
    return jnp.where(jnp.abs(r) < 1, jnp.exp(1 - 1 / (1 - jnp.square(r))), 0)
