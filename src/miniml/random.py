import jax
from abc import ABC, abstractmethod
from jax import Array as JXArray
import jax.numpy as jnp


class MiniMLRandom(ABC):
    """Utility class for random number generation in MiniML."""

    _shape: tuple[int, ...]
    _dtype: jnp.dtype

    def __init__(self, shape: tuple[int, ...], dtype: jnp.dtype = jnp.float32) -> None:
        """Initialize the MiniMLRandom with a given shape and data type.

        Args:
            shape (tuple[int, ...]): The shape of the random array to generate.
            dtype (jnp.dtype, optional): The data type of the random array. Defaults to jnp.float32.
        """
        self._shape = shape
        self._dtype = dtype

    @property
    def shape(self) -> tuple[int, ...]:
        """Get the shape of the random array.

        Returns:
            tuple[int, ...]: The shape of the random array.
        """
        return self._shape

    @property
    def dtype(self) -> jnp.dtype:
        """Get the data type of the random array.

        Returns:
            jnp.dtype: The data type of the random array.
        """
        return self._dtype

    @abstractmethod
    def generate(self, rng_key: JXArray) -> JXArray:
        """Abstract method to generate random numbers.

        Args:
            rng_key (jax.random.KeyArray): The JAX random key for generating random numbers.

        Returns:
            JXArray: The generated random array.
        """
        ...

    def __call__(self, rng_key: JXArray) -> tuple[JXArray, JXArray]:
        """Generate random numbers using the provided random key.

        Args:
            rng_key (jax.random.KeyArray): The JAX random key for generating random numbers.

        Returns:
            tuple[JXArray, JXArray]: A tuple containing the generated random array and the new random key.
        """
        new_key, subkey = jax.random.split(rng_key)
        random_array = self.generate(subkey).astype(self._dtype)
        return random_array, new_key


class RandomMask(MiniMLRandom):
    """Class for generating random binary masks."""

    _p: float

    def __init__(
        self, shape: tuple[int, ...], p: float, dtype: jnp.dtype = jnp.float32
    ) -> None:
        """Initialize the RandomMask with a given shape, probability, and data type.

        Args:
            shape (tuple[int, ...]): The shape of the mask to generate.
            p (float): The probability of an element being 1 in the mask.
            dtype (jnp.dtype, optional): The data type of the mask. Defaults to jnp.float32.
        """
        super().__init__(shape, dtype)
        if not (0.0 <= p <= 1.0):
            raise ValueError("Probability p must be in the range [0.0, 1.0]")
        self._p = p

    def generate(self, rng_key: JXArray) -> JXArray:
        return jax.random.bernoulli(rng_key, p=self._p, shape=self._shape).astype(
            self._dtype
        )
