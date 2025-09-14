import jax.numpy as jnp
from jax import Array as JXArray
from numpy.typing import DTypeLike
from miniml.param import MiniMLParam, MiniMLParamRef, BufferContainer


class CayleyMatrix(MiniMLParam):
    r"""Cayley Matrix Parameter.

    A Cayley matrix is a square orthogonal matrix that can be parameterized using a skew-symmetric matrix.
    This parameterization is useful in various applications, including 3D rotations and computer graphics.

    The Cayley transform is defined as:

    $$
    C = (I - A)(I + A)^{-1}
    $$

    where \( C \) is the Cayley matrix, \( I \) is the identity matrix, and \( A \) is a skew-symmetric matrix.

    Args:
        dim (int): The dimension of the Cayley matrix.
        dtype (DTypeLike, optional): The data type of the parameter. Defaults to np.float32.
    """

    def __init__(self, dim: int, dtype: DTypeLike = jnp.float32) -> None:

        # Record indices of the upper triangular part of the skew-symmetric matrix
        self._idx = jnp.triu_indices(dim, k=1)
        n_params = len(self._idx[0])
        self._dim = dim
        self._shape = (dim, dim)
        self._params = MiniMLParam(shape=(n_params,), dtype=dtype)
        self._dtype = self._params._dtype
        self._dtype_name = self._params._dtype_name
        self._size = self._params._size

    def bind(self, i0: int, bufc: BufferContainer) -> None:
        self._params.bind(i0, bufc)

    def __call__(self, buffer: JXArray | None = None) -> jnp.ndarray:
        """Compute the Cayley matrix from the skew-symmetric parameters.

        Returns:
            jnp.ndarray: The Cayley matrix.
        """

        jnp_dtype = jnp.dtype(self._dtype)
        A = jnp.zeros((self._dim, self._dim), dtype=jnp_dtype)
        A = A.at[self._idx].set(self._params(buffer))
        A = A - A.T  # Make it skew-symmetric

        Ieye = jnp.eye(self._dim, dtype=jnp_dtype)
        C = jnp.linalg.solve(
            Ieye + A, Ieye - A
        )  # More stable than (I - A) @ inv(I + A)
        return C

    def _get_inner_params(self) -> list[MiniMLParamRef]:
        return self._params._get_inner_params()
