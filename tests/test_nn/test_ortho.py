import pytest
import numpy as np
import jax.numpy as jnp
from miniml.nn.ortho import CayleyMatrix


@pytest.mark.parametrize(
    "dim,seed",
    [
        (2, 42),
        (3, 123),
        (4, 2021),
        (5, 7),
    ],
)
def test_cayley_matrix(dim: int, seed: int):
    rng = np.random.default_rng(seed)
    cayley = CayleyMatrix(dim=dim, dtype=jnp.float32)

    assert cayley.shape == (dim, dim)
    assert cayley.size == dim * (dim - 1) // 2

    buffer = jnp.array(rng.normal(size=cayley.size), dtype=jnp.float32)

    class BufferContainer:
        def __init__(self, buffer: jnp.ndarray):
            self._buffer = buffer

    cayley.bind(0, BufferContainer(buffer))

    C = cayley.value

    # Check orthogonality
    Ieye = jnp.eye(dim, dtype=jnp.float32)
    assert jnp.allclose(Ieye, C @ C.T, atol=1e-7), f"C @ C.T != I for dim={dim}"
