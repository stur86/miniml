from miniml.nn import Identity
import jax.numpy as jnp

def test_identity():
    identity = Identity()
    identity.bind()

    in_data = jnp.array([[1, 2, 3], [4, 5, 6]], dtype=jnp.float32)
    out_data = identity.predict(in_data)

    assert jnp.array_equal(out_data, in_data)