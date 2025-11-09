from miniml.nn import Identity
import jax.numpy as jnp


def test_identity():
    identity = Identity()
    identity.bind()

    in_data = jnp.array([[1, 2, 3], [4, 5, 6]], dtype=jnp.float32)
    out_data = identity.predict(in_data)

    assert jnp.array_equal(out_data, in_data)

    # Test with scaling
    scale = 2.0
    identity_scaled = Identity(scale=scale)
    identity_scaled.bind()
    out_data_scaled = identity_scaled.predict(in_data)
    assert jnp.array_equal(out_data_scaled, in_data * scale)
