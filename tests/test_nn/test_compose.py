import numpy as np
from miniml.nn import Stack, Linear, Identity, Parallel, Take
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


def test_stack():
    # Create two linear models in sequence
    l1 = Linear(n_in=2, n_out=2)
    l2 = Linear(n_in=2, n_out=2)

    m1 = np.ones((2, 2)) - np.eye(2)
    b1 = np.array([1, 0])
    m2 = np.array([[1, 0], [1, 1]])
    b2 = np.array([-0.5, 0.5])

    stack = Stack([l1, l2])
    stack.bind()
    stack.set_buffer(np.concatenate([m1.flatten(), b1, m2.flatten(), b2]))

    in_data = np.array([[1, 2], [3, 4], [5, 6]])
    out_data = stack.predict(jnp.array(in_data))

    out_target = in_data @ m1 + b1
    out_target = out_target @ m2 + b2

    assert np.allclose(out_data, out_target)


def test_parallel_sum():
    model1 = Identity()
    model2 = Identity()
    parallel = Parallel([model1, model2], mode="sum")
    parallel.bind()

    in_data = jnp.array([[1, 2, 3], [4, 5, 6]], dtype=jnp.float32)
    out_data = parallel.predict(in_data)

    assert jnp.array_equal(out_data, in_data * 2)


def test_parallel_concat():
    model1 = Identity()
    model2 = Identity()
    parallel = Parallel([model1, model2], mode="concat", concat_axis=1)
    parallel.bind()

    in_data = jnp.array([[1, 2, 3], [4, 5, 6]], dtype=jnp.float32)
    out_data = parallel.predict(in_data)

    expected_output = jnp.array(
        [[1, 2, 3, 1, 2, 3], [4, 5, 6, 4, 5, 6]], dtype=jnp.float32
    )
    assert jnp.array_equal(out_data, expected_output)


def test_take():
    indices = [0, 2]
    take_model = Take(indices=indices, axis=1)
    take_model.bind()

    in_data = jnp.array([[10, 20, 30], [40, 50, 60]], dtype=jnp.float32)
    out_data = take_model.predict(in_data)

    expected_output = jnp.array([[10, 30], [40, 60]], dtype=jnp.float32)
    assert jnp.array_equal(out_data, expected_output)

    # Try with a scalar index
    scalar_index = 1
    take_model_scalar = Take(indices=scalar_index, axis=1)
    take_model_scalar.bind()

    in_data_scalar = jnp.array([[10, 20, 30], [40, 50, 60]], dtype=jnp.float32)
    out_data_scalar = take_model_scalar.predict(in_data_scalar)

    expected_output_scalar = jnp.array([20, 50], dtype=jnp.float32)
    assert jnp.array_equal(out_data_scalar, expected_output_scalar)
