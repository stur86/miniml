from miniml.nn import Identity, Parallel
import jax.numpy as jnp


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
