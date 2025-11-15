import numpy as np
import jax.numpy as jnp
from miniml.nn import Stack, Linear


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
