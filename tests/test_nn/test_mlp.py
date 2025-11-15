import jax.numpy as jnp
import numpy as np
import pytest
from miniml.nn.mlp import MLP
from miniml.param import MiniMLError


def test_mlp_forward_pass():
    # Simple 2-layer MLP: input 3, output 2
    mlp = MLP([3, 4, 2])
    X = jnp.ones((5, 3))  # batch of 5
    mlp.randomize()
    out = mlp.predict(X)
    assert out.shape == (5, 2)
    assert jnp.isfinite(out).all()


def test_mlp_invalid_layer_sizes():
    # Less than 2 layers should raise error
    with pytest.raises(MiniMLError):
        MLP([3])
    # Negative layer size should raise error
    with pytest.raises(MiniMLError):
        MLP([3, -2])
    with pytest.raises(MiniMLError):
        MLP([0, 2])


def test_mlp_custom_activation():
    def custom_act(x):
        return x * 2

    mlp = MLP([2, 2], activation=custom_act)
    mlp.randomize()
    X = jnp.ones((1, 2))
    out = mlp.predict(X)
    assert out.shape == (1, 2)
    # Should be finite and not all zeros
    assert jnp.isfinite(out).all()
    assert not jnp.all(out == 0)


def test_mlp_training_on_max_sum():
    # Generate simple dataset: input (x1, x2), output is max(x1 + x2, 0)
    rng = np.random.default_rng(42)
    X = jnp.array(rng.normal(size=(100, 2)))
    y = jnp.maximum(jnp.sum(X, axis=1, keepdims=True), 0.0)

    # 2-layer MLP: input 2, hidden 1, output 1
    mlp = MLP([2, 1, 1])
    mlp.randomize(seed=99)
    # Train
    mlp.fit(X, y, reg_lambda=0.0)
    # Predict
    y_pred = mlp.predict(X)
    # Check that predictions are close to true values
    assert y_pred.shape == y.shape
    # Check that they match
    assert jnp.allclose(y_pred, y, atol=1e-2)
