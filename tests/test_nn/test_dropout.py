import pytest
import jax.numpy as jnp
import jax
from miniml.nn.dropout import Dropout
from miniml.model import PredictMode

def test_dropout_forward_training() -> None:
    key = jax.random.PRNGKey(0)
    x = jnp.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    dropout_rate = 0.5
    dropout_model = Dropout(shape=x.shape, rate=dropout_rate)
    buffer = jnp.array([])

    out = dropout_model._predict_kernel(x, buffer=buffer, rng_key=key, mode=PredictMode.TRAINING)
    
    # Expected mask
    expected_mask = jax.random.bernoulli(key, p=1-dropout_rate, shape=x.shape)
    expected_out = x * expected_mask / (1.0 - dropout_rate)
    
    assert jnp.array_equal(out, expected_out), "Dropout forward pass in training mode failed."
    
    # In inference mode, output should be same as input
    out_inference = dropout_model._predict_kernel(x, buffer=buffer, mode=PredictMode.INFERENCE)
    assert jnp.array_equal(out_inference, x), "Dropout forward pass in inference mode failed."
    
    # Should fail if rng_key is not provided in training mode
    with pytest.raises(ValueError):
        dropout_model._predict_kernel(x, buffer=buffer, mode=PredictMode.TRAINING)