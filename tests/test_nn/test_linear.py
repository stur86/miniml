import numpy as np
import jax.numpy as jnp
from miniml.nn import Linear

def test_linear():
    
    W = np.array([[2,3], [-1,1]])
    b = np.array([1,0])
    
    m = Linear(n_in=W.shape[1], n_out=W.shape[0], apply_bias_reg=False)
    # Now bind
    m.bind()
    # And set the buffer manually
    m._buffer = jnp.array(np.concatenate([W.flatten(), b]))
    
    assert m.regularization_loss() == np.sum(W**2)
    
    # Try with bias regularization
    m = Linear(n_in=W.shape[1], n_out=W.shape[0], apply_bias_reg=True)
    # Now bind
    m.bind()
    # And set the buffer manually
    m._buffer = jnp.array(np.concatenate([W.flatten(), b]))
    
    assert m.regularization_loss() == np.sum(W**2)+np.sum(b**2)
    
    # Prediction
    X = np.array([[3,1]])
    y_pred = m.predict(jnp.array(X))
    
    assert np.allclose(y_pred, X @ W + b)
