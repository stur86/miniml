from miniml.nn.positional import PositionalEmbedding, RotaryPositionalEmbedding
import jax.numpy as jnp
import pytest


def test_positional_embedding_basic():
    max_length = 10
    dim = 4
    model = PositionalEmbedding(max_length=max_length, dim=dim)
    model.bind()
    
    pos_embeddings = jnp.arange(max_length)[:,None] + jnp.linspace(0, 1.0, dim)[None,:]
    model.set_params({'_pos_embeddings.v': pos_embeddings})

    # Create dummy input data: batch_size=2, seq_length=5, dim=4
    in_data = jnp.zeros((2, 5, dim))
    out_data = model.predict(in_data)

    # Check output shape
    assert out_data.shape == in_data.shape

    # Check that positional embeddings are added correctly
    expected_out = in_data + pos_embeddings[:5, :]
    assert jnp.allclose(out_data, expected_out)
    
    # Check that a sequence that exceeds max_length raises an error
    in_data_exceed = jnp.zeros((2, 15, dim))
    with pytest.raises(ValueError):
        model.predict(in_data_exceed)
        
def test_positional_embedding_invalid_init():
    with pytest.raises(ValueError):
        PositionalEmbedding(max_length=0, dim=4)
    with pytest.raises(ValueError):
        PositionalEmbedding(max_length=10, dim=0)
        
def test_rope():
    
    dim = 6
    rot_length = 20
    model = RotaryPositionalEmbedding(dim=dim, rot_length=rot_length)
    model.bind()

    # Create dummy input data: batch_size=2, seq_length=10, dim=6
    n_seq = 2
    in_data = jnp.ones((n_seq, dim))/2**0.5
    out_data = model.predict(in_data)

    # Check output shape
    assert out_data.shape == in_data.shape

    # Compute Gram matrix of input and output pairs
    gram = jnp.sum((out_data[:,None,:]*out_data[None,:,:]).reshape(n_seq, n_seq, 2, -1), axis=-2)

    i = jnp.arange(n_seq)
    # Expected angles
    freqs = 1.0 / (rot_length ** (jnp.arange(0, dim, 2) / dim))    
    angles = (i[:,None]-i[None,:])[:,:,None] * freqs[None,None,:]
    
    expected_gram = jnp.cos(angles)
        
    assert jnp.allclose(gram, expected_gram, atol=1e-5)

def test_rope_invalid_init():
    with pytest.raises(ValueError):
        RotaryPositionalEmbedding(dim=0, rot_length=10000)
    with pytest.raises(ValueError):
        RotaryPositionalEmbedding(dim=5, rot_length=10000)
    with pytest.raises(ValueError):
        RotaryPositionalEmbedding(dim=6, rot_length=0)