import pytest
from miniml.nn.embedding import Embedding
import jax.numpy as jnp

def test_embedding_no_unknown():
    n_vocab = 5
    dim = 3
    emb = Embedding(n_vocab=5, dim=3, use_unknown=False)
    emb.bind()
    
    embd_matrix = jnp.arange(n_vocab * dim).reshape((n_vocab, dim)).astype(jnp.float32)
    emb.set_params({
        "_embeddings.v": embd_matrix
    })
    
    X = jnp.array([[0, 1], [2, 3]])
    output = emb(X)
    
    assert output.shape == (2, 2, dim)
    assert jnp.array_equal(output[0, 0], embd_matrix[0])
    assert jnp.array_equal(output[0, 1], embd_matrix[1])
    assert jnp.array_equal(output[1, 0], embd_matrix[2])
    assert jnp.array_equal(output[1, 1], embd_matrix[3])
    
    # Unknown should raise error
    X_invalid = jnp.array([[0, 5], [2, 3]])
    with pytest.raises(ValueError):
        emb(X_invalid)
        
    # Negative indices should raise error
    X_invalid_neg = jnp.array([[0, -1], [2, 3]])
    with pytest.raises(ValueError):
        emb(X_invalid_neg)
        
def test_embedding_with_unknown():
    n_vocab = 5
    dim = 3
    emb = Embedding(n_vocab=5, dim=3, use_unknown=True)
    emb.bind()
    
    embd_matrix = jnp.arange((n_vocab + 1) * dim).reshape((n_vocab + 1, dim)).astype(jnp.float32)
    emb.set_params({
        "_embeddings.v": embd_matrix
    })
    
    X = jnp.array([[0, 1], [5, 6]])  # 5 and 6 should map to unknown
    output = emb(X)
    
    assert output.shape == (2, 2, dim)
    assert jnp.array_equal(output[0, 0], embd_matrix[0])
    assert jnp.array_equal(output[0, 1], embd_matrix[1])
    assert jnp.array_equal(output[1, 0], embd_matrix[5])  # Unknown index
    assert jnp.array_equal(output[1, 1], embd_matrix[5])  # Unknown index
    
    # Negative indices should raise error
    X_invalid = jnp.array([[0, -1], [2, 3]])
    with pytest.raises(ValueError):
        emb(X_invalid)