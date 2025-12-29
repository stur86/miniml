from miniml.random import MiniMLRandom, RandomMask
import jax.numpy as jnp
from jax import Array as JXArray
import jax
import pytest


def test_miniml_random_properties():
    class DummyRandom(MiniMLRandom):
        def generate(self, rng_key: JXArray) -> JXArray:
            return jax.numpy.zeros(self.shape, dtype=self.dtype)
        
    shape = (2, 3)
    dtype = jnp.int16
    
    random_instance = DummyRandom(shape, dtype)
    
    assert random_instance.shape == shape
    assert random_instance.dtype == dtype
    
    # Try calling the instance
    rng_key = jax.random.PRNGKey(0)
    random_array, new_key = random_instance(rng_key)
    
    assert random_array.shape == shape
    assert random_array.dtype == dtype
    assert isinstance(new_key, JXArray)
    assert new_key.shape == rng_key.shape
    assert jnp.all(new_key != rng_key)  # Ensure the key has changed
    
def test_random_mask_generation():
    shape = (20, 20)
    dtype = jnp.float32
    prob = 0.5
    
    random_mask = RandomMask(shape, prob, dtype)
    
    rng_key = jax.random.PRNGKey(42)
    mask, _ = random_mask(rng_key)
    
    assert mask.shape == shape
    assert mask.dtype == dtype
    assert jnp.all((mask == 0) | (mask == 1))  # Mask should only contain 0s and 1s
    
    # Check approximate proportion of ones
    proportion_ones = jnp.mean(mask)
    assert jnp.isclose(proportion_ones, prob, atol=0.1)  # Allow some tolerance due to randomness
    
    # Test edge cases for probability
    with pytest.raises(ValueError):
        RandomMask(shape, -0.1, dtype)
    with pytest.raises(ValueError):
        RandomMask(shape, 1.1, dtype)