import pytest
import numpy as np
from jax.scipy.special import erf
import jax.numpy as jnp
from miniml.nn import activations

def test_relu():
	assert activations.relu(jnp.array(1.0)) == 1.0
	assert activations.relu(jnp.array(-1.0)) == 0.0
	assert activations.relu(jnp.array(0.0)) == 0.0

def test_leaky_relu():
	leaky = activations.LeakyReLU(alpha=0.1)
	assert leaky(jnp.array(1.0)) == 1.0
	assert leaky(jnp.array(-1.0)) == -0.1
	assert leaky(jnp.array(0.0)) == 0.0

def test_sigmoid():
	assert pytest.approx(activations.sigmoid(jnp.array(0.0)), 0.0001) == 0.5
	assert pytest.approx(activations.sigmoid(jnp.array(1.0)), 0.0001) == 1/(1+jnp.exp(-1.0))
	assert pytest.approx(activations.sigmoid(jnp.array(-1.0)), 0.0001) == 1/(1+jnp.exp(1.0))

def test_silu():
	# SiLU(0) = 0
	assert pytest.approx(activations.silu(jnp.array(0.0)), 0.0001) == 0.0
	# SiLU(1) = 1 * sigmoid(1)
	assert pytest.approx(activations.silu(jnp.array(1.0)), 0.0001) == float(1.0 * activations.sigmoid(jnp.array(1.0)))

def test_gelu():
	# GELU(0) = 0
	assert pytest.approx(activations.gelu(jnp.array(0.0)), 0.0001) == 0.0
	# GELU(1) = 0.5 * 1 * (1 + erf(1/sqrt(2)))
	expected = 0.5 * 1.0 * (1 + float(erf(1.0/np.sqrt(2))))
	assert pytest.approx(activations.gelu(jnp.array(1.0)), 0.0001) == expected

def test_tanh():
	assert pytest.approx(activations.tanh(jnp.array(0.0)), 0.0001) == 0.0
	assert pytest.approx(activations.tanh(jnp.array(1.0)), 0.0001) == float(jnp.tanh(1.0))
	assert pytest.approx(activations.tanh(jnp.array(-1.0)), 0.0001) == float(jnp.tanh(-1.0))

def test_softplus():
	# softplus(0) = log(2)
	assert pytest.approx(activations.softplus(jnp.array(0.0)), 0.0001) == np.log(2)
	# softplus(1) = log(1+exp(1))
	assert pytest.approx(activations.softplus(jnp.array(1.0)), 0.0001) == np.log(1+np.exp(1.0))
