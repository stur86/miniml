"""Utilities for common ML model architectures"""

from miniml.nn.activations import (
    relu,
    sigmoid,
    tanh,
    LeakyReLU,
    Activation,
    ActivationFunction,
    gelu,
    silu,
)
from miniml.nn.rbf import (
    RBFunction,
    gaussian_rbf,
    multiquadric_rbf,
    inverse_quadratic_rbf,
    inverse_multiquadric_rbf,
    linear_rbf,
    r_tanh_rbf,
    bump_rbf,
    PolyharmonicRBF,
)
from miniml.nn.linear import Linear
from miniml.nn.compose import Stack, Identity, Parallel, Take
from miniml.nn.mlp import MLP
from miniml.nn.rbfnet import RBFLayer
from miniml.nn.embedding import Embedding
from miniml.nn.layer_norm import LayerNorm
from miniml.nn.mha import MultiHeadAttention

__all__ = [
    "Linear",
    "Stack",
    "MLP",
    "Embedding",
    "LayerNorm",
    "Activation",
    "ActivationFunction",
    "relu",
    "sigmoid",
    "tanh",
    "gelu",
    "silu",
    "LeakyReLU",
    "RBFunction",
    "gaussian_rbf",
    "multiquadric_rbf",
    "inverse_quadratic_rbf",
    "inverse_multiquadric_rbf",
    "linear_rbf",
    "r_tanh_rbf",
    "bump_rbf",
    "PolyharmonicRBF",
    "RBFLayer",
    "Identity",
    "Parallel",
    "Take",
    "MultiHeadAttention",
]
