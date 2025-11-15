from jax import Array as JXArray
from miniml.model import MiniMLModel
from miniml.loss import (
    RegLossFunction,
    LNormRegularization,
    LossFunction,
    squared_error_loss,
)
from miniml.param import MiniMLError, DTypeLike
import jax.numpy as jnp
from miniml.nn.activations import relu, ActivationFunction, Activation
from miniml.nn.linear import Linear
from miniml.nn.compose import Stack


class MLP(MiniMLModel):

    def __init__(
        self,
        layer_sizes: list[int],
        activation: ActivationFunction = relu,
        loss: LossFunction = squared_error_loss,
        reg_loss: RegLossFunction = LNormRegularization(2),
        dtype: DTypeLike = jnp.float32,
    ) -> None:
        """Multi-Layer Perceptron model.

        Args:
            layer_sizes (list[int]): List of layer sizes, including input and output layers.
                Must have at least two elements. Activation functions are applied between layers,
                except after the last layer.
            activation (ActivationFunction, optional): Activation function to use between layers. Defaults to relu.
                Can be any callable that takes a JAX array and returns a JAX array of the same shape.
            loss (LossFunction, optional): Loss function for the model. Defaults to squared_error_loss.
            reg_loss (RegLossFunction, optional): Regularization function for the layers.
                Defaults to LNormRegularization(2).
            dtype (DTypeLike, optional): Data type for the model parameters. Defaults to jnp.float32.
        """

        if len(layer_sizes) < 2:
            raise MiniMLError("MLP must have at least two layers (input and output)")

        layers: list[MiniMLModel] = []
        self._n = len(layer_sizes) - 1
        for i in range(self._n):
            in_size, out_size = layer_sizes[i], layer_sizes[i + 1]
            if in_size <= 0 or out_size <= 0:
                raise MiniMLError("Layer sizes must be positive integers")
            layers.append(
                Linear(
                    in_size,
                    out_size,
                    reg_loss=reg_loss,
                    dtype=dtype,
                    apply_bias_reg=False,
                )
            )
            if i < self._n - 1:
                layers.append(Activation(activation))

        self._layer_stack = Stack(layers)

        super().__init__(loss=loss)

    def _predict_kernel(self, X: JXArray, buffer: JXArray) -> JXArray:
        return self._layer_stack._predict_kernel(X, buffer)
