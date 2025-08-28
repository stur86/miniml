from jax import Array as JXArray
from miniml.model import MiniMLModel
from miniml.loss import RegLossFunction, LNormRegularization
from miniml.param import MiniMLParam, MiniMLError, MiniMLParamList, DTypeLike
import jax.numpy as jnp
from miniml.nn.activations import relu, ActivationFunction


class MLP(MiniMLModel):
    
    def __init__(self, layer_sizes: list[int], activation: ActivationFunction = relu, 
                 reg_loss: RegLossFunction = LNormRegularization(2), 
                 dtype: DTypeLike = jnp.float32) -> None:
        """Multi-Layer Perceptron model.

        Args:
            layer_sizes (list[int]): List of layer sizes, including input and output layers.
                Must have at least two elements. Activation functions are applied between layers,
                except after the last layer.
            activation (ActivationFunction, optional): Activation function to use between layers. Defaults to relu.
                Can be any callable that takes a JAX array and returns a JAX array of the same shape.
            reg_loss (RegLossFunction, optional): Regularization function for the layers.
                Defaults to LNormRegularization(2).
        """
        
        
        if len(layer_sizes) < 2:
            raise MiniMLError("MLP must have at least two layers (input and output)")
        
        layers: list[MiniMLParam] = []
        self._n = len(layer_sizes) - 1
        for i in range(self._n):
            in_size, out_size = layer_sizes[i], layer_sizes[i + 1]
            if in_size <= 0 or out_size <= 0:
                raise MiniMLError("Layer sizes must be positive integers")
            layers.append(MiniMLParam((in_size, out_size), dtype, reg_loss=reg_loss))
            layers.append(MiniMLParam((out_size,), dtype))
        self._layers = MiniMLParamList(layers)
        self._activation = activation
        
        super().__init__()
        
    def predict(self, X: JXArray) -> JXArray:
        for i in range(self._n):
            W = self._layers[2 * i].value
            b = self._layers[2 * i + 1].value
            X = X@W + b
            if i < self._n - 1:
                X = self._activation(X)
        return X

        
        
