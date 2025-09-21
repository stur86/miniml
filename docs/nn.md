# The `miniml.nn` module

The [`miniml.nn` module](api/miniml/nn/index.md) provides some basic utilities to build simple common ML models. Among these:

* [`miniml.nn.activations`](api/miniml/nn/activations.md) includes common activation functions and an `Activation` layer class that can be used as a container for them;
* [`miniml.nn.linear`](api/miniml/nn/linear.md) contains a basic linear layer;
* [`miniml.nn.mlp`](api/miniml/nn/mlp.md) contains a simple [Multilayer Perceptron](https://en.wikipedia.org/wiki/Multilayer_perceptron);
* [`miniml.nn.stack`](api/miniml/nn/stack.md) contains a `Stack` model that allows to execute sequentially a list of models, passing the output of the previous one as input to the following;
* [`miniml.nn.rbf`](api/miniml/nn/rbf.md) contains various radial basis functions;
* [`miniml.nn.rbfnet`](api/miniml/nn/rbf.md) contains a basic radial basis function net layer.


Check [the API docs](api/miniml/nn/index.md) for more information.