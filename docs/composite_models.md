# Composite models

It's possible in MiniML to build composite models, which include themselves models inside. For example, a simple two-layer neural network could be:

```python
from miniml.nn.activations import Activation
from miniml.nn.linear import Linear

class NN(MiniMLModel):
    def __init__(n_in, n_hidden, n_out, activation):
        self._L1 = Linear(n_in, n_hidden)
        self._act = Activation(activation)
        self._L2 = Linear(n_hidden, n_out)

    def _predict_kernel(self, X, buffer, rng_key=None, mode=None):
        X = self._L1._predict_kernel(X, buffer, rng_key=rng_key, mode=mode)
        X = self._act._predict_kernel(X, buffer, rng_key=rng_key, mode=mode)
        return self._L2._predict_kernel(X, buffer, rng_key=rng_key, mode=mode)
```

Notice the use of `_predict_kernel` instead of `predict`. It's important to use it and pass around the `buffer` argument (and propagate `rng_key` and `mode` if present) because this keeps the function pure and compatible with Jax's traceability rules (for more info see for example [this Jax documentation page](https://docs.jax.dev/en/latest/errors.html#jax.errors.UnexpectedTracerError)).

## Sharing a model between parents

A `MiniMLModel` owns the parameters of every child model it stores, so storing the same child in two places makes MiniML count, and try to bind, its parameters twice. Wrap it in a [SharedModel](api/miniml/model.md#miniml.model.SharedModel) to state that this parent only *uses* it:

```py
class Decoder(MiniMLModel):
    def __init__(self, encoder):
        self._encoder = SharedModel(encoder) # borrowed, not owned
        super().__init__()
```

A `SharedModel` declares no parameters of its own, so it adds nothing to the buffer. The model inside it must still be stored directly somewhere else in the same tree, otherwise its parameters never get bound. Offsets are kept on the parameters themselves, so a borrowed model always reads the same values as its owner.
