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