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

    def predict(self, X):
        X = self._L1.predict_kernel(X)
        X = self._act.predict_kernel(X)
        return self._L2.predict_kernel(X)
```

Notice the use of `predict_kernel` instead of `predict`. This is the original, "raw" version of predict that is *not* JIT compiled by Jax. It's useful to call it when composing models because sometimes nesting JIT functions can cause issues of leaking variables in Jax (for more info see for example [this Jax documentation page](https://docs.jax.dev/en/latest/errors.html#jax.errors.UnexpectedTracerError)).