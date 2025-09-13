# Building a model

Here is a basic working example of a model built using MiniML:

```python { .yaml .copy }
from miniml import MiniMLModel, MiniMLParam

class LinearModel(MiniMLModel):
    A: MiniMLParam
    b: MiniMLParam

    def __init__(self, n_in: int, n_out: int):
        self.A = MiniMLParam((n_in,n_out))
        self.b = MiniMLParam((n_out,))
        super().__init__()

    def _predict_kernel(self, X, buffer):
        return X@self.A(buffer)+self.b(buffer)

if __name__ == "__main__":
    import numpy as np
    import jax.numpy as jnp

    rng = np.random.default_rng(seed=1)
    X = jnp.array(rng.normal(size=(100,3)))
    y = jnp.sum(X, axis=1, keepdims=True) - 0.5 # Linear function

    lin_model = LinearModel(X.shape[1], y.shape[1])
    lin_model.randomize()
    success = lin_model.fit(X, y).success
    y_hat = lin_model.predict(X)
    
    print(f"Fit converged: {success}")
    print(f"Final loss: {lin_model.loss(y, y_hat)}")
```

Let's go through it step by step.

#### Imports

```py
from miniml import MiniMLModel, MiniMLParam
```

The top level `miniml` provides imports for model and parameter classes; other useful imports are found in `miniml.loss` and `miniml.nn`.

#### Class definition

```py
class LinearModel(MiniMLModel):
    A: MiniMLParam
    b: MiniMLParam
```

Our model must inherit from `MiniMLModel`, or one of its child classes. Type hints for the parameters are optional.

#### Constructor

```py
def __init__(self, n_in: int, n_out: int):
    self.A = MiniMLParam((n_in,n_out))
    self.b = MiniMLParam((n_out,))
    super().__init__()
```

The constructor structure is extremely important:

* ***all*** child parameters and models must be defined inside the constructor;
* the parent constructor `super().__init__()` must be called as the very last thing.

If either of these things aren't done, the model won't work. Parameters are tensors initialized by passing the shape as a tuple of integers, and optionally the `jax.numpy.dtype` and the regularization loss (see [regularization loss](loss.md#regularization-loss) for more details)

#### Predict kernel implementation

```py
def _predict_kernel(self, X, buffer):
    return X@self.A(buffer)+self.b(buffer)
```

`MiniMLModel` is an abstract base class with `._predict_kernel` as an abstract method, meaning any child class has to provide its implementation of it. This is the "forward" inference method. It takes in the input and a buffer (which contains all the model's parameters), and it must return the output.

!!! note
    The parameter `buffer` is necessary because Jax requires all differentiable functions to be "pure", meaning they can't modify state.
    This means every time `_predict_kernel` is called (inside `predict` or `fit`), it gets passed an array with all the parameters it needs
    to use. When implementing `_predict_kernel` you must take care to pass this array forward to all parameters and/or child models so that they can also use it! This means:

    * for parameters, retrieve their value by calling them with the buffer as argument (as in `self.A(buffer)`);

    * for models, if there are any, call them using their own `_predict_kernel` methods passing the buffer argument in the same way it was passed to this one.

`X` should be a Jax array, and the parameters can be accessed by calling them with the buffer as an argument; they will also be Jax arrays.
If any parameter is not called this way it will not use the correct values during fitting. Write this function sticking to Jax philosophy for differentiability (use Jax functions and functional constructs).

#### Example data

```py
if __name__ == "__main__":
    import numpy as np
    import jax.numpy as jnp

    rng = np.random.default_rng(seed=1)
    X = jnp.array(rng.normal(size=(100,3)))
    y = jnp.sum(X, axis=1, keepdims=True) - 0.5 # Linear function
```

Here we just import a couple libraries and generate test data in Jax array form.

#### Preparing the model

```py
    lin_model = LinearModel(X.shape[1], y.shape[1])
    lin_model.randomize()
```

Here we create the model object (passing in the input and output dimensions), and then invoke `.randomize()`. This method does two things:

* internally invoke `lin_model.bind()`, which is an essential step that links each parameter to the proper "address" in the global linearized buffer that stores all the model's parameters;
* assign random, normally-distributed values to that buffer (it can be passed a `seed` argument for determinism if desired).

For models created from scratch, `.randomize()` is recommended. If loading a model from a file, `.bind()` would be sufficient here, before invoking `.load`. 

#### Fitting

```py
    success = lin_model.fit(X, y).success
```

Fitting is performed here on a batch of `X` and `y`. The required shape of these tensors is entirely down to how the `.predict` method and the loss functions are implemented, so it can be customized if necessary.

#### Predicting

```py
    y_hat = lin_model.predict(X)
    
    print(f"Fit converged: {success}")
    print(f"Final loss: {lin_model.loss(y, y_hat)}")
```

Here finally we see how to invoke predict directly, and then we can write out the loss of the model (note that `.loss` returns only the base loss, not the regularization loss).