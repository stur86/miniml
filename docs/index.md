# Introduction

MiniML (pronounced "minimal") is a tiny machine-learning framework which uses [Jax](https://github.com/jax-ml/jax) as its core engine, but mixes a PyTorch inspired approach to building model with Scikit-learn's interface (using the `.fit` and `.predict` methods), and is powered by SciPy's optimization algorithms. It's meant for simple prototyping of small ML architectures that allows more flexibility than Scikit's built-in models without sacrificing too much on performance.

Training a linear model in MiniML for example looks as simple as this:

```py
class LinearModel(MiniMLModel):
    A: MiniMLParam
    b: MiniMLParam

    def __init__(self, n_in: int, n_out: int):
        self.A = MiniMLParam((n_in,n_out))
        self.b = MiniMLParam((n_out,))
        super().__init__()

    def _predict_kernel(self, X, buffer):
        return X@self.A(buffer)+self.b(buffer)
        
lin_model = LinearModel(X.shape[1], y.shape[1])
lin_model.randomize()
lin_model.fit(X, y)
y_hat = lin_model.predict(X)
```

## How does it work?

MiniML is a simple wrapper for Jax's differentiation capabilities and SciPy's `minimize` optimizer. When you invoke the base constructor of the `MiniMLModel` class, it scans its `__dict__` for any fittable parameters (`MiniMLParam` objects, other models, and lists of either parameters or models). It then compiles them all into a single JAX array, a "buffer", which can be then optimized by the `.fit` method, as well as saved and loaded. The specified loss functions are used appropriately in the process and the model is fitted.