# MiniML

[![Run tests](https://github.com/stur86/miniml/actions/workflows/test.yml/badge.svg?branch=main)](https://github.com/stur86/miniml/actions/workflows/test.yml)

MiniML (pronounced "minimal") is a tiny machine-learning framework which uses [Jax](https://github.com/jax-ml/jax) as its core engine, but mixes a PyTorch
inspired approach to building model with Scikit-learn's interface (using the `.fit` and `.predict` methods), and is powered by SciPy's optimization algorithms. It's meant for simple prototyping of small ML architectures that allows more flexibility than Scikit's built-in models without sacrificing too much on performance.

 Training a linear model in MiniML for example looks as simple as this:

```py
class LinearModel(MiniMLModel):
    A: MiniMLParam
    b: MiniMLParam

    def __init__(self, n_in: int, n_out: int):
        self.A = MiniMLParam((n_in,n_out))
        self.b = MiniMLParam((n_out,))
        super().__init__()

    def predict(self, X):
        return X@self.A.value+self.b.value

lin_model = LinearModel(X.shape[1], y.shape[1])
lin_model.randomize()
lin_model.fit(X, y)
y_hat = lin_model.predict(X)
```

## Installation

Simply install this package from PyPi:

```bash
pip install miniml-jax
```

## Usage

The two core types are `MiniMLParam` and `MiniMLModel`. There are also `MiniMLParamList` and `MiniMLModelList` containers to store multiple of either inside.

To define a model in MiniML, subclass `MiniMLModel` and define your parameters as `MiniMLParam` attributes in the `__init__` method. Remember to make sure that:

* every parameter or child model is stored either directly as a class member, or inside a corresponding `List` class;
* the `super().__init__()` constructor is called at the end.


Then, implement the `predict` method, which takes an input array and returns the model's prediction. After instantiating your model, call `bind()` to initialize parameter buffers, or use directly `randomize()` to initialize parameter values. You can then use methods like `fit`, `save`, and `load`.

### Example: Linear Model

```python
import jax.numpy as jnp
from miniml.param import MiniMLParam
from miniml.model import MiniMLModel

class LinearModel(MiniMLModel):
    def __init__(self):
        self.a = MiniMLParam((1,))
        self.b = MiniMLParam((1,))
        super().__init__()
    def predict(self, X):
        return self.a.value * X + self.b.value

# Create and bind the model
model = LinearModel()
model.bind()
model.randomize()

# Fit to data (e.g., y = 2x + 1)
X = jnp.linspace(0, 10, 20)
y = 2 * X + 1
model.fit(X, y)

# Save and load
model.save('model.npz')
model.load('model.npz')
```

### Nested Models

You can compose models by including other `MiniMLModel` instances as attributes. For example:

```python
class ConstantModel(MiniMLModel):

    def __init__(self):
        self._c = MiniMLParam((1,))
        super().__init__()

    def predict(self, X):
        return self._c.value

class LinearWithConstant(MiniMLModel):
    def __init__(self):
        self._b = MiniMLParam((5,))
        self._M = MiniMLParam((5, 5))
        self._c = ConstantModel()
        super().__init__()

    def predict(self, X):
        return self._M.value @ X + self._b.value[:, None] + self._c.predict(X)
```

See [the full documentation](https://stur86.github.io/miniml/).