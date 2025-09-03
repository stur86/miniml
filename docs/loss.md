# Loss functions

When training any models it's important to set up the correct loss function, which will determine how the fit to the target data is scored. The default loss function for `MiniMLModel` is the squared error loss:

$$
\mathcal{L}(y, \hat{y}) = \sum_i(y_i-\hat{y}_i)^2
$$

However, it's very easy to customize this option for your models, using other provided loss functions or even creating your own.

## Base loss

### Setting the loss function

The loss function is an argument taken by the base constructor of `MiniMLModel`. Simply pass it when invoking it:

```py
from miniml import MiniMLModel
from miniml.loss import cross_entropy_loss

class MyClassifier(MiniMLModel):
    def __init__(self):
        ... # Your initialization code
        super().__init__(loss=cross_entropy_loss)
```

### Computing the loss value for a prediction

Just invoke the `.loss` method to see the model's loss function computed on a pair `(true, prediction)` (note the order):

```py
print(f"Final loss = {model.loss(y, y_hat)}")
```

## Defining your own loss functions

The `LossFunction` interface implemented in `miniml.loss` is just a type hint:

```py
Callable[[JXArray, JXArray], JXArray]
```

meaning a loss function can be any callable function or object that accepts two Jax arrays and returns one (which is expected to be a scalar, namely, a zero-dimensional Jax array).

If your loss function requires parameters, you can make it a class (there's also `miniml.loss.LossFunctionBase` as an inheritable interface for this), or use `functools.partial` to embed some arguments.

## Regularization loss

In addition to a target loss, it's customary, especially in larger models, to add a regularization loss that enforces a certain scale, or smoothness constraints, for the parameters of the model. This in MiniML can be set on each individual parameter:

```py
class LinearModel(MiniMLModel):
    def __init__(self, n_in: int, n_out: int):
        self._W = MiniMLParam((n_in, n_out), reg_loss=LNormRegularization(2))
```

The rules, even for custom loss functions, are as above - any callable with the right interface can be passed. The main difference is that the interface only accepts a single array:

```py
RegLossFunction = Callable[[JXArray], JXArray]
```

There is also a `RegLossFunctionBase` abstract class to derive for parametrized losses.

### Regularization scale

The model's total loss can be retrieved with the method `.total_loss`. This method also accepts a global regularization strength parameter `reg_lambda`: 

```TotalLoss = BaseLoss + reg_lambda * RegularizationLoss```.

The `.fit` model accepts the same parameter, and it will be used for regularization in that fitting process. If you want different regularization strengths on a parameter-by-parameter basis you should take care of it by creating appropriate individual loss functions.

For more information see [the `loss` module API reference](api/miniml/loss.md).