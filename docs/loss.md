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

The `.fit` model accepts the same parameter, and it will be used for regularization in that fitting process.

It is also possible to pass an additional scaling parameter for the regularization loss to each individual parameter:

```py
M = MiniMLParam((3,2), reg_loss=LNormRegularization(2), reg_scale=0.1)
```

means the loss on M will be multiplied by an additional factor of `0.1`. The overall `reg_lambda` will still apply as well.

For more information see [the `loss` module API reference](api/miniml/loss.md).

## Activity regularization

Sometimes the regularization you need depends on intermediate values computed during the forward pass — layer activations, attention weights, or other internal state — rather than on the raw parameters. For these cases, `_predict_kernel` can return a `PredictKernelOutput` object instead of a plain array:

```python
from miniml import MiniMLModel, MiniMLParam, PredictKernelOutput, PredictMode
from miniml.loss import squared_error_loss, LNormRegularization

class SparseActivationModel(MiniMLModel):
    def __init__(self, n_in: int, n_out: int):
        self._W = MiniMLParam((n_in, n_out))
        self._b = MiniMLParam((n_out,))
        self._l1 = LNormRegularization(p=1)
        super().__init__(loss=squared_error_loss)

    def _predict_kernel(self, X, buffer, rng_key=None, mode=PredictMode.INFERENCE, **kw):
        activations = X @ self._W(buffer) + self._b(buffer)
        if mode == PredictMode.TRAINING:
            return PredictKernelOutput(
                y_pred=activations,
                activity_loss=self._l1(activations),
            )
        return activations
```

The `activity_loss` field is added to the total training objective scaled by a separate `active_reg_lambda` parameter in `.fit()`:

$$
\mathcal{L}_{\text{total}} = \mathcal{L}(y, \hat{y}) + \lambda\sum_i\mathcal{R}_i(w_i) + \lambda_{\text{act}}\,\mathcal{L}_{\text{activity}}
$$

```python
res = model.fit(X, y, reg_lambda=0.01, active_reg_lambda=0.1)
```

During inference (`.predict()`), `activity_loss` is discarded; the model behaves identically to one returning a plain array.

!!! note
    It is strongly recommended to compute the `activity_loss` only when `mode == PredictMode.TRAINING`, as shown above. During inference the value is ignored, so computing it is a pure waste of time.

When using `Stack` or `Parallel`, activity losses from all children are automatically summed and propagated upward, so composite models work without any extra effort. For manually written composite models that call children's `_predict_kernel` directly, use `MiniMLModel._unpack_kernel_output(result)` to safely extract `(y_pred, activity_loss)` from whatever the child returns.