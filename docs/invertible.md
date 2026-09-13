# Invertible models

Some models can undo their own transformation. These derive from the [InvertibleModel](api/miniml/model.md#miniml.model.InvertibleModel) mixin, alongside `MiniMLModel`, and implement `_invert_kernel()` with the same signature as `_predict_kernel()`:

```py
class MyLayer(InvertibleModel, MiniMLModel):
    def _predict_kernel(self, X, buffer, rng_key=None, mode=None): ...
    def _invert_kernel(self, Y, buffer, rng_key=None, mode=None): ...
```

The public counterpart of `predict()` is then `invert()`, which recovers the input from the output:

```py
model.invert(model.predict(X)) # == X
```

[AffineCouplingLayer](api/miniml/nn/affine_coupling.md#miniml.nn.affine_coupling.AffineCouplingLayer) is an example: it transforms part of its input with a scale and a shift computed from the part it leaves untouched, which makes it invertible whatever its parameters are.

## Using the inverse as a model

The method `.inverse_model()` returns the inverse transformation as a model in its own right, an [InverseModel](api/miniml/model.md#miniml.model.InverseModel) with `predict` and `invert` swapped:

```py
layer = AffineCouplingLayer(4)
inverse = layer.inverse_model()

inverse.predict(layer.predict(X)) # == X
```

The view borrows the wrapped model's parameters instead of copying them (see [sharing a model between parents](composite_models.md#sharing-a-model-between-parents)), so it costs nothing to build and it follows those parameters as they are fitted. Calling `inverse_model()` on it gives the original model back, instead of a view on a view.

Because it declares no parameters, you can put it in a container together with the model it inverts:

```py
Stack([layer, layer.inverse_model()]) # the identity, for any parameter values
```

!!! note
    The wrapped model owns the parameters, so bind, unbind and save it, not the view. Binding a view binds its owner.
