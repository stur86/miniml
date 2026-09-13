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

## Stacking coupling layers

A single coupling layer leaves part of its input untouched, so `AffineCouplingStack` chains several of them, changing the transformed subset from one layer to the next. The stack is invertible as a whole, running its layers backwards, and its `.log_det_jac()` adds up the contribution of each layer at its own input:

```py
flow = AffineCouplingStack(dim=5, n_layers=4)
flow.invert(flow.predict(X)) # == X
```

Which indices each layer transforms can be given directly, one entry per layer, as index sets or slices:

```py
AffineCouplingStack(dim=4, partition=[{0, 1}, slice(1, None, 2), {2, 3}])
```

Or it can be left to a rule, taking the dimension and the index of the layer in the stack:

```py
def my_partition(dim: int, layer_idx: int) -> slice | set[int]:
    ...

AffineCouplingStack(dim=5, n_layers=4, partition=my_partition)
```

The default, `alternating_partition`, is the usual checkerboard: even indices, then odd ones, then even again. `RandomPartition` instead draws a random subset for each layer, and is a class rather than a plain function because it carries a seed and a fraction:

```py
AffineCouplingStack(dim=8, n_layers=6, partition=RandomPartition(seed=0, frac=0.25))
```

A rule is called once per layer while the stack is built, and only the indices it returns are kept. You can read them back with `.partitions`, and they are what gets saved, so loading a model never calls the rule again and does not depend on it still existing. This also means a rule that can not be pickled, such as a lambda, does not stop the stack being saved.

The stack warns if some dimension is never transformed, since it would then pass through the whole flow unchanged.

### Randomization scale

Each layer multiplies its target set by the exponential of a small network, so a dimension transformed by $k$ layers accumulates $k$ such exponents: their spread grows as $\sqrt{k}$, and a randomized stack of any depth would quickly leave the range where single precision can invert it. The stack therefore damps the randomization of each layer by $1/\sqrt{2k}$, taking $k$ from the partitions it has already resolved, so that the total stays in range whatever the depth. Both the scale and the shift network are damped, as both compound. For the checkerboard this is exactly $1/\sqrt{n_{layers}}$, every dimension being transformed by half of the layers.

Pass `rnd_scale` to choose the damping yourself, on either a single layer or a whole stack:

```py
AffineCouplingStack(dim=6, n_layers=8, rnd_scale=0.1)
```
