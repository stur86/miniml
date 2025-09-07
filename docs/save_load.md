# Saving and loading

A fitted model can be saved to disk as simply as:

```py
model.save("model_file_name.npz")
```

Loading is a static alternative to a constructor. Make sure to use the same class type
as the saved model, but the construction parameters are saved in the file together with the weights:

```py
model = MyModel.load("model_file_name.npz")
```

## Storing parameters as a dictionary

If you want to manipulate the parameters more directly, or save them in your own more readable format, you can instead get them as a dictionary, with keys corresponding to the "paths" of the parameters in the object hierarchy. You can just use:

```py
param_dict: dict[str, jax.Array] = model.get_params()
```

These parameters are *copies* of the ones inside the model, so modifying them directly has no effect on the model itself. You can however update all or some of them by passing back a dictionary with the modified arrays to the model like this:

```py
model.set_params({'param.path.v': x})
```

Take care that the size and dtypes match those you got from `.get_params()`, and that the paths are the same. The paths will be a series of names of nested objects (or a number for indexing in [containers](containers.md)) and always end with `.v`.

For more information see [the API reference for the model class](api/miniml/model.md).
