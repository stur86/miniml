# Containers

For convenience, MiniML provides containers to store multiple parameters and models within a bigger one:

* [MiniMLParamList](api/miniml/param.md#miniml.param.MiniMLParamList) for parameters,
* [MiniMLModelList](api/miniml/model.md#miniml.model.MiniMLModelList) for models.

You should use these whenever you want to pack multiple parameters or models inside of a bigger model. If you simply add as child of a `MiniMLModel` a bare list, it won't get properly bound to the buffer and the model won't work.

```py
class MyModel(MiniMLModel):
    def __init__(self):
        self._layers = [Linear(3,5), Activation(), Linear(5,1)] # WRONG
        self._layers = MiniMLModelList([Linear(3,5), 
                                        Activation(), 
                                        Linear(5,1)]) # Correct
```

