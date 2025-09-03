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