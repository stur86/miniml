# MiniML

MiniML (pronounced "minimal") is a tiny machine-learning framework based on JAX for its gradients that allows building models with a PyTorch inspired approach, but with a Scikit-like interface (using `.predict` and `.fit`), and powered by SciPy's optimization algorithms. Training a linear model in MiniML looks as simple as this:

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