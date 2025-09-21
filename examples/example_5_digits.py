import marimo

__generated_with = "0.16.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import jax.numpy as jnp
    from ucimlrepo import fetch_ucirepo
    from miniml.nn import MLP, Identity, Parallel, Linear, Activation, relu, Stack
    from miniml import MiniMLModel
    from miniml.loss import CrossEntropyLogLoss
    return (
        Activation,
        CrossEntropyLogLoss,
        Identity,
        Linear,
        Parallel,
        Stack,
        fetch_ucirepo,
        jnp,
        np,
        relu,
    )


@app.cell
def _(fetch_ucirepo):
    optical_recognition_of_handwritten_digits = fetch_ucirepo(id=80)
    return (optical_recognition_of_handwritten_digits,)


@app.cell
def _(jnp, optical_recognition_of_handwritten_digits):
    # data (as pandas dataframes) 
    _X = optical_recognition_of_handwritten_digits.data.features 
    _y = optical_recognition_of_handwritten_digits.data.targets

    X = jnp.array(_X.to_numpy()*1.0/255, dtype=jnp.float32)
    y = jnp.zeros((len(_y), 10), dtype=jnp.float32)
    y = y.at[range(len(y)), _y.to_numpy()[:,0]].set(1.0)
    return X, y


@app.cell
def _(X, jnp, np, y):
    # Train/test split
    N = len(X)
    _rng = np.random.default_rng(seed=111)
    train_idx = _rng.choice(range(N), size=int(N*0.8), replace=False)
    test_idx = jnp.array(list(set(range(N))-set(train_idx)))

    X_train = X[train_idx]
    X_test = X[test_idx]

    y_train = y[train_idx]
    y_test = y[test_idx]
    return X_test, X_train, y_test, y_train


@app.cell
def _(
    Activation,
    CrossEntropyLogLoss,
    Identity,
    Linear,
    Parallel,
    Stack,
    relu,
):
    # Example of a stacked model: a simple MLP with "leakage" in the intermediate layer

    n_hidden = 20
    mlpc = Stack(models=[
        Linear(64, n_hidden),
        Activation(relu),
        Parallel([Linear(n_hidden, n_hidden), Identity()], mode="sum"),
        Activation(relu),
        Linear(n_hidden, 10)
    ], loss=CrossEntropyLogLoss(zero_ref=False))
    mlpc.randomize()

    print(f"Parameter count: {mlpc.size}")
    return (mlpc,)


@app.cell
def _(X_test, X_train, jnp, mlpc, y_test, y_train):
    res = mlpc.fit(X_train, y_train, reg_lambda=1.0)

    print(res)

    y_pred = mlpc.predict(X_test)

    print(f"Accuracy: {jnp.mean(jnp.argmax(y_test, axis=1) == jnp.argmax(y_pred, axis=1)):.2%}")
    return


if __name__ == "__main__":
    app.run()
