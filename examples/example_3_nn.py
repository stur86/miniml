import marimo

__generated_with = "0.15.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import jax.numpy as jnp
    from miniml import MiniMLParam, MiniMLParamList, MiniMLModel
    from miniml.loss import LNormRegularization, CrossEntropyLogLoss
    from ucimlrepo import fetch_ucirepo 
    from miniml.nn.mlp import MLP
    return MLP, fetch_ucirepo, jnp, mo, np


@app.cell
def _(fetch_ucirepo):
    # Load dataset
    iris = fetch_ucirepo(id=53) 

    # data (as pandas dataframes) 
    iris_X = iris.data.features
    iris_y = iris.data.targets
    return iris_X, iris_y


@app.cell
def _(iris_X, iris_y, jnp, np):
    # Train/test split
    _rng = np.random.default_rng(0)
    _f = 0.8
    _n = len(iris_X)
    _train_idx = list(set(map(int, _rng.choice(np.arange(_n), replace=False, size=int(_f*_n)))))
    _test_idx = list(set(range(_n))-set(_train_idx))

    train_X = jnp.array(iris_X.to_numpy()[_train_idx].astype(np.float32))
    test_X = jnp.array(iris_X.to_numpy()[_test_idx].astype(np.float32))

    _keys_y = iris_y.to_numpy()[:,0]
    class_keys = sorted(list(set(_keys_y)))
    _all_y = np.zeros((_n, len(class_keys)), dtype=np.float32)
    for _i in range(_n):
        _all_y[_i,class_keys.index(_keys_y[_i])] = 1

    train_y = jnp.array(_all_y[_train_idx])
    test_y = jnp.array(_all_y[_test_idx])
    return test_X, test_y, train_X, train_y


@app.cell
def _(MLP, np, train_X, train_y):
    # Train a MLP with a single hidden layer
    _n_hidden = 10

    nnm = MLP([4, _n_hidden, 3])
    nnm.randomize()
    nnm.fit(train_X, train_y, reg_lambda=0.5)

    def classification_accuracy(y_true, y_pred):
        return np.mean(np.argmax(y_true, axis=1) == np.argmax(y_pred, axis=1))
    return classification_accuracy, nnm


@app.cell
def _(classification_accuracy, mo, nnm, test_X, test_y, train_X, train_y):
    mo.md(
        f"""
    | Set       |               Accuracy                                       |
    |-----------|--------------------------------------------------------------|
    | **Train** | {classification_accuracy(train_y, nnm.predict(train_X)):.2%} |
    | **Test**  |  {classification_accuracy(test_y, nnm.predict(test_X)):.2%}  |
    """
    )
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
