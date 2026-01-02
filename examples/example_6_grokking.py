import marimo

__generated_with = "0.17.8"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    import numpy as np
    import jax
    import jax.numpy as jnp
    from jax.nn import one_hot as jax_one_hot
    import seaborn as sns
    import matplotlib.pyplot as plt

    from miniml.nn import MLP
    from miniml.nn.activations import relu
    from miniml.loss import CrossEntropyStableMaxLogLoss
    from miniml.optim import AdamWOptimizer


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Grokking with OrthoGrad and StableMax

    "Grokking" is the concept of a ML model going beyond overfitting to converge to a highly generalised algorithmic solution to a problem; the term was originally introduced in [this paper from 2022](https://arxiv.org/abs/2201.02177). This original work used transformers and required over a million iterations to converge on a solution for a simple problem of modular arithmetic.

    However, in 2025, [L. Prieto et al. suggested some clever improvements](https://arxiv.org/abs/2501.04697) to this approach that achieved "grokking" much faster and much more reliably. These were StableMax (a replacement for SoftMax that vanishes much less decisively and thus is better for gradients) and OrthoGrad (a modification to an AdamW optimizer using only the component of the gradient orthogonal to the weight vector).

    This notebook shows how to use this functionality with MiniML.
    """)
    return


@app.class_definition(hide_code=True)
class ModularDataset:
    def __init__(self, modulo: int = 113, one_hot: bool = True, operation: str = "add", shuffle_seed: int = 0):
        # Generate all combinations
        self.n = modulo
        self.op = operation
        grid = jnp.array(jnp.meshgrid(jnp.arange(modulo), jnp.arange(modulo))).reshape((2,-1)).T
        # Shuffle
        prng = jax.random.PRNGKey(shuffle_seed)
        perm = jax.random.permutation(prng, jnp.arange(len(grid)))
        grid = grid[perm]

        self.X = grid
        self.y = self._operate(grid[:,0], grid[:,1])

        if one_hot:
            # Turn these into one-hot encodings
            self.X = jax_one_hot(self.X, num_classes=self.n).reshape((self.X.shape[0], -1))

    def _operate(self, x1, x2):
        if self.op == "add":
            ans = x1+x2
        elif self.op == "mul":
            ans = x1*x2
        elif self.op == "sub":
            ans = x1-x2
        else:
            raise ValueError(f"Invalid operation {self.op}")
        return ans%self.n

    def splits(self, train_frac: float = 0.4) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
        train_n = int(len(self.X)*train_frac)
        test_n = len(self.X)-train_n
        return self.X[:train_n], self.X[train_n:], self.y[:train_n], self.y[train_n:]


@app.cell(hide_code=True)
def _():
    dset = ModularDataset()

    def make_model(n: int, n_inner: int = 200, hidden_count: int = 2) -> MLP:
        model = MLP(layer_sizes=[2*n] + [n_inner]*hidden_count + [n], activation=relu, loss=CrossEntropyStableMaxLogLoss(expect_labels=True, zero_ref=False))
        return model

    model = make_model(dset.n)
    model.randomize()

    train_X, test_X, train_y, test_y = dset.splits(0.4)
    return model, test_X, test_y, train_X, train_y


@app.cell(hide_code=True)
def _():
    train_btn = mo.ui.run_button("neutral", label="Train")
    return (train_btn,)


@app.cell(hide_code=True)
def _(train_btn):
    mo.center(train_btn)
    return


@app.cell(hide_code=True)
def _(model, test_X, test_y, train_X, train_btn, train_y):
    loss_iters = []
    loss_curves = []
    accuracy_curves = []
    if train_btn.value:
        model.randomize()
        _opt = AdamWOptimizer(alpha=0.001, beta_1=0.9, beta_2=0.99, ortho_grad=True, maxiter=100)
        for _i in range(100):
            res = model.fit(train_X, train_y, optimizer=_opt, reg_lambda=0.0)
            _train_loss = res.objective_value
            _train_pred = model.predict(train_X)
            _train_acc = jnp.mean(jnp.argmax(_train_pred, axis=-1) == train_y)
            _test_pred = model.predict(test_X)
            _test_loss = model.total_loss(test_y, _test_pred)
            _test_acc = jnp.mean(jnp.argmax(_test_pred, axis=-1) == test_y)
            print(f"Iteration {_i}: Train loss = {_train_loss} / Test loss = {_test_loss}")
            loss_iters.append(_i)
            loss_curves.append([_train_loss, _test_loss])
            accuracy_curves.append([_train_acc, _test_acc])

    loss_iters = np.array(loss_iters)
    loss_curves = np.array(loss_curves)
    accuracy_curves = np.array(accuracy_curves)
    return accuracy_curves, loss_curves, loss_iters


@app.cell
def _(accuracy_curves, loss_curves, loss_iters):
    _fig, _ax = plt.subplots(ncols=2, figsize=(12,6))

    _ax[0].semilogy(loss_iters, loss_curves)
    _ax[0].set_xlabel("Iteration")
    _ax[0].set_ylabel("Loss")

    _ax[1].plot(loss_iters, accuracy_curves*100.0)
    _ax[1].set_xlabel("Iteration")
    _ax[1].set_ylabel("Accuracy (%)")
    return


if __name__ == "__main__":
    app.run()
