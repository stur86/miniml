import marimo

__generated_with = "0.18.4"
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
    mo.vstack([mo.md("""#### The model

    Here we reproduce a model very similar to the one used in Figure 1 of the paper, as [seen in the paper's code repository](https://github.com/LucasPrietoAl/grokking-at-the-edge-of-numerical-stability). Our dataset is algorithmically generated; we use a one-hot encoded vector to represent two integers between 0 and 113 as inputs, and their sum, modulo 113, as output.

    We then train a Multi-Layer Perceptron with two hidden layers of size 200 on this, using AdamW with orthogonalized gradients and cross-entropy with stablemax as our loss.

    You can click the button below to start the training; even on CPU it should only take a few minutes at most. Each "iteration" printed corresponds to 30 steps of AdamW optimizer.
    """), mo.center(train_btn)])
    return


@app.cell(hide_code=True)
def _(model, test_X, test_y, train_X, train_btn, train_y):
    loss_iters = []
    loss_curves = []
    accuracy_curves = []
    if train_btn.value:
        model.randomize()
        _opt = AdamWOptimizer(alpha=0.01, beta_1=0.9, beta_2=0.99, ortho_grad=True, maxiter=30)
        with _opt.persistent():
            for _i in range(100):
                res = model.fit(train_X, train_y, optimizer=_opt, reg_lambda=0.0)
                _train_loss = res.objective_value
                _train_pred = model.predict(train_X)
                _train_acc = jnp.mean(jnp.argmax(_train_pred, axis=-1) == train_y)
                _test_pred = model.predict(test_X)
                _test_loss = model.total_loss(test_y, _test_pred)
                _test_acc = jnp.mean(jnp.argmax(_test_pred, axis=-1) == test_y)
                print(f"Iteration {_i+1}: Train loss = {_train_loss} / Test loss = {_test_loss}")
                loss_iters.append(_i)
                loss_curves.append([_train_loss, _test_loss])
                accuracy_curves.append([_train_acc, _test_acc])

    loss_iters = np.array(loss_iters)
    loss_curves = np.array(loss_curves)
    accuracy_curves = np.array(accuracy_curves)
    return accuracy_curves, loss_curves, loss_iters


@app.cell(hide_code=True)
def _(accuracy_curves, loss_curves, loss_iters):
    _fig, _ax = plt.subplots(ncols=2, figsize=(12,6))

    if len(loss_iters) > 0:
        _ax[0].semilogy(loss_iters, loss_curves[:,0], label="Train")
        _ax[0].semilogy(loss_iters, loss_curves[:,1], label="Test")
        _ax[0].set_xlabel("Iteration")
        _ax[0].set_ylabel("Loss")
        _ax[0].legend()

        _ax[1].plot(loss_iters, accuracy_curves[:,0]*100.0, label="Train")
        _ax[1].plot(loss_iters, accuracy_curves[:,1]*100.0, label="Test")
        _ax[1].set_xlabel("Iteration")
        _ax[1].set_ylabel("Accuracy (%)")
        _ax[1].legend()

    mo.vstack([
        mo.md("""
    Here we see the output. On the left, we have the plot of the loss. We should see that the train loss goes down very sharply, while the test loss stays up and only at a later stage has a lesser drop - this is where the "grokking" occurs. The train loss may have a sharp peak around this time too.

    On the right, we have the accuracy plot, which should show a much more dramatic result; while training accuracy goes up to 100% almost immediately, testing accuracy lags behind, to then increase to 100% too once the grokking occurs.
    """),
        _fig
    ])
    return


if __name__ == "__main__":
    app.run()
