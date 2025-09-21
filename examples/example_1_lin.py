import marimo

__generated_with = "0.16.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import jax.numpy as jnp
    from miniml import MiniMLModel, MiniMLParam
    import seaborn as sns
    return MiniMLModel, MiniMLParam, jnp, mo, np, sns


@app.cell
def _(jnp, np):
    # Create data
    seed = 35
    nl = 0.1
    rng = np.random.default_rng(seed)
    X = jnp.array(rng.uniform(size=(1000,2)))
    A = jnp.array(rng.normal(size=(2,1)))
    b = jnp.array(rng.normal(size=(1,)))
    y = X@A+b+jnp.array(rng.normal(size=(len(X),1), scale=nl))
    return A, X, b, y


@app.cell
def _(X, sns, y):
    sns.scatterplot(x=X[:,0], y=X[:,1], c=y)
    return


@app.cell
def _(A, MiniMLModel, MiniMLParam, X, b, mo, y):
    # Try fitting a model
    class LinearModel(MiniMLModel):
        A: MiniMLParam
        b: MiniMLParam

        def __init__(self, n_in: int, n_out: int):
            self.A = MiniMLParam((n_in,n_out))
            self.b = MiniMLParam((n_out,))
            super().__init__()

        def _predict_kernel(self, X, buffer):
            return X@self.A(buffer)+self.b(buffer)

    lin_model = LinearModel(X.shape[1], y.shape[1])
    lin_model.randomize()
    lin_model.fit(X, y)
    y_hat = lin_model.predict(X)

    mo.md(f"""

    Real equation:

    $$
    {A[:,0]}x+{b[0]}
    $$

    Fitted equation:

    $$
    {lin_model.A.value[:,0]}x+{lin_model.b.value[0]}
    $$

    Final average loss per sample: {lin_model.total_loss(y, y_hat)/len(y):.2e}

    """)
    return


if __name__ == "__main__":
    app.run()
