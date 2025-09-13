import marimo

__generated_with = "0.15.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import jax
    import jax.numpy as jnp
    import seaborn as sns
    import matplotlib.pyplot as plt
    from miniml import MiniMLModel, MiniMLParam
    from miniml.loss import CrossEntropyLogLoss, LNormRegularization
    return (
        CrossEntropyLogLoss,
        LNormRegularization,
        MiniMLModel,
        MiniMLParam,
        jnp,
        mo,
        np,
        plt,
        sns,
    )


@app.cell
def _(jnp, np, plt, sns):
    seed = 42
    rng = np.random.default_rng(seed)
    N = 500
    X = jnp.array(rng.normal(size=(N,2)), dtype=jnp.float32)
    # Use a circular boundary for classification
    class_R = 1.2
    _is_out = (jnp.linalg.norm(X, axis=1)+rng.normal(size=N, scale=0.3*class_R)) > class_R
    y = jnp.stack([~_is_out, _is_out], dtype=jnp.float32).T

    _fig, _ax = plt.subplots()
    _ax.set_aspect(1.0)
    sns.scatterplot(x=X[:,0], y=X[:,1], c=y[:,0], ax=_ax)
    _th = np.linspace(0, 2*np.pi, 100)
    _ax.plot(class_R*np.cos(_th), class_R*np.sin(_th), c='r')
    return N, X, class_R, y


@app.cell
def _(CrossEntropyLogLoss, LNormRegularization, MiniMLModel, MiniMLParam, jnp):
    class LogisticRegressor(MiniMLModel):
        def __init__(self):
            self.M = MiniMLParam((6,1), reg_loss=LNormRegularization(p=2, root=False))
            super().__init__(CrossEntropyLogLoss(zero_ref=True))

        def predict_kernel(self, X, buf):
            # Augment X with polynomial features
            X_aug = jnp.array([jnp.ones(len(X)), X[:,0], X[:,1], X[:,0]**2, X[:,0]*X[:,1], X[:,1]**2])
            return (X_aug.T@self.M(buf))
    return (LogisticRegressor,)


@app.cell
def _(LogisticRegressor, X, y):
    logreg = LogisticRegressor()
    logreg.randomize(seed=42)

    logreg.fit(X, y, reg_lambda=1.0)
    y_hat_logits = logreg.predict(X)
    y_hat = (y_hat_logits[:,0] > 0.0)
    return logreg, y_hat, y_hat_logits


@app.cell
def _(X, class_R, jnp, logreg, np, plt, sns, y_hat):
    _fig, _ax = plt.subplots(1, 2, figsize=(12,6))
    _ax[0].set_aspect(1.0)
    _ax[0].set_xlim(-3, 3)
    _ax[0].set_ylim(-3, 3)

    sns.scatterplot(x=X[:,0], y=X[:,1], c=y_hat, ax=_ax[0])
    _th = np.linspace(0, 2*np.pi, 100)
    _ax[0].plot(class_R*np.cos(_th), class_R*np.sin(_th), c='r')

    _ax[1].set_aspect(1.0)
    _ax[1].set_xlim(-3, 3)
    _ax[1].set_ylim(-3, 3)

    # Plot the decision surface
    _xgrid = np.linspace(-3, 3, 50)
    _xgrid = np.meshgrid(_xgrid, _xgrid)
    _zval = jnp.exp(logreg.predict(jnp.array(_xgrid).reshape((2,-1)).T))
    _zval /= (1.0+_zval)
    _ax[1].pcolormesh(_xgrid[0], _xgrid[1], _zval.reshape((len(_xgrid[0]), -1)), cmap="Blues", vmax=1, vmin=0)
    _ax[1].plot(class_R*np.cos(_th), class_R*np.sin(_th), c='k', lw=4, ls="--")

    _fig
    return


@app.cell
def _(N, jnp, logreg, mo, y, y_hat, y_hat_logits):
    mo.md(
        f"""
    Accuracy: {jnp.mean(y_hat==y[:,0]):.2%}
    Average loss: {logreg.total_loss(y, y_hat_logits)/N:.2e}
    """
    )
    return


if __name__ == "__main__":
    app.run()
