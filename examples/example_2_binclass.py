import marimo

__generated_with = "0.15.0"
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
    from miniml.loss import cross_entropy_log_loss, l2_regularization
    return (
        MiniMLModel,
        MiniMLParam,
        cross_entropy_log_loss,
        jnp,
        l2_regularization,
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
    y = (jnp.linalg.norm(X, axis=1) > class_R).astype(jnp.float32)

    _fig, _ax = plt.subplots()
    _ax.set_aspect(1.0)
    sns.scatterplot(x=X[:,0], y=X[:,1], c=y, ax=_ax)
    _th = np.linspace(0, 2*np.pi, 100)
    _ax.plot(class_R*np.cos(_th), class_R*np.sin(_th), c='r')
    return N, X, class_R, y


@app.cell
def _(
    MiniMLModel,
    MiniMLParam,
    cross_entropy_log_loss,
    jnp,
    l2_regularization,
):
    class LogisticRegressor(MiniMLModel):
        def __init__(self):
            self.M = MiniMLParam((1,6), reg_loss=l2_regularization)
            super().__init__(cross_entropy_log_loss)

        def predict(self, X):
            # Augment X with polynomial features
            X_aug = jnp.array([jnp.ones(len(X)), X[:,0], X[:,1], X[:,0]**2, X[:,0]*X[:,1], X[:,1]**2])
            return (self.M.value@X_aug)[0]
    return (LogisticRegressor,)


@app.cell
def _(LogisticRegressor, X, y):
    logreg = LogisticRegressor()
    logreg.randomize(seed=42)

    logreg.fit(X, y, reg_lambda=1.0)
    y_hat_logits = logreg.predict(X)
    y_hat = (y_hat_logits > 0.0)
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
    Accuracy: {jnp.mean(y_hat==y):.2%}  
    Average loss: {logreg.total_loss(y, y_hat_logits)/N:.2e}
    """
    )
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
