import marimo

__generated_with = "0.16.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import jax.numpy as jnp
    from miniml.nn.rbfnet import RBFLayer
    from miniml.nn.rbf import gaussian_rbf, PolyharmonicRBF, inverse_quadratic_rbf
    from miniml.loss import CrossEntropyLogLoss
    import matplotlib.pyplot as plt
    return CrossEntropyLogLoss, RBFLayer, gaussian_rbf, jnp, np, plt


@app.cell
def _(jnp, np):
    rng = np.random.default_rng(0)
    X0 = np.array([0.4, -0.2], dtype=np.float32)
    S = np.array([[0.9, 0.6], [-0.2, 0.8]], dtype=np.float32)
    X = jnp.array(rng.uniform(-3, 3, size=(1000, 2)), dtype=jnp.float32)
    y = jnp.exp(-jnp.sum((S@(X-X0).T)**2, axis=0)/2)[:,None]

    def make_moons(n_samples=200, noise=0.1, random_state=None):
        rng = np.random.RandomState(random_state)

        n_samples_out = n_samples // 2
        n_samples_in = n_samples - n_samples_out

        # First moon
        theta = np.linspace(0, np.pi, n_samples_out)
        x_outer = np.c_[np.cos(theta), np.sin(theta)]

        # Second moon
        theta = np.linspace(0, np.pi, n_samples_in)
        x_inner = np.c_[1 - np.cos(theta), 1 - np.sin(theta) - 0.5]

        X = np.vstack([x_outer, x_inner])
        y = np.hstack([np.zeros(n_samples_out, dtype=int),
                       np.ones(n_samples_in, dtype=int)])

        # Add noise
        X += noise * rng.randn(*X.shape)

        return X, y

    # Example usage
    X, y = make_moons(n_samples=200, noise=0.1, random_state=42)
    yclass = np.zeros((len(y), 2))
    yclass[np.arange(len(yclass)), y] = 1
    return X, y, yclass


@app.cell
def _(CrossEntropyLogLoss, RBFLayer, X, gaussian_rbf, yclass):
    rbfm = RBFLayer(n_in=2, n_centers=10, n_out=1, projection="full", rbf=gaussian_rbf, loss=CrossEntropyLogLoss(zero_ref=True), 
                   normalize=True)
    rbfm.randomize(0)

    res = rbfm.fit(X, yclass)
    return rbfm, res


@app.cell
def _(X, np, plt, rbfm, y):
    _xgrid = np.linspace(-3, 3, 50)
    _xgrid = np.meshgrid(_xgrid, _xgrid)

    _xy = np.array(_xgrid).reshape((2,-1)).T
    _z = rbfm.predict(_xy)

    _fig, _ax = plt.subplots()

    _ax.pcolormesh(_xgrid[0], _xgrid[1], _z.reshape((len(_xgrid[0]), -1)), cmap="Blues", vmax=1, vmin=0)
    _ax.scatter(X[:,0], X[:,1], c=y)
    _ax.set_xlim(-3, 3)
    _ax.set_ylim(-3, 3)

    _x0 = rbfm._X0.value

    _ax.scatter(_x0[:,0], _x0[:,1], c='red')
    return


@app.cell
def _(res):
    res
    return


if __name__ == "__main__":
    app.run()
