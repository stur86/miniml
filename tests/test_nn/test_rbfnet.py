import numpy as np
import pytest
import jax.numpy as jnp
from miniml.param import MiniMLParamList
from miniml.nn.rbfnet import RBFLayer, ProjectionType
from miniml.nn.rbf import inverse_quadratic_rbf


@pytest.mark.parametrize("proj_type", ["scaling", "none", "ortho", "full"])
def test_rbf_layer_basic(proj_type: ProjectionType) -> None:
    n_in, n_out, n_centers = 2, 3, 4
    layer = RBFLayer(n_in=n_in, n_out=n_out, n_centers=n_centers, projection=proj_type)
    layer.randomize(seed=0)
    params = layer.get_params()

    if proj_type == "scaling":
        assert layer.param_names == ["_W.v", "_X0.v", "_s.v"]
        assert params["_s.v"].shape == (n_centers, n_in)
    elif proj_type == "none":
        assert layer.param_names == ["_W.v", "_X0.v"]
    elif proj_type == "ortho":
        assert layer.param_names == [
            "_W.v", 
            "_X0.v",
            "_pmats_list.0.v",
            "_pmats_list.1.v",
            "_pmats_list.2.v",
            "_pmats_list.3.v",
            "_s.v",
        ]
        assert isinstance(layer._pmats_list, MiniMLParamList)
        assert len(layer._pmats_list) == n_centers
        for i in range(n_centers):
            assert layer._pmats_list[i].shape == (n_in, n_in)
            assert params[f"_pmats_list.{i}.v"].shape == (n_in*(n_in - 1)//2,)
        assert params["_s.v"].shape == (n_centers, n_in)
    elif proj_type == "full":
        assert layer.param_names == [
            "_W.v", 
            "_X0.v",
            "_pmats.v",
        ]
        assert params["_pmats.v"].shape == (n_centers, n_in, n_in)
    assert params["_W.v"].shape == (n_centers, n_out)
    assert params["_X0.v"].shape == (n_centers, n_in)
    layer.randomize(seed=0)
        
    Z = jnp.arange(24).reshape((-1,4,2))
    assert layer._xproj(Z, layer._buffer).shape == (Z.shape[0], n_centers, n_in)

    X = jnp.arange(10).reshape((-1, n_in))
    assert layer.predict(X).shape == (X.shape[0], n_out)

@pytest.mark.parametrize("seed", range(10)) 
def test_rbf_layer_fit(seed: int) -> None:
    n_in, n_centers, n_out = 3, 1, 1
    # Generate gaussian-distributed data
    rng = np.random.default_rng(seed)
    X0 = rng.normal(size=(n_in))*0.1
    S = rng.normal(size=(n_in, n_in))
    S = S.T@S
    
    X = jnp.array(rng.uniform(-5, 5, size=(5000, n_in)))
    y = 1.0/(1.0 + jnp.einsum('ij,jk,ik->i', X-X0, S, X-X0))[:,None]
    
    # Inverse quadratic RBF has longer tails and makes for more stable
    # fitting if we're trying to recover exactly the generating function    
    rbf = RBFLayer(n_in=n_in, n_out=n_out, n_centers=n_centers, 
                   rbf=inverse_quadratic_rbf,
                   projection="full")
    rbf.randomize(seed=seed+10)
    res = rbf.fit(X, y, reg_lambda=0.0)
    y_pred = rbf.predict(X)
    fitted_params = rbf.get_params()

    assert res.success
    assert jnp.allclose(y, y_pred, atol=1e-2)
    
    assert jnp.allclose(fitted_params["_X0.v"], X0, atol=1e-3)
    assert jnp.allclose(fitted_params["_pmats.v"][0].T@fitted_params["_pmats.v"][0], S, atol=1e-2)
    assert jnp.allclose(fitted_params["_W.v"], jnp.ones((n_centers, n_out)), atol=1e-2)
    
    