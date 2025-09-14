import pytest
import jax.numpy as jnp
from miniml.param import MiniMLParamList
from miniml.nn.rbfnet import RBFLayer, ProjectionType


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