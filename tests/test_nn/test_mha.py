from jax import Array as JxArray
import jax.numpy as jnp
import numpy as np
import pytest
from pathlib import Path
from dataclasses import dataclass
from miniml.nn.mha import MultiHeadAttention

DATA_PATH = Path(__file__).parent / "data"

@dataclass
class MHAData:
    params: dict
    m_weights: dict
    t_input_0: JxArray
    t_input_1: JxArray
    t_input_2: JxArray
    t_output: JxArray

    def apply_weights(self, mha: MultiHeadAttention):
        # Does this have separate Q, K, V?
        is_homogeneous = self.m_weights.get("_QKV") is not None
        if is_homogeneous != mha.homogeneous:
            raise ValueError("Weights do not match model homogeneity.")
        params = {}
        if is_homogeneous:
            params["_QKV.v"] = self.m_weights["_QKV"].T
        else:
            params["_Q.v"] = self.m_weights["_Q"].T
            params["_K.v"] = self.m_weights["_K"].T
            params["_V.v"] = self.m_weights["_V"].T
        params["_QKV_bias.v"] = self.m_weights["_QKV_bias"]
        params["_out_proj._W.v"] = self.m_weights["_out_proj._W"]
        params["_out_proj._b.v"] = self.m_weights["_out_proj._b"]
        mha.set_params(params)

@pytest.fixture
def mha_data(name: str) -> MHAData:
    data_file = DATA_PATH / f"{name}.npz"
    data = np.load(data_file, allow_pickle=True)
    return MHAData(
        params=data["params"].item(),
        m_weights=data["m_weights"].item(),
        t_input_0=data["t_input_0"],
        t_input_1=data["t_input_1"],
        t_input_2=data["t_input_2"],
        t_output=data["t_output"],
    )

def test_mha_basic():
    embed_dim = 6
    mha_1h = MultiHeadAttention(embed_dim=embed_dim, num_heads=1)
    mha_1h.bind()
    
    # Set all weights to identity-like
    W_qkv = np.tile(np.eye(embed_dim, dtype=np.float32), (1,3))
    mha_1h.set_params({
        "_QKV.v": jnp.array(W_qkv),
        "_QKV_bias.v": jnp.zeros((3*embed_dim,), dtype=jnp.float32),
        "_out_proj._W.v": jnp.eye(embed_dim, dtype=jnp.float32),
        "_out_proj._b.v": jnp.zeros((embed_dim,), dtype=jnp.float32),
    })
    
    X1 = jnp.zeros((4,embed_dim), dtype=jnp.float32)
    Y1 = mha_1h.predict(X1)
    
    assert Y1.shape == X1.shape

@pytest.mark.parametrize(
    "name",
    [
        "mha_1"
    ]
)
def test_mha_w_data(mha_data: MHAData):
    mha = MultiHeadAttention(**mha_data.params)
    mha.bind()
    mha_data.apply_weights(mha)
    output = mha.predict((mha_data.t_input_0, mha_data.t_input_1, mha_data.t_input_2))
    
    assert np.allclose(output, mha_data.t_output)
