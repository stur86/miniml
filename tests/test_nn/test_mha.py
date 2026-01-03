from jax import Array as JxArray
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from pathlib import Path
from dataclasses import dataclass
from miniml.nn.mha import MultiHeadAttention, _MultiHeadArg
from miniml.model import PredictMode
from miniml.random import RandomMask

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
        params["_out_proj._W.v"] = self.m_weights["_out_proj._W"].T
        params["_out_proj._b.v"] = self.m_weights["_out_proj._b"]
        mha.set_params(params)

    @property
    def init_args(self) -> dict:
        return {
            "embed_dim": self.params["embed_dim"],
            "num_heads": self.params.get("num_heads", 1),
            "kdim": self.params.get("kdim", None),
            "vdim": self.params.get("vdim", None),
        }

    @property
    def X(self) -> _MultiHeadArg:
        if self.params.get("is_self_attention", False):
            return self.t_input_0
        else:
            return (self.t_input_0, self.t_input_1, self.t_input_2)

    @property
    def call_args(self) -> dict:
        return {"attn_mask": self.params.get("attn_mask", None)}


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
    W_qkv = np.tile(np.eye(embed_dim, dtype=np.float32), (1, 3))
    mha_1h.set_params(
        {
            "_QKV.v": jnp.array(W_qkv),
            "_QKV_bias.v": jnp.zeros((3 * embed_dim,), dtype=jnp.float32),
            "_out_proj._W.v": jnp.eye(embed_dim, dtype=jnp.float32),
            "_out_proj._b.v": jnp.zeros((embed_dim,), dtype=jnp.float32),
        }
    )

    X1 = jnp.zeros((4, embed_dim), dtype=jnp.float32)
    Y1 = mha_1h.predict(X1)

    assert Y1.shape == X1.shape


@pytest.mark.parametrize("name", ["mha_1", "mha_2", "mha_3", "mha_4", "mha_5"])
def test_mha_w_data(mha_data: MHAData):
    mha = MultiHeadAttention(**mha_data.init_args)
    mha.bind()
    mha_data.apply_weights(mha)
    output = mha.predict(mha_data.X, **mha_data.call_args)
    assert np.allclose(output, mha_data.t_output)


def _allclose_with_nan(a: jnp.ndarray, b: jnp.ndarray, atol: float = 1e-6) -> bool:
    a_np = np.asarray(a)
    b_np = np.asarray(b)
    # NaNs must be in the same positions
    if not np.array_equal(np.isnan(a_np), np.isnan(b_np)):
        return False
    return np.allclose(
        np.nan_to_num(a_np, nan=0.0), np.nan_to_num(b_np, nan=0.0), atol=atol
    )


@pytest.mark.parametrize("p_drop", [0.0, 0.3, 0.5])
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_mha_dropout(p_drop: float, seed: int) -> None:
    embed_dim = 4
    num_heads = 1
    seq_len = 4

    # Construct MHA with dropout and identity-like projections
    mha = MultiHeadAttention(embed_dim=embed_dim, num_heads=num_heads, dropout=p_drop)
    mha.bind()

    W_qkv = np.tile(np.eye(embed_dim, dtype=np.float32), (1, 3))
    mha.set_params(
        {
            "_QKV.v": jnp.array(W_qkv),
            "_QKV_bias.v": jnp.zeros((3 * embed_dim,), dtype=jnp.float32),
            "_out_proj._W.v": jnp.eye(embed_dim, dtype=jnp.float32),
            "_out_proj._b.v": jnp.zeros((embed_dim,), dtype=jnp.float32),
        }
    )

    # Simple deterministic sequence input
    X = jnp.arange(seq_len * embed_dim, dtype=jnp.float32).reshape(seq_len, embed_dim)

    rng_key = jax.random.PRNGKey(seed)

    # Actual output from the model in training mode (so dropout is active)
    y_model = mha._predict_kernel(
        X,
        buffer=mha._buffer,
        rng_key=rng_key,
        mode=PredictMode.TRAINING,
    )

    # Manual computation of expected output using the same RandomMask
    head_dim = embed_dim // num_heads

    # With identity projections and zero bias, Q = K = V = X
    X_heads = X.reshape(X.shape[:-1] + (num_heads, head_dim))

    attn_scores = jnp.einsum("...qhd,...khd->...qhk", X_heads, X_heads)
    attn_scores = attn_scores / jnp.sqrt(head_dim)

    # Expected attention weights/output in training mode (with dropout)
    attn_weights_train = jnp.exp(
        attn_scores - jnp.max(attn_scores, axis=-1, keepdims=True)
    )

    if p_drop > 0.0:
        keep_prob = 1.0 - p_drop
        dropout_mask = RandomMask(
            attn_weights_train.shape, p=keep_prob, dtype=attn_weights_train.dtype
        ).generate(rng_key)
        attn_weights_train = attn_weights_train * dropout_mask

    attn_weights_train = attn_weights_train / jnp.sum(
        attn_weights_train, axis=-1, keepdims=True
    )
    attn_output_heads_train = jnp.einsum(
        "...qhk,...khd->...qhd", attn_weights_train, X_heads
    )
    y_expected_train = attn_output_heads_train.reshape(
        attn_output_heads_train.shape[:-2] + (embed_dim,)
    )

    # Expected attention weights/output without dropout (used for inference mode)
    attn_weights_nodrop = jnp.exp(
        attn_scores - jnp.max(attn_scores, axis=-1, keepdims=True)
    )
    attn_weights_nodrop = attn_weights_nodrop / jnp.sum(
        attn_weights_nodrop, axis=-1, keepdims=True
    )
    attn_output_heads_nodrop = jnp.einsum(
        "...qhk,...khd->...qhd", attn_weights_nodrop, X_heads
    )
    y_expected_nodrop = attn_output_heads_nodrop.reshape(
        attn_output_heads_nodrop.shape[:-2] + (embed_dim,)
    )

    # Training-mode output must match manual dropout computation
    assert _allclose_with_nan(y_model, y_expected_train)

    # Inference-mode output must match no-dropout computation
    y_inference = mha.predict(X)
    assert _allclose_with_nan(y_inference, y_expected_nodrop)

def test_mha_dropout_invalid():
    embed_dim = 4
    num_heads = 1
    seq_len = 4

    mha = MultiHeadAttention(embed_dim=embed_dim, num_heads=num_heads, dropout=0.5)
    mha.bind()

    X = jnp.arange(seq_len * embed_dim, dtype=jnp.float32).reshape(seq_len, embed_dim)

    with pytest.raises(
        Exception,
        match="rng_key must be provided during training when dropout is enabled.",
    ):
        mha._predict_kernel(
            X,
            buffer=mha._buffer,
            rng_key=None,
            mode=PredictMode.TRAINING,
        )