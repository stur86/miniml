import numpy as np
import pytest
from dataclasses import dataclass
from pathlib import Path
from jax import Array as JxArray, numpy as jnp
from miniml.nn.conv import Conv
from miniml.param import MiniMLError

DATA_PATH = Path(__file__).parent / "data"


@dataclass
class ConvData:
    params: dict
    m_weights: dict
    t_input_0: JxArray
    t_output: JxArray

    def apply_weights(self, conv: Conv) -> None:
        params = {"_W.v": jnp.array(self.m_weights["_W"])}
        if "_b" in self.m_weights:
            params["_b.v"] = jnp.array(self.m_weights["_b"])
        conv.set_params(params)

    @property
    def init_args(self) -> dict:
        return {
            "n_dim": self.params["n_dim"],
            "in_channels": self.params["in_channels"],
            "out_channels": self.params["out_channels"],
            "kernel_size": self.params["kernel_size"],
            "stride": self.params.get("stride", 1),
            "padding": self.params.get("padding", "valid"),
            "dilation": self.params.get("dilation", 1),
            "groups": self.params.get("groups", 1),
            "bias": self.params.get("bias", True),
        }

    @property
    def X(self) -> JxArray:
        return jnp.array(self.t_input_0)


@pytest.fixture
def conv_data(name: str) -> ConvData:
    data_file = DATA_PATH / f"{name}.npz"
    data = np.load(data_file, allow_pickle=True)

    return ConvData(
        params=data["params"].item(),
        m_weights=data["m_weights"].item(),
        t_input_0=data["t_input_0"],
        t_output=data["t_output"],
    )


def _reference_conv1d(
    X: np.ndarray, W: np.ndarray, b: np.ndarray, stride: int = 1
) -> np.ndarray:
    """Naive reference 1D convolution (cross-correlation) in channels-first layout."""
    n_batch, _, length = X.shape
    n_out, n_in, k = W.shape
    out_len = (length - k) // stride + 1
    y = np.zeros((n_batch, n_out, out_len))
    for n in range(n_batch):
        for o in range(n_out):
            for i in range(out_len):
                acc = 0.0
                for c in range(n_in):
                    for j in range(k):
                        acc += W[o, c, j] * X[n, c, i * stride + j]
                y[n, o, i] = acc + b[o]
    return y


def test_shapes_and_params() -> None:
    """Parameter shapes and count for a 2D convolution."""
    conv = Conv(2, 3, 8, kernel_size=(3, 5))
    conv.bind()

    assert conv.n_dim == 2
    assert conv.in_channels == 3
    assert conv.out_channels == 8
    assert conv.kernel_size == (3, 5)
    assert conv.stride == (1, 1)
    assert conv.dilation == (1, 1)
    assert conv.padding == "VALID"
    assert conv.has_bias

    params = conv.get_params()
    assert params["_W.v"].shape == (8, 3, 3, 5)
    assert params["_b.v"].shape == (8,)
    assert conv.size == 8 * 3 * 3 * 5 + 8


def test_no_bias() -> None:
    """A layer built with bias=False has only the weight parameter."""
    conv = Conv(1, 2, 4, kernel_size=3, bias=False)
    conv.bind()

    assert not conv.has_bias
    assert set(conv.get_params().keys()) == {"_W.v"}
    assert conv.size == 4 * 2 * 3


@pytest.mark.parametrize("stride", [1, 2])
def test_conv1d_against_reference(stride: int) -> None:
    """The 1D convolution matches a naive explicit implementation."""
    rng = np.random.default_rng(42)
    X = rng.normal(size=(2, 3, 9)).astype(np.float32)
    W = rng.normal(size=(4, 3, 3)).astype(np.float32)
    b = rng.normal(size=(4,)).astype(np.float32)

    conv = Conv(1, 3, 4, kernel_size=3, stride=stride)
    conv.bind()
    conv.set_params({"_W.v": jnp.array(W), "_b.v": jnp.array(b)})

    y = conv.predict(jnp.array(X))
    y_ref = _reference_conv1d(X, W, b, stride=stride)

    assert y.shape == y_ref.shape
    assert np.allclose(y, y_ref, atol=1e-5)


@pytest.mark.parametrize(
    "name", ["conv_1", "conv_2", "conv_3", "conv_4", "conv_5", "conv_6"]
)
def test_conv_w_data(conv_data: ConvData) -> None:
    """The convolution matches PyTorch output on stored test data."""
    conv = Conv(**conv_data.init_args)
    conv.bind()
    conv_data.apply_weights(conv)
    output = conv.predict(conv_data.X)
    assert output.shape == conv_data.t_output.shape
    assert np.allclose(output, conv_data.t_output, atol=1e-5)


def test_ndim_output_shapes() -> None:
    """Output spatial shapes are correct for 1, 2 and 3 spatial dimensions."""
    for n_dim in (1, 2, 3):
        spatial = (7,) * n_dim
        conv = Conv(n_dim, 2, 3, kernel_size=3)
        conv.randomize(seed=0)
        X = jnp.zeros((4, 2) + spatial)
        y = conv.predict(X)
        assert y.shape == (4, 3) + (5,) * n_dim


def test_padding_modes() -> None:
    """'same' preserves the spatial size, explicit padding grows it."""
    X = jnp.zeros((2, 1, 6, 6))

    same_conv = Conv(2, 1, 1, kernel_size=3, padding="same")
    same_conv.randomize(seed=1)
    assert same_conv.predict(X).shape == (2, 1, 6, 6)

    pad_conv = Conv(2, 1, 1, kernel_size=3, padding=2)
    pad_conv.randomize(seed=1)
    assert pad_conv.padding == ((2, 2), (2, 2))
    assert pad_conv.predict(X).shape == (2, 1, 8, 8)

    asym_conv = Conv(2, 1, 1, kernel_size=3, padding=[(1, 0), 2])
    asym_conv.randomize(seed=1)
    assert asym_conv.padding == ((1, 0), (2, 2))
    assert asym_conv.predict(X).shape == (2, 1, 5, 8)


def test_dilation() -> None:
    """Dilation widens the receptive field, shrinking the output accordingly."""
    conv = Conv(1, 1, 1, kernel_size=3, dilation=2)
    conv.bind()
    conv.set_params(
        {
            "_W.v": jnp.ones((1, 1, 3), dtype=jnp.float32),
            "_b.v": jnp.zeros((1,), dtype=jnp.float32),
        }
    )

    X = jnp.arange(7, dtype=jnp.float32).reshape((1, 1, 7))
    y = conv.predict(X)

    # Effective kernel spans 5 elements, so 3 output positions
    assert y.shape == (1, 1, 3)
    expected = jnp.array([[[0 + 2 + 4, 1 + 3 + 5, 2 + 4 + 6]]], dtype=jnp.float32)
    assert jnp.allclose(y, expected)


def test_groups() -> None:
    """Grouped convolution keeps channel groups independent."""
    conv = Conv(1, 4, 4, kernel_size=1, groups=2)
    conv.bind()
    conv.set_params(
        {
            "_W.v": jnp.ones((4, 2, 1), dtype=jnp.float32),
            "_b.v": jnp.zeros((4,), dtype=jnp.float32),
        }
    )

    X = jnp.array([[[1.0], [2.0], [10.0], [20.0]]], dtype=jnp.float32)
    y = conv.predict(X)

    # First two output channels see only channels 0-1, last two only channels 2-3
    expected = jnp.array([[[3.0], [3.0], [30.0], [30.0]]], dtype=jnp.float32)
    assert jnp.allclose(y, expected)


def test_unbatched_input() -> None:
    """An unbatched input returns an unbatched output equal to the batched one."""
    conv = Conv(2, 2, 3, kernel_size=3, padding="same")
    conv.randomize(seed=7)

    X = jnp.array(np.random.default_rng(3).normal(size=(2, 5, 5)), dtype=jnp.float32)
    y_unbatched = conv.predict(X)
    y_batched = conv.predict(X[None])

    assert y_unbatched.shape == (3, 5, 5)
    assert jnp.allclose(y_unbatched, y_batched[0], atol=1e-6)


def test_fit_recovers_kernel() -> None:
    """The layer can be trained to recover a known convolution kernel."""
    rng = np.random.default_rng(11)
    X = jnp.array(rng.normal(size=(32, 1, 10)), dtype=jnp.float32)

    target = Conv(1, 1, 2, kernel_size=3)
    target.randomize(seed=5)
    y = target.predict(X)

    conv = Conv(1, 1, 2, kernel_size=3, reg_loss=None)
    conv.randomize(seed=123)
    result = conv.fit(X, y)

    assert result.success
    assert jnp.allclose(conv.predict(X), y, atol=1e-3)


def test_regularization() -> None:
    """Weights are regularized by default, the bias is not."""
    conv = Conv(1, 1, 1, kernel_size=3)
    conv.bind()
    conv.set_params(
        {
            "_W.v": jnp.ones((1, 1, 3), dtype=jnp.float32),
            "_b.v": jnp.full((1,), 3.0, dtype=jnp.float32),
        }
    )

    assert set(conv.get_regularization_scales().keys()) == {"_W.v"}
    assert jnp.allclose(conv.regularization_loss(), 3.0)


def test_invalid_arguments() -> None:
    """Invalid constructor arguments raise a MiniMLError."""
    with pytest.raises(MiniMLError, match="n_dim"):
        Conv(0, 1, 1, kernel_size=3)
    with pytest.raises(MiniMLError, match="divisible by groups"):
        Conv(1, 3, 4, kernel_size=3, groups=2)
    with pytest.raises(MiniMLError, match="kernel_size"):
        Conv(2, 1, 1, kernel_size=(3, 3, 3))
    with pytest.raises(MiniMLError, match="padding mode"):
        Conv(1, 1, 1, kernel_size=3, padding="reflect")
    with pytest.raises(MiniMLError, match="non-negative"):
        Conv(1, 1, 1, kernel_size=3, padding=-1)


def test_invalid_input_shape() -> None:
    """Inputs with the wrong rank or channel count are rejected."""
    conv = Conv(2, 3, 1, kernel_size=3)
    conv.randomize(seed=0)

    with pytest.raises(MiniMLError, match="dimensions"):
        conv.predict(jnp.zeros((5,)))
    with pytest.raises(MiniMLError, match="channels"):
        conv.predict(jnp.zeros((2, 1, 5, 5)))


def test_save_load(tmp_path) -> None:
    """A convolution layer can be saved and reloaded."""
    conv = Conv(2, 2, 3, kernel_size=(3, 2), stride=2, padding="same", groups=1)
    conv.randomize(seed=17)

    fname = tmp_path / "conv.npz"
    conv.save(fname)
    loaded = Conv.load(fname)

    X = jnp.array(np.random.default_rng(9).normal(size=(2, 2, 6, 6)), dtype=jnp.float32)
    assert jnp.allclose(loaded.predict(X), conv.predict(X))
