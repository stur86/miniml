from typing import Any
from jax import numpy as jnp, Array as JxArray
from miniml.model import MiniMLModel
from miniml.nn.linear import Linear
from miniml.param import MiniMLParam, DTypeLike, MiniMLError
from miniml.loss import (
    LossFunction,
    RegLossFunction,
    squared_error_loss,
    LNormRegularization,
)

_MultiHeadArg = JxArray | tuple[JxArray, JxArray, JxArray]


class MultiHeadAttention(MiniMLModel):
    """A MiniML model that implements multi-head attention mechanism."""

    _embed_dim: int
    _num_heads: int
    _kdim: int
    _vdim: int

    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 1,
        kdim: int | None = None,
        vdim: int | None = None,
        loss: LossFunction = squared_error_loss,
        reg_loss: RegLossFunction = LNormRegularization(2),
        dtype: DTypeLike = jnp.float32,
    ) -> None:
        r"""Initialize the MultiHeadAttention model.

        The multi-head attention is computed as:
        $$
        \text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, \ldots, \text{head}_h) W^T
        $$
        where each head is computed as:
        $$
        \text{head}_i = \text{Attention}(Q W_i^Q, K W_i^K, V W_i^V)
        $$
        and the attention function is defined as:
        $$
        \text{Attention}(Q, K, V) = \text{softmax}\left(\frac{Q K^T}{\sqrt{d_k}}\right) V
        $$

        For self-attention, Q, K, and V are all the same input, and this model can be invoked as:

        ```py
        mha = MultiHeadAttention(embed_dim=64, num_heads=8)
        output = mha.predict(X)  # where X has shape (..., embed_dim)
        ```

        Whereas for general attention, Q, K, and V can be different inputs, and the model can be invoked as:

        ```py
        mha = MultiHeadAttention(embed_dim=64, num_heads=8, kdim=32, vdim=32)
        output = mha.predict((Xq, Xk, Xv))  # where Xq has shape (..., embed_dim), Xk and Xv have shape (..., kdim) and (..., vdim) respectively
        ```

        Args:
            embed_dim (int): Total dimension of the model.
            num_heads (int, optional): Number of attention heads. Defaults to 1.
            kdim (int | None, optional): Dimension of the key vectors. Defaults to None, which sets it equal to embed_dim.
            vdim (int | None, optional): Dimension of the value vectors. Defaults to None, which sets it equal to embed_dim.
            loss (LossFunction, optional): Loss function for the model. Defaults to squared_error_loss.
            reg_loss (RegLossFunction, optional): Regularization function for the weights. Defaults to LNormRegularization(2).
            dtype (DTypeLike, optional): Data type for the model parameters. Defaults to jnp.float32.
        """

        self._kdim = kdim if kdim is not None else embed_dim
        self._vdim = vdim if vdim is not None else embed_dim
        self._embed_dim = embed_dim
        self._num_heads = num_heads
        self._head_dim = embed_dim // num_heads
        if self._head_dim * num_heads != embed_dim:
            raise MiniMLError("embed_dim must be divisible by num_heads.")

        homogeneous = (self._kdim == embed_dim) and (self._vdim == embed_dim)
        if homogeneous:
            self._QKV = MiniMLParam(
                (embed_dim, 3 * embed_dim), dtype=dtype, reg_loss=reg_loss
            )
        else:
            self._Q = MiniMLParam(
                (embed_dim, embed_dim), dtype=dtype, reg_loss=reg_loss
            )
            self._K = MiniMLParam(
                (self._kdim, embed_dim), dtype=dtype, reg_loss=reg_loss
            )
            self._V = MiniMLParam(
                (self._vdim, embed_dim), dtype=dtype, reg_loss=reg_loss
            )
        self._QKV_bias = MiniMLParam((3 * embed_dim,), dtype=dtype)
        self._out_proj = Linear(
            embed_dim, embed_dim, reg_loss=reg_loss, dtype=dtype, apply_bias_reg=False
        )

        super().__init__(loss=loss)

    @property
    def embed_dim(self) -> int:
        return self._embed_dim

    @property
    def num_heads(self) -> int:
        return self._num_heads

    @property
    def kdim(self) -> int:
        return self._kdim

    @property
    def vdim(self) -> int:
        return self._vdim

    @property
    def homogeneous(self) -> bool:
        return hasattr(self, "_QKV")

    def _get_Q_K_V(self, buffer: JxArray) -> tuple[JxArray, JxArray, JxArray]:
        if self.homogeneous:
            QKV_weight = self._QKV(buffer)
            Q, K, V = jnp.split(QKV_weight, 3, axis=-1)
        else:
            Q = self._Q(buffer)
            K = self._K(buffer)
            V = self._V(buffer)
        return Q, K, V

    def predict(self, X: _MultiHeadArg, **predict_kwargs: dict[str, Any]) -> JxArray:
        return super().predict(X, **predict_kwargs)  # type: ignore

    def _predict_kernel(
        self, X: _MultiHeadArg, buffer: JxArray, attn_mask: JxArray | None = None
    ) -> JxArray:
        is_self_attn = not isinstance(X, tuple)
        if self.homogeneous and is_self_attn:
            QKV_weight = self._QKV(buffer)
            Xqkv = X @ QKV_weight + self._QKV_bias(buffer)
            Xq, Xk, Xv = jnp.split(Xqkv, 3, axis=-1)
        else:
            if is_self_attn:
                Xq = X
                Xk = X
                Xv = X
            else:
                Xq, Xk, Xv = X
            Q, K, V = self._get_Q_K_V(buffer)
            Qb, Kb, Vb = jnp.split(self._QKV_bias(buffer), 3, axis=-1)
            Xq = Xq @ Q + Qb
            Xk = Xk @ K + Kb
            Xv = Xv @ V + Vb

        # Split into heads
        Xq_heads = Xq.reshape(Xq.shape[:-1] + (self._num_heads, self._head_dim))
        Xk_heads = Xk.reshape(Xk.shape[:-1] + (self._num_heads, self._head_dim))
        Xv_heads = Xv.reshape(Xv.shape[:-1] + (self._num_heads, self._head_dim))

        # Build attention scores
        attn_scores = jnp.einsum("...qhd,...khd->...qhk", Xq_heads, Xk_heads)
        attn_scores = attn_scores / jnp.sqrt(self._head_dim)
        if attn_mask is not None:
            if attn_mask.dtype == jnp.bool_:
                attn_mask = jnp.where(attn_mask, -jnp.inf, 0.0)
            attn_scores = attn_scores + attn_mask[:, None, :]
        attn_weights = jnp.exp(
            attn_scores - jnp.max(attn_scores, axis=-1, keepdims=True)
        )
        attn_weights = attn_weights / jnp.sum(attn_weights, axis=-1, keepdims=True)
        attn_output_heads = jnp.einsum("...qhk,...khd->...qhd", attn_weights, Xv_heads)
        attn_output = attn_output_heads.reshape(
            attn_output_heads.shape[:-2] + (self._embed_dim,)
        )

        return self._out_proj._predict_kernel(attn_output, buffer)
