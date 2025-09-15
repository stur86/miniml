from typing import Literal, Callable
from numpy.typing import DTypeLike

import jax
from jax import Array as JXArray
import jax.numpy as jnp
from miniml.loss import LossFunction, squared_error_loss
from miniml.model import MiniMLModel
from miniml.param import MiniMLParam, MiniMLParamList
from miniml.nn.rbf import RBFunction, gaussian_rbf
from miniml.nn.ortho import CayleyMatrix

ProjectionType = Literal["none", "scaling", "ortho", "full"]


class RBFLayer(MiniMLModel):

    _rbf: RBFunction
    _X0: MiniMLParam
    _s: MiniMLParam
    _pmats_list: MiniMLParamList
    _pmats: MiniMLParam
    _xproj: Callable[[JXArray, JXArray], JXArray]
    _normalize: bool

    def __init__(
        self,
        n_in: int,
        n_out: int,
        n_centers: int,
        rbf: RBFunction = gaussian_rbf,
        projection: ProjectionType = "scaling",
        loss: LossFunction = squared_error_loss,
        normalize: bool = False,
        dtype: DTypeLike = jnp.float32,
    ) -> None:

        # Radial basis function
        self._rbf = rbf

        # X centers
        self._X0 = MiniMLParam((n_centers, n_in), dtype=dtype)

        # Projection parameters
        if projection == "scaling":
            self._s = MiniMLParam((n_centers, n_in), dtype=dtype)
        elif projection == "ortho":
            self._pmats_list = MiniMLParamList(
                [CayleyMatrix(n_in, dtype=dtype) for _ in range(n_centers)]
            )
            self._s = MiniMLParam((n_centers, n_in), dtype=dtype)
        elif projection == "full":
            self._pmats = MiniMLParam((n_centers, n_in, n_in), dtype=dtype)
        elif projection == "none":
            pass

        try:
            self._xproj = getattr(self, f"_xproj_{projection}")
        except AttributeError:
            raise ValueError(f"Unknown projection type: {projection}")

        # Output weights
        self._W = MiniMLParam((n_centers, n_out), dtype=dtype)
        self._normalize = normalize

        super().__init__(loss=loss)

    # Projection methods
    def _xproj_none(self, X: JXArray, buffer: JXArray) -> JXArray:
        # No projection, return input as is
        return X

    def _xproj_scaling(self, X: JXArray, buffer: JXArray) -> JXArray:
        # Element-wise scaling
        s = self._s(buffer)
        return X * s

    def _xproj_ortho(self, X: JXArray, buffer: JXArray) -> JXArray:
        # Orthogonal projection using Cayley matrices
        pmats = jnp.stack([p(buffer) for p in self._pmats_list.contents])
        s = self._s(buffer)
        # i = centers, n = batch, k = input features, j = projected features
        return jnp.einsum("ijk,nik->nij", pmats, X) * s

    def _xproj_full(self, X: JXArray, buffer: JXArray) -> JXArray:
        # Full linear projection
        pmats = self._pmats(buffer)
        # i = centers, n = batch, k = input features, j = projected features
        return jnp.einsum("ijk,nik->nij", pmats, X)

    def _predict_kernel(self, X: JXArray, buffer: JXArray) -> JXArray:
        X0 = self._X0(buffer)  # (centers, n_in)
        # Translate inputs by centers
        X = X[:, None, :] - X0[None, :, :]  # (n, centers, n_in)
        # Scale/project inputs
        X = self._xproj(X, buffer)  # (n, centers, n_in)
        # Find radius
        r = jnp.linalg.norm(X, axis=-1)  # (n, centers)
        # Apply radial basis function
        H = self._rbf(r)  # (n, centers)
        # Apply output weights
        W = self._W(buffer)  # (centers, n_out)
        X = H @ W  # (n, n_out)

        # Normalize if required
        jax.lax.cond(
            self._normalize,
            lambda X: X / jnp.sum(H, axis=-1, keepdims=True),
            lambda X: X,
            X,
        )

        return X
