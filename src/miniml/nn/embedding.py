from jax import Array as JXArray
import jax.numpy as jnp
from miniml.model import MiniMLModel
from miniml.param import MiniMLParam


class Embedding(MiniMLModel):
    """A MiniML model that maps discrete input indices to dense vector representations."""

    def __init__(self, n_vocab: int, dim: int, use_unknown: bool = False):
        """Initialize the Embedding model.

        Args:
            n_vocab (int): The size of the vocabulary (number of unique indices).
            dim (int): The dimensionality of the embedding vectors.
            use_unknown (bool, optional): Whether to include an additional embedding for unknown indices. Defaults to False.
        """
        if n_vocab <= 0:
            raise ValueError("n_vocab must be a positive integer")
        if dim <= 0:
            raise ValueError("dim must be a positive integer")

        self._n_vocab = n_vocab
        self._dim = dim
        self._embeddings = MiniMLParam((n_vocab + int(use_unknown), dim))
        self._use_unknown = use_unknown
        super().__init__()

    def _predict_kernel(self, X: JXArray, buffer: JXArray) -> JXArray:
        embd = self._embeddings(buffer)
        if jnp.any(X < 0):
            raise ValueError("Input indices must be non-negative")
        if self._use_unknown:
            X = jnp.clip(X, 0, self._n_vocab)
        else:
            if jnp.any(X >= self._n_vocab):
                raise ValueError(f"Input indices must be less than {self._n_vocab}")
        return jnp.take(embd, X, axis=0)

    @property
    def n_vocab(self) -> int:
        """Vocabulary size (number of unique indices)."""
        return self._n_vocab

    @property
    def dim(self) -> int:
        """Dimensionality of the embedding vectors."""
        return self._dim

    @property
    def use_unknown(self) -> bool:
        """Whether an additional embedding for unknown indices is used."""
        return self._use_unknown
