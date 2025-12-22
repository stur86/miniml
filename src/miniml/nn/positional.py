from jax import Array as JXArray
import jax.numpy as jnp
from miniml.model import MiniMLModel
from miniml.param import MiniMLParam

class PositionalEmbedding(MiniMLModel):
    """A MiniML model that adds positional embeddings to input sequences."""

    def __init__(self, max_length: int, dim: int):
        """Initialize the PositionalEmbedding model.

        Args:
            max_length (int): The maximum length of input sequences.
            dim (int): The dimensionality of the embedding vectors.
        """
        if max_length <= 0:
            raise ValueError("max_length must be a positive integer")
        if dim <= 0:
            raise ValueError("dim must be a positive integer")

        self._max_length = max_length
        self._dim = dim
        self._pos_embeddings = MiniMLParam((max_length, dim))
        super().__init__()

    def _predict_kernel(self, X: JXArray, buffer: JXArray) -> JXArray:
        pos_embd = self._pos_embeddings(buffer)
        seq_length = X.shape[1]
        if seq_length > self._max_length:
            raise ValueError(f"Input sequence length {seq_length} exceeds maximum length {self._max_length}")
        return X + pos_embd[:seq_length, :]

    @property
    def max_length(self) -> int:
        """Maximum length of input sequences."""
        return self._max_length

    @property
    def dim(self) -> int:
        """Dimensionality of the embedding vectors."""
        return self._dim
    
class RotaryPositionalEmbedding(MiniMLModel):
    """A MiniML model that applies rotary positional embeddings to input sequences."""

    def __init__(self, dim: int, rot_length: int = 10000):
        """Initialize the RotaryPositionalEmbedding model.

        Args:
            dim (int): The dimensionality of the embedding vectors.
            rot_length (int): The maximum length of input sequences,
                after which the embeddings will repeat.
        """
        if dim <= 0:
            raise ValueError("dim must be a positive integer")
        if dim%2 != 0:
            raise ValueError("dim must be an even integer for rotary embeddings")
        if rot_length <= 0:
            raise ValueError("rot_length must be a positive integer")

        self._dim = dim
        self._rot_length = rot_length
        
        # Precalculate the positional embeddings
        max_length = rot_length
        position_ids = jnp.arange(max_length)[:, None]
        indices = jnp.arange(dim // 2)[None, :]
        angle_rates = 1 / (rot_length ** (2 * indices / dim))
        angle_rads = position_ids * angle_rates
        self._sin_embeddings = jnp.sin(angle_rads)
        self._cos_embeddings = jnp.cos(angle_rads)


        super().__init__()

    def _predict_kernel(self, X: JXArray, buffer: JXArray) -> JXArray:
        seq_length = X.shape[-2]
        sin_emb = self._sin_embeddings
        cos_emb = self._cos_embeddings
        if seq_length > self._sin_embeddings.shape[0]:
            # Repeat the embeddings if sequence length exceeds rot_length
            sin_emb = jnp.tile(sin_emb, (seq_length // self._rot_length + 1, 1))
            cos_emb = jnp.tile(cos_emb, (seq_length // self._rot_length + 1, 1))
        sin_emb = sin_emb[:seq_length, :]
        cos_emb = cos_emb[:seq_length, :]

        x1, x2 = jnp.split(X, 2, axis=-1)
        x_rotated_1 = x1 * cos_emb - x2 * sin_emb
        x_rotated_2 = x1 * sin_emb + x2 * cos_emb
        return jnp.concatenate([x_rotated_1, x_rotated_2], axis=-1)

    @property
    def dim(self) -> int:
        """Dimensionality of the embedding vectors."""
        return self._dim