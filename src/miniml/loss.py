from abc import ABC, abstractmethod
from jax.nn import log_softmax, softplus
from jax import Array as JXArray
import jax.numpy as jnp

from typing import Callable

RegLossFunction = Callable[[JXArray], JXArray]
LossFunction = Callable[[JXArray, JXArray], JXArray]


class LossFunctionBase(ABC):

    @abstractmethod
    def __call__(self, y_true: JXArray, y_pred: JXArray) -> JXArray:
        """Compute the loss between true and predicted values."""
        pass


class RegLossFunctionBase(ABC):

    @abstractmethod
    def __call__(self, y: JXArray) -> JXArray:
        """Compute the regularization loss for the given values."""
        pass


def squared_error_loss(y_true: JXArray, y_pred: JXArray) -> JXArray:
    r"""Compute the squared loss between true and predicted values:

    $$
    \mathcal{L}(y, \hat{y}) = \sum_i (y_i-\hat{y}_i)^2
    $$
    """
    if y_true.shape != y_pred.shape:
        raise ValueError("Shapes of true and predicted values must match.")
    return jnp.sum((y_true - y_pred) ** 2)


def cross_entropy_loss(y_true: JXArray, y_pred: JXArray) -> JXArray:
    r"""Compute the cross-entropy loss between true and predicted values.

    $$
    \mathcal{L}(y, \hat{y}) = -\sum_i y_i \log(\hat{y}_i)
    $$
    """
    if y_true.shape != y_pred.shape:
        raise ValueError("Shapes of true and predicted values must match.")
    # Avoid log(0) by clipping predictions
    log_y_pred = jnp.log(jnp.clip(y_pred, 1e-15, 1 - 1e-15))
    return -jnp.sum(y_true * log_y_pred)


def _stablemax_elem(x: JXArray) -> JXArray:
    r"""Compute the StableMax element-wise transformation $s(x)$ for numerical stability:
    
    $$
    s(x) = \begin{cases}
    x + 1, & x > 0 \\
    \frac{1}{1 - x}, & x \leq 0
    \end{cases}
    $$
    """
    return jnp.where(x > 0, x + 1.0, 1.0 / (1.0 - x))


def stablemax(y: JXArray) -> JXArray:
    r"""Apply the StableMax transformation element-wise to the input array.

    $$
    \text{StableMax}(x) = \frac{s(x)}{\sum_i s(x_i)}
    $$

    where $s(x)$ is the StableMax element-wise transformation defined in
    `_stablemax_elem`.
    """
    ans = _stablemax_elem(y)
    ans = ans / jnp.sum(ans, axis=-1, keepdims=True)
    return ans

def log_stablemax(y: JXArray, epsilon: float = 1e-12) -> JXArray:
    r"""Compute the log of the StableMax transformation for numerical stability.

    $$
    \log(\text{StableMax}(x)) = \log(s(x)) - \log\left(\sum_i s(x_i)\right)
    $$

    where $s(x)$ is the StableMax element-wise transformation defined in
    `_stablemax_elem`.
    """
    s_x = _stablemax_elem(y)
    s_x = jnp.clip(s_x, min=epsilon)  # Avoid log(0)
    log_s_x = jnp.log(s_x)
    log_sum_s_x = jnp.log(jnp.sum(s_x, axis=-1, keepdims=True))
    return log_s_x - log_sum_s_x

class CrossEntropyLogLoss(LossFunctionBase):
    r"""Compute the cross-entropy loss between true and predicted values,
    assuming that $\hat{y}$ is an array of unnormalized logits instead of
    probabilities.

    $$
    \mathcal{L}(y, \hat{y}) = -\sum_i y_i \hat{y}_i + \log\left(\sum_i \exp(\hat{y}_i)\right)
    $$

    """

    def __init__(self, zero_ref: bool = False, expect_labels: bool = False) -> None:
        """Initialize the loss function.

        Args:
            zero_ref (bool, optional): Whether to add a zero reference category.
                If true, log_y_pred must have one less element along the category dimension
                than y_true. Defaults to False.
            expect_labels (bool, optional): Whether y_true contains class labels
                instead of a probability distribution. Defaults to False.
        """
        self.zero_ref = zero_ref
        self.expect_labels = expect_labels
        
    def _normalize_logits(self, log_y_pred: JXArray) -> JXArray:
        return log_softmax(log_y_pred, axis=-1)

    def __call__(self, y_true: JXArray, log_y_pred: JXArray) -> JXArray:
        if self.zero_ref:
            # Add a zero reference category
            zero_shape = log_y_pred.shape[:-1] + (1,)
            log_y_pred = jnp.concatenate([log_y_pred, jnp.zeros(zero_shape)], axis=-1)
        log_y_pred = self._normalize_logits(log_y_pred)

        if not self.expect_labels:
            if y_true.shape != log_y_pred.shape:
                raise ValueError("Shapes of true and predicted values must match.")

            return -jnp.sum(y_true * log_y_pred)
        else:
            if y_true.ndim != log_y_pred.ndim - 1:
                raise ValueError("y_true must have one less dimension than log_y_pred when expect_labels is True.")
            return  -jnp.take_along_axis(log_y_pred, y_true.astype(jnp.int32)[..., None], axis=-1).sum()


class CrossEntropyStableMaxLogLoss(CrossEntropyLogLoss):
    """Compute the cross-entropy loss between true and predicted values,
    using the 'StableMax' in place of softmax for numerical stability.
    See [L. Prieto et al., 2025](https://arxiv.org/abs/2501.04697) for details.
    """
    
    def _normalize_logits(self, log_y_pred: JXArray) -> JXArray:
        return log_stablemax(log_y_pred)


def binary_match_loss(y_true: JXArray, log_y_pred: JXArray) -> JXArray:
    r"""Loss function for binary classification tasks on multiple independent outputs.
    Computes the sum of binary cross-entropy losses for each output:

    $$
    \mathcal{L}(y, \hat{y}) = \sum_i \left[ \log(1 + \exp(\hat{y}_i)) - y_i \hat{y}_i \right]
    $$

    Args:
        y_true (JXArray): True binary labels (0 or 1).
        log_y_pred (JXArray): Logit predictions for the positive class (y_true=1).

    Returns:
        JXArray: Computed binary cross-entropy loss.
    """

    if y_true.shape != log_y_pred.shape:
        raise ValueError("Shapes of true and predicted values must match.")
    # Here log_y_pred represents the logit predictions of y_true being 1 in each position
    log_y_norms = softplus(log_y_pred)  # log(1 + exp(y_pred))
    return jnp.mean(log_y_norms) - jnp.mean(y_true * log_y_pred)


class LNormRegularization(RegLossFunctionBase):
    r"""L^p norm regularization. Optionally includes taking the p-th root.

    Without root:

    $$
    \mathcal{R}_p(w) = \sum_i |w_i|^p
    $$

    With root:

    $$
    \mathcal{R}_p(w) = (\sum_i |w_i|^p)^{1/p}
    $$
    """

    def __init__(self, p: int = 2, root: bool = False) -> None:
        """Initialize the LNormRegularization class.

        Args:
            p (int, optional): The order of the norm. Defaults to 2.
            root (bool, optional): Whether to take the root of the norm. Defaults to False.
        """
        self.p = p
        self.return_root = root

    def __call__(self, y: JXArray) -> JXArray:
        ans = jnp.sum(jnp.abs(y) ** self.p)
        if self.return_root:
            return ans ** (1.0 / self.p)
        return ans
