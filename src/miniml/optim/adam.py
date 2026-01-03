import jax
import jax.numpy as jnp
from jax import Array as JxArray
from miniml.optim.base import (
    MiniMLOptimizer,
    MiniMLOptimResult,
    OptimizationMethods,
    DerivRequire,
)
from typing import Generator
from contextlib import contextmanager
from dataclasses import dataclass

def _adam_grad_apply_weight_decay(grad, weight_decay, x):
    return grad + weight_decay * x

def _adamw_grad_apply_weight_decay(grad, weight_decay, x):
    return grad  # Weight decay is applied separately in AdamW

def _adam_update_step(update, weight_decay, x):
    return update  # Weight decay is applied inside the gradient in Adam

def _adamw_update_step(update, weight_decay, x):
    return update + weight_decay * x  # Decoupled weight decay in AdamW


@dataclass
class AdamState:
    """Wrapper for persistent Adam optimizer state.

    Attributes:
        m: First moment estimates.
        v: Second moment estimates.
        flag: Stopping flag (local iteration index where tolerance was met,
            or 0.0 if not yet met).
        iteration: Total number of iterations performed so far (global count).
    """

    m: JxArray
    v: JxArray
    flag: float
    iteration: int

    def compatible(self, length: int) -> bool:
        """Return True if the state is compatible with a parameter vector
        of the given length.
        """
        return self.m.size == length and self.v.size == length


class AdamBaseOptimizer(MiniMLOptimizer):
    """Base class for Adam-style optimizers with optional weight decay.

    This class can implement classical Adam or decoupled-weight-decay AdamW
    behaviour depending on the ``decouple_weight_decay`` flag.
    """
    
    # State persistence
    _persist: bool = False
    _state: AdamState | None = None

    def __init__(
        self,
        alpha: float = 0.001,
        beta_1: float = 0.9,
        beta_2: float = 0.999,
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        ortho_grad: bool = False,
        tol: float = 0.0,
        maxiter: int = 1000,
        decouple_weight_decay: bool = False,
    ) -> None:
        """Initialize the base Adam-style optimizer.

        Args:
            alpha (float, optional): Learning rate. Defaults to 0.001.
            beta_1 (float, optional): Exponential decay rate for the first
                moment estimates. Defaults to 0.9.
            beta_2 (float, optional): Exponential decay rate for the second
                moment estimates. Defaults to 0.999.
            eps (float, optional): Small constant for numerical stability.
                Defaults to 1e-8.
            weight_decay (float, optional): L2 weight decay coefficient.
                For AdamOptimizer this is coupled to the gradient; for
                AdamWOptimizer it is decoupled. Defaults to 0.0.
            ortho_grad (bool, optional): If True, project gradients to be
                orthogonal to the parameters at each step. Defaults to False.
            tol (float, optional): Tolerance for stopping criterion based on
                the norm of the first moment. Defaults to 0.0.
            maxiter (int, optional): Maximum number of iterations. Defaults
                to 1000.
            decouple_weight_decay (bool, optional): If True, use AdamW-style
                decoupled weight decay, otherwise classical Adam behaviour.
                Defaults to False.
        """
        # Default configuration: we need the Jacobian, nothing else
        config = OptimizationMethods.Config(
            deriv_require=DerivRequire.JACOBIAN, join_jac_and_value=False, 
            ortho_grad=ortho_grad,
        )
        super().__init__(config)
        self._alpha = alpha
        self._beta_1 = beta_1
        self._beta_2 = beta_2
        self._eps = eps
        self._weight_decay = weight_decay
        self._tol = tol
        self._maxiter = maxiter
        self._decouple_weight_decay = decouple_weight_decay
        
        if decouple_weight_decay:
            _grad_apply = _adamw_grad_apply_weight_decay
            _update_step = _adamw_update_step
        else:
            _grad_apply = _adam_grad_apply_weight_decay
            _update_step = _adam_update_step
        
        def _update_impl(
            x: JxArray,
            m: JxArray,
            v: JxArray,
            t: int,
            update_key: JxArray | None,
            gradfun,
            alpha: float,
            beta_1: float,
            beta_2: float,
            eps: float,
            weight_decay: float,
        ) -> tuple[JxArray, JxArray, JxArray]:
            grad = gradfun(x, update_key)
            grad = _grad_apply(grad, weight_decay, x)
            m = beta_1 * m + (1 - beta_1) * grad
            v = beta_2 * v + (1 - beta_2) * (grad**2)
            m_hat = m / (1 - beta_1**t)
            v_hat = v / (1 - beta_2**t)
            update = alpha * m_hat / (jnp.sqrt(v_hat) + eps)
            update = _update_step(update, weight_decay, x)
            x = x - update
            return x, m, v
        
        self._update_impl = _update_impl
    
    def _clear_state(self) -> None:
        """Clear any persisted optimizer state."""
        self._state = None

    def _get_state(self, x: JxArray) -> AdamState:
        """Get the current optimizer state.

        If no compatible state is persisted, initialize to zeros.

        Args:
            x (JxArray): Current parameters.

        Returns:
            AdamState: Wrapper containing first/second moments, flag and
                global iteration count.
        """
        n = x.shape[0]
        if self._state is None or (not self._state.compatible(n)):
            # Drop any incompatible state and start fresh
            self._clear_state()
            m = jnp.zeros_like(x)
            v = jnp.zeros_like(x)
            return AdamState(m=m, v=v, flag=0.0, iteration=0)

        return self._state

    def _minimize_kernel(
        self, x0: JxArray, methods: OptimizationMethods,
        seed: int | None = None,
    ) -> MiniMLOptimResult:
        n = x0.shape[0]
        x = x0
        state = self._get_state(x)
        m = state.m
        v = state.v
                
        assert (
            methods.jac is not None
        ), "Jacobian function must be provided for Adam optimizer."
        gradfun = methods.jac

        def update_fn(x, m, v, t, update_key):
            # Use a global step index for bias correction that accounts for
            # any iterations performed in previous persistent calls.
            return self._update_impl(
                x,
                m,
                v,
                t,
                update_key,
                gradfun,
                self._alpha,
                self._beta_1,
                self._beta_2,
                self._eps,
                self._weight_decay,
            )

        def no_update_fn(x, m, v, t, update_key):
            return x, m, v

        def adam_step(
            t: int, internals: tuple[JxArray, JxArray | None]
        ) -> tuple[JxArray, JxArray | None]:
            state, rng_key = internals
            flag = state[-1]
            x, m, v = jnp.split(state[:-1], 3)
            if rng_key is not None:
                rng_key, update_key = jax.random.split(rng_key)
            else:
                update_key = None

            x, m, v = jax.lax.cond(
                flag == 0, update_fn, no_update_fn, x, m, v, t, update_key
            )
            flag = jnp.where(
                jnp.linalg.norm(m) < self._tol, jnp.where(flag == 0, t, flag), 0.0
            )
            return (jnp.concatenate([x, m, v, jnp.array([flag])], axis=0), rng_key)

        adam_step_jit = jax.jit(adam_step, inline=True)

        prng_key: JxArray | None = None
        if seed is not None:
            prng_key = jax.random.PRNGKey(seed)

        # Start with a fresh local stopping flag each call; global iteration
        # count is tracked separately in the persistent state wrapper.
        state_vec = jnp.concatenate([x, m, v, jnp.array([state.flag])], axis=0)
        loop_state = (state_vec, prng_key)

        iter_start = state.iteration + 1
        iter_end = state.iteration + self._maxiter + 1
        out_state, last_rng_key = jax.lax.fori_loop(
            iter_start, iter_end, adam_step_jit, loop_state
        )

        success = out_state[-1] > 0
        n_iters = int(out_state[-1]) if success else self._maxiter
        x_opt = out_state[:n]
        
        if self._persist:
            # Save the optimizer state for future calls
            m_opt = out_state[n : 2 * n]
            v_opt = out_state[2 * n : 3 * n]
            flag_opt = float(out_state[-1]) if success else 0.0
            total_iteration = iter_end - 1
            self._state = AdamState(
                m=m_opt,
                v=v_opt,
                flag=flag_opt,
                iteration=total_iteration,
            )

        return MiniMLOptimResult(
            x_opt=x_opt,
            success=bool(success),
            message=(
                "Optimization converged." if success else "Maximum iterations reached."
            ),
            objective_value=float(methods.obj(x_opt, last_rng_key)),
            n_iterations=n_iters,
            n_function_evaluations=None,
            n_jacobian_evaluations=n_iters,
            n_hessian_evaluations=None,
        )
    
    @contextmanager
    def persistent(self) -> Generator["AdamBaseOptimizer", None, None]:
        """Open a context in which optimizer state is persisted across multiple
        minimize calls.
        """
        self._persist = True
        try:
            yield self
        finally:
            self._persist = False
            self._clear_state()


class AdamOptimizer(AdamBaseOptimizer):
    """Adaptive Moment Estimation (Adam) optimizer.

    This implements classical Adam. Optional ``weight_decay`` is applied
    via L2 regularisation inside the gradient (coupled weight decay).

    References:
        - Diederik P. Kingma and Jimmy Ba. ["Adam: A Method for Stochastic Optimization."](https://arxiv.org/abs/1412.6980)
    """

    def __init__(
        self,
        alpha: float = 0.001,
        beta_1: float = 0.9,
        beta_2: float = 0.999,
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        ortho_grad: bool = False,
        tol: float = 0.0,
        maxiter: int = 1000,
    ) -> None:
        """Initialize the classical Adam optimizer.
        
        !!! note
            Weight decay here is a way to implement L2 regularization,
            which is redundant compared to the use of reg_lambda in
            MiniML models if per-parameter regularization has been set.
            Be mindful of not mixing the two approaches unintentionally;
            if you use weight_decay > 0, use reg_lambda=0 in the fitting call.

        Args:
            alpha (float, optional): Learning rate. Defaults to 0.001.
            beta_1 (float, optional): Exponential decay rate for the first
                moment estimates. Defaults to 0.9.
            beta_2 (float, optional): Exponential decay rate for the second
                moment estimates. Defaults to 0.999.
            eps (float, optional): Small constant for numerical stability.
                Defaults to 1e-8.
            weight_decay (float, optional): L2 weight decay coefficient
                applied inside the gradient (coupled). Defaults to 0.0.
            ortho_grad (bool, optional): If True, project gradients to be
                orthogonal to the parameters at each step. Defaults to False.
            tol (float, optional): Tolerance for stopping criterion based on
                the norm of the first moment. Defaults to 0.0.
            maxiter (int, optional): Maximum number of iterations. Defaults
                to 1000.
        """
        super().__init__(
            alpha=alpha,
            beta_1=beta_1,
            beta_2=beta_2,
            eps=eps,
            weight_decay=weight_decay,
            ortho_grad=ortho_grad,
            tol=tol,
            maxiter=maxiter,
            decouple_weight_decay=False,
        )


class AdamWOptimizer(AdamBaseOptimizer):
    """AdamW optimizer with decoupled weight decay."""

    def __init__(
        self,
        alpha: float = 0.001,
        beta_1: float = 0.9,
        beta_2: float = 0.999,
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        ortho_grad: bool = False,
        tol: float = 0.0,
        maxiter: int = 1000,
    ) -> None:
        """Initialize the AdamW optimizer with decoupled weight decay.

        !!! note
            Weight decay here is a way to implement decoupled regularization,
            which is redundant compared to the use of reg_lambda in
            MiniML models if per-parameter regularization has been set.
            Be mindful of not mixing the two approaches unintentionally;
            if you use weight_decay > 0, you may want to use reg_lambda=0
            in the fitting call.

        Args:
            alpha (float, optional): Learning rate. Defaults to 0.001.
            beta_1 (float, optional): Exponential decay rate for the first
                moment estimates. Defaults to 0.9.
            beta_2 (float, optional): Exponential decay rate for the second
                moment estimates. Defaults to 0.999.
            eps (float, optional): Small constant for numerical stability.
                Defaults to 1e-8.
            weight_decay (float, optional): L2 weight decay coefficient,
                applied in a decoupled AdamW fashion. Defaults to 0.0.
            ortho_grad (bool, optional): If True, project gradients to be
                orthogonal to the parameters at each step. Defaults to False.
            tol (float, optional): Tolerance for stopping criterion based on
                the norm of the first moment. Defaults to 0.0.
            maxiter (int, optional): Maximum number of iterations. Defaults
                to 1000.
        """
        super().__init__(
            alpha=alpha,
            beta_1=beta_1,
            beta_2=beta_2,
            eps=eps,
            weight_decay=weight_decay,
            ortho_grad=ortho_grad,
            tol=tol,
            maxiter=maxiter,
            decouple_weight_decay=True,
        )
