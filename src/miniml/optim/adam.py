import jax
import jax.numpy as jnp
from jax import Array as JxArray
from miniml.optim.base import (
    MiniMLOptimizer,
    MiniMLOptimResult,
    OptimizationMethods,
    DerivRequire,
)


class AdamOptimizer(MiniMLOptimizer):
    """Adaptive Moment Estimation (Adam) optimizer.
    
    References:
        - Diederik P. Kingma and Jimmy Ba. ["Adam: A Method for Stochastic Optimization."](https://arxiv.org/abs/1412.6980)
    """

    def __init__(
        self,
        alpha: float = 0.001,
        beta_1: float = 0.9,
        beta_2: float = 0.999,
        eps: float = 1e-8,
        tol: float = 0.0,
        maxiter: int = 1000,
    ) -> None:
        """Initialize the Adam optimizer.
        
        Args:
            alpha (float, optional): Learning rate. Defaults to 0.001.
            beta_1 (float, optional): Exponential decay rate for the first moment estimates. Defaults to 0.9.
            beta_2 (float, optional): Exponential decay rate for the second moment estimates. Defaults to 0.999.
            eps (float, optional): Small constant for numerical stability. Defaults to 1e-8.
            tol (float, optional): Tolerance for stopping criterion based on the norm of the first moment. Defaults to 0.0.
            maxiter (int, optional): Maximum number of iterations. Defaults to 1000.
        """
        # Default configuration: we need the Jacobian, nothing else
        config = OptimizationMethods.Config(
            deriv_require=DerivRequire.JACOBIAN, join_jac_and_value=False
        )
        super().__init__(config)
        self._alpha = alpha
        self._beta_1 = beta_1
        self._beta_2 = beta_2
        self._eps = eps
        self._tol = tol
        self._maxiter = maxiter

    def _minimize_kernel(
        self, x0: JxArray, methods: OptimizationMethods
    ) -> MiniMLOptimResult:
        n = x0.shape[0]
        m = jnp.zeros_like(x0)
        v = jnp.zeros_like(x0)
        x = x0
        assert (
            methods.jac is not None
        ), "Jacobian function must be provided for Adam optimizer."
        gradfun = methods.jac

        def adam_step(t: int, state: JxArray) -> JxArray:
            flag = state[-1]
            x, m, v = jnp.split(state[:-1], 3)

            def update_fn(x, m, v):
                grad = gradfun(x)
                m = self._beta_1 * m + (1 - self._beta_1) * grad
                v = self._beta_2 * v + (1 - self._beta_2) * (grad**2)
                m_hat = m / (1 - self._beta_1**t)
                v_hat = v / (1 - self._beta_2**t)
                x = x - self._alpha * m_hat / (jnp.sqrt(v_hat) + self._eps)
                return x, m, v

            def no_update_fn(x, m, v):
                return x, m, v

            x, m, v = jax.lax.cond(flag == 0, update_fn, no_update_fn, x, m, v)
            flag = jnp.where(jnp.linalg.norm(m) < self._tol, jnp.where(flag == 0, t, flag), 0.0)
            return jnp.concatenate([x, m, v, jnp.array([flag])], axis=0)

        adam_step_jit = jax.jit(adam_step, inline=True)
        state = jnp.concatenate([x, m, v, jnp.array([0.0])], axis=0)

        out_state = jax.lax.fori_loop(1, self._maxiter + 1, adam_step_jit, state)

        success = out_state[-1] > 0
        n_iters = int(out_state[-1]) if success else self._maxiter
        x_opt = out_state[:-1][:n]

        return MiniMLOptimResult(
            x_opt=x_opt,
            success=bool(success),
            message=(
                "Optimization converged." if success else "Maximum iterations reached."
            ),
            objective_value=float(methods.obj(x_opt)),
            n_iterations=n_iters,
            n_function_evaluations=None,
            n_jacobian_evaluations=n_iters,
            n_hessian_evaluations=None,
        )
