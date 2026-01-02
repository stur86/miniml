import jax
import jax.numpy as jnp
from jax import Array as JxArray
from abc import ABC, abstractmethod
from typing import Callable
from enum import Enum
from dataclasses import dataclass

OptionalRngKey = JxArray | None

ObjectiveFunction = Callable[[JxArray, OptionalRngKey], JxArray]
JacobianFunction = Callable[[JxArray, OptionalRngKey], JxArray]
ObjJacFunction = Callable[[JxArray, OptionalRngKey], tuple[JxArray, JxArray]]
HessianProductFunction = Callable[[JxArray, JxArray, OptionalRngKey], JxArray]
HessianFunction = Callable[[JxArray, OptionalRngKey], JxArray]

class DerivRequire(Enum):
    """Enumeration of derivative requirements for the optimizer.

    Attributes:
        NONE: No derivatives required.
        JACOBIAN: Jacobian (gradient) required.
        HESSIAN_PRODUCT: Hessian-vector product required.
        HESSIAN: Full Hessian matrix required.
    """

    NONE = 0
    JACOBIAN = 1
    HESSIAN_PRODUCT = 2
    HESSIAN = 3


@dataclass
class MiniMLOptimResult:
    """Result of the optimization process.

    Attributes:
        x_opt: The optimized parameters.
        success: Whether the optimization was successful.
        message: A message describing the outcome of the optimization.
        loss: The final loss value, if available.
        n_iterations: Number of iterations performed, if available.
        n_function_evaluations: Number of function evaluations, if available.
        n_jacobian_evaluations: Number of jacobian evaluations, if available.
        n_hessian_evaluations: Number of hessian evaluations, if available.
    """

    x_opt: JxArray
    success: bool
    message: str = ""
    objective_value: float | None = None
    n_iterations: int | None = None
    n_function_evaluations: int | None = None
    n_jacobian_evaluations: int | None = None
    n_hessian_evaluations: int | None = None

def _get_ortho_component(grad: JxArray, w: JxArray) -> JxArray:
    w_norm_sq = jnp.sum(w * w)
    grad_ortho = jnp.where(
        w_norm_sq > 0,
        grad - (jnp.sum(grad * w) / w_norm_sq) * w,
        grad,
    )
    return grad_ortho

@dataclass
class OptimizationMethods:
    """Container for JIT-compiled optimization methods.

    Attributes:
        obj: The JIT-compiled objective function.
        jac: The JIT-compiled jacobian function, if required.
        obj_and_jac: The JIT-compiled combined objective and jacobian function, if required.
        hessp: The JIT-compiled Hessian-vector product function, if required.
        hess: The JIT-compiled Hessian function, if required.
    """

    obj: ObjectiveFunction
    jac: JacobianFunction | None = None
    obj_and_jac: ObjJacFunction | None = None
    hessp: HessianProductFunction | None = None
    hess: HessianFunction | None = None

    @dataclass
    class Config:
        """Configuration for OptimizationMethods.

        Attributes:
            deriv_require: The level of derivative required by the optimizer.
            join_jac_and_value: Whether to compute the objective and jacobian together.
            ortho_grad: Whether to orthogonalize gradients (take only the component orthogonal to the weight vector).
                Used for generalization in some contexts (see L. Prieto et al, "Grokking at the edge of numerical stability", 2025).
        """

        deriv_require: DerivRequire = DerivRequire.JACOBIAN
        join_jac_and_value: bool = True
        ortho_grad: bool = False
        
        def __post_init__(self) -> None:
            if self.deriv_require == DerivRequire.HESSIAN and self.ortho_grad:
                raise ValueError("Orthogonalized gradients are not compatible with full Hessian computation.")

    @classmethod
    def from_objective(
        cls, objective: ObjectiveFunction, config: Config
    ) -> "OptimizationMethods":
        """Create OptimizationMethods from an objective function and config.

        Args:
            objective: The objective function to optimize.
            config: Configuration specifying which derivatives to compute.

        Returns:
            OptimizationMethods: Container with JIT-compiled methods.
        """
        # Always JIT the objective
        obj = jax.jit(objective, inline=True)

        jac = None
        obj_and_jac = None
        hessp = None
        hess = None

        # Compute the required derivatives
        if config.deriv_require == DerivRequire.NONE:
            pass  # Only obj is needed
        elif config.deriv_require.value >= DerivRequire.JACOBIAN.value:
            if config.join_jac_and_value:
                obj_and_jac = jax.value_and_grad(objective)
                if config.ortho_grad:
                    _obj_and_jac = obj_and_jac

                    def obj_and_jac_ortho(x: JxArray, rng: OptionalRngKey = None) -> tuple[JxArray, JxArray]:
                        val, grad = _obj_and_jac(x, rng)
                        grad_ortho = _get_ortho_component(grad, x)
                        return val, grad_ortho

                    obj_and_jac = obj_and_jac_ortho
                
                obj_and_jac = jax.jit(obj_and_jac, inline=True)
                
            if (not config.join_jac_and_value) or (config.deriv_require == DerivRequire.HESSIAN_PRODUCT):
                jac = jax.jit(jax.grad(objective), inline=True)
                if config.ortho_grad:
                    _jac_original = jac

                    def jac_ortho(x: JxArray, rng: OptionalRngKey = None) -> JxArray:
                        grad = _jac_original(x, rng)
                        grad_ortho = _get_ortho_component(grad, x)
                        return grad_ortho

                    jac = jax.jit(jac_ortho, inline=True)

            if config.deriv_require == DerivRequire.HESSIAN_PRODUCT:
                assert jac is not None, "Jacobian must be computed for Hessian-vector product"

                _jac_hessp = jac

                def jac_dir(x, p, *args, **kwargs):
                    return _jac_hessp(x, *args, **kwargs) @ p

                hessp = jax.grad(jac_dir, argnums=0)
                hessp = jax.jit(hessp, inline=True)
            elif config.deriv_require == DerivRequire.HESSIAN:
                hess = jax.hessian(objective)
                hess = jax.jit(hess, inline=True)

        return cls(
            obj=obj, jac=jac, obj_and_jac=obj_and_jac, hessp=hessp, hess=hess
        )


class MiniMLOptimizer(ABC):
    """Base class for MiniML optimizers."""

    _config: OptimizationMethods.Config

    def __init__(self, config: OptimizationMethods.Config) -> None:
        """Construct the optimizer.

        Args:
            config (OptimizationMethods.Config): Configuration for the optimizer.
        """
        self._config = config

    @abstractmethod
    def _minimize_kernel(
        self, x0: JxArray, methods: OptimizationMethods, 
        seed: int | None = None
    ) -> MiniMLOptimResult:
        """Execute the optimization kernel.

        Args:
            x0: Starting point for optimization.
            methods: Container with JIT-compiled optimization methods.
            seed: Optional random seed for stochastic methods.

        Returns:
            MiniMLOptimResult: Result of the optimization.
        """
        pass

    def __call__(
        self, objective: ObjectiveFunction, x0: JxArray, seed: int | None = None
    ) -> MiniMLOptimResult:
        """Run the optimizer given the objective function and starting value.

        Args:
            objective: The objective function to optimize.
            x0: Starting point for optimization.
            seed: Optional random seed for stochastic methods.

        Returns:
            MiniMLOptimResult: Result of the optimization.
        """
        methods = OptimizationMethods.from_objective(objective, self._config)
        return self._minimize_kernel(x0, methods, seed=seed)
