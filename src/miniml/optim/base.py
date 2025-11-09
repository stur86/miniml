import jax
from jax import Array as JxArray
from abc import ABC, abstractmethod
from typing import Callable
from enum import Enum
from dataclasses import dataclass

ObjectiveFunction = Callable[[JxArray], JxArray]
JacobianFunction = Callable[[JxArray], JxArray]
ObjJacFunction = Callable[[JxArray], tuple[JxArray, JxArray]]
HessianProductFunction = Callable[[JxArray, JxArray], JxArray]
HessianFunction = Callable[[JxArray], JxArray]


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
        """

        deriv_require: DerivRequire = DerivRequire.JACOBIAN
        join_jac_and_value: bool = True

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
                obj_and_jac = jax.jit(obj_and_jac, inline=True)
            else:
                jac = jax.jit(jax.grad(objective), inline=True)

            if config.deriv_require == DerivRequire.HESSIAN_PRODUCT:
                # Create jacobian if not already done for hessp
                if jac is None:
                    jac = jax.jit(jax.grad(objective), inline=True)

                _jac = jac

                def jac_dir(x, p):
                    return _jac(x) @ p

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
        self, x0: JxArray, methods: OptimizationMethods
    ) -> MiniMLOptimResult:
        """Execute the optimization kernel.

        Args:
            x0: Starting point for optimization.
            methods: Container with JIT-compiled optimization methods.

        Returns:
            MiniMLOptimResult: Result of the optimization.
        """
        pass

    def __call__(
        self, objective: ObjectiveFunction, x0: JxArray
    ) -> MiniMLOptimResult:
        """Run the optimizer given the objective function and starting value.

        Args:
            objective: The objective function to optimize.
            x0: Starting point for optimization.

        Returns:
            MiniMLOptimResult: Result of the optimization.
        """
        methods = OptimizationMethods.from_objective(objective, self._config)
        return self._minimize_kernel(x0, methods)
