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


class MiniMLOptimizer(ABC):
    """Base class for MiniML optimizers."""

    # Methods that can be JIT-compiled from objective function
    _obj: ObjectiveFunction
    _jac: JacobianFunction | None = None
    _obj_and_jac: ObjJacFunction | None = None
    _hessp: HessianProductFunction | None = None
    _hess: HessianFunction | None = None

    @dataclass
    class Config:
        """Configuration for MiniMLOptimizer.

        Attributes:
            deriv_require: The level of derivative required by the optimizer.
            join_jac_and_value: Whether to compute the objective and jacobian together.
        """

        deriv_require: DerivRequire = DerivRequire.JACOBIAN
        join_jac_and_value: bool = True

    def __init__(self, objective: ObjectiveFunction, config: Config) -> None:
        """Construct the optimizer.

        Args:
            objective (ObjectiveFunction): The objective function to optimize.
            config (Config): Configuration for the optimizer.
        """
        self._obj = objective
        self._config = config

    def _jit_objective(self) -> None:
        self._obj = jax.jit(self._obj, inline=True)

    def _jit_jacobian(self) -> None:
        if self._jac is None:
            self._jac = jax.jit(jax.grad(self._obj), inline=True)

    def _jit_obj_and_jac(self) -> None:
        if self._obj_and_jac is None:
            self._obj_and_jac = jax.value_and_grad(self._obj)
            self._obj_and_jac = jax.jit(self._obj_and_jac, inline=True)

    def _jit_hessian_product(self) -> None:
        # First, JIT the jacobian if not already done
        if self._hessp is None:
            self._jit_jacobian()
            _jac = self._jac

            def jac_dir(x, p):
                return _jac(x) @ p  # type: ignore

            self._hessp = jax.grad(jac_dir, argnums=0)
            self._hessp = jax.jit(self._hessp, inline=True)

    def _jit_hessian(self) -> None:
        if self._hess is None:
            self._hess = jax.hessian(self._obj)
            self._hess = jax.jit(self._hess, inline=True)

    @abstractmethod
    def _minimize_kernel(self, x0: JxArray) -> JxArray:
        pass

    def __call__(self, x0: JxArray) -> JxArray:
        """Run the optimizer given the starting value

        Args:
            x0 (JxArray): Starting point for optimization

        Returns:
            JxArray: Optimized value
        """
        # Compute the required derivatives
        if self._config.deriv_require == DerivRequire.NONE:
            self._jit_objective()
        elif self._config.deriv_require.value >= DerivRequire.JACOBIAN.value:
            if self._config.join_jac_and_value:
                self._jit_obj_and_jac()
            else:
                self._jit_objective()
                self._jit_jacobian()
            if self._config.deriv_require == DerivRequire.HESSIAN_PRODUCT:
                self._jit_hessian_product()
            elif self._config.deriv_require == DerivRequire.HESSIAN:
                self._jit_hessian()

        return self._minimize_kernel(x0)
