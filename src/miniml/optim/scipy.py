from jax import Array as JxArray
import jax.numpy as jnp
from scipy.optimize import minimize
from miniml.optim.base import MiniMLOptimizer, DerivRequire, MiniMLOptimResult, ObjectiveFunction


class ScipyOptimizer(MiniMLOptimizer):

    _method: str
    _options: dict
    _tol: float | None
    
    _obj_only_methods = {"Nelder-Mead", "Powell", "COBYLA"}
    _jac_methods = {"CG", "BFGS", "L-BFGS-B"}
    _hessp_methods = {"Newton-CG", "trust-ncg", "trust-krylov", "trust-constr"}
    _hess_methods = {"dogleg", "trust-exact"}
    
    _all_methods = (
        _obj_only_methods
        | _jac_methods
        | _hessp_methods
        | _hess_methods
    )

    def __init__(
        self,
        objective: ObjectiveFunction,
        method: str = "L-BFGS-B",
        options: dict = {},
        tol: float | None = None,
    ) -> None:
        config = self._get_method_config(method)
        super().__init__(objective, config)
        self._method = method
        self._options = options
        self._tol = tol

    @staticmethod
    def _get_method_config(method: str) -> MiniMLOptimizer.Config:
        # Methods that don't require derivatives
        if method in ScipyOptimizer._obj_only_methods:
            return MiniMLOptimizer.Config(
                deriv_require=DerivRequire.NONE, join_jac_and_value=False
            )
        # Methods that require Hessian-vector product
        elif method in ScipyOptimizer._hessp_methods:
            return MiniMLOptimizer.Config(
                deriv_require=DerivRequire.HESSIAN_PRODUCT, join_jac_and_value=True
            )
        # Methods that require full Hessian
        elif method in ScipyOptimizer._hess_methods:
            return MiniMLOptimizer.Config(
                deriv_require=DerivRequire.HESSIAN, join_jac_and_value=True
            )
        # Methods that require Jacobian (most common case)
        elif method in ScipyOptimizer._jac_methods:
            return MiniMLOptimizer.Config(
                deriv_require=DerivRequire.JACOBIAN, join_jac_and_value=True
            )
        else:
            raise ValueError(f"Unsupported optimization method: {method}")
    
    @property
    def supported_methods(self) -> set[str]:
        return self._all_methods

    def _minimize_kernel(self, x0: JxArray) -> MiniMLOptimResult:
        jac = None
        if self._config.deriv_require.value >= DerivRequire.JACOBIAN.value:
            jac = self._jac if not self._config.join_jac_and_value else True

        result = minimize(
            fun=self._obj_and_jac if self._config.join_jac_and_value else self._obj,
            x0=x0,
            method=self._method,
            jac=jac,
            hessp=(
                self._hessp
                if self._config.deriv_require == DerivRequire.HESSIAN_PRODUCT
                else None
            ),
            hess=(
                self._hess
                if self._config.deriv_require == DerivRequire.HESSIAN
                else None
            ),
            options=self._options,
            tol=self._tol,
        )
        return MiniMLOptimResult(
            x_opt=jnp.array(result.x),
            success=result.success,
            message=result.message,
            n_iterations=result.get("nit", None),
            n_function_evaluations=result.get("nfev", None),
            n_jacobian_evaluations=result.get("njev", None),
            n_hessian_evaluations=result.get("nhev", None),
        )
