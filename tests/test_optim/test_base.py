import pytest
from itertools import product
from jax import Array
from jax import numpy as jnp
from miniml.optim.base import (
    MiniMLOptimizer,
    DerivRequire,
    MiniMLOptimResult,
    OptimizationMethods,
)


@pytest.mark.parametrize(
    "config",
    [
        OptimizationMethods.Config(deriv_require=dr, join_jac_and_value=join)
        for dr, join in product(DerivRequire, [True, False])
    ],
)
def test_miniml_optimizer_init(config: OptimizationMethods.Config) -> None:

    class TestOptimizer(MiniMLOptimizer):

        def _minimize_kernel(
            self, x0: Array, methods: OptimizationMethods
        ) -> MiniMLOptimResult:
            assert methods.obj is not None
            if self._config.deriv_require == DerivRequire.NONE:
                assert methods.jac is None
                assert methods.obj_and_jac is None
                assert methods.hessp is None
                assert methods.hess is None
            elif self._config.deriv_require == DerivRequire.JACOBIAN:
                if self._config.join_jac_and_value:
                    assert methods.obj_and_jac is not None
                    assert methods.jac is None
                else:
                    assert methods.jac is not None
                    assert methods.obj_and_jac is None
                assert methods.hessp is None
                assert methods.hess is None
            elif self._config.deriv_require == DerivRequire.HESSIAN_PRODUCT:
                if self._config.join_jac_and_value:
                    assert methods.obj_and_jac is not None
                else:
                    assert methods.jac is not None
                    assert methods.obj_and_jac is None
                assert methods.hessp is not None
                assert methods.hess is None
            elif self._config.deriv_require == DerivRequire.HESSIAN:
                if self._config.join_jac_and_value:
                    assert methods.obj_and_jac is not None
                    assert methods.jac is None
                else:
                    assert methods.jac is not None
                    assert methods.obj_and_jac is None
                assert methods.hessp is None
                assert methods.hess is not None

            return MiniMLOptimResult(x_opt=x0, success=True, message="Test successful")

    # Create a dummy function
    def dummy_objective(x: Array) -> Array:
        return (x**2).sum()

    optimizer = TestOptimizer(config)
    x0 = jnp.zeros(5)
    result = optimizer(dummy_objective, x0)
    assert isinstance(result, MiniMLOptimResult)
    assert jnp.all(result.x_opt == x0)
    assert result.success
    assert result.message == "Test successful"