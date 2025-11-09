import pytest
from itertools import product
from jax import Array
from jax import numpy as jnp
from miniml.optim.base import MiniMLOptimizer, DerivRequire


@pytest.mark.parametrize(
    "config",
    [
        MiniMLOptimizer.Config(deriv_require=dr, join_jac_and_value=join)
        for dr, join in product(DerivRequire, [True, False])
    ],
)
def test_miniml_optimizer_init(config: MiniMLOptimizer.Config) -> None:

    class TestOptimizer(MiniMLOptimizer):

        def _minimize_kernel(self, x0: Array) -> Array:
            assert self._obj is not None
            if self._config.deriv_require == DerivRequire.NONE:
                assert self._jac is None
                assert self._obj_and_jac is None
                assert self._hessp is None
                assert self._hess is None
            elif self._config.deriv_require == DerivRequire.JACOBIAN:
                if self._config.join_jac_and_value:
                    assert self._obj_and_jac is not None
                    assert self._jac is None
                else:
                    assert self._jac is not None
                    assert self._obj_and_jac is None
                assert self._hessp is None
                assert self._hess is None
            elif self._config.deriv_require == DerivRequire.HESSIAN_PRODUCT:
                if self._config.join_jac_and_value:
                    assert self._obj_and_jac is not None
                else:
                    assert self._jac is not None
                    assert self._obj_and_jac is None
                assert self._hessp is not None
                assert self._hess is None
            elif self._config.deriv_require == DerivRequire.HESSIAN:
                if self._config.join_jac_and_value:
                    assert self._obj_and_jac is not None
                    assert self._jac is None
                else:
                    assert self._jac is not None
                    assert self._obj_and_jac is None
                assert self._hessp is None
                assert self._hess is not None

            return x0

    # Create a dummy function
    def dummy_objective(x: Array) -> Array:
        return (x**2).sum()

    optimizer = TestOptimizer(dummy_objective, config)
    x0 = jnp.zeros(5)
    optimizer(x0)
