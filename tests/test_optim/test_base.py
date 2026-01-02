import pytest
from itertools import product
from jax import Array as JxArray
from jax import numpy as jnp
from miniml.optim.base import (
    MiniMLOptimizer,
    DerivRequire,
    MiniMLOptimResult,
    OptimizationMethods,
    _get_ortho_component,
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
            self, x0: JxArray, methods: OptimizationMethods, seed: int | None = None
        ) -> MiniMLOptimResult:
            assert methods.obj is not None
            assert seed is None
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
    def dummy_objective(x: JxArray, rng_key = None) -> JxArray:
        return (x**2).sum()

    optimizer = TestOptimizer(config)
    x0 = jnp.zeros(5)
    result = optimizer(dummy_objective, x0)
    assert isinstance(result, MiniMLOptimResult)
    assert jnp.all(result.x_opt == x0)
    assert result.success
    assert result.message == "Test successful"
    
def test_miniml_optimizer_use_seed() -> None:
    
    TEST_SEED = 42
    
    class SeedTestOptimizer(MiniMLOptimizer):
        def _minimize_kernel(self, x0: JxArray, methods: OptimizationMethods, seed: int | None = None) -> MiniMLOptimResult:
            assert seed == TEST_SEED
            return MiniMLOptimResult(x_opt=x0, success=True, message="Seed test successful")
        
    def dummy_objective(x: JxArray, rng_key = None) -> JxArray:
        return (x**2).sum()
    
    config = OptimizationMethods.Config(
        deriv_require=DerivRequire.NONE, join_jac_and_value=False
    )
    optimizer = SeedTestOptimizer(config)
    x0 = jnp.zeros(5)
    result = optimizer(dummy_objective, x0, seed=TEST_SEED)
    assert isinstance(result, MiniMLOptimResult)
    assert jnp.all(result.x_opt == x0)
    assert result.success
    assert result.message == "Seed test successful"

def test_get_ortho_component() -> None:
    w = jnp.array([1.0, 2.0, 3.0])
    grad = jnp.array([4.0, 5.0, 6.0])
    
    grad_ortho = _get_ortho_component(grad, w)
    
    # Check that grad_ortho is orthogonal to w
    dot_product = jnp.dot(grad_ortho, w)
    assert jnp.isclose(dot_product, 0.0, atol=1e-5)
    
    # Check that grad_ortho + projection equals original grad
    w_norm_sq = jnp.sum(w * w)
    grad_proj = (jnp.sum(grad * w) / w_norm_sq) * w
    reconstructed_grad = grad_ortho + grad_proj
    assert jnp.allclose(reconstructed_grad, grad, atol=1e-5)
    
def test_optim_config() -> None:
    # Test that ValueError is raised when ortho_grad is True and deriv_require is HESSIAN
    with pytest.raises(ValueError):
        OptimizationMethods.Config(
            deriv_require=DerivRequire.HESSIAN,
            ortho_grad=True
        )

def test_opt_methods_ortho():
    # Test that the orthogonalized gradient is computed correctly in OptimizationMethods.from_objective
    v = jnp.array([1.0, 1.0, 0.0])
    def dummy_objective(x: JxArray, rng_key = None) -> JxArray:
        return (x**2).sum()+jnp.dot(x, v)
    
    config = OptimizationMethods.Config(
        deriv_require=DerivRequire.HESSIAN_PRODUCT,
        join_jac_and_value=True,
        ortho_grad=True
    )
    
    methods = OptimizationMethods.from_objective(dummy_objective, config)
    assert methods.obj_and_jac is not None
    assert methods.jac is not None
    assert methods.hessp is not None
    
    x = jnp.array([0.0, 1.0, 0.0])    
    val, grad_ortho = methods.obj_and_jac(x, None)
    
    assert jnp.isclose(val, dummy_objective(x), atol=1e-6)
    assert jnp.isclose(grad_ortho, jnp.array([1.0, 0.0, 0.0]), atol=1e-6).all()
    
    grad_from_jac = methods.jac(x, None)
    assert jnp.isclose(grad_ortho, grad_from_jac, atol=1e-6).all()

    