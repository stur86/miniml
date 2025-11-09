import pytest
import warnings
from jax import Array as JxArray
import jax.numpy as jnp
from miniml.optim.scipy import ScipyOptimizer
from miniml.optim.base import MiniMLOptimResult

@pytest.mark.parametrize(
    "method",
    [
        "Nelder-Mead",
        "Powell",
        "CG",
        "BFGS",
        "L-BFGS-B",
        "COBYLA",
        "Newton-CG",
        "trust-ncg",
        "trust-krylov",
        "trust-constr",
        "dogleg",
        "trust-exact",
        "Unsupported-Method",  # This should raise an error
    ],
)
def test_scipy_optimizer_init(method: str) -> None:
    # Create a dummy function
    def dummy_objective(x: JxArray) -> JxArray:
        return (x**2).sum()
    
    if method == "Unsupported-Method":
        with pytest.raises(ValueError):
            ScipyOptimizer(dummy_objective, method=method)
    else:
        tol = 1e-6
        optimizer = ScipyOptimizer(dummy_objective, method=method, tol=tol)
        x0 = jnp.ones(5, dtype=jnp.float32)
        # Silence any warnings from scipy about convergence for this test
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = optimizer(x0)
        assert isinstance(result, MiniMLOptimResult)
        assert jnp.allclose(result.x_opt, jnp.zeros(5), atol=tol)
        assert result.success
