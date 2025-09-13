import numpy as np
from typing import Callable
from abc import ABC, abstractmethod
from pathlib import Path
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable, Type, TypeVar
import time
import jax
from jax import Array as JXArray
import jax.numpy as jnp
from numpy.typing import DTypeLike, NDArray
from miniml.param import MiniMLError, _supported_types, MiniMLParamRef
from miniml.loss import LossFunction, squared_error_loss
from scipy.optimize import minimize


# Generic interface for something that has parameters
@runtime_checkable
class ParametrizedObject(Protocol):

    def _get_inner_params(self) -> list[MiniMLParamRef]: ...


@dataclass
class MiniMLFitResult:
    success: bool
    message: str
    loss: float
    niter: int
    nfev: int
    njev: int | None = None
    nhev: int | None = None


T = TypeVar("T", bound="MiniMLModel")


class MiniMLModel(ABC):
    """MiniML Model

    Base for any MiniML model. It should be subclassed as follows:

    * the constructor must declare all MiniMLParam and MiniMLModels
      directly as members of the MiniMLModel object;
    * the super() constructor must be called at the end;
    * the predict() method must be implemented.

    """

    _dtype: DTypeLike
    _dtype_name: str
    _buffer_size: int
    _buffer: JXArray
    _params: list[MiniMLParamRef]
    _loss_f: LossFunction | None = None

    # Stored call arguments
    _init_args: list[Any]
    _init_kwargs: dict[str, Any]

    # Kernel
    _predict_kernel: Callable[[JXArray], JXArray]

    def __new__(cls: Type[T], *args, **kwargs) -> T:
        instance = super().__new__(cls)  # type: ignore
        instance._init_args = list(args)
        instance._init_kwargs = dict(kwargs)
        return instance

    def __init__(self, loss: LossFunction | None = squared_error_loss) -> None:
        """Construct a MiniML Model.

        Args:
            loss (LossFunction, optional): The loss function to use. Defaults to squared_error_loss.

        Raises:
            MiniMLError: If the model parameters are not properly initialized.
            MiniMLError: If a child model is not properly initialized.
            MiniMLError: If a child model is not bound to a buffer.
        """

        self._loss_f = loss

        # Scan self for parameters
        pfound: list[tuple[str, ParametrizedObject]] = []
        for k, v in self.__dict__.items():
            if isinstance(v, ParametrizedObject):
                pfound.append((k, v))
        pfound = sorted(pfound, key=lambda kv: kv[0])

        self._params = []
        for k, v in pfound:
            try:
                self._params.extend([ref.as_child(k) for ref in v._get_inner_params()])
            except Exception as e:
                raise MiniMLError(f"Child member {k} was not properly initialized: {e}")

        # Scan for dtype consistency
        dtype: DTypeLike = None
        for pref in self._params:
            p = pref.param
            if dtype is None:
                dtype = p.dtype
            elif dtype != p.dtype:
                raise MiniMLError(
                    f"Model parameter dtype mismatch: found {dtype} and {p.dtype}"
                )

        self._dtype = dtype
        self._dtype_name = _supported_types.get_inverse(dtype)  # type: ignore
        # Calculate total size
        self._buffer_size = sum(pref.param.size for pref in self._params)

        # Apply JIT to predict method
        self._predict_kernel = self.predict
        self.predict = jax.jit(self.predict)  # type: ignore

    @property
    def bound(self) -> bool:
        """Check if the model parameters are bound to a buffer.

        Returns:
            bool: True if the model parameters are bound, False otherwise.
        """
        return hasattr(self, "_buffer")

    @property
    def size(self) -> int:
        """Get the total number of parameters in the model.

        Returns:
            int: The total number of parameters.
        """
        return self._buffer_size

    @property
    def ready(self) -> bool:
        """Check if the model parameters are initialized and bound.

        Returns:
            bool: True if parameters are initialized and bound, False otherwise.
        """
        return hasattr(self, "_params") and self.bound

    @property
    def dtype(self) -> DTypeLike:
        """Get the data type of the model parameters.

        Returns:
            DTypeLike: The data type of the parameters.
        """
        return self._dtype

    @property
    def dtype_name(self) -> str:
        """Get the name of the data type of the model parameters.

        Returns:
            str: The name of the data type.
        """
        return self._dtype_name
    
    @property
    def param_names(self) -> list[str]:
        """Get the list of parameter names in the model.

        Returns:
            list[str]: A list of parameter names.
        """
        return [p.path for p in self._params]

    def bind(self) -> None:
        """Bind the model parameters to a contiguous buffer."""
        if not hasattr(self, "_params"):
            raise MiniMLError(
                "Model parameters have not been initialized; remember to call super().__init__() at the end of the constructor"
            )

        if not self.bound:
            # Initialize buffers
            self._buffer = jnp.empty(self._buffer_size, dtype=jnp.dtype(self._dtype))

        # Bind buffers to parameters
        i0 = 0
        for p in self._params:
            p.param.bind(i0, self)
            i0 += p.param.size

    def randomize(self, seed: int | None = None) -> None:
        """Randomize the parameters of the model using JAX random generators.

        Args:
            seed (int | None, optional): The random seed. Defaults to None.
        """
        if not self.bound:
            self.bind()
        key = jax.random.key(seed or time.time_ns() % (2**31 - 1))
        shape = self._buffer.shape
        dtype = self._buffer.dtype
        if dtype.kind == "f":
            vals = jax.random.normal(key, shape, dtype=dtype)
            self._buffer = vals
        elif dtype.kind == "c":
            # JAX does not support complex dtypes directly in random.normal, so handle manually
            ftype = {
                "complex64": jnp.float32,
                "complex128": jnp.float64,
            }.get(dtype.name)
            if ftype is None:
                raise MiniMLError(
                    f"Randomization of parameters with dtype {dtype} not supported"
                )
            re = jax.random.normal(key, shape, dtype=ftype)
            # Use a new key for imaginary part
            key2 = jax.random.split(key)[1]
            im = jax.random.normal(key2, shape, dtype=ftype)
            self._buffer = re + 1.0j * im
        else:
            raise MiniMLError(
                f"Randomization of parameters with dtype {dtype} not supported"
            )

    def loss(self, y_true: JXArray, y_pred: JXArray) -> JXArray:
        """Compute the loss $\\mathcal{L}(y, \\hat{y})$ between true and predicted values using the model's loss function.

        Args:
            y_true (JXArray): Ground truth values.
            y_pred (JXArray): Predicted values.

        Returns:
            JXArray: The computed loss.
        """
        if self._loss_f is None:
            return jnp.array(0.0, dtype=jnp.dtype(self._dtype))
        return self._loss_f(y_true, y_pred)

    def regularization_loss(self) -> JXArray:
        """Compute the total regularization loss $\\sum_i\\mathcal{R}_i(w_i)$ for all parameters and child models.

        Returns:
            JXArray: The total regularization loss.
        """
        reg_loss = jnp.array(0.0, dtype=jnp.dtype(self._dtype))
        for p in self._params:
            reg_loss += p.param.regularization_loss()
        return reg_loss

    def total_loss(
        self, y_true: JXArray, y_pred: JXArray, reg_lambda: float = 1.0
    ) -> JXArray:
        """Compute the total loss as the sum of prediction loss and regularization loss, with
        a strength parameter:

        $$
        \\mathcal{L}(y, \\hat{y}) + \\lambda\\left(\\sum_i\\mathcal{R}_i(w_i)\\right)
        $$

        Args:
            y_true (JXArray): Ground truth values.
            y_pred (JXArray): Predicted values.
            reg_lambda (float, optional): Regularization strength. Defaults to 1.0.

        Returns:
            JXArray: The total loss.
        """
        return self.loss(y_true, y_pred) + reg_lambda * self.regularization_loss()

    @property
    def predict_kernel(self) -> Callable[[JXArray], JXArray]:
        """The non-JIT version of the predict method; use
        to call inside other models, whenever JIT compilation
        produces UnexpectedTracerError errors.
        """
        return self._predict_kernel

    @abstractmethod
    def predict(self, X: JXArray) -> JXArray:
        pass
    
    def _pre_fit(self, X: JXArray, y: JXArray) -> set[str]:
        """Initialize the fit by pre-fitting some parameters based on the data.
        This method is called once before the fitting process starts.
        It should return a set of parameter names that have been initialized,
        and these will not be optimized in the successive fitting procedure.

        Args:
            X (JXArray): Input features.
            y (JXArray): Target values.

        Returns:
            set[str]: A set of parameter names that have been initialized.
        """
        return set()

    def fit(
        self,
        X: JXArray,
        y: JXArray,
        reg_lambda: float = 1.0,
        fit_args: dict[str, Any] = {"method": "L-BFGS-B"},
    ) -> MiniMLFitResult:
        """Fit the model parameters to the data by minimizing the total loss.
        See [the SciPy docs](https://docs.scipy.org/doc/scipy-1.16.1/reference/generated/scipy.optimize.minimize.html)
        for details on the optimization arguments.

        Args:
            X (JXArray): Input features.
            y (JXArray): Target values.
            reg_lambda (float, optional): Regularization strength. Defaults to 1.0.
            fit_args (dict[str, Any], optional): Arguments for the optimizer.
                Refer to the documentation for scipy.minimize for details.
                Defaults to {"method": "L-BFGS-B"}.

        Returns:
            MiniMLFitResult: An object containing information about the fitting process.
        """
        if not self.bound:
            self.bind()
            
        all_params = set(self.param_names)
        prefit_params = set(self._pre_fit(X, y))
        fit_params = all_params - prefit_params

        if len(fit_params) == 0:
            return MiniMLFitResult(
                success=True,
                message="No parameters left to fit after pre-fitting",
                loss=float(self.total_loss(y, self._predict_kernel(X), reg_lambda)),
                niter=0,
                nfev=0,
            )
            
        # Create a mask for the parameters to fit
        i0 = 0
        mask_params: list[JXArray] = []
        for p in self._params:
            if p.path in fit_params:
                mask_params.append(jnp.arange(i0, i0 + p.param.size))
            i0 += p.param.size
        p_mask = jnp.concatenate(mask_params)

        def _targ_fun(p: JXArray) -> JXArray:
            self._buffer = self._buffer.at[p_mask].set(p)
            loss = self.total_loss(y, self._predict_kernel(X), reg_lambda)
            return loss

        # Does the method require a jacobian?
        method: str = fit_args.get("method", "L-BFGS-B")
        requires_jac = method not in {"Nelder-Mead", "Powell"}
        # Does it require a hessian product?
        requires_hessp = method in {
            "Newton-CG",
            "trust-ncg",
            "trust-krylov",
            "trust-constr",
        }
        # Does it require directly a hessian?
        requires_hess = method in {"dogleg", "trust-exact"}

        if requires_jac:
            _targ_fun_opt = jax.jit(jax.value_and_grad(_targ_fun), inline=True)
        else:
            _targ_fun_opt = jax.jit(_targ_fun, inline=True)

        if requires_hessp:
            _targ_jac = jax.jit(jax.grad(_targ_fun), inline=True)

            def jac_dir(x, p):
                return _targ_jac(x) @ p

            _targ_hessp = jax.jit(jax.grad(jac_dir, argnums=0), inline=True)
            fit_args["hessp"] = _targ_hessp
        if requires_hess:
            _targ_hess = jax.jit(jax.hessian(_targ_fun), inline=True)
            fit_args["hess"] = _targ_hess
        p0 = self._buffer[p_mask]

        sol = minimize(_targ_fun_opt, p0, jac=requires_jac, **fit_args)
        self._buffer = self._buffer.at[p_mask].set(sol.x)

        return MiniMLFitResult(
            success=sol.success,
            message=sol.message,
            loss=sol.fun,
            niter=sol.nit,
            nfev=sol.nfev,
            njev=sol.get("njev", None),
            nhev=sol.get("nhev", None),
        )

    def save(self, filename: str | Path) -> None:
        """Save the model parameters to a file.

        Args:
            filename (str | Path): The file name.
        """
        if not self.bound:
            raise MiniMLError(
                "Model parameters have not been bound to buffers; can not save"
            )
        metadata = {"model_name": self.__class__.__name__}
        init = [
            {
                "args": self._init_args,
                "kwargs": self._init_kwargs,
            }
        ]

        np.savez_compressed(
            filename,
            buffer=self._buffer,
            init=init,  # type: ignore
            metadata=[metadata],  # type: ignore
        )

    @classmethod
    def load(cls: Type[T], filename: str | Path) -> T:
        """Load a model from a file.

        Args:
            filename (str | Path): The file name.
        """
        load_dict = np.load(filename, allow_pickle=True)
        mdata = load_dict["metadata"][0]
        assert mdata["model_name"] == cls.__name__, "Model is not same class"

        init = load_dict["init"][0]
        args = init["args"]
        kwargs = init["kwargs"]

        model = cls(*args, **kwargs)  # type: ignore
        model.bind()
        model.set_buffer(load_dict["buffer"])
        return model

    def get_buffer(self, copy: bool = True) -> JXArray:
        if copy:
            return self._buffer.copy()
        return self._buffer

    def set_buffer(self, buf: JXArray | NDArray) -> None:

        buf = jnp.array(buf)
        if self._dtype != buf.dtype:
            raise MiniMLError(
                f"Parameter buffer dtype mismatch: model has {self._dtype}, buffer has {buf.dtype}"
            )
        if buf.shape != (self._buffer_size,):
            raise MiniMLError(
                f"Parameter buffer shape mismatch: model has {self._buffer_size}, buffer has {buf.shape}"
            )
        self._buffer = buf

    def get_params(self) -> dict[str, JXArray]:
        """Get a dictionary of parameter names and their values.
        All values are copies of the internal buffers.

        Returns:
            dict[str, JXArray]: A dictionary mapping parameter names to their values.
        """
        return {p.path: p.param.value.copy() for p in self._params}

    def set_params(self, params: dict[str, JXArray]) -> None:
        """Set the model parameters from a dictionary of parameter names and their values.

        Args:
            params (dict[str, JXArray]): A dictionary mapping parameter names to their values.
        """
        param_paths = [p.path for p in self._params]

        for key, val in params.items():
            idx = param_paths.index(key) if key in param_paths else -1
            if idx < 0:
                raise MiniMLError(f"Parameter name not found: {key}")
            p = self._params[idx].param
            if p.dtype != val.dtype:
                raise MiniMLError(
                    f"Parameter dtype mismatch for {key}: model has {p.dtype}, provided value has {val.dtype}"
                )
            if p.size != val.size:
                raise MiniMLError(
                    f"Parameter size mismatch for {key}: model has {p.size}, provided value has {val.size}"
                )
            idx = slice(p._buf_i0, p._buf_i0 + p.size)
            self._buffer = self._buffer.at[idx].set(val.reshape(-1))

    def _get_inner_params(self) -> list[MiniMLParamRef]:
        return self._params


class MiniMLModelList:
    """A list of MiniMLModels."""

    _contents: list[MiniMLModel]

    def __init__(self, models: list[MiniMLModel]) -> None:
        """Construct the list of models.

        Args:
            models (list[MiniMLModel]): List of models to include
        """
        self._contents = models

    @property
    def contents(self) -> list[MiniMLModel]:
        """Get the list of models."""
        return self._contents

    def __getitem__(self, i: int) -> MiniMLModel:
        """Access a model by index."""
        return self._contents[i]

    def __len__(self) -> int:
        """Total length of the list."""
        return len(self._contents)

    def _get_inner_params(self) -> list[MiniMLParamRef]:
        return [p.as_child(f"{i}") for i, m in enumerate(self._contents) for p in m._get_inner_params()]
