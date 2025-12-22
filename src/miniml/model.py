import numpy as np
import pickle
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Protocol, runtime_checkable, Type, TypeVar, Generic
import time
import jax
from jax import Array as JXArray
import jax.numpy as jnp
from numpy.typing import DTypeLike, NDArray
from miniml.param import MiniMLError, _supported_types, MiniMLParamRef
from miniml.loss import LossFunction
from miniml.optim.base import MiniMLOptimizer, MiniMLOptimResult
from miniml.optim.scipy import ScipyOptimizer

# Import Self from typing or typing_extensions based on Python version
import sys

if sys.version_info >= (3, 11):
    from typing import Self
else:
    from typing_extensions import Self


# Generic interface for something that has parameters
@runtime_checkable
class ParametrizedObject(Protocol):

    def _get_inner_params(self) -> list[MiniMLParamRef]: ...


T = TypeVar("T", bound="MiniMLModel")

class MiniMLModelPlan(Generic[T]):
    """A plan to create a MiniMLModel later."""

    _model_cls: Type[T]
    _args: list[Any]
    _kwargs: dict[str, Any]

    def __init__(self, model_cls: Type[T], *args: Any, **kwargs: Any) -> None:
        """Construct a MiniMLModelPlan.

        Args:
            model_cls (Type[T]): The class of the model to create.
            *args: Positional arguments for the model constructor.
            **kwargs: Keyword arguments for the model constructor.
        """
        self._model_cls = model_cls
        self._args = list(args)
        self._kwargs = dict(kwargs)

    def create(self) -> T:
        """Create the MiniMLModel instance.

        Returns:
            T: The created MiniMLModel instance.
        """
        return self._model_cls(*self._args, **self._kwargs)  # type: ignore

class MiniMLModel(ABC):
    """MiniML Model

    Base for any MiniML model. It should be subclassed as follows:

    * the constructor must declare all MiniMLParam and MiniMLModels
      directly as members of the MiniMLModel object;
    * the super() constructor must be called at the end;
    * the _predict_kernel() method must be implemented.

    """

    _dtype: DTypeLike
    _dtype_name: str
    _buffer_size: int
    _buffer: JXArray
    _params: list[MiniMLParamRef]
    _loss_f: LossFunction | None = None

    # Stored call arguments
    _init_args: bytes | None

    def __new__(cls: Type[T], *args, **kwargs) -> T:
        instance = super().__new__(cls)  # type: ignore
        # Store init arguments for saving/loading pickled
        try:
            instance._init_args = pickle.dumps({
                "args": args,
                "kwargs": kwargs,
            })
        except Exception:
            # Any reason why pickling fails, just set to None
            instance._init_args = None
        return instance

    def __init__(self, loss: LossFunction | None = None) -> None:
        """Construct a MiniML Model.

        Args:
            loss (LossFunction, optional): The loss function to use. Defaults to None.

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

        self._dtype = dtype or jnp.float32
        self._dtype_name = _supported_types.get_inverse(dtype)  # type: ignore
        # Calculate total size
        self._buffer_size = sum(pref.param.size for pref in self._params)

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
            
    def unbind(self) -> None:
        """Unbind the model parameters from the buffer."""
        if not self.bound:
            raise MiniMLError("Model parameters are not bound to a buffer")

        for p in self._params:
            p.param.unbind()
        del self._buffer

    def randomize(self, seed: int | None = None) -> None:
        """Randomize the parameters of the model using JAX random generators.

        Args:
            seed (int | None, optional): The random seed. Defaults to None.
        """
        if not self.bound:
            self.bind()
        key = jax.random.key(
            seed if seed is not None else (time.time_ns() % (2**31 - 1))
        )
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

    def regularization_loss(self, buffer: JXArray | None = None) -> JXArray:
        """Compute the total regularization loss $\\sum_i\\mathcal{R}_i(w_i)$ for all parameters and child models.

        Args:
            buffer (JXArray | None, optional): An optional buffer to use instead of the internal one.
                Defaults to None.

        Returns:
            JXArray: The total regularization loss.
        """
        reg_loss = jnp.array(0.0, dtype=jnp.dtype(self._dtype))
        for p in self._params:
            reg_loss += p.param.regularization_loss(buffer=buffer)
        return reg_loss

    def total_loss(
        self,
        y_true: JXArray,
        y_pred: JXArray,
        reg_lambda: float = 1.0,
        buffer: JXArray | None = None,
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
            buffer (JXArray | None, optional): An optional buffer to use instead of the internal one.
                Defaults to None.

        Returns:
            JXArray: The total loss.
        """
        return self.loss(y_true, y_pred) + reg_lambda * self.regularization_loss(
            buffer=buffer
        )

    @abstractmethod
    def _predict_kernel(self, X: JXArray, buffer: JXArray) -> JXArray:
        pass
    
    def _fit_predict_kernel(self, X: JXArray, buffer: JXArray) -> JXArray:
        """Prediction kernel used during fitting. By default, it calls _predict_kernel.
        Can be overridden in subclasses to provide different behavior during fitting.
        
        Args:
            X (JXArray): Input data.
            buffer (JXArray): Parameter buffer.
            
        Returns:
            JXArray: Predicted output.
        """
        return self._predict_kernel(X, buffer=buffer)

    def predict(self, X: JXArray, **predict_kwargs: dict[str, Any]) -> JXArray:
        """Predict the output for the given input data.

        Args:
            X (JXArray): Input data.
            **predict_kwargs: Additional named arguments for prediction.
                Defaults to {}.

        Returns:
            JXArray: Predicted output.
        """
        if not hasattr(self, "_jit_predict_kernel"):
            self._jit_predict_kernel = jax.jit(self._predict_kernel, inline=True)
        return self._jit_predict_kernel(X, buffer=self._buffer, **predict_kwargs)

    def __call__(self, X: JXArray) -> JXArray:
        """Syntactic sugar for predict."""
        return self.predict(X)

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
        optimizer: MiniMLOptimizer | None = None,
        predict_kwargs: dict[str, Any] = {},
    ) -> MiniMLOptimResult:
        """Fit the model parameters to the data by minimizing the total loss.

        Args:
            X (JXArray): Input features.
            y (JXArray): Target values.
            reg_lambda (float, optional): Regularization strength. Defaults to 1.0.
            optimizer (MiniMLOptimizer | None, optional): The optimizer to use.
                If None, uses L-BFGS-B ScipyOptimizer. Defaults to None.
            predict_kwargs (dict[str, Any], optional): Additional arguments to pass to the predict method.
                Defaults to {}.

        Returns:
            MiniMLOptimResult: An object containing information about the fitting process.
        """
        if not self.bound:
            self.bind()

        # Use L-BFGS-B ScipyOptimizer as default
        if optimizer is None:
            optimizer = ScipyOptimizer(method="L-BFGS-B")

        all_params = set(self.param_names)
        prefit_params = set(self._pre_fit(X, y))
        fit_params = all_params - prefit_params

        if len(fit_params) == 0:
            return MiniMLOptimResult(
                x_opt=self._buffer,
                success=True,
                message="No parameters left to fit after pre-fitting",
                objective_value=float(self.total_loss(y, self.predict(X), reg_lambda)),
                n_iterations=0,
                n_function_evaluations=0,
            )

        # Create a mask for the parameters to fit
        i0 = 0
        mask_params: list[JXArray] = []
        for p in self._params:
            if p.path in fit_params:
                mask_params.append(jnp.arange(i0, i0 + p.param.size))
            i0 += p.param.size
        p_mask = jnp.concatenate(mask_params)

        buffer = self._buffer

        def _targ_fun(p: JXArray) -> JXArray:
            nonlocal buffer
            buf_in = buffer.at[p_mask].set(p)
            y_pred = self._fit_predict_kernel(X, buffer=buf_in, **predict_kwargs)
            loss = self.total_loss(y, y_pred, reg_lambda, buf_in)
            return loss

        p0 = self._buffer[p_mask]

        result = optimizer(_targ_fun, p0)
        self._buffer = self._buffer.at[p_mask].set(result.x_opt)

        # Update result with the full buffer and recompute loss if not available
        if result.objective_value is None:
            result.objective_value = float(_targ_fun(result.x_opt))
        
        # Return result with updated x_opt pointing to full buffer
        return MiniMLOptimResult(
            x_opt=self._buffer,
            success=result.success,
            message=result.message,
            objective_value=result.objective_value,
            n_iterations=result.n_iterations,
            n_function_evaluations=result.n_function_evaluations,
            n_jacobian_evaluations=result.n_jacobian_evaluations,
            n_hessian_evaluations=result.n_hessian_evaluations,
        )

    def save(self, filename: str | Path, state_only: bool = False) -> None:
        """Save the model parameters to a file.

        Args:
            filename (str | Path): The file name.
            state_only (bool, optional): If True, do not save initialization arguments.
                This means only load_state() can be used to restore the model, and
                the user must guarantee that the model structure is the same.
                Helps in cases in which the regular save/load mechanism fails.
                Defaults to False.
        """
        if not self.bound:
            raise MiniMLError(
                "Model parameters have not been bound to buffers; can not save"
            )
        metadata = {"model_name": self.__class__.__name__}
        
        save_args = {
            "buffer": self._buffer,
            "metadata": [metadata],
        }
        if not state_only:
            if self._init_args is None:
                raise MiniMLError(
                    "Model initialization arguments could not be pickled; can not save full model. Consider using state_only=True."
                )
            save_args["init"] = self._init_args
        
        np.savez_compressed(
            filename,
            **save_args
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

        init = pickle.loads(load_dict["init"])
        args = init["args"]
        kwargs = init["kwargs"]

        try:
            model = cls(*args, **kwargs)  # type: ignore
            model.bind()
            model.set_buffer(load_dict["buffer"])
            return model
        except Exception as e:
            # When this happens, it's often because some of the arguments
            # are not well-serialized by numpy, or include stateful models.
            # In this case, we should suggest using load_state() instead.
            raise MiniMLError(
                f"Failed to load model using full state. Consider using manual initialization and load_state(). Original error:\n{e}"
            )
    
    @classmethod
    def plan(cls: Type[T], *args: Any, **kwargs: Any) -> MiniMLModelPlan[T]:
        """Create a MiniMLModelPlan to create the model later.
        
        Args:
            *args: Positional arguments for the model constructor.
            **kwargs: Keyword arguments for the model constructor.
        Returns:
            MiniMLModelPlan[T]: A plan to create the model later.
        """
        
        return MiniMLModelPlan(cls, *args, **kwargs)
    
    def load_state(self, filename: str | Path) -> None:
        """Load only the model parameters from a file
        created with state_only=True in save().

        Args:
            filename (str | Path): The file name.
        """
        load_dict = np.load(filename, allow_pickle=True)
        mdata = load_dict["metadata"][0]
        assert mdata["model_name"] == self.__class__.__name__, "Model is not same class"

        if not self.bound:
            self.bind()
        self.set_buffer(load_dict["buffer"])

    def clone(self, with_params: bool = False) -> Self:
        """Create a clone of the model with the same parameters.

        Args:
            with_params (bool, optional): If True, clone the model with the same parameters.
                Otherwise, parameters are uninitialized. Defaults to False.

        Returns:
            Self: A clone of the model.
        """
        if self._init_args is None:
            raise MiniMLError(
                "Model initialization arguments could not be pickled; can not clone. Consider manual initialization."
            )
        init = pickle.loads(self._init_args)
        clone_model = self.__class__(*init["args"], **init["kwargs"])  # type: ignore
        if with_params:
            clone_model.bind()
            clone_model.set_buffer(self.get_buffer(copy=True))
        return clone_model

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
        return {p.path: p.param().copy() for p in self._params}

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
            if p.shape != val.shape:
                raise MiniMLError(
                    f"Parameter shape mismatch for {key}: model has {p.shape}, provided value has {val.shape}"
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
        return [
            p.as_child(f"{i}")
            for i, m in enumerate(self._contents)
            for p in m._get_inner_params()
        ]
