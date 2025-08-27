import numpy as np
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any
import time
import jax
from jax import Array as JXArray
import jax.numpy as jnp
from numpy.typing import DTypeLike
from miniml.param import MiniMLParam, MiniMLParamList, MiniMLError, _supported_types
from miniml.loss import LossFunction, squared_error_loss
from scipy.optimize import minimize

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
    _buffer: JXArray
    _params: list[MiniMLParam]
    _models: list["MiniMLModel"]
    _loss_f: LossFunction | None = None
    
    def __init__(self, loss: LossFunction = squared_error_loss) -> None:
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
        pfound: list[tuple[str, MiniMLParam]] = []
        plsfound: list[tuple[str, MiniMLParamList]] = []
        mfound: list[tuple[str, MiniMLModel]] = []
        for k, v in self.__dict__.items():
            if isinstance(v, MiniMLParam):
                pfound.append((k, v))
            elif isinstance(v, MiniMLParamList):
                plsfound.append((k, v))
            elif isinstance(v, MiniMLModel):
                mfound.append((k, v))
        pfound = sorted(pfound, key=lambda kv: kv[0])
        plsfound = sorted(plsfound, key=lambda kv: kv[0])
        mfound = sorted(mfound, key=lambda kv: kv[0])

        dtype: DTypeLike = None
        self._params = []
        for _, v in pfound:
            if dtype is None:
                dtype = v.dtype
            else:
                if (v.dtype != dtype):
                    raise MiniMLError("All parameters in a model must have the same dtype")
            self._params.append(v)
        for _, v in plsfound:
            self._params.extend(v.contents)

        # Now merge in all parameters from the child models
        self._models = []
        for k, m in mfound:
            try:
                mp = m._params
                if dtype is None:
                    dtype = m._dtype
                else:
                    if (m._dtype != dtype):
                        raise MiniMLError("All parameters in a model must have the same dtype")
            except AttributeError:
                raise MiniMLError(f"Child model {k} was not properly initialized; remember to call super().__init__() at the end of the constructor")
            self._params.extend(mp)
            self._models.append(m)
            
        self._dtype = dtype
        self._dtype_name = _supported_types.get_inverse(dtype)  # type: ignore

    @property
    def bound(self) -> bool:
        """Check if the model parameters are bound to a buffer.

        Returns:
            bool: True if the model parameters are bound, False otherwise.
        """
        return hasattr(self, "_buffer")
    
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

    def bind(self) -> None:
        """Bind the model parameters to a contiguous buffer."""
        if not hasattr(self, "_params"):
            raise MiniMLError("Model parameters have not been initialized; remember to call super().__init__() at the end of the constructor")

        if not self.bound:
            # Calculate total size
            total_size = sum(p.size for p in self._params)
            # Initialize buffers
            self._buffer = jnp.empty(total_size, dtype=jnp.dtype(self._dtype))

        # Bind buffers to parameters
        i0 = 0
        for p in self._params:
            p.bind(i0, self)
            i0 += p.size
    
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
                raise MiniMLError(f"Randomization of parameters with dtype {dtype} not supported")
            re = jax.random.normal(key, shape, dtype=ftype)
            # Use a new key for imaginary part
            key2 = jax.random.split(key)[1]
            im = jax.random.normal(key2, shape, dtype=ftype)
            self._buffer = re + 1.0j * im
        else:
            raise MiniMLError(f"Randomization of parameters with dtype {dtype} not supported")
        
    def loss(self, y_true: JXArray, y_pred: JXArray) -> JXArray:
        """Compute the loss between true and predicted values using the model's loss function.

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
        """Compute the total regularization loss for all parameters and child models.

        Returns:
            JXArray: The total regularization loss.
        """
        reg_loss = jnp.array(0.0, dtype=jnp.dtype(self._dtype))
        for p in self._params:
            reg_loss += p.regularization_loss()
        for m in self._models:
            reg_loss += m.regularization_loss()
        return reg_loss

    def total_loss(self, y_true: JXArray, y_pred: JXArray, reg_lambda: float = 1.0) -> JXArray:
        """Compute the total loss as the sum of prediction loss and regularization loss.

        Args:
            y_true (JXArray): Ground truth values.
            y_pred (JXArray): Predicted values.
            reg_lambda (float, optional): Regularization strength. Defaults to 1.0.

        Returns:
            JXArray: The total loss.
        """
        return self.loss(y_true, y_pred) + reg_lambda * self.regularization_loss()

    @abstractmethod
    def predict(self, X: JXArray) -> JXArray:
        pass

    def fit(self, X: JXArray, y: JXArray, reg_lambda: float = 1.0, 
            fit_args: dict[str, Any] = {"method": "L-BFGS-B"}) -> None:
        """Fit the model parameters to the data by minimizing the total loss.

        Args:
            X (JXArray): Input features.
            y (JXArray): Target values.
            reg_lambda (float, optional): Regularization strength. Defaults to 1.0.
            fit_args (dict[str, Any], optional): Arguments for the optimizer.
                Refer to the documentation for scipy.minimize for details.
                Defaults to {"method": "L-BFGS-B"}.
        """
        if not self.bound:
            self.bind()
            
        def _targ_fun(p: JXArray) -> JXArray:
            self._buffer = p
            loss = self.total_loss(y, self.predict(X), reg_lambda)
            return loss
        
        _targ_fun_opt = jax.jit(jax.value_and_grad(_targ_fun))
        
        sol = minimize(_targ_fun_opt, self._buffer, jac=True, **fit_args)
        self._buffer = jnp.array(sol.x, dtype=jnp.dtype(self._dtype))

    def save(self, filename: str | Path) -> None:
        """Save the model parameters to a file.

        Args:
            filename (str | Path): The file name.
        """
        if not self.bound:
            raise MiniMLError("Model parameters have not been bound to buffers; can not save")
        
        np.savez_compressed(filename, buffer=self._buffer) # type: ignore
        
    def load(self, filename: str | Path) -> None:
        """Load the model parameters from a file.

        Args:
            filename (str | Path): The file name.
        """
        if not self.bound:
            self.bind()
        
        load_dict = np.load(filename)
        buf = load_dict["buffer"]
        if self._dtype != buf.dtype:
            raise MiniMLError(f"Parameter buffer dtype mismatch: model has {self._dtype}, file has {buf.dtype}")
        self._buffer = jnp.array(buf)