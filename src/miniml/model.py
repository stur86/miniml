import numpy as np
import pickle
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Protocol, runtime_checkable, Type, TypeVar, Generic
import time
import jax
from jax import Array as JXArray
import jax.numpy as jnp
from numpy.typing import DTypeLike, NDArray
from miniml.param import MiniMLError, MiniMLParam, _supported_types, MiniMLParamRef
from miniml.loss import LossFunction
from miniml.optim.base import MiniMLOptimizer, MiniMLOptimResult
from miniml.optim.scipy import ScipyOptimizer

# Import Self from typing or typing_extensions based on Python version
import sys
import fnmatch
import warnings

if sys.version_info >= (3, 11):
    from typing import Self
else:
    from typing_extensions import Self


class PredictMode(Enum):
    """Prediction mode for MiniML models.

    Attributes:
        TRAINING: Prediction performed during training/optimization.
        INFERENCE: Prediction performed during inference/evaluation.
    """

    TRAINING = "training"
    INFERENCE = "inference"


@dataclass
class PredictKernelOutput:
    """Output of a ``_predict_kernel`` call that optionally carries an activity loss.

    When a model returns this from ``_predict_kernel`` instead of a plain array,
    the ``activity_loss`` (if set) is added to the training objective scaled by
    ``active_reg_lambda``.  The inference path discards ``activity_loss`` entirely.

    Attributes:
        y_pred: The model's prediction array.
        activity_loss: An optional scalar activity regularization loss.  Should
            only be computed when ``mode == PredictMode.TRAINING`` — it is ignored
            during inference, so computing it then wastes time.
    """

    y_pred: JXArray
    activity_loss: JXArray | None = field(default=None)


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
        * the _predict_kernel() method must be implemented, supporting both
            training and inference modes and an optional RNG key.

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
        instance._replace_init_args(*args, **kwargs)
        return instance

    def _replace_init_args(self, *args: Any, **kwargs: Any) -> None:
        """Record the arguments that ``load()`` rebuilds this model from.

        They are stored as passed to the constructor.  A model that resolves an
        argument into plain data while constructing itself can call this again to
        store the data instead, so that loading it back does not depend on
        rebuilding the original object.  The arguments must still describe the
        same model.

        Args:
            *args: Positional arguments to record.
            **kwargs: Keyword arguments to record.
        """
        try:
            self._init_args = pickle.dumps({"args": args, "kwargs": kwargs})
        except Exception:
            # Any reason why pickling fails, just set to None: it only blocks save()
            self._init_args = None

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

    @property
    def loss_function(self) -> LossFunction | None:
        """Get the loss function of the model.

        Returns:
            LossFunction | None: The loss function, or None if not set.
        """
        return self._loss_f

    def bind(self) -> None:
        """Bind the model parameters to a contiguous buffer.

        Binding is atomic: if any parameter can not be bound, the model is left
        exactly as it was found, rather than holding an uninitialized buffer.

        Raises:
            MiniMLError: If the model parameters have not been initialized.
            MiniMLError: If any parameter is already bound, e.g. because it is
                shared with another model that owns it.
        """
        if not hasattr(self, "_params"):
            raise MiniMLError(
                "Model parameters have not been initialized; remember to call super().__init__() at the end of the constructor"
            )

        buffer_created = not self.bound
        if buffer_created:
            # Initialize buffers
            self._buffer = jnp.empty(self._buffer_size, dtype=jnp.dtype(self._dtype))

        # Bind buffers to parameters, rolling back if any of them fails
        done: list[MiniMLParam] = []
        try:
            i0 = 0
            for p in self._params:
                p.param.bind(i0, self)
                done.append(p.param)
                i0 += p.param.size
        except Exception:
            for param in done:
                param.unbind()
            if buffer_created:
                del self._buffer
            raise

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

        for ref in self._params:
            p = ref.param
            if p.rnd_scale != 1.0:
                idx = slice(p._buf_i0, p._buf_i0 + p.size)
                self._buffer = self._buffer.at[idx].multiply(p.rnd_scale)

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

    @staticmethod
    def _unpack_kernel_output(
        result: JXArray | PredictKernelOutput,
        activity_loss: JXArray | None = None,
    ) -> tuple[JXArray, JXArray | None]:
        """Unpack a ``_predict_kernel`` return value into ``(y_pred, activity_loss)``.

        ``activity_loss`` is ``None`` when neither ``result`` nor the passed in
        accumulator carries one; no zero term is created in that case.  Pass the
        running total as ``activity_loss`` to accumulate over several children::

            activity_loss = None
            for model in models:
                result = model._predict_kernel(X, buffer)
                X, activity_loss = MiniMLModel._unpack_kernel_output(
                    result, activity_loss
                )

        Args:
            result: The raw return value of ``_predict_kernel``.
            activity_loss: An optional activity loss to add the one carried by
                ``result`` to. Defaults to None.

        Returns:
            Tuple of ``(y_pred, activity_loss)`` where ``activity_loss`` is a
            JAX scalar, or None if there is no activity loss at all.
        """
        if not isinstance(result, PredictKernelOutput):
            return result, activity_loss
        if result.activity_loss is None:
            return result.y_pred, activity_loss
        if activity_loss is None:
            return result.y_pred, result.activity_loss
        return result.y_pred, activity_loss + result.activity_loss

    @staticmethod
    def _with_activity_loss(
        y_pred: JXArray,
        activity_loss: JXArray | None,
    ) -> JXArray | PredictKernelOutput:
        """Attach an activity loss to a prediction, if there is one to attach.

        Returns the bare ``y_pred`` when ``activity_loss`` is None, so that models
        which happen to carry no activity loss are indistinguishable from ones that
        never produce any.

        Args:
            y_pred: The model's prediction array.
            activity_loss: The activity loss to carry, if any.

        Returns:
            ``y_pred`` itself, or a :class:`PredictKernelOutput` wrapping both.
        """
        if activity_loss is None:
            return y_pred
        return PredictKernelOutput(y_pred=y_pred, activity_loss=activity_loss)

    @abstractmethod
    def _predict_kernel(
        self,
        X: JXArray,
        buffer: JXArray,
        rng_key: JXArray | None = None,
        mode: PredictMode = PredictMode.INFERENCE,
        **predict_kwargs: Any,
    ) -> JXArray | PredictKernelOutput:
        """Core prediction kernel used for both training and inference.

        Subclasses can branch on ``mode`` and optionally use ``rng_key``
        for stochastic behaviour (e.g. dropout) during training.  May return
        either a plain ``JXArray`` or a :class:`PredictKernelOutput` to carry
        an additional activity regularization loss.

        Args:
            X: Input data.
            buffer: Parameter buffer.
            rng_key: Optional JAX random key for stochastic models.
            mode: Prediction mode (training or inference).
            **predict_kwargs: Additional keyword-only arguments for model-specific
                behaviour.
        """
        raise NotImplementedError

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

            def _inference_kernel(
                X: JXArray, buffer: JXArray, **kwargs: Any
            ) -> JXArray:
                result = self._predict_kernel(
                    X,
                    buffer=buffer,
                    rng_key=None,
                    mode=PredictMode.INFERENCE,
                    **kwargs,
                )
                y_pred, _ = MiniMLModel._unpack_kernel_output(result)
                return y_pred

            self._jit_predict_kernel = jax.jit(_inference_kernel, inline=True)

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
        active_reg_lambda: float = 1.0,
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
            active_reg_lambda (float, optional): Scaling factor for the activity
                regularization loss returned by ``_predict_kernel``.  Defaults to 1.0.

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

        def _targ_fun(p: JXArray, rng_key: JXArray | None) -> JXArray:
            nonlocal buffer
            buf_in = buffer.at[p_mask].set(p)
            result = self._predict_kernel(
                X,
                buf_in,
                rng_key,
                PredictMode.TRAINING,
                **predict_kwargs,
            )
            y_pred, activity_loss = MiniMLModel._unpack_kernel_output(result)
            total = self.total_loss(y, y_pred, reg_lambda, buf_in)
            if activity_loss is not None:
                total = total + active_reg_lambda * activity_loss
            return total

        p0 = self._buffer[p_mask]

        result = optimizer(_targ_fun, p0)
        self._buffer = self._buffer.at[p_mask].set(result.x_opt)

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

        np.savez_compressed(filename, **save_args)

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

    def get_regularization_scales(self) -> dict[str, float]:
        """Return a dict of {param_path: reg_scale} for all regularized parameters.

        Parameters without a regularizer are excluded. Returns an empty dict if
        the model has no regularized parameters.

        Returns:
            dict[str, float]: Mapping of parameter path to its current reg_scale.
        """
        return {
            ref.path: ref.param.reg_scale
            for ref in self._params
            if ref.param._reg_loss is not None
        }

    def set_regularization_scale(self, path: str, value: float) -> None:
        """Set the regularization scale for all regularized parameters matching a glob pattern.

        Uses fnmatch glob semantics: '*' matches any characters including '.', so
        '*W.v' matches both 'W.v' and 'layer.W.v'. Parameters without a
        regularizer are silently skipped. Emits a warning if no regularized
        parameters were updated.

        Args:
            path (str): A glob pattern matched against parameter dot-paths
                (e.g. 'W.v', '_layer.*', '*').
            value (float): The new reg_scale value to set.
        """
        any_path_matched = False
        any_reg_updated = False
        for ref in self._params:
            if fnmatch.fnmatch(ref.path, path):
                any_path_matched = True
                if ref.param._reg_loss is not None:
                    ref.param.reg_scale = value
                    any_reg_updated = True
        if not any_reg_updated:
            if not any_path_matched:
                warnings.warn(
                    f"set_regularization_scale('{path}'): pattern matched no parameter paths.",
                    stacklevel=2,
                )
            else:
                warnings.warn(
                    f"set_regularization_scale('{path}'): pattern matched parameter(s) but none have a regularizer.",
                    stacklevel=2,
                )


class InvertibleModel(ABC):
    """Mixin interface for models that can be inverted.

    A model that implements this interface can reverse its own ``predict()``
    transformation through ``invert()``.  Subclasses must implement
    ``_invert_kernel()`` with the same signature as ``_predict_kernel``,
    but computing the inverse transformation instead.

    Use it as a mixin alongside MiniMLModel::

        class AffineCouplingLayer(InvertibleModel, MiniMLModel):
            def _predict_kernel(self, ...): ...

            def _invert_kernel(self, ...): ...
    """

    @abstractmethod
    def _invert_kernel(
        self,
        Y: JXArray,
        buffer: JXArray,
        rng_key: JXArray | None = None,
        mode: PredictMode = PredictMode.INFERENCE,
        **invert_kwargs: Any,
    ) -> "JXArray | PredictKernelOutput":
        """Core inversion kernel used by ``invert()``.

        Mirrors ``_predict_kernel`` but computes the inverse transformation.
        May return either a plain ``JXArray`` or a :class:`PredictKernelOutput`.

        Args:
            Y: Input data in the model's output space.
            buffer: Parameter buffer.
            rng_key: Optional JAX random key for stochastic models.
            mode: Prediction mode (training or inference).
            **invert_kwargs: Additional keyword-only arguments.
        """
        raise NotImplementedError

    def invert(self, Y: JXArray, **invert_kwargs: dict[str, Any]) -> JXArray:
        """Invert the model transformation, recovering the input from its output.

        Args:
            Y (JXArray): Input data in the model's output space.
            **invert_kwargs: Additional named arguments for inversion.

        Returns:
            JXArray: The recovered input.
        """
        if not hasattr(self, "_jit_invert_kernel"):

            def _inference_invert(
                Y: JXArray, buffer: JXArray, **kwargs: Any
            ) -> JXArray:
                result = self._invert_kernel(
                    Y,
                    buffer=buffer,
                    rng_key=None,
                    mode=PredictMode.INFERENCE,
                    **kwargs,
                )
                y_pred, _ = MiniMLModel._unpack_kernel_output(result)
                return y_pred

            self._jit_invert_kernel = jax.jit(_inference_invert, inline=True)

        return self._jit_invert_kernel(Y, buffer=self._buffer, **invert_kwargs)

    def inverse_model(self) -> "InvertibleModel":
        """Return a model that applies this one's inverse transformation.

        The returned model shares this model's parameters instead of copying
        them, so it always reflects the current parameter values, including
        during and after a fit.  Inverting it returns the original model back.

        Returns:
            InvertibleModel: A view on this model with predict and invert
            swapped.
        """
        return InverseModel(self)


class SharedModel:
    """A reference to a model whose parameters are owned somewhere else.

    A ``MiniMLModel`` collects the parameters of every member that exposes
    ``_get_inner_params``, so storing a model directly as a member makes the
    holder a co-owner of its parameters.  Wrapping it here instead states that
    the holder only *uses* the model: the wrapper reports no parameters of its
    own, so the same model can be referenced from several places without being
    counted, or bound, more than once.

    The referenced model must still appear, in its own right, somewhere in the
    tree that owns the buffer; otherwise its parameters are never bound.
    """

    _model: MiniMLModel

    def __init__(self, model: MiniMLModel) -> None:
        """Construct a reference to an externally owned model.

        Args:
            model (MiniMLModel): The model to refer to.
        """
        self._model = model

    @property
    def model(self) -> MiniMLModel:
        """The referenced model."""
        return self._model

    def _get_inner_params(self) -> list[MiniMLParamRef]:
        """No parameters: they belong to whoever owns the referenced model."""
        return []


class InverseModel(InvertibleModel, MiniMLModel):
    """A view on an :class:`InvertibleModel` with predict and invert swapped.

    It holds no parameters of its own — it reads those of the model it wraps,
    through a :class:`SharedModel` reference — so it costs nothing to create and
    it follows the wrapped model's parameters as they are fitted.  Build one with
    ``model.inverse_model()``.

    It can be used on its own, or placed in a container alongside the model it
    inverts, e.g. to tie an encoder to its decoder::

        layer = AffineCouplingLayer(4)
        Stack([layer, layer.inverse_model()])  # the identity, for any parameters
    """

    def __init__(self, model: MiniMLModel) -> None:
        """Construct the inverse view of an invertible model.

        Args:
            model (MiniMLModel): The model to invert.  It must also be an
                InvertibleModel.

        Raises:
            MiniMLError: If the model is not an invertible MiniMLModel.
        """
        if not isinstance(model, InvertibleModel) or not isinstance(model, MiniMLModel):
            raise MiniMLError(
                "InverseModel can only wrap a MiniMLModel that is also an InvertibleModel"
            )
        self._inverted = SharedModel(model)
        super().__init__(loss=model.loss_function)

    @property
    def inverted(self) -> MiniMLModel:
        """The model being inverted."""
        return self._inverted.model

    @property
    def _buffer(self) -> JXArray:  # type: ignore[override]
        """The buffer of the wrapped model, which owns the parameters."""
        return self.inverted._buffer

    @_buffer.setter
    def _buffer(self, buffer: JXArray) -> None:
        self.inverted._buffer = buffer

    def bind(self) -> None:
        """Bind the wrapped model, which owns every parameter this view reads."""
        self.inverted.bind()

    def unbind(self) -> None:
        """Unbind the wrapped model, which owns every parameter this view reads."""
        self.inverted.unbind()

    def inverse_model(self) -> "InvertibleModel":
        """Return the wrapped model itself, rather than a view on a view."""
        return self.inverted  # type: ignore[return-value]

    def save(self, filename: str | Path, state_only: bool = False) -> None:
        """Refuse to save: the parameters belong to the wrapped model.

        Args:
            filename (str | Path): Unused.
            state_only (bool, optional): Unused. Defaults to False.

        Raises:
            MiniMLError: Always.  Save the wrapped model instead, and rebuild
                the view from it with ``inverse_model()``.
        """
        raise MiniMLError(
            "An InverseModel owns no parameters and can not be saved; save the "
            "model it inverts instead, then call inverse_model() on it again"
        )

    def _predict_kernel(
        self,
        X: JXArray,
        buffer: JXArray,
        rng_key: JXArray | None = None,
        mode: PredictMode = PredictMode.INFERENCE,
        **predict_kwargs: Any,
    ) -> JXArray | PredictKernelOutput:
        return self.inverted._invert_kernel(  # type: ignore[attr-defined]
            X, buffer, rng_key=rng_key, mode=mode, **predict_kwargs
        )

    def _invert_kernel(
        self,
        Y: JXArray,
        buffer: JXArray,
        rng_key: JXArray | None = None,
        mode: PredictMode = PredictMode.INFERENCE,
        **invert_kwargs: Any,
    ) -> JXArray | PredictKernelOutput:
        return self.inverted._predict_kernel(
            Y, buffer, rng_key=rng_key, mode=mode, **invert_kwargs
        )


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
