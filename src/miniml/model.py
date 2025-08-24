import numpy as np
from pathlib import Path
from numpy.typing import NDArray
from numpy.typing import DTypeLike
from miniml.param import MiniMLParam, MiniMLError, _supported_types


class MiniMLModel:
    
    _dtype: DTypeLike
    _dtype_name: str
    _buffer: NDArray
    _params: list[MiniMLParam]
    
    def __init__(self) -> None:
        
        # Scan self for parameters
        pfound: list[tuple[str, MiniMLParam]] = []
        mfound: list[tuple[str, MiniMLModel]] = []
        for k, v in self.__dict__.items():
            if isinstance(v, MiniMLParam):
                pfound.append((k, v))
            elif isinstance(v, MiniMLModel):
                mfound.append((k, v))
        pfound = sorted(pfound, key=lambda kv: kv[0])
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
            
        # Now merge in all parameters from the child models
        for k, m in mfound:
            if dtype is None:
                dtype = m._dtype
            else:
                if (m._dtype != dtype):
                    raise MiniMLError("All parameters in a model must have the same dtype")
            try:
                mp = m._params
            except AttributeError:
                raise MiniMLError(f"Child model {k} was not properly initialized; remember to call super().__init__() at the end of the constructor")
            self._params.extend(mp)
            
        self._dtype = dtype
        self._dtype_name = _supported_types.get_inverse(dtype)  # type: ignore

    @property
    def bound(self) -> bool:
        return hasattr(self, "_buffer")
    
    @property
    def ready(self) -> bool:
        return hasattr(self, "_params") and self.bound
    
    @property
    def dtype(self) -> DTypeLike:
        return self._dtype
    
    @property
    def dtype_name(self) -> str:
        return self._dtype_name

    def bind(self) -> None:
        if not hasattr(self, "_params"):
            raise MiniMLError("Model parameters have not been initialized; remember to call super().__init__() at the end of the constructor")

        if not self.bound:
            # Calculate total size
            total_size = sum(p.size for p in self._params)
            # Initialize buffers
            self._buffer = np.empty(total_size, dtype=self._dtype)

        # Bind buffers to parameters
        i0 = 0
        for p in self._params:
            p.bind(i0, self._buffer)
            i0 += p.size
    
    def randomize(self, seed: int | None = None) -> None:
        """Randomize the parameters of the model.

        Args:
            seed (int | None, optional): The random seed. Defaults to None.
        """
        if not self.bound:
            self.bind()
        rng = np.random.default_rng(seed)
        if self._buffer.dtype.kind == "f":
            self._buffer[:] = rng.standard_normal(self._buffer.shape, dtype=self._buffer.dtype)
        elif self._buffer.dtype.kind == "c":
            ftype = {
                "complex64": np.float32,
                "complex128": np.float64,
                "complex256": np.float128,
            }[self._buffer.dtype.name]
            re = rng.standard_normal(self._buffer.shape, dtype=ftype)
            im = rng.standard_normal(self._buffer.shape, dtype=ftype)
            self._buffer[:] = re + 1.0j * im
        else:
            raise MiniMLError(f"Randomization of parameters with dtype {self._buffer.dtype} not supported")
            
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
        self._buffer[:] = buf 