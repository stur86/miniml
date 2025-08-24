import numpy as np
from pathlib import Path
from numpy.typing import NDArray
from numpy.typing import DTypeLike
from miniml.param import MiniMLParam, MiniMLError

_save_type_keys = {
    np.float32: "float32",
    np.float64: "float64",
    np.int32: "int32",
    np.int64: "int64",
    np.uint32: "uint32",
    np.uint64: "uint64",
    np.bool_: "bool",
    np.complex64: "complex64",
    np.complex128: "complex128",
}

_load_type_keys = {
    v: k for k, v in _save_type_keys.items()
}

class MiniMLModel:
    
    _buffers: dict[DTypeLike, NDArray]
    _params: dict[DTypeLike, list[MiniMLParam]]
    
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

        self._params = {}
        for _, v in pfound:
            try:
                self._params.setdefault(v.dtype, []).append(v)
            except TypeError:
                raise MiniMLError("Parameters with a non-hashable dtype are not supported")
            
        # Now merge in all parameters from the child models
        for k, m in mfound:
            try:
                mp = m._params
            except AttributeError:
                raise MiniMLError(f"Child model {k} was not properly initialized; remember to call super().__init__() at the end of the constructor")
            for dt, plist in mp.items():
                self._params.setdefault(dt, []).extend(plist)
                
    @property
    def bound(self) -> bool:
        return hasattr(self, "_buffers")
    
    @property
    def ready(self) -> bool:
        return hasattr(self, "_params") and self.bound

    def bind(self) -> None:
        if not hasattr(self, "_params"):
            raise MiniMLError("Model parameters have not been initialized; remember to call super().__init__() at the end of the constructor")

        if not self.bound:
            # Calculate total size
            total_sizes = {dt: sum(p.size for p in plist) for dt, plist in self._params.items()}
            # Initialize buffers
            self._buffers = {dt: np.empty((sz,), dtype=dt) for dt, sz in total_sizes.items()}

        # Bind buffers to parameters
        for dt, plist in self._params.items():
            i0 = 0
            for p in plist:
                p.bind(i0, self._buffers[dt])
                i0 += p.size
    
    def randomize(self, seed: int | None = None) -> None:
        """Randomize the parameters of the model.

        Args:
            seed (int | None, optional): The random seed. Defaults to None.
        """
        if not self.bound:
            self.bind()
        rng = np.random.default_rng(seed)
        for buf in self._buffers.values():
            if buf.dtype.kind == "f":
                buf[:] = rng.standard_normal(buf.shape, dtype=buf.dtype)
            elif buf.dtype.kind in ("i", "u"):
                buf[:] = rng.integers(0, 100, size=buf.shape, dtype=buf.dtype)
            elif buf.dtype.kind == "b":
                buf[:] = rng.integers(0, 2, size=buf.shape, dtype=buf.dtype)
            elif buf.dtype.kind == "c":
                ftype = np.float32 if buf.dtype == np.complex64 else np.float64
                buf[:] = rng.standard_normal(buf.shape, dtype=ftype) + 1.0j * rng.standard_normal(buf.shape, dtype=ftype)
            else:
                raise MiniMLError(f"Randomization of parameters with dtype {buf.dtype} not supported")
            
    def save(self, filename: str | Path) -> None:
        """Save the model parameters to a file.

        Args:
            filename (str | Path): The file name.
        """
        if not self.bound:
            raise MiniMLError("Model parameters have not been bound to buffers; can not save")
        
        save_dict: dict[str, NDArray] = {}
        for dt, buf in self._buffers.items():
            try:
                k = _save_type_keys[dt]
            except KeyError:
                raise MiniMLError(f"Parameters with dtype {dt} can not be saved")
            if k in save_dict:
                raise MiniMLError(f"Multiple parameter buffers with dtype {dt} can not be saved")
            save_dict[k] = buf

        np.savez_compressed(filename, **save_dict) # type: ignore
        
    def load(self, filename: str | Path) -> None:
        """Load the model parameters from a file.

        Args:
            filename (str | Path): The file name.
        """
        if not self.bound:
            self.bind()
        
        load_dict = np.load(filename)
        for k, buf in load_dict.items():
            try:
                dt = _load_type_keys[k]
            except KeyError:
                raise MiniMLError(f"Parameters with dtype key {k} can not be loaded")
            try:
                mybuf = self._buffers[dt]
            except KeyError:
                raise MiniMLError(f"No parameters with dtype {dt} in model; can not load")
            if mybuf.shape != buf.shape:
                raise MiniMLError(f"Parameter buffer shape mismatch for dtype {dt}: model has {mybuf.shape}, file has {buf.shape}")
            mybuf[:] = buf
        