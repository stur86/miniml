from typing import Generic, TypeVar
from copy import deepcopy

KT = TypeVar("KT")
VT = TypeVar("VT")

class ImmutableBiDict(Generic[KT, VT]):

    def __init__(self, values: list[tuple[KT, VT]]) -> None:
        self._fwd: dict[KT, VT] = {}
        self._bwd: dict[VT, KT] = {}
        for k, v in values:
            if k in self._fwd:
                raise ValueError(f"Duplicate key {k} in ImmutableBiDict")
            if v in self._bwd:
                raise ValueError(f"Duplicate value {v} in ImmutableBiDict")
            self._fwd[k] = v
            self._bwd[v] = k

    def __getitem__(self, key: KT) -> VT:
        return self._fwd[key]

    def get(self, key: KT, default: VT | None = None) -> VT | None:
        return self._fwd.get(key, default)

    def get_inverse(self, key: VT, default: KT | None = None) -> KT | None:
        return self._bwd.get(key, default)
    
    def keys(self) -> list[KT]:
        return list(self._fwd.keys())
    
    def values(self) -> list[VT]:
        return list(self._fwd.values())
    
class StaticConstants(type):
    STATIC = "static"

    def __new__(cls, name, bases, attrs):
        # Create a new base metaclass with properties for each
        # variable annotated with STATIC and given a default value
        static_vars = {}
        for key, value in attrs.get("__annotations__", {}).items():
            mdata = getattr(value, "__metadata__", None)
            if mdata and StaticConstants.STATIC in mdata:
                static_vars[key] = attrs.pop(key, None)
        meta_properties = {}
        for key, val in static_vars.items():
            attrs["__annotations__"].pop(key, None)
            def func(_, v=val):
                return deepcopy(v)
            meta_properties[key] = property(func)

        new_meta = super().__new__(
            cls,
            f"{name}Meta",
            (type,),
            meta_properties,
        )
        new_class = super().__new__(new_meta, name, bases, attrs)
        return new_class
