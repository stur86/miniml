from typing import Generic, TypeVar

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
