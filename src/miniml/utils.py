import inspect
from types import ModuleType
from typing import Generic, TypeVar

KT = TypeVar("KT")
VT = TypeVar("VT")


class ImmutableBiDict(Generic[KT, VT]):
    """An immutable bidirectional dictionary."""

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
        """Get the value for the given key, or return default if the key is not found.
        
        Args:
            key (KT): The key to look up.
            default (VT | None, optional): The default value to return if the key is not found. Defaults to None.
            
        Returns:
            VT | None: The value associated with the key, or the default value.
        """
        return self._fwd.get(key, default)

    def get_inverse(self, key: VT, default: KT | None = None) -> KT | None:
        """Get the key for the given value, or return default if the value is not found.
        
        Args:
            key (VT): The value to look up.
            default (KT | None, optional): The default key to return if the value is not found. Defaults to None.
            
        Returns:
            KT | None: The key associated with the value, or the default key.
        """
        return self._bwd.get(key, default)

    def keys(self) -> list[KT]:
        """Get the list of keys.

        Returns:
            list[KT]: A list of keys in the dictionary.
        """
        return list(self._fwd.keys())

    def values(self) -> list[VT]:
        """Get the list of values.

        Returns:
            list[VT]: A list of values in the dictionary.
        """
        
        return list(self._fwd.values())
    
    def insert(self, key: KT, value: VT) -> "ImmutableBiDict[KT, VT]":
        """Return a new ImmutableBiDict with the given key-value pair added.

        Args:
            key (KT): The key to add.
            value (VT): The value to add.

        Returns:
            ImmutableBiDict[KT, VT]: A new ImmutableBiDict with the added key-value pair.

        Raises:
            ValueError: If the key or value already exists in the dictionary.
        """
        if key in self._fwd:
            raise ValueError(f"Duplicate key {key} in ImmutableBiDict")
        if value in self._bwd:
            raise ValueError(f"Duplicate value {value} in ImmutableBiDict")
        
        new_items = list(self._fwd.items()) + [(key, value)]
        return ImmutableBiDict(new_items)

class CallablesRegistry:
    """Registry for callable functions and classes in MiniML."""
    
    _bidict: ImmutableBiDict[str, object]
    
    def __init__(self, target_modules: list[ModuleType]) -> None:  
        """Initialize the CallablesRegistry.
        
        Args:
            target_modules (list[ModuleType]): List of modules to scan for callables.
        """
        callables: list[tuple[str, object]] = []
        for module in target_modules:
            prefix = module.__name__ + "."
            for name, member in inspect.getmembers(module):
                if inspect.isfunction(member) or inspect.isclass(member):
                    if inspect.getmodule(member) == module:
                        callables.append((prefix + name, member))
        self._bidict = ImmutableBiDict(callables)
        
    @property
    def dict(self) -> ImmutableBiDict[str, object]:
        """Get the bidirectional dictionary of registered callables.
        
        Returns:
            ImmutableBiDict[str, object]: The bidirectional dictionary of callables.
        """
        return self._bidict
    