import pytest
from typing import Annotated
from miniml.utils import ImmutableBiDict, StaticConstants


def test_immut_bidict():
    pairs = [("one", 1), ("two", 2)]
    
    ibidict = ImmutableBiDict(pairs)
    
    assert ibidict["one"] == 1
    assert ibidict["two"] == 2
    
    assert ibidict.get("one") == 1
    assert ibidict.get("three") is None
    assert ibidict.get("three", 3) == 3
    assert ibidict.get_inverse(1) == "one"
    assert ibidict.get_inverse(3) is None

    assert sorted(ibidict.keys()) == ["one", "two"]
    assert sorted(ibidict.values()) == [1, 2]
    


def test_static_constants_metaclass() -> None:
    class TestClass(metaclass=StaticConstants):
        x: Annotated[int, StaticConstants.STATIC] = 10
        y: Annotated[str, StaticConstants.STATIC] = "hello"
        z: float = 3.14  # Not static
        v: Annotated[list, StaticConstants.STATIC] = [1, 2, 3]
        
    assert TestClass.__class__.__name__ == "TestClassMeta"

    # Static variables should be defined on the class only
    assert hasattr(TestClass, "x")
    assert hasattr(TestClass, "y")
    assert hasattr(TestClass, "z")
    assert hasattr(TestClass, "v")
    
    assert TestClass.x == 10
    assert TestClass.y == "hello"
    assert TestClass.z == 3.14
    assert TestClass.v == [1, 2, 3]
    
    obj = TestClass()
    assert not hasattr(obj, "x")
    assert not hasattr(obj, "y")
    assert hasattr(obj, "z")
    
    with pytest.raises(AttributeError):
        TestClass.x = 20
    with pytest.raises(AttributeError):
        TestClass.y = "world"
        
    # Try assigning to an element of v
    TestClass.v[0] = 10
    assert TestClass.v == [1, 2, 3]  # Original should remain unchanged