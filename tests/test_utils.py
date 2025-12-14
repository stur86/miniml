from miniml.utils import ImmutableBiDict, CallablesRegistry


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

def test_callables_registry():
    # Let's use our own module
    import miniml.utils
    
    registry = CallablesRegistry([miniml.utils])
    
    # Check that some known functions are registered
    assert "miniml.utils.ImmutableBiDict" in registry.dict.keys()
    assert registry.dict.get("miniml.utils.ImmutableBiDict") == miniml.utils.ImmutableBiDict
    assert registry.dict.get_inverse(miniml.utils.CallablesRegistry) == "miniml.utils.CallablesRegistry"