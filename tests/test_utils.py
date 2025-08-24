from miniml.utils import ImmutableBiDict


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