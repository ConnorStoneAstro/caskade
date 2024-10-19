from caskade import Module, Param


def test_module_creation():
    m1 = Module("test1")
    m2 = Module("test2")
    m1.mod = m2
    assert m1["mod"] == m2
    p1 = Param("test1")
    m2.p = p1
    assert m2["p"] == p1
    assert m1.mod.p == p1
    assert m1.dynamic_params == (p1,)
    assert m2.dynamic_params == (p1,)
