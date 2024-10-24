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


def test_module_del():
    m1 = Module("deltest1")

    def f():
        m2 = Module("deltest2")
        Module._module_names.remove("deltest2")
        print(m2)

    f()

    def g():
        m3 = Module("deltest3")
        print(m3)

    g()
    assert "deltest3" not in Module._module_names
    assert "deltest2" not in Module._module_names
    assert "deltest1" in Module._module_names
    assert m1.name == "deltest1"
