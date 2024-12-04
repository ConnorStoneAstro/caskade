import torch

from caskade import Module, Param, ActiveStateError, forward

import pytest


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

    m3 = Module("test1")
    assert m3.name == "test1_0"


def test_module_methods():

    m1 = Module("test1")

    with pytest.raises(ActiveStateError):
        m1.fill_params([1.0, 2.0, 3.0])

    with pytest.raises(ActiveStateError):
        m1.clear_params()


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


def test_module_delattr():
    class TestModule(Module):
        def __init__(self, name, init_param):
            super().__init__(name)
            self.p = init_param

    initparam = Param("p")
    m = TestModule("test", initparam)
    newparam = Param("p")
    m.p = newparam
    assert m.p is initparam, "Module should not allow overwriting of parameters"
    del m.p
    m.p = newparam
    assert m.p is not initparam, "Module should allow deletion of parameters"
    assert m.p is newparam, "Module should allow setting of parameters"


def test_shared_param():

    class TestModule(Module):
        def __init__(self, name, param):
            super().__init__(name)
            self.p = param

        @forward
        def test(self, p):
            return 2 * p

    shared_param = Param("shared")

    m1 = TestModule("m1", shared_param)
    m2 = TestModule("m2", shared_param)

    class CombineModules(Module):
        def __init__(self, name, m1, m2):
            super().__init__(name)
            self.m1 = m1
            self.m2 = m2

        @forward
        def big_test(self):
            return self.m1.test() + self.m2.test()

    c1 = CombineModules("c1", m1, m2)
    assert c1.big_test([torch.tensor(1.0)]).item() == 4.0, "Shared parameter not working"
