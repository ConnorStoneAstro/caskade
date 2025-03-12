import torch

from caskade import (
    Module,
    Param,
    ActiveStateError,
    ParamConfigurationError,
    InvalidValueWarning,
    forward,
)

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


@pytest.mark.filterwarnings("ignore")
def test_dynamic_value():

    class TestSim(Module):
        def __init__(self, a, b_shape, c, m1):
            super().__init__("test_sim")
            self.a = Param("a", a)
            self.b = Param("b", None, b_shape)
            self.c = Param("c", dynamic_value=c)
            self.m1 = m1

        @forward
        def testfun(self, x, a=None, b=None, c=None):
            y = self.m1(live_c=c + x)
            return (a + b).prod() + y

    class TestSubSim(Module):
        def __init__(self, d=None, e=None, f=None):
            super().__init__()
            self.d = Param("d", dynamic_value=d)
            self.e = Param("e", e)
            self.f = Param("f", dynamic_value=f, valid=(0, 10))

        @forward
        def __call__(self, d=None, e=None, live_c=None):
            return d + e + live_c.sum()

    sub1 = TestSubSim(d=2.0, e=2.5, f=None)
    main1 = TestSim(a=1.0, b_shape=(2,), c=4.0, m1=sub1)

    assert not main1.all_dynamic_value
    main1.b = torch.tensor([1.0, 2.0])
    # Try to get auto params when not all dynamic values available
    with pytest.raises(ParamConfigurationError):
        p00 = main1.build_params_tensor()
    with pytest.raises(ParamConfigurationError):
        p00 = main1.build_params_list()
    with pytest.raises(ParamConfigurationError):
        p00 = main1.build_params_dict()
    with pytest.raises(ParamConfigurationError):
        p00 = sub1.build_params_dict()
    sub1.f.dynamic_value = 3.0
    assert main1.all_dynamic_value

    # Check dynamic value
    assert main1.c.dynamic_value.item() == 4.0
    assert main1.c.value.item() == 4.0
    assert main1.c._value is None

    # Auto tensor
    p0 = main1.build_params_tensor()
    x = torch.tensor([0.1, 0.2])
    assert p0.shape == (3,)
    assert torch.allclose(main1.testfun(x, p0), torch.tensor(18.8))
    assert torch.allclose(main1.testfun(x, p0), main1.testfun(x=x))
    p02 = p0 * 2
    main1.fill_dynamic_values(p02)
    assert torch.allclose(main1.testfun(x=x), torch.tensor(28.8))
    main1.fill_dynamic_values(p0)

    # Auto list
    p0 = main1.build_params_list()
    x = torch.tensor([0.1, 0.2])
    assert len(p0) == 3
    assert torch.allclose(main1.testfun(x, p0), torch.tensor(18.8))
    assert torch.allclose(main1.testfun(x, p0), main1.testfun(x=x))
    p02 = [p * 2 for p in p0]
    main1.fill_dynamic_values(p02)
    assert torch.allclose(main1.testfun(x=x), torch.tensor(28.8))
    main1.fill_dynamic_values(p0)

    # Auto dict
    p0 = main1.build_params_dict()
    x = torch.tensor([0.1, 0.2])
    print(p0)
    assert len(p0) == 2
    s1, s2 = p0.keys()
    assert len(p0[s1]) in [1, 2]
    assert len(p0[s2]) in [1, 2]
    assert len(p0[s1]) != len(p0[s2])
    assert torch.allclose(main1.testfun(x, p0), torch.tensor(18.8))
    assert torch.allclose(main1.testfun(x, p0), main1.testfun(x=x))
    p02 = {}
    for k in p0:
        p02[k] = {p: p0[k][p] * 2 for p in p0[k]}
    main1.fill_dynamic_values(p02)
    assert torch.allclose(main1.testfun(x=x), torch.tensor(28.8))

    # Check active state error
    with pytest.raises(ActiveStateError):
        main1.active = True
        main1.fill_dynamic_values(p0)
    main1.active = False

    # Check invalid dynamic value
    with pytest.warns(InvalidValueWarning):
        sub1.f.dynamic_value = 11.0

    # All static make params
    main1.c.to_static()
    main1.m1.d.to_static()
    main1.m1.f.to_static()
    p0 = main1.build_params_tensor()
    assert p0.shape == (0,)
    p0 = main1.build_params_list()
    assert len(p0) == 0
    p0 = main1.build_params_dict()
    assert len(p0) == 0

    # Module level to_dynamic/static
    main1.to_dynamic()
    assert main1.c.dynamic
    assert main1.m1.d.static
    main1.to_dynamic(False)
    assert main1.c.dynamic
    assert main1.m1.d.dynamic
    main1.to_static()
    assert main1.c.static
    assert main1.m1.d.dynamic
    main1.to_static(False)
    assert main1.c.static
    assert main1.m1.d.static
