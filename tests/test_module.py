import numpy as np

from caskade import (
    Module,
    Param,
    NodeTuple,
    ActiveStateError,
    ActiveContext,
    ParamConfigurationError,
    InvalidValueWarning,
    forward,
    backend,
    BackendError,
    ValidContext,
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


def test_module_methods():

    m1 = Module("test1")

    with pytest.raises(ActiveStateError):
        m1.fill_params([1.0, 2.0, 3.0])

    with pytest.raises(ActiveStateError):
        m1.clear_params()


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
    if backend.backend == "object":
        return
    assert c1.big_test([backend.make_array(1.0)]).item() == 4.0, "Shared parameter not working"


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
            return backend.module.prod(a + b) + y

    class TestSubSim(Module):
        def __init__(self, d=None, e=None, f=None):
            super().__init__()
            self.d = Param("d", dynamic_value=d)
            self.e = Param("e", e)
            self.f = Param("f", dynamic_value=f, valid=(0, 10))

        @forward
        def __call__(self, d=None, e=None, live_c=None):
            return d + e + backend.sum(live_c)

    sub1 = TestSubSim(d=2.0, e=2.5, f=None)
    main1 = TestSim(a=1.0, b_shape=(2,), c=4.0, m1=sub1)

    assert not main1.all_dynamic_value
    main1.b = backend.make_array([1.0, 2.0])
    if backend.backend == "object":
        with pytest.raises(BackendError):
            main1.testfun(np.array([1.0, 2.0]), np.ones(3))
        with pytest.raises(BackendError):
            main1.build_params_array()
        x = main1.to_valid(np.array([1, 2, 3]))
        assert x[1] == 2.0
        x = main1.from_valid(x)
        assert x[1] == 2.0
        return
    # Try to get auto params when not all dynamic values available
    with pytest.raises(ParamConfigurationError):
        p00 = main1.build_params_array()
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
    p0 = main1.build_params_array()
    x = backend.make_array([0.1, 0.2])
    assert p0.shape == (3,)
    assert backend.module.allclose(main1.testfun(x, p0), backend.make_array(18.8))
    assert backend.module.allclose(main1.testfun(x, p0), main1.testfun(x=x))
    p02 = p0 * 2
    main1.fill_dynamic_values(p02)
    assert backend.module.allclose(main1.testfun(x=x), backend.make_array(28.8))
    main1.fill_dynamic_values(p0)

    # Auto list
    p0 = main1.build_params_list()
    x = backend.make_array([0.1, 0.2])
    assert len(p0) == 3
    assert backend.module.allclose(main1.testfun(x, p0), backend.make_array(18.8))
    assert backend.module.allclose(main1.testfun(x, p0), main1.testfun(x=x))
    p02 = [p * 2 for p in p0]
    main1.fill_dynamic_values(p02)
    assert backend.module.allclose(main1.testfun(x=x), backend.make_array(28.8))
    main1.fill_dynamic_values(p0)

    # Auto dict
    p0 = main1.build_params_dict()
    x = backend.make_array([0.1, 0.2])
    print(p0)
    assert len(p0) == 2
    assert p0["c"].item() == 4.0
    assert p0["m1"]["d"] == 2.0
    assert p0["m1"]["f"] == 3.0
    print(main1.m1.d.dynamic)
    assert backend.module.allclose(main1.testfun(x, p0), backend.make_array(18.8))
    assert backend.module.allclose(main1.testfun(x, p0), main1.testfun(x=x))
    p02 = {}
    p02["c"] = p0["c"] * 2
    p02["m1"] = {}
    p02["m1"]["d"] = p0["m1"]["d"] * 2
    p02["m1"]["f"] = p0["m1"]["f"] * 2
    main1.fill_dynamic_values(p02)
    assert backend.module.allclose(main1.testfun(x=x), backend.make_array(28.8))

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
    p0 = main1.build_params_array()
    assert p0.shape == (0,)
    p0 = main1.build_params_list()
    assert len(p0) == 0
    p0 = main1.build_params_dict()
    assert len(p0) == 0

    # Module level to_dynamic/static
    main1.m1.f = main1.m1.d
    main1.to_dynamic()
    assert main1.c.dynamic
    assert main1.m1.d.static
    assert main1.m1.f.pointer
    main1.to_dynamic(False)
    assert main1.c.dynamic
    assert main1.m1.d.dynamic
    assert main1.m1.f.pointer
    main1.to_static()
    assert main1.c.static
    assert main1.m1.d.dynamic
    assert main1.m1.f.pointer
    main1.to_static(False)
    assert main1.c.static
    assert main1.m1.d.static
    assert main1.m1.f.pointer


def test_batched_build_params_array():
    if backend.backend == "object":
        return
    M = Module("M")
    M.p1 = Param("p1")
    M.p2 = Param("p2")

    M.p1.dynamic_value = [1.0, 2.0]
    M.p1.shape = ()
    M.p2.dynamic_value = [3.0, 4.0]
    M.p2.shape = ()

    a = M.build_params_array()
    assert a.shape == (2, 2)

    with pytest.raises(ParamConfigurationError):
        M.p1.dynamic_value = [1.0, 2.0]
        M.p1.shape = (2,)
        M.p2.dynamic_value = [3.0, 4.0]
        M.p2.shape = ()
        M.build_params_array()
    with pytest.raises(ParamConfigurationError):
        M.p1.dynamic_value = [1.0, 2.0]
        M.p1.shape = ()
        M.p2.dynamic_value = [1.0, 2.0]
        M.p2.shape = (2,)
        M.build_params_array()


def test_module_and_collection():

    M = Module("M")
    M.p = Param("p")
    S = Module("S")
    S.p = Param("p")
    N = NodeTuple((Param("c"), S), name="N")
    M.lp = NodeTuple((Param("a"), Param("b"), Param("c"), S, N), name="lp")
    D = Module("D")
    S.d = D
    D.p = Param("p")
    D.p2 = Param("p2")

    params = {
        "p": 1.0,
        "lp": {
            "a": 2.0,
            "b": 3.0,
            "c": 4.0,
            "S": {"p": 5.0, "d": {"p": 5.5, "p2": 5.75}},
            "N": {"c": 6.0},
        },
    }

    with ActiveContext(M):
        M.fill_params(params)

    params = {
        "p": 1.0,
        "lp": {"a": 2.0, "b": 3.0, "c": 4.0, "S": [[5.0], {"p": 5.5, "p2": 5.75}], "N": {"c": 6.0}},
    }

    with ActiveContext(M):
        M.fill_params(params)

    assert not M.static
    assert N.dynamic
    assert not N.static


def test_valid():
    if backend.backend == "object":
        return
    M = Module("M")
    p1 = Param("p1", 1.0, valid=(0, None))
    M.p1 = p1
    M.p2 = Param("p2", [1.0, 1.5], valid=(None, 2))
    M.p3 = Param("p3", [[1.0, 1.1], [1.2, 1.3]], valid=(0, 2))
    M.m2 = Module("m2")
    M.m2.p1 = Param("p1", 1.0, valid=(0, None))
    M.m2.p2 = Param("p2", [1.0, 1.5], valid=(None, 2))
    M.m2.p3 = Param("p3", M.p3, valid=(0, 2))
    M.m2.m3 = Module("m3")
    M.m2.m3.p1 = Param("p1", 1.0, valid=(0, 3), cyclic=True)
    M.m2.m3.p2 = Param("p2", [1.0, 1.5], valid=(-1, 2), cyclic=True)
    M.to_dynamic(False)
    with ValidContext(M):
        # Array
        params = M.build_params_array()
        M.fill_dynamic_values(params)
        assert np.isclose(M.p1.value.item(), 1.0)
        assert np.isclose(M.p2.value[1].item(), 1.5)
        assert np.isclose(M.m2.p3.value[0][1].item(), 1.1)
        assert np.isclose(M.m2.m3.p2.value[1].item(), 1.5)

        # List
        params = M.build_params_list()
        M.fill_dynamic_values(params)
        assert np.isclose(M.p1.value.item(), 1.0)
        assert np.isclose(M.p2.value[1].item(), 1.5)
        assert np.isclose(M.m2.p3.value[0][1].item(), 1.1)
        assert np.isclose(M.m2.m3.p2.value[1].item(), 1.5)

        # Dict
        params = M.build_params_dict()
        M.fill_dynamic_values(params)
        assert np.isclose(M.p1.value.item(), 1.0)
        assert np.isclose(M.p2.value[1].item(), 1.5)
        assert np.isclose(M.m2.p3.value[0][1].item(), 1.1)
        assert np.isclose(M.m2.m3.p2.value[1].item(), 1.5)
