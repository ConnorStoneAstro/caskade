from caskade import Module, Param, forward, ActiveContext, OverrideParam, backend
import numpy as np


def test_active_context():

    class TestSim(Module):
        def __init__(self):
            super().__init__()
            self.a = Param("a", 1.0)
            self.b = Param("b", None)
            self.c = Param("c", None)

        @forward
        def testfunc(self, a, b, c):
            return a + b + c

    testsim = TestSim()
    if backend.backend == "object":
        return
    params1 = backend.make_array([2.0, 3.0])
    params2 = backend.make_array([4.0, 5.0])
    with ActiveContext(testsim):
        assert testsim.active
        assert testsim.a.active
        assert testsim.b.active
        assert testsim.c.active
        testsim.fill_params(params1)
        assert testsim.testfunc().item() == 6.0
        with ActiveContext(testsim):
            assert testsim.testfunc().item() == 6.0
        with ActiveContext(testsim, False):
            assert not testsim.active
            assert not testsim.a.active
            assert not testsim.b.active
            assert not testsim.c.active
            assert testsim.testfunc(params2).item() == 10.0
        assert testsim.testfunc().item() == 6.0


def test_override_param():

    if backend.backend == "object":
        return

    class TestSim(Module):
        def __init__(self):
            super().__init__()
            self.a = Param("a", 3.0)
            self.b = Param("b", lambda p: p["a"].value)
            self.b.link(self.a)
            self.c = Param("c", None)
            self.a_vals = (backend.make_array(1.0), backend.make_array(2.0))

        @forward
        def testsubfunc(self, a, b, c):
            return a + b + c

        @forward
        def testfunc(self):
            d = self.testsubfunc()
            d = d + self.testsubfunc(a=backend.make_array(4.0))
            with OverrideParam(self.a, self.a_vals[0]):
                d = d + self.b.value
            with OverrideParam(self.a, self.a_vals[1]):
                d = d + self.b.value
            return d

    testsim = TestSim()
    assert testsim.testfunc(backend.make_array([5.0])).item() == 27.0
    assert testsim.a.value.item() == 3.0
