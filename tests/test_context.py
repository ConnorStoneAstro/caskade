from caskade import Module, Param, forward, ActiveContext, OverrideParam, backend


def test_context_in_context(sim):
    # A simulation may be started inside a running one. It starts from the
    # values of the outer simulation, which are put back when it has finished.
    params = sim.get_values()
    with ActiveContext(sim):
        sim.fill_params(params)
        held = [p._active_value for p in sim.dynamic_params]
        for module in (sim, sim.helper):
            with ActiveContext(module):
                assert module.online
                assert [p._active_value for p in sim.dynamic_params] == held
                for p in module.dynamic_params:
                    p._active_value = None
            assert sim.online and sim.helper.online
            for p, h in zip(sim.dynamic_params, held):
                assert p._active_value is h
    assert not sim.active and not sim.helper.active
    assert all(p._active_value is None for p in sim.all_params)


def test_active_context():

    class TestSim(Module):
        def __init__(self):
            super().__init__()
            self.a = Param("a", 1.0)
            self.b = Param("b")
            self.c = Param("c")

        @forward
        def testfunc(self, a, b, c):
            return a + b + c

    testsim = TestSim()
    params1 = backend.make_array([2.0, 3.0])
    params2 = backend.make_array([4.0, 5.0])
    with ActiveContext(testsim):
        assert testsim.active
        assert testsim.a.active
        assert testsim.b.active
        assert testsim.c.active
        testsim.fill_params(params1)
        assert testsim.testfunc().item() == 6.0
    assert testsim.testfunc(params2).item() == 10.0
    assert testsim.a.value.item() == 1.0
    assert testsim.b.value is None


def test_override_param():

    class TestSim(Module):
        def __init__(self):
            super().__init__()
            self.a = Param("a", 3.0)
            self.b = Param("b", lambda p: p["a"].value)
            self.b.link(self.a)
            self.c = Param("c")
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
