from caskade import Module, Param, forward, ActiveContext, ValidContext
import torch


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

    params1 = torch.tensor([2.0, 3.0])
    params2 = torch.tensor([4.0, 5.0])
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
