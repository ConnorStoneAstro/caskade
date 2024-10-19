import torch

from caskade import Module, Param, forward

__all__ = ("test",)


def _test_full_integration():

    class TestSim(Module):
        def __init__(self, a, b, c, c_shape, m1):
            super().__init__("test_sim")
            self.a = a
            self.b = Param("b", b)
            self.c = Param("c", c, c_shape)
            self.m1 = m1

        @forward
        def testfun(self, x, b=None):
            y = self.m1()
            return x + self.a + b + y

    class TestSubSim(Module):
        def __init__(self, d, e, f):
            super().__init__("test_sub_sim")
            self.d = Param("d", d)
            self.e = Param("e", e)
            self.f = Param("f", f)

        @forward
        def __call__(self, d=None, e=None, f=None):
            return d + e + f

    sub1 = TestSubSim(d=1.0, e=lambda s: s.children["flink"].value, f=None)
    sub1.e.link("flink", sub1.f)
    main1 = TestSim(a=2.0, b=None, c=None, c_shape=(), m1=sub1)
    main1.c = main1.b
    sub1.f = main1.c

    b_value = torch.tensor(3.0)
    res = main1.testfun(1.0, params=[b_value])
    assert res.item() == 13.0


def test():
    _test_full_integration()
    print("Success!")
