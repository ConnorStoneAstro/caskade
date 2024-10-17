import torch

from caskade import Module, Param, forward, LiveParam


def test_forward():

    class TestSim(Module):
        def __init__(self, a, b_shape, c, c_shape, m1):
            super().__init__("test_sim")
            self.a = Param("a", a)
            self.b = Param("b", None, b_shape)
            self.c = Param("c", c, c_shape)
            self.m1 = m1

        @forward
        def testfun(self, x, a=None, b=None):
            self.c.value = b + x
            y = self.m1()
            return x + a + b + y

    class TestSubSim(Module):
        def __init__(self, d=None, e=None, f=None):
            super().__init__()
            self.d = Param("d", d)
            self.e = Param("e", e)
            self.f = Param("f", f)

        @forward
        def __call__(self, d=None, e=None):
            return d + e

    sub1 = TestSubSim()
    main1 = TestSim(2.0, (2, 2), LiveParam, (2,), sub1)

    # List as params
    params = [torch.ones((2, 2)), torch.tensor(3.0), torch.tensor(4.0)]
    result = main1.testfun(1.0, params=params)
    assert result.shape == (2, 2)
    result = main1.testfun(params, 1.0)
    assert result.shape == (2, 2)

    # Tensor as params
    params = torch.cat((p.flatten() for p in params))
    result = main1.testfun(1.0, params=params)
    assert result.shape == (2, 2)
    result = main1.testfun(params, 1.0)
    assert result.shape == (2, 2)

    # Batched tesnor as params
    params = params.repeat(3, 1)
    main1.batch = True
    result = main1.testfun(1.0, params=params)
    assert result.shape == (3, 2, 2)
    result = main1.testfun(params, 1.0)
    assert result.shape == (3, 2, 2)
    main1.batch = False

    # Dict as params, sub element is tensor
    params = {"b": torch.ones((2, 2)), "m1": torch.tensor((3.0, 4.0))}
    result = main1.testfun(1.0, params=params)
    assert result.shape == (2, 2)
    result = main1.testfun(params, 1.0)
    assert result.shape == (2, 2)

    # Dict as params, sub element is list
    params = {"b": torch.ones((2, 2)), "m1": [torch.tensor(3.0), torch.tensor(4.0)]}
    result = main1.testfun(1.0, params=params)
    assert result.shape == (2, 2)
    result = main1.testfun(params, 1.0)
    assert result.shape == (2, 2)

    # Dict as params, sub element is dict
    params = {"b": torch.ones((2, 2)), "m1": {"d": torch.tensor(3.0), "e": torch.tensor(4.0)}}
    result = main1.testfun(1.0, params=params)
    assert result.shape == (2, 2)
    result = main1.testfun(params, 1.0)
    assert result.shape == (2, 2)
