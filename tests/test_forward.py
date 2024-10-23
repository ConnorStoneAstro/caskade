import torch

from caskade import Module, Param, forward, ValidContext

import pytest


def test_forward():

    class TestSim(Module):
        def __init__(self, a, b_shape, c, m1):
            super().__init__("test_sim")
            self.a = Param("a", a)
            self.b = Param("b", None, b_shape)
            self.c = Param("c", c)
            self.m1 = m1

        @forward
        def testfun(self, x, a=None, b=None, c=None):
            y = self.m1(live_c=c + x)
            return x + a + b + y.unsqueeze(-1)

    class TestSubSim(Module):
        def __init__(self, d=None, e=None, f=None):
            super().__init__()
            self.d = Param("d", d)
            self.e = Param("e", e)
            self.f = Param("f", f)

        @forward
        def __call__(self, d=None, e=None, live_c=None):
            return d + e + live_c.sum()

    sub1 = TestSubSim()
    main1 = TestSim(2.0, (2, 2), None, sub1)
    main1.c = main1.b

    # check graph generation
    graph = main1.graphviz()
    assert graph is not None, "should return a graphviz object"

    # Dont provide params
    with pytest.raises(ValueError):
        main1.testfun()

    # List as params
    params = [torch.ones((2, 2)), torch.tensor(3.0), torch.tensor(4.0), torch.tensor(1.0)]
    result = main1.testfun(1.0, params=params)
    assert result.shape == (2, 2)
    result = main1.testfun(1.0, params)
    assert result.shape == (2, 2)
    # valid context
    assert main1.from_valid(main1.to_valid(params)) == params
    with ValidContext(main1):
        valid_result = main1.testfun(1.0, params=main1.to_valid(params))
        assert valid_result.shape == (2, 2)
        assert torch.all(valid_result == result).item()
    # Wrong number of params, too few
    with pytest.raises(AssertionError):
        result = main1.testfun(1.0, params=params[:3])
    with pytest.raises(AssertionError):
        main1.to_valid(params[:3])
    with pytest.raises(AssertionError):
        main1.from_valid(params[:3])
    # Wrong number of params, too many
    badparams = params + params + params
    with pytest.raises(AssertionError):
        result = main1.testfun(1.0, params=badparams)
    with pytest.raises(AssertionError):
        main1.to_valid(badparams)
    with pytest.raises(AssertionError):
        main1.from_valid(badparams)

    # List by children
    params = [torch.ones((2, 2)).flatten(), torch.tensor([3.0, 4.0, 1.0])]
    result = main1.testfun(1.0, params=params)
    assert result.shape == (2, 2)
    result = main1.testfun(1.0, params)
    assert result.shape == (2, 2)
    # valid context
    for param1, param2 in zip(main1.from_valid(main1.to_valid(params)), params):
        assert torch.all(param1 == param2).item()
    with ValidContext(main1):
        valid_result = main1.testfun(1.0, params=main1.to_valid(params))
        assert valid_result.shape == (2, 2)
        assert torch.all(valid_result == result).item()

    # Tensor as params
    params = torch.cat(tuple(p.flatten() for p in params))
    result = main1.testfun(1.0, params=params)
    assert result.shape == (2, 2)
    result = main1.testfun(1.0, params)
    assert result.shape == (2, 2)
    # valid context
    assert torch.all(main1.from_valid(main1.to_valid(params)) == params).item()
    with ValidContext(main1):
        valid_result = main1.testfun(1.0, params=main1.to_valid(params))
        assert valid_result.shape == (2, 2)
        assert torch.all(valid_result == result).item()
    # Wrong number of params, too few
    with pytest.raises(AssertionError):
        result = main1.testfun(1.0, params[:-3])
    # Wrong number of params, too many
    with pytest.raises(AssertionError):
        result = main1.testfun(1.0, torch.cat((params, params)))

    # Batched tensor as params
    params = params.repeat(3, 1).unsqueeze(1)
    result = main1.testfun(torch.tensor((1.0, 1.0)), params=params)
    assert result.shape == (3, 3, 2, 2)
    result = main1.testfun(torch.tensor((1.0, 1.0)), params)
    assert result.shape == (3, 3, 2, 2)
    # valid context
    assert torch.all(main1.from_valid(main1.to_valid(params)) == params).item()
    with ValidContext(main1):
        valid_result = main1.testfun(1.0, params=main1.to_valid(params))
        assert valid_result.shape == (3, 3, 2, 2)
        assert torch.all(valid_result == result).item()

    # Dict as params, sub element is tensor
    params = {"b": torch.ones((2, 2)), "TestSubSim": torch.tensor((3.0, 4.0, 1.0))}
    result = main1.testfun(1.0, params=params)
    assert result.shape == (2, 2)
    result = main1.testfun(1.0, params)
    assert result.shape == (2, 2)
    # valid context
    reparam = main1.from_valid(main1.to_valid(params))
    for key in params:
        assert torch.all(reparam[key] == params[key]).item()
    with ValidContext(main1):
        valid_result = main1.testfun(1.0, params=main1.to_valid(params))
        assert valid_result.shape == (2, 2)
        assert torch.all(valid_result == result).item()
    # Wrong name for params
    params = {"q": torch.ones((2, 2)), "TestSubSim": torch.tensor((3.0, 4.0, 1.0))}
    with pytest.raises(ValueError):
        result = main1.testfun(1.0, params=params)
    with pytest.raises(ValueError):
        main1.to_valid(params)
    with pytest.raises(ValueError):
        main1.from_valid(params)

    # Dict as params, sub element is list
    params = {
        "b": torch.ones((2, 2)),
        "TestSubSim": [torch.tensor(3.0), torch.tensor(4.0), torch.tensor(1.0)],
    }
    result = main1.testfun(1.0, params=params)
    assert result.shape == (2, 2)
    result = main1.testfun(1.0, params)
    assert result.shape == (2, 2)
    # valid context
    with ValidContext(main1):
        valid_result = main1.testfun(1.0, params=main1.to_valid(params))
        assert valid_result.shape == (2, 2)
        assert torch.all(valid_result == result).item()

    # Dict as params, sub element is dict
    params = {
        "b": torch.ones((2, 2)),
        "TestSubSim": {"d": torch.tensor(3.0), "e": torch.tensor(4.0), "f": torch.tensor(1.0)},
    }
    result = main1.testfun(1.0, params=params)
    assert result.shape == (2, 2)
    result = main1.testfun(1.0, params)
    assert result.shape == (2, 2)
    # valid context
    with ValidContext(main1):
        valid_result = main1.testfun(1.0, params=main1.to_valid(params))
        assert valid_result.shape == (2, 2)
        assert torch.all(valid_result == result).item()

    # All params static
    main1.b = torch.ones((2, 2))
    sub1.d = torch.tensor(3.0)
    sub1.e = torch.tensor(4.0)
    sub1.f = torch.tensor(1.0)
    result = main1.testfun(1.0)
    assert result.shape == (2, 2)

    # dynamic with no shape
    main1.b = None
    main1.b.shape = None
    with pytest.raises(ValueError):
        main1.testfun(1.0, params=torch.ones(4))
    result = main1.testfun(1.0, params=[torch.ones((2, 2))])
    assert result.shape == (2, 2)

    # wrong number of params
    with pytest.raises(AssertionError):
        main1.testfun(1.0, params=[torch.ones((2, 2)), torch.tensor(3.0)])

    # wrong parameter type
    with pytest.raises(ValueError):
        main1.testfun(1.0, params=None)
    with pytest.raises(ValueError):
        main1.to_valid(None)
    with pytest.raises(ValueError):
        main1.from_valid(None)

    # param key doesn't exist
    with pytest.raises(ValueError):
        main1.testfun(1.0, params={"q": torch.ones((2, 2))})
