from caskade import (
    Module,
    Param,
    forward,
    ValidContext,
    FillDynamicParamsError,
    FillDynamicParamsSequenceError,
    FillDynamicParamsMappingError,
    FillDynamicParamsArrayError,
    ParamConfigurationError,
    backend,
)

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
            return x + a + b + y[..., None]

    class TestSubSim(Module):
        def __init__(self, d=None, e=None, f=None):
            super().__init__()
            self.d = Param("d", d)
            self.e = Param("e", e)
            self.f = Param("f", f)

        @forward
        def __call__(self, d=None, e=None, live_c=None):
            return d + e + backend.sum(live_c)

    sub1 = TestSubSim()
    main1 = TestSim(2.0, (2, 2), None, sub1)
    main1.c = main1.b

    # check graph generation
    print(main1)
    graph = main1.graphviz()
    assert graph is not None, "should return a graphviz object"

    # Dont provide params
    with pytest.raises(FillDynamicParamsError):
        main1.testfun()

    if backend.backend == "object":
        return

    # List as params
    params = [
        backend.module.ones((2, 2)),
        backend.make_array(3.0),
        backend.make_array(4.0),
        backend.make_array(1.0),
    ]
    result = main1.testfun(1.0, params=params)
    assert result.shape == (2, 2)
    result = main1.testfun(1.0, params)
    assert result.shape == (2, 2)
    # valid context
    assert main1.from_valid(main1.to_valid(params)) == params
    with ValidContext(main1):
        valid_result = main1.testfun(1.0, params=main1.to_valid(params))
        assert valid_result.shape == (2, 2)
        assert backend.all(valid_result == result).item()
    # Wrong number of params, too few
    with pytest.raises(FillDynamicParamsError):
        result = main1.testfun(1.0, params=[])
    with pytest.raises(FillDynamicParamsSequenceError):
        result = main1.testfun(1.0, params=params[:3])
    with pytest.raises(FillDynamicParamsSequenceError):
        main1.to_valid(params[:3])
    with pytest.raises(FillDynamicParamsSequenceError):
        main1.from_valid(params[:3])
    # Wrong number of params, too many
    badparams = params + params + params
    with pytest.raises(FillDynamicParamsSequenceError):
        result = main1.testfun(1.0, params=badparams)
    with pytest.raises(FillDynamicParamsSequenceError):
        main1.to_valid(badparams)
    with pytest.raises(FillDynamicParamsSequenceError):
        main1.from_valid(badparams)

    # List by children
    params = [backend.module.ones((2, 2)).flatten(), backend.make_array([3.0, 4.0, 1.0])]
    result = main1.testfun(1.0, params=params)
    assert result.shape == (2, 2)
    result = main1.testfun(1.0, params)
    assert result.shape == (2, 2)
    # valid context
    for param1, param2 in zip(main1.from_valid(main1.to_valid(params)), params):
        assert backend.all(param1 == param2).item()
    with ValidContext(main1):
        valid_result = main1.testfun(1.0, params=main1.to_valid(params))
        assert valid_result.shape == (2, 2)
        assert backend.all(valid_result == result).item()

    # Tensor as params
    params = backend.concatenate(tuple(p.flatten() for p in params))
    result = main1.testfun(1.0, params=params)
    assert result.shape == (2, 2)
    result = main1.testfun(1.0, params)
    assert result.shape == (2, 2)
    # valid context
    assert backend.all(main1.from_valid(main1.to_valid(params)) == params).item()
    with ValidContext(main1):
        valid_result = main1.testfun(1.0, params=main1.to_valid(params))
        assert valid_result.shape == (2, 2)
        assert backend.all(valid_result == result).item()
    # Wrong number of params, too few
    with pytest.raises(FillDynamicParamsError):
        result = main1.testfun(1.0, backend.as_array([]))
    with pytest.raises(FillDynamicParamsArrayError):
        result = main1.testfun(1.0, params[:-3])
    # Wrong number of params, too many
    with pytest.raises(FillDynamicParamsArrayError):
        result = main1.testfun(1.0, backend.concatenate((params, params)))

    # Batched tensor as params
    params = backend.module.stack([params] * 3, axis=0)[:, None]  # shape (3, 1, nparams)
    result = main1.testfun(backend.make_array((1.0, 1.0)), params=params)
    assert result.shape == (3, 3, 2, 2)
    result = main1.testfun(backend.make_array((1.0, 1.0)), params)
    assert result.shape == (3, 3, 2, 2)
    # valid context
    assert backend.all(main1.from_valid(main1.to_valid(params)) == params).item()
    with ValidContext(main1):
        valid_result = main1.testfun(1.0, params=main1.to_valid(params))
        assert valid_result.shape == (3, 3, 2, 2)
        assert backend.all(valid_result == result).item()

    # Dict as params, sub element is tensor
    params = {"b": backend.module.ones((2, 2)), "m1": backend.make_array((3.0, 4.0, 1.0))}
    result = main1.testfun(1.0, params=params)
    assert result.shape == (2, 2)
    result = main1.testfun(1.0, params)
    assert result.shape == (2, 2)
    # valid context
    reparam = main1.from_valid(main1.to_valid(params))
    for key in params:
        assert backend.all(reparam[key] == params[key]).item()
    with ValidContext(main1):
        valid_result = main1.testfun(1.0, params=main1.to_valid(params))
        assert valid_result.shape == (2, 2)
        assert backend.all(valid_result == result).item()
    # Wrong name for params
    params = {"q": backend.module.ones((2, 2)), "m1": backend.make_array((3.0, 4.0, 1.0))}
    with pytest.raises(FillDynamicParamsMappingError):
        result = main1.testfun(1.0, params=params)
    with pytest.raises(FillDynamicParamsMappingError):
        main1.to_valid(params)
    with pytest.raises(FillDynamicParamsMappingError):
        main1.from_valid(params)

    # Dict as params, sub element is list
    params = {
        "b": backend.module.ones((2, 2)),
        "m1": [backend.make_array(3.0), backend.make_array(4.0), backend.make_array(1.0)],
    }
    result = main1.testfun(1.0, params=params)
    assert result.shape == (2, 2)
    result = main1.testfun(1.0, params)
    assert result.shape == (2, 2)
    # valid context
    with ValidContext(main1):
        valid_result = main1.testfun(1.0, params=main1.to_valid(params))
        assert valid_result.shape == (2, 2)
        assert backend.all(valid_result == result).item()

    # Dict as params, sub element is dict
    params = {
        "b": backend.module.ones((2, 2)),
        "m1": {
            "d": backend.make_array(3.0),
            "e": backend.make_array(4.0),
            # "f": backend.make_array(1.0), # missing but not needed
        },
    }
    result = main1.testfun(1.0, params=params)
    assert result.shape == (2, 2)
    result = main1.testfun(1.0, params)
    assert result.shape == (2, 2)
    # valid context
    with ValidContext(main1):
        valid_result = main1.testfun(1.0, params=main1.to_valid(params))
        assert valid_result.shape == (2, 2)
        assert backend.all(valid_result == result).item()
    # Missing param
    params = {
        # "b": backend.module.ones((2, 2)),
        "m1": {
            "d": backend.make_array(3.0),
            "e": backend.make_array(4.0),
            "f": backend.make_array(1.0),
        },
    }
    with pytest.raises(FillDynamicParamsError):
        result = main1.testfun(1.0, params=params)

    # All params static
    main1.b = backend.module.ones((2, 2))
    sub1.d = backend.make_array(3.0)
    sub1.e = backend.make_array(4.0)
    sub1.f = backend.make_array(1.0)
    result = main1.testfun(1.0)
    assert result.shape == (2, 2)
    result = main1.testfun(1.0, [])
    assert result.shape == (2, 2)

    # dynamic with no shape
    main1.b = None
    main1.b.dynamic_value = None
    main1.b.shape = None
    with pytest.raises(ParamConfigurationError):
        main1.testfun(1.0, params=backend.module.ones(4))
    result = main1.testfun(1.0, params=[backend.module.ones((2, 2))])
    assert result.shape == (2, 2)

    # wrong parameter type
    with pytest.raises(TypeError):
        main1.testfun(1.0, params=None)
    with pytest.raises(TypeError):
        main1.to_valid(None)
    with pytest.raises(TypeError):
        main1.from_valid(None)
