import os
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


def test_module_graphviz(sim):
    graph = sim.graphviz(saveto="test_graph.pdf")
    assert graph is not None, "should return a graphviz object"
    assert os.path.exists("test_graph.pdf")
    os.remove("test_graph.pdf")


def test_module_print(sim):
    result = str(sim)
    assert all(node.name in result for node in sim.topological_ordering())


def test_module_methods(sim):
    with pytest.raises(ActiveStateError):
        sim.fill_params([1.0, 2.0, 3.0])

    with ActiveContext(sim):
        with pytest.raises(ActiveStateError):
            sim.set_values(())


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


@pytest.mark.parametrize("params_type", ["array", "list", "dict"])
def test_input_methods(sim, params_type):
    p0 = sim.get_values(params_type)
    val = sim.run_sim(10, 11, p0)
    # Check value
    assert backend.module.allclose(val, backend.make_array(781))
    # Check last arg vs kwarg
    assert backend.module.allclose(val, sim.run_sim(10, 11, params=p0))
    # Check last arg vs no arg
    sim.to_static(False)
    assert backend.module.allclose(val, sim.run_sim(10, 11))
    assert backend.module.allclose(val, sim.run_sim(10, 11, ()))


def nested_double(params):
    new_params = {}
    for param in params:
        if isinstance(params[param], dict):
            new_params[param] = nested_double(params[param])
        else:
            new_params[param] = 2 * params[param]
    return new_params


# @pytest.mark.filterwarnings("ignore")
@pytest.mark.parametrize("group", [0, 1])
@pytest.mark.parametrize("params_type", ["array", "list", "dict"])
def test_get_set_values(sim, group, params_type, capsys):

    sim.helper.h1.group = group
    for i in range(5):
        sim.workers[i].w1.group = i * group

    sim.to_dynamic(False)
    sim.s1 = None
    # Try to get auto params when not all dynamic values available
    with pytest.raises(ParamConfigurationError):
        sim.get_values(params_type)
    sim.s1 = 9

    p0 = sim.get_values(params_type)
    res = sim.run_sim(10, 11, p0)
    assert res == 781

    if group == 0:  # fixme, group version test
        if params_type == "array":
            assert p0.shape == (28,)
            assert isinstance(p0, backend.array_type)
            p1 = p0 * 2
        elif params_type == "list":
            assert len(p0) == 12
            assert isinstance(p0, list)
            p1 = list(p * 2 for p in p0)
        else:
            assert isinstance(p0, dict)
            p0str = str(p0)
            assert all(p.name in p0str for p in sim.dynamic_params)
            p1 = nested_double(p0)

        sim.set_values(p1)
        p = sim.get_values()
        assert backend.module.allclose(sim.run_sim(20, 22, p), 2 * res)

    sim.set_values(p0)
    p = sim.get_values()
    assert backend.module.allclose(sim.run_sim(10, 11, p), res)

    sim.to_static(False)
    assert backend.module.allclose(sim.run_sim(10, 11), res)
    assert len(sim.get_values(params_type)) == 0

    sim.helper.to_dynamic()
    p = sim.get_values()
    assert backend.module.allclose(sim.run_sim(10, 11, p), res)

    # Ensure no spurious output
    captured = capsys.readouterr()
    assert captured.out == ""


def test_module_graph_tracking(sim):
    sim.to_dynamic(False)
    assert len(sim.dynamic_params) == 12
    assert len(sim.static_params) == 0
    assert len(sim.pointer_params) == 1


def test_batched_build_params_array(sim):
    sim.to_dynamic(False)
    sim.helper.h2 = np.ones((4, 2))  # batch_shape = (4,)
    vals = sim.get_values()
    assert vals.shape == (4, 28)

    res = sim.run_sim(10, 11, vals[0])

    if backend.backend == "torch":
        assert backend.module.allclose(
            res, backend.module.vmap(sim.run_sim, in_dims=(None, None, 0))(10, 11, vals)
        )

    # Missmatched batch shapes
    sim.workers[4].w1 = np.ones((5, 5))
    with pytest.raises((RuntimeError, TypeError, ValueError)):
        sim.get_values("array")

    # Multi-dim batching
    sim.workers[4].w1 = np.ones((5, 4))
    vals = sim.get_values()
    assert vals.shape == (5, 4, 28)


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
        "lp": {"a": 2.0, "b": 3.0, "c": 4.0, "S": [5.0, 5.5, 5.75], "N": {"c": 6.0}},
    }

    with ActiveContext(M):
        M.fill_params(params)

    assert not M.static
    assert N.dynamic
    assert not N.static


@pytest.mark.parametrize("group", [0, 1])
@pytest.mark.parametrize("params_type", ["array", "list", "dict"])
def test_valid(sim, params_type, group):
    sim.to_dynamic(False)

    sim.helper.h1.group = group
    for i in range(5):
        sim.workers[i].w1.group = i * group

    init_params = sim.get_values()
    with ValidContext(sim):
        params = sim.get_values(params_type)
        sim.set_values(params)

    if group == 0:
        assert backend.module.allclose(init_params, sim.get_values())
    else:
        assert len(sim.dynamic_param_groups) > 1
        final_params = sim.get_values()
        for i in range(len(sim.dynamic_param_groups)):
            assert backend.module.allclose(init_params[i], final_params[i])
