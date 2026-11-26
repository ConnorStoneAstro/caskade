"""How an ``@forward`` method is given its values.

A call with no arguments at all uses the current values of the params, and is
vectorised when they hold a batch of simulations. So is a call that is given
nothing but params. Any other call hands the method exactly what it was given. ``params`` and ``static_params`` start a
simulation with those values, also from inside a running simulation.
"""

import numpy as np
import pytest

from caskade import (
    FillParamsError,
    Module,
    Param,
    ParamConfigurationError,
    ValidContext,
    active_cache,
    backend,
    forward,
)

X = np.arange(4.0)


def x():
    return backend.as_array(X)


def numpy(value):
    return backend.to_numpy(value)


def each(func, values):
    """``func`` for every entry along the first dimension of ``values``, this
    is ``vmap(func)(values)`` (a loop with numpy)."""
    return backend.vmap(lambda v: func(v[0]), [0])([values])


class Line(Module):
    def __init__(self, name="line"):
        super().__init__(name)
        self.m = Param("m", shape=(), dynamic=True)
        self.b = Param("b", shape=(), dynamic=True)
        self.c = Param("c", 0.5, shape=())
        self.x = x()

    @forward
    def total(self, m, b, c):
        return (m * self.x + b).sum() + c  # one number for one line

    @forward
    def total_at(self, x, m, b, c):
        return (m * x + b).sum() + c

    @forward
    def parts(self, m, b):
        x = self.x
        return {"y": m * x + b, "sums": ((m * x).sum(), b + 0 * x)}

    @forward
    def m_total(self, m, scale=1.0):
        return scale * m.sum()

    @forward
    def broken(self, m):
        raise RuntimeError("stop")


def expect_total(m, b, c=0.5):
    return 6.0 * np.asarray(m) + 4.0 * np.asarray(b) + c


def test_single_simulation_is_called_as_before():
    line = Line()
    line.m, line.b = 2.0, 1.0
    assert line._batched_values() is None
    assert np.allclose(numpy(line.total()), 16.5)


@pytest.mark.parametrize(
    "m, b",
    [
        ([1.0, 2.0, 3.0], [0.0, 10.0, 20.0]),  # every param has the batch
        ([1.0, 2.0, 3.0], 10.0),  # one value shared by the batch
        (np.arange(6.0).reshape(2, 3), [0.0, 10.0, 20.0]),  # two batch dimensions
        (np.arange(6.0).reshape(2, 3), np.ones((2, 1))),  # broadcast along a dimension
        (np.arange(6.0).reshape(1, 2, 3), 1.0),  # three batch dimensions
    ],
)
def test_batch_of_values_gives_batch_of_results(m, b):
    line = Line()
    line.m, line.b = m, b
    expect = expect_total(*np.broadcast_arrays(m, b))

    result = line.total()
    assert tuple(result.shape) == expect.shape
    assert np.allclose(numpy(result), expect)

    # The same as the simulations run one at a time with params given
    flat = numpy(line.get_values()).reshape(-1, 2)
    each = [numpy(line.total(backend.as_array(p))) for p in flat]
    assert np.allclose(numpy(result).reshape(-1), each)

    # Nothing is left behind
    assert not line.active
    assert line.m._active_value is None and line.b._active_value is None
    assert np.allclose(numpy(line.m.value), m)


def test_results_keep_their_structure():
    line = Line()
    line.m, line.b = [1.0, 2.0, 3.0], [0.0, 10.0, 20.0]
    result = line.parts()
    assert tuple(result["y"].shape) == (3, 4)
    assert np.allclose(numpy(result["y"]), np.outer([1, 2, 3], X) + np.array([[0], [10], [20]]))
    sums, b = result["sums"]
    assert np.allclose(numpy(sums), [6.0, 12.0, 18.0])
    assert tuple(b.shape) == (3, 4)


def test_a_call_with_any_other_argument_is_not_vectorised():
    line = Line()
    line.m, line.b = [1.0, 2.0, 3.0], 0.0
    assert np.allclose(numpy(line.m_total()), [1.0, 2.0, 3.0])
    # There is no telling what should be mapped, so the method gets the values
    # as they are, the current ones or those that are passed in
    assert np.allclose(numpy(line.m_total(scale=2.0)), 12.0)
    m, b = backend.as_array([1.0, 2.0, 3.0]), backend.as_array(0.0)
    assert np.allclose(numpy(line.m_total([m, b], scale=2.0)), 12.0)
    assert np.allclose(numpy(line.m_total(params=[m, b], scale=2.0)), 12.0)
    static = line.get_values(dynamic=False)
    assert np.allclose(numpy(line.m_total(static_params=static, scale=2.0)), 12.0)
    batch = backend.as_array(np.array([[1.0, 0.0], [2.0, 0.0], [3.0, 0.0]]))
    assert np.allclose(numpy(line.m_total(batch, scale=2.0)), 12.0)


M, B, C = [1.0, 2.0, 3.0], [0.0, 10.0, 20.0], [0.0, 100.0, 200.0]


@pytest.mark.parametrize("scheme", ["array", "list", "dict"])
def test_a_call_with_nothing_but_params_is_vectorised(scheme):
    line = Line()
    line.m, line.b = M, B
    expect = expect_total(M, B)
    params = line.get_values(scheme)
    line.m, line.b = 7.0, 7.0  # the stored values play no part

    # The same as storing the values and making the call with no arguments
    for result in (line.total(params), line.total(params=params)):
        assert tuple(result.shape) == (3,)
        assert np.allclose(numpy(result), expect)
    assert not line.active and line.m._active_value is None
    assert np.allclose(numpy(line.m.value), 7.0)

    # Values without a batch are run as before
    line.m, line.b = 2.0, 1.0
    assert np.allclose(numpy(line.total(line.get_values(scheme))), 16.5)

    # The static values are part of the batch, given or stored
    line.c = C
    static = line.get_values(scheme, dynamic=False)
    line.c = 0.5
    result = line.total(params, static_params=static)
    assert np.allclose(numpy(result), expect_total(M, B, C))
    result = line.total(line.get_values(scheme), static_params=static)
    assert np.allclose(numpy(result), expect_total(2.0, 1.0, C))
    assert np.allclose(numpy(line.total(static_params=static)), expect_total(2.0, 1.0, C))
    line.c = C
    assert np.allclose(numpy(line.total(line.get_values(scheme))), expect_total(2.0, 1.0, C))
    assert np.allclose(numpy(line.total(params)), expect_total(M, B, C))


def test_params_with_more_than_one_batch_dimension():
    line = Line()
    m = np.arange(6.0).reshape(2, 3)
    line.m, line.b = m, B
    params = line.get_values()
    assert tuple(params.shape) == (2, 3, 2)
    result = line.total(params)
    assert tuple(result.shape) == (2, 3)
    assert np.allclose(numpy(result), expect_total(m, B))
    assert np.allclose(numpy(result), numpy(line.total()))


def test_params_in_a_valid_context_and_in_groups():
    line = Line()
    line.m, line.b = M, B
    line.m.valid = (0.0, None)
    expect = expect_total(M, B)
    with ValidContext(line):
        params = line.get_values()
        assert not np.allclose(numpy(params)[:, 0], M)  # the logarithm
        assert np.allclose(numpy(line.total(params)), expect)
    line.b.group = 1
    params = line.get_values()
    assert len(params) == 2 and tuple(params[0].shape) == (3, 1)
    assert np.allclose(numpy(line.total(params)), expect)
    assert np.allclose(numpy(line.total(params=line.get_values("list"))), expect)


def test_params_that_do_not_go_together():
    line = Line()
    line.m, line.b = 1.0, 1.0
    params = [backend.as_array(M), backend.as_array([0.0, 10.0])]
    with pytest.raises(ParamConfigurationError, match=r"m: \(3,\), b: \(2,\)"):
        line.total(params)
    assert not line.active and line.m._active_value is None


def test_empty_params_for_a_module_with_only_static_params():
    line = Line()
    line.m, line.b = 2.0, 1.0
    line.to_static()
    line.c = C
    expect = expect_total(2.0, 1.0, C)
    # Code written for any module passes the params along, also when there are none
    assert np.allclose(numpy(line.total(backend.make_array([]))), expect)
    assert np.allclose(numpy(line.total([])), expect)
    assert np.allclose(numpy(line.total(params=[])), expect)


def test_batches_that_do_not_go_together():
    line = Line()
    line.m, line.b = [1.0, 2.0, 3.0], [0.0, 10.0]
    with pytest.raises(ParamConfigurationError, match=r"m: \(3,\), b: \(2,\)"):
        line.total()
    assert not line.active


def test_missing_value_is_reported_as_before():
    line = Line()
    line.m = [1.0, 2.0, 3.0]
    with pytest.raises(FillParamsError, match="has no value"):
        line.total()
    assert not line.active


def test_param_without_a_declared_shape_has_no_batch():
    class Free(Module):
        def __init__(self):
            super().__init__("free")
            self.table = Param("table", np.ones((3, 2)), dynamic=True)  # any shape

        @forward
        def total(self, table):
            return table.sum()

    assert np.allclose(numpy(Free().total()), 6.0)


def test_state_is_restored_after_an_error():
    line = Line()
    line.m, line.b = [1.0, 2.0, 3.0], 0.0
    with pytest.raises(RuntimeError, match="stop"):
        line.broken()
    assert not line.active
    assert line.m._active_value is None


def test_valid_context_and_groups():
    line = Line()
    line.m, line.b = [1.0, 2.0, 3.0], [0.0, 10.0, 20.0]
    line.m.valid = (0.0, 10.0)
    expect = expect_total([1.0, 2.0, 3.0], [0.0, 10.0, 20.0])
    with ValidContext(line):  # the stored values are used directly
        assert np.allclose(numpy(line.total()), expect)
    line.b.group = 1
    assert np.allclose(numpy(line.total()), expect)


def test_gradients_reach_the_stored_values():
    if backend.backend != "torch":
        pytest.skip("stored values only carry gradients with torch")
    line = Line()
    line.m, line.b = [1.0, 2.0, 3.0], 1.0
    line.m.value.requires_grad_(True)
    line.total().sum().backward()
    assert np.allclose(numpy(line.m.value.grad), 6.0)


# Static params
#################################################################
def test_batched_static_params_are_part_of_the_batch():
    line = Line()
    line.m, line.b = [1.0, 2.0, 3.0], 1.0
    line.c = [0.0, 100.0, 200.0]
    assert line.c.static and line.c.batch_shape == (3,)
    assert np.allclose(numpy(line.total()), expect_total([1, 2, 3], 1.0, [0, 100, 200]))

    # A batch that only the static params hold
    line.m = 2.0
    assert np.allclose(numpy(line.total()), expect_total(2.0, 1.0, [0, 100, 200]))
    line.to_static()
    assert len(line.dynamic_params) == 0
    assert np.allclose(numpy(line.total()), expect_total(2.0, 1.0, [0, 100, 200]))
    assert all(p._active_value is None for p in line.all_params)

    line.b = [0.0, 10.0]
    with pytest.raises(ParamConfigurationError, match=r"b: \(2,\), c: \(3,\)"):
        line.total()


def test_static_params_keyword():
    line = Line()
    line.m, line.b = 2.0, 1.0
    params = line.get_values()
    static = line.get_values(dynamic=False)
    assert np.allclose(numpy(static), [0.5])
    assert np.allclose(numpy(line.total(params, static_params=static)), 16.5)
    assert np.allclose(numpy(line.total(params, static_params=10 * static)), 21.0)
    assert np.allclose(
        numpy(line.total(params=params, static_params=[backend.as_array(1.5)])), 17.5
    )
    assert np.allclose(numpy(line.total(static_params={"c": backend.as_array(2.5)})), 18.5)
    # The stored value is untouched
    assert np.allclose(numpy(line.c.value), 0.5) and line.c._active_value is None

    # so the user can go over static values with every argument in their hands
    stack = backend.as_array(np.array([[0.5], [1.5], [2.5]]))
    result = each(lambda s: line.total_at(x(), params, static_params=s), stack)
    assert np.allclose(numpy(result), [16.5, 17.5, 18.5])

    # Without dynamic params
    line.to_static()
    static = line.get_values(dynamic=False)
    assert tuple(static.shape) == (3,)
    assert np.allclose(numpy(line.total_at(x(), static_params=2 * static)), 33.0)
    assert np.allclose(numpy(line.total_at(x())), 16.5)

    with ValidContext(line):  # static values take the same route as dynamic ones
        line.c.valid = (0.0, None)
        static = line.get_values(dynamic=False)
        assert not np.allclose(numpy(static)[-1], 0.5)
        assert np.allclose(numpy(line.total_at(x(), static_params=static)), 16.5)


def test_get_dims():
    line = Line()
    line.m, line.b = [1.0, 2.0, 3.0], 1.0
    assert line.get_dims() == [0, None]
    assert line.get_dims("dict") == {"m": 0, "b": None}
    assert line.get_dims("array") == 0
    assert line.get_dims(dynamic=False) == [None]
    line.m = 2.0
    assert line.get_dims() == [None, None] and line.get_dims("array") is None
    line.b.group = 1
    line.b = [1.0, 2.0]
    assert line.get_dims() == [[None], [0]]
    assert line.get_dims("array") == [None, 0]
    assert line.get_dims(group=1) == [0]


# Starting a simulation
#################################################################
def test_param_given_by_keyword_to_a_call_that_starts_a_simulation():
    line = Line()
    line.m, line.b = 2.0, 1.0
    for call in (
        lambda: line.total(m=3.0),
        lambda: line.total(line.get_values(), m=3.0),
        lambda: line.total(params=line.get_values(), c=3.0),
        lambda: line.total_at(x=x(), b=3.0),
    ):
        with pytest.raises(FillParamsError, match="given by keyword"):
            call()
        assert not line.active
    assert np.allclose(numpy(line.m_total(scale=3.0)), 6.0)  # not a param


class Pair(Module):
    """Runs ``line`` as part of its own simulation, and as simulations of its own."""

    def __init__(self, line):
        super().__init__("pair")
        self.line = line
        self.k = Param("k", 2.0, shape=(), dynamic=True)
        self.shifted = Param("shifted", lambda p: p.m.value + 100.0)
        self.shifted.link("m", line.m)
        self.calls = 0

    @active_cache
    def once(self):
        self.calls += 1
        return self.line.total()

    @forward
    def inside(self, k):
        return k * self.line.total()

    @forward
    def nested(self, k, shifted, other):
        first = self.line.total() + shifted + self.once()
        second = self.line.total(params=other)  # line as a simulation of its own
        both = self.nested_static(other)
        again = self.line.total() + self.shifted.value + self.once()
        assert self.calls == 1
        return k * first, second, both, k * again

    @forward
    def nested_static(self, other):
        c = backend.as_array(np.array([100.0]))
        return self.line.total(params=other, static_params=c), self.line.total(static_params=c)

    @forward
    def override(self, k):
        return self.line.total(m=10.0), self.line.total(params=self.line.get_values(), m=10.0)


def test_params_given_inside_a_running_simulation():
    line = Line()
    pair = Pair(line)
    line.m, line.b = 2.0, 1.0
    params = pair.get_values()
    other = backend.as_array(np.array([5.0, 7.0]))

    first, second, (both, static_only), again = pair.nested(params, other=other)
    # the simulation that was running is as it was, pointers and cached results included
    assert np.allclose(numpy(first), 2 * (16.5 + 102.0 + 16.5))
    assert np.allclose(numpy(again), numpy(first))
    assert np.allclose(numpy(second), expect_total(5.0, 7.0))
    assert np.allclose(numpy(both), expect_total(5.0, 7.0, 100.0))
    assert np.allclose(numpy(static_only), expect_total(2.0, 1.0, 100.0))
    assert not pair.active and all(p._active_value is None for p in pair.all_params)
    assert pair._active_cache == {} and line._active_cache == {}

    # a keyword replaces a value in the running simulation, not in one that is started
    with pytest.raises(FillParamsError, match="given by keyword"):
        pair.override(params)
    assert not pair.active


def test_pointers_follow_the_values_of_the_inner_simulation():
    class Shifted(Line):
        def __init__(self):
            super().__init__("shifted")
            self.m10 = Param("m10", lambda p: 10 * p.m.value)
            self.m10.link("m", self.m)

        @forward
        def top(self, m10):
            return m10

    class Holder(Module):
        def __init__(self, line):
            super().__init__("holder")
            self.line = line

        @forward
        def run(self, other):
            return self.line.top(), self.line.top(params=other), self.line.top()

    line = Shifted()
    line.m, line.b = 2.0, 1.0
    other = {"m": backend.as_array(5.0), "b": backend.as_array(7.0)}
    out = Holder(line).run(line.get_values(), other=other)
    assert [float(numpy(v)) for v in out] == [20.0, 50.0, 20.0]


# Hierarchical links
#################################################################
class Blob(Module):
    def __init__(self):
        super().__init__("blob")
        self.pos = Param("pos", shape=(2,), dynamic=True)
        self.amp = Param("amp", shape=(), dynamic=True)
        self.x = x()
        self.calls = 0

    @forward
    def flux(self, x, pos, amp):
        return amp * (x - pos[0]) * pos[1]  # one blob, the shape of x

    @active_cache
    @forward
    def profile(self, pos):
        self.calls += 1
        return (self.x - pos[0]) * pos[1]

    @forward
    def on_grid(self, amp):
        return amp * self.profile()


class Many(Module):
    def __init__(self, blob, name="many", how="array"):
        super().__init__(name)
        self.hierarchical_link("blob", blob)
        self.gain = Param("gain", shape=(), dynamic=True)
        self.how = how

    @forward
    def each(self, x):
        # one simulation of the blob for each entry along the link dimension
        if self.how == "array":
            return each(lambda p: self.blob.flux(x, params=p), self.blob.get_values())
        if self.how == "positional":
            return each(lambda p: self.blob.flux(x, p), self.blob.get_values())
        if self.how == "dims":
            values, dims = self.blob.get_values("list"), self.blob.get_dims()
            return backend.vmap(lambda v: self.blob.flux(x, params=v), dims)(values)
        if self.how == "grid":
            return each(lambda p: self.blob.on_grid(params=p), self.blob.get_values())
        if self.how == "given":  # nothing but params, the blob goes over them
            return self.blob.on_grid(params=self.blob.get_values())
        return self.blob.on_grid()  # no arguments, the blob goes over its own batch

    @forward
    def flux(self, x, gain):
        return gain * self.each(x).sum(0)


class Field(Module):
    def __init__(self, many):
        super().__init__("field")
        self.hierarchical_link("many", many)

    @forward
    def flux(self, x):
        return each(lambda p: self.many.flux(x, params=p), self.many.get_values()).sum(0)


def blob_flux(pos, amp):
    pos, amp = np.asarray(pos), np.asarray(amp)
    return amp[..., None] * (X - pos[..., 0:1]) * pos[..., 1:2]


POS = np.arange(10.0).reshape(5, 2) / 3
HOW = ["array", "positional", "dims", "grid", "given", "no arguments"]


@pytest.mark.parametrize("how", HOW)
@pytest.mark.parametrize("amp", [2.0, np.arange(5.0)])
def test_hierarchical_child_is_run_by_its_parent(amp, how):
    blob = Blob()
    many = Many(blob, how=how)
    blob.pos, blob.amp, many.gain = POS, amp, 3.0
    expect = blob_flux(POS, amp * np.ones(5))

    # From many the 5 is a link dimension, one simulation of many has no batch
    assert many._batched_values() is None
    result = many.each(x(), many.get_values())
    assert tuple(result.shape) == (5, 4)
    assert np.allclose(numpy(result), expect)
    assert np.allclose(numpy(many.flux(x(), many.get_values())), 3.0 * expect.sum(0))
    assert not many.active and blob.pos._active_value is None
    assert blob._active_cache == {}

    # The blob on its own sees the 5 as a batch
    assert np.allclose(numpy(blob.on_grid()), expect)


def test_values_shared_by_every_copy_stay_out_of_the_map():
    blob = Blob()
    many = Many(blob)
    blob.pos, blob.amp, many.gain = POS, 2.0, 3.0
    captured = {}

    class Check(Module):
        def __init__(self):
            super().__init__("check")
            self.many = many

        @forward
        def run(self):
            captured["dims"] = blob.get_dims()
            captured["array"] = tuple(blob.get_values().shape)
            captured["many"] = many.get_dims(), many.get_dims("array")

    Check().run(many.get_values())
    assert captured == {"dims": [0, None], "array": (5, 3), "many": ([None, None, None], None)}


def test_cached_results_belong_to_one_simulation():
    # Every simulation of the child that the parent starts has its own cache,
    # also when they are run one after the other
    blob = Blob()
    many = Many(blob, how="grid")
    blob.pos, blob.amp, many.gain = POS, 2.0, 3.0

    class Loop(Many):
        @forward
        def each(self, x):
            values = self.blob.get_values()
            before = self.blob.profile()  # the batch as a whole
            calls = self.blob.calls
            results = [self.blob.on_grid(params=values[i]) for i in range(5)]
            assert self.blob.calls == calls + 5  # none of them was handed a cached result
            assert self.blob.profile() is before
            return backend.module.stack(results)

    loop = Loop(blob, name="loop")
    loop.gain = 3.0
    result = loop.each(x(), loop.get_values())
    assert np.allclose(numpy(result), blob_flux(POS, 2.0 * np.ones(5)))
    assert blob._active_cache == {}


@pytest.mark.parametrize("how", HOW)
def test_batch_of_simulations_with_a_hierarchical_child(how):
    blob = Blob()
    many = Many(blob, how=how)
    pos = np.arange(60.0).reshape(6, 5, 2) / 7
    gain = np.arange(6.0) + 1
    blob.pos, blob.amp, many.gain = pos, 2.0, gain
    expect = gain[:, None] * blob_flux(pos, 2.0 * np.ones((6, 5))).sum(1)

    many.x = x()

    class Run(Module):  # a method of many without arguments
        def __init__(self):
            super().__init__("run")
            self.many = many

        @forward
        def flux(self):
            return self.many.flux(x())

    result = Run().flux()
    assert tuple(result.shape) == (6, 4)
    assert np.allclose(numpy(result), expect)
    assert all(p._active_value is None for p in many.all_params)


def test_two_levels():
    blob = Blob()
    many = Many(blob)
    field = Field(many)
    pos = np.arange(30.0).reshape(3, 5, 2) / 7
    gain = np.arange(3.0) + 1
    blob.pos, blob.amp, many.gain = pos, 2.0, gain
    expect = (gain[:, None] * blob_flux(pos, 2.0 * np.ones((3, 5))).sum(1)).sum(0)

    assert np.allclose(numpy(field.flux(x(), field.get_values())), expect)
    assert np.allclose(numpy(field.flux(x=x())), expect)

    # and a batch of fields on top
    blob.pos = np.stack([pos, 2 * pos])
    other = (gain[:, None] * blob_flux(2 * pos, 2.0 * np.ones((3, 5))).sum(1)).sum(0)
    result = each(lambda p: field.flux(x(), p), field.get_values())
    assert tuple(result.shape) == (2, 4)
    assert np.allclose(numpy(result), np.stack([expect, other]))


def test_values_of_a_running_simulation_keep_their_gradients():
    if backend.backend != "torch":
        pytest.skip("only torch cuts values off from their history")
    import torch

    blob = Blob()
    many = Many(blob)
    blob.pos, blob.amp, many.gain = POS, 2.0, 3.0
    assert not many.get_values().requires_grad
    params = many.get_values().requires_grad_(True)
    many.flux(x(), params).sum().backward()

    def by_hand(p):
        pos, amp, gain = p[:10].reshape(5, 2), p[10], p[11]
        return (gain * amp * (x() - pos[:, :1]) * pos[:, 1:]).sum()

    assert torch.allclose(params.grad, torch.func.grad(by_hand)(params.detach()))
    assert params.grad.abs().sum() > 0


def test_deprecated_arguments_for_a_hierarchical_child():
    if backend.backend == "numpy":
        pytest.skip("needs vmap")

    class Old(Module):
        def __init__(self, blob):
            super().__init__("old")
            self.hierarchical_link("blob", blob)

        @forward
        def each(self, x, blob_params, blob_dims):
            return backend.vmap(lambda v: self.blob.flux(x, v), blob_dims)(blob_params)

    blob = Blob()
    old = Old(blob)
    blob.pos, blob.amp = POS, 2.0
    with pytest.warns(DeprecationWarning, match="blob_params"):
        result = old.each(x(), old.get_values())
    assert np.allclose(numpy(result), blob_flux(POS, 2.0 * np.ones(5)))


def test_inner_method_compiled_on_its_own():
    if backend.backend != "jax":
        pytest.skip("jit is a jax feature")
    import jax

    class Inner(Line):
        traces = 0

        @forward
        def heavy(self, x, m, b, c):
            Inner.traces += 1
            return m * x + b + c

    class Outer(Module):
        def __init__(self, inner):
            super().__init__("outer")
            self.inner = inner
            # values in, result out: nothing is read from the module when it is called
            self.heavy = jax.jit(lambda x, p, s: inner.heavy(x, params=p, static_params=s))

        @forward
        def run(self, x):
            inner = self.inner
            return self.heavy(x, inner.get_values(), inner.get_values(dynamic=False))

    inner = Inner("inner")
    outer = Outer(inner)
    inner.m, inner.b = 2.0, 1.0
    result = outer.run(x(), backend.as_array(np.array([2.0, 1.0])))
    assert np.allclose(numpy(result), 2 * X + 1.5)
    assert Inner.traces == 1
    inner.c = 10.0  # a static value that changes after the method was compiled
    result = outer.run(x(), backend.as_array(np.array([5.0, 7.0])))
    assert np.allclose(numpy(result), 5 * X + 17.0)
    assert Inner.traces == 1  # the compiled function was used again
    assert not outer.active


def test_random_pointer_in_a_batch_of_simulations():
    """A pointer may draw random numbers. Every use inside one simulation gets
    the same draw, and each simulation of a batch has a draw of its own."""
    kind = backend.backend
    keys = None
    if kind == "jax":
        import jax

        keys = jax.random.split(jax.random.PRNGKey(0), 3)

        def draw(p):
            return jax.random.normal(p.key.value, (4,))

    elif kind == "torch":
        import torch

        def draw(p):
            return torch.randn(4)

    else:
        rng = np.random.default_rng(0)

        def draw(p):
            return backend.as_array(rng.normal(size=4))

    class Noisy(Module):
        def __init__(self):
            super().__init__("noisy")
            self.m = Param("m", shape=(), dynamic=True)
            self.noise = Param("noise", draw)
            if keys is not None:  # jax has no global generator, the draw follows a key
                self.key = Param("key", keys[0], shape=(2,))
                self.noise.link(self.key)

        @forward
        def noise_alone(self, noise):
            return noise

        @forward
        def difference(self, m, noise):
            return m + noise - self.noise_alone()

        @forward
        def sample(self, m, noise):
            return m + noise

    noisy = Noisy()
    m = backend.as_array(np.array([1.0, 2.0, 3.0]))
    static = {} if keys is None else {"static_params": [keys]}

    result = numpy(noisy.difference([m], **static))
    assert np.allclose(result, np.array([1.0, 2.0, 3.0])[:, None] * np.ones(4), atol=1e-6)

    noise = numpy(noisy.sample([m], **static)) - np.array([1.0, 2.0, 3.0])[:, None]
    assert noise.shape == (3, 4)
    assert not np.allclose(noise[0], noise[1])
    assert not np.allclose(noise[1], noise[2])
    assert not noisy.active
