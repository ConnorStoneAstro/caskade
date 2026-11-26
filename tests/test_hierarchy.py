"""Batch dimensions and state in graphs with hierarchical links.

The rule under test (see ``Node.hierarchical_link``): a module reads the batch
shape of a param as ``(batch of the module ..., link 1, ..., link d)`` where
``d`` is the fewest hierarchical links on any path from the module to the
param. The link dimensions are part of the array of the module, the dimensions
before them are a batch of whole simulations of the module.
"""

import numpy as np
import pytest

from caskade import (
    ActiveContext,
    FillParamsArrayError,
    Memo,
    Module,
    NodeList,
    Param,
    ParamConfigurationError,
    ValidContext,
    backend,
    forward,
)

N_MID, N_SUB = 2, 3  # copies of ``mid`` in ``outer``, copies of ``sub`` in each ``mid``


class Sub(Module):
    def __init__(self, name="sub"):
        super().__init__(name)
        self.s = Param("s", shape=(), dynamic=True)

    @forward
    def f(self, s):
        return s


class Mid(Module):
    def __init__(self, sub, name="mid", array_call=False):
        super().__init__(name)
        self.hierarchical_link("sub", sub)
        self.m = Param("m", shape=(), dynamic=True)
        self.array_call = array_call
        self.seen = None

    @forward
    def f(self, m):
        vmap = backend.module.vmap
        if self.array_call:  # hand ``sub`` its params as one array
            x = self.sub.get_values()
            self.seen = tuple(x.shape)
            return m + vmap(self.sub.f)(x).sum()
        sub_params, sub_dims = self.sub.get_values("list"), self.sub.get_dims()
        return m + vmap(self.sub.f, in_dims=(sub_dims,))(sub_params).sum()


class Outer(Module):
    def __init__(self, mid, name="outer", array_call=False):
        super().__init__(name)
        self.hierarchical_link("mid", mid)
        self.array_call = array_call
        self.seen = None

    @forward
    def f(self):
        vmap = backend.module.vmap
        if self.array_call:  # hand ``mid`` its params as one array
            x = self.mid.get_values()
            self.seen = tuple(x.shape)
            return vmap(self.mid.f)(x).sum()
        mid_params, mid_dims = self.mid.get_values("list"), self.mid.get_dims()
        return vmap(self.mid.f, in_dims=(mid_dims,))(mid_params).sum()


def two_levels(batch=(), outer_array=False, mid_array=False):
    """``outer -h-> mid -h-> sub``, with ``batch`` extra leading dimensions."""
    sub = Sub()
    mid = Mid(sub, array_call=mid_array)
    outer = Outer(mid, array_call=outer_array)
    sub.s = np.arange(6.0).reshape(N_MID, N_SUB) * np.ones(batch + (N_MID, N_SUB))
    mid.m = np.array([10.0, 20.0]) * np.ones(batch + (N_MID,))
    return outer, mid, sub


TOTAL = 15.0 + 30.0  # every s and every m, summed


@pytest.mark.parametrize("batch", [(), (4,), (4, 5)])
def test_batch_dims_split_from_each_module(batch):
    outer, mid, sub = two_levels(batch)
    assert sub.s.batch_shape == batch + (N_MID, N_SUB)

    # Both link dimensions are inside one simulation of outer
    assert tuple(outer.get_values().shape) == batch + (N_MID + N_MID * N_SUB,)
    # mid has one link below it, the dimension outer maps over is batch to mid
    assert tuple(mid.get_values().shape) == batch + (N_MID, 1 + N_SUB)
    # sub has no link below it, everything is batch
    assert tuple(sub.get_values().shape) == batch + (N_MID, N_SUB, 1)

    # The array of each module is taken back by that module
    for module in (outer, mid, sub):
        values = module.get_values()
        module.set_values(2 * values)
        assert np.allclose(backend.to_numpy(module.get_values()), 2 * backend.to_numpy(values))
        module.set_values(values)

    # The index of each element follows the same split
    if batch == ():
        assert outer.find_index(sub.s) == slice(0, N_MID * N_SUB)
        assert outer.find_param(N_SUB + 1) == (sub.s, (1, 1))
        assert outer.find_param(N_MID * N_SUB + 1) == (mid.m, (1,))
        assert mid.find_param(1) == (sub.s, (1,))
        assert mid.find_param(N_SUB) == (mid.m, ())
        assert sub.find_param(0) == (sub.s, ())


def test_batch_dims_do_not_depend_on_what_is_running():
    outer, mid, sub = two_levels()
    alone = {m: tuple(m.get_values().shape) for m in (outer, mid, sub)}
    for running in (outer, mid, sub):
        with ActiveContext(running):
            for module in (outer, mid, sub):
                assert tuple(module.get_values().shape) == alone[module]
    with ActiveContext(outer), ActiveContext(mid), ActiveContext(sub):
        for module in (outer, mid, sub):
            assert tuple(module.get_values().shape) == alone[module]


def test_node_shared_by_separate_graphs():
    # A node knows nothing of the graphs above it. The count of links is kept
    # by each module that looks down at it, so every graph has its own answer.
    outer, mid, sub = two_levels()  # two links above sub.s

    class Plain(Module):
        def __init__(self, sub):
            super().__init__("plain")
            self.sub = sub

    plain = Plain(sub)  # no link above sub.s
    other = Mid(sub, name="other")  # a second graph with one link above sub.s
    other.m = 5.0
    assert sub.parents == {mid, plain, other}

    expect = {outer: 2, mid: 1, other: 1, plain: 0, sub: 0}
    shapes = {
        outer: (N_MID + N_MID * N_SUB,),
        mid: (N_MID, 1 + N_SUB),
        other: (N_MID, 1 + N_SUB),
        plain: (N_MID, N_SUB, 1),
        sub: (N_MID, N_SUB, 1),
    }

    def check():
        for module, depth in expect.items():
            assert module._hierarchy_depths()[sub.s] == depth
            assert tuple(module.get_values().shape) == shapes[module]

    check()
    # Nothing changes while any one of the graphs is running
    for running in expect:
        with ActiveContext(running):
            check()
    if backend.backend == "torch":
        assert outer.f(outer.get_values()).item() == TOTAL
        each = backend.module.vmap(other.f)(other.get_values())  # one call for each row
        assert np.allclose(backend.to_numpy(each), [5.0 + 3.0, 5.0 + 12.0])

    # A new graph, or a change to one, is seen at once by the modules it concerns
    top = Outer(other, name="top")
    assert top._hierarchy_depths()[sub.s] == 2
    assert tuple(top.get_values().shape) == (1 + N_MID * N_SUB,)
    other.unlink("sub")
    other.sub = sub  # now a plain link
    assert top._hierarchy_depths()[sub.s] == 1
    assert other._hierarchy_depths()[sub.s] == 0
    assert outer._hierarchy_depths()[sub.s] == 2


@pytest.mark.parametrize("outer_array", [False, True])
@pytest.mark.parametrize("mid_array", [False, True])
@pytest.mark.parametrize("batched", [False, True])
def test_array_params_at_every_level(outer_array, mid_array, batched):
    if backend.backend != "torch":
        pytest.skip("uses torch.vmap")
    outer, mid, sub = two_levels((4,) if batched else (), outer_array, mid_array)
    params = outer.get_values()
    if batched:
        result = backend.module.vmap(outer.f)(params)
        assert np.allclose(backend.to_numpy(result), TOTAL * np.ones(4))
    else:
        assert np.allclose(backend.to_numpy(outer.f(params)), TOTAL)
    # Inside the simulation each module saw one copy of the module below it
    if outer_array:
        assert outer.seen == (N_MID, 1 + N_SUB)
    if mid_array:
        assert mid.seen == (N_SUB, 1)


def test_shared_value_below_links():
    # No batch dimensions, one value shared by every copy at every level
    outer, mid, sub = two_levels()
    sub.s = 1.0
    assert tuple(outer.get_values().shape) == (N_MID + 1,)
    assert tuple(mid.get_values().shape) == (N_MID, 2)
    outer.set_values(2 * outer.get_values())
    assert np.allclose(backend.to_numpy(sub.s.value), 2.0)
    assert np.allclose(backend.to_numpy(mid.m.value), [20.0, 40.0])


@pytest.mark.parametrize("scheme", ["array", "list", "dict"])
def test_fewer_batch_dims_than_links(scheme):
    # One batch dimension below two links could be for either link
    outer, mid, sub = two_levels()
    sub.s = np.arange(float(N_MID))
    for use in (
        outer.get_values,
        lambda: outer.set_values(backend.as_array(np.zeros(4))),
        lambda: outer.find_param(0),
        lambda: outer.to_valid(backend.as_array(np.zeros(4))),
    ):
        with pytest.raises(ParamConfigurationError, match="below 2 hierarchical links"):
            use()

    # mid on its own has one link, so one dimension is enough
    assert tuple(mid.get_values().shape) == (N_MID, 1 + N_MID)

    # Running the simulation checks it however the params are passed
    if backend.backend == "torch":
        with pytest.raises(ParamConfigurationError, match="below 2 hierarchical links"):
            outer.f(
                outer.get_values(scheme) if scheme != "array" else backend.as_array(np.zeros(4))
            )
    assert not outer.active and not sub.s.active


def test_param_shared_between_levels_takes_the_shallowest():
    outer, mid, sub = two_levels()
    shared = Param("shared", np.ones((N_MID, N_SUB)), shape=(), dynamic=True)
    sub.shared = shared
    # Only below both links, so it has a dimension for each of them
    assert outer.find_index(shared) == slice(N_MID * N_SUB, 2 * N_MID * N_SUB)
    assert tuple(mid.get_values().shape) == (N_MID, 1 + 2 * N_SUB)

    # Also used by mid itself, mid needs one value per call
    mid.shared = shared
    shared.value = np.arange(float(N_MID))
    assert mid._hierarchy_depths()[shared] == 0
    assert outer._hierarchy_depths()[shared] == 1
    assert tuple(outer.get_values().shape) == (N_MID + N_MID + N_MID * N_SUB,)
    assert tuple(mid.get_values().shape) == (N_MID, 2 + N_SUB)

    # Also used by outer, so its dimension is a batch of outer
    outer.shared = shared
    assert outer._hierarchy_depths()[shared] == 0
    assert tuple(outer.get_values().shape) == (N_MID, 1 + N_MID + N_MID * N_SUB)


def test_array_inside_mapping_is_a_piece_of_the_array_of_its_module():
    # A mapping given to outer is read by outer all the way down
    outer, mid, sub = two_levels()
    flat = outer.get_values()  # every param of outer is below mid
    expect = backend.to_numpy(flat)
    s, m = outer.find_index(sub.s), outer.find_index(mid.m)

    outer.set_values({"mid": 2 * flat})
    assert np.allclose(backend.to_numpy(outer.get_values()), 2 * expect)
    outer.set_values({"mid": {"sub": flat[s], "m": flat[m]}})
    assert np.allclose(backend.to_numpy(outer.get_values()), expect)

    # The array of mid has the link of outer as a batch dimension, outer does not
    with pytest.raises(FillParamsArrayError, match=rf"flattened params \({N_MID * (1 + N_SUB)}\)"):
        outer.set_values({"mid": mid.get_values()})

    # A mapping given to mid is read by mid
    values = mid.get_values()  # (N_MID, 1 + N_SUB)
    mid.set_values({"sub": 3 * values[..., mid.find_index(sub.s)]})
    assert np.allclose(backend.to_numpy(sub.s.value), 3 * np.arange(6.0).reshape(N_MID, N_SUB))
    outer.set_values(flat)

    # The transformation to and from valid values reads the arrays the same way
    sub.s.valid = (-100.0, 100.0)
    mid.m.valid = (-100.0, 100.0)
    with ValidContext(outer):
        valid = outer.get_values()
        outer.set_values({"mid": 0 * valid})
        outer.set_values({"mid": valid})
        assert np.allclose(backend.to_numpy(outer.get_values()), backend.to_numpy(valid))
        outer.set_values({"mid": {"sub": valid[s], "m": valid[m]}})
    assert np.allclose(backend.to_numpy(outer.get_values()), expect)

    if backend.backend == "torch":
        assert outer.f({"mid": flat}).item() == TOTAL
        assert outer.f({"mid": {"sub": flat[s], "m": flat[m]}}).item() == TOTAL


def test_array_inside_mapping_keeps_a_shared_value_shared():
    # The array of mid on its own cannot hold this: it repeats q for each copy
    outer, mid, sub = two_levels()
    mid.q = Param("q", 100.0, shape=(), dynamic=True)  # one value for every copy of mid
    assert tuple(mid.get_values().shape) == (N_MID, 2 + N_SUB)

    flat = outer.get_values()
    assert tuple(flat.shape) == (N_MID * N_SUB + N_MID + 1,)
    outer.set_values({"mid": 2 * flat})
    assert mid.q.batch_shape == ()
    assert backend.to_numpy(mid.q.value) == 200.0
    assert np.allclose(backend.to_numpy(outer.get_values()), 2 * backend.to_numpy(flat))
    if backend.backend == "torch":
        assert outer.f({"mid": flat}).item() == TOTAL


def test_array_inside_mapping_when_the_child_has_no_array_of_its_own():
    outer, mid, sub = two_levels(batch=(4,))
    g = Param("g", np.arange(4.0), shape=(), dynamic=True)
    outer.g = g  # one value for each simulation in the batch of outer
    mid.g = g  # that every copy of mid uses
    assert outer._hierarchy_depths()[g] == 0

    # Seen from mid the batch shapes (4,) and (4, N_MID) do not go together
    with pytest.raises((RuntimeError, ValueError, TypeError)):
        mid.get_values()

    flat = outer.get_values()  # every param of outer is also below mid
    assert tuple(flat.shape) == (4, N_MID * N_SUB + N_MID + 1)
    outer.set_values({"mid": 2 * flat})
    assert g.batch_shape == (4,)
    assert np.allclose(backend.to_numpy(outer.get_values()), 2 * backend.to_numpy(flat))


def test_array_for_a_collection_inside_mapping():
    class Top(Module):
        def __init__(self):
            super().__init__("top")
            self.parts = NodeList([Simple("a"), Simple("b")])
            self.c = Param("c", 5.0, shape=(), dynamic=True)

    top = Top()
    top.set_values({"parts": backend.as_array(np.arange(4.0)), "c": 7.0})
    assert np.allclose(backend.to_numpy(top.parts[0].a.value), [0.0, 1.0])
    assert np.allclose(backend.to_numpy(top.parts[1].a.value), [2.0, 3.0])
    with ValidContext(top):
        top.set_values({"parts": backend.as_array(np.arange(4.0) + 1), "c": 7.0})
    assert np.allclose(backend.to_numpy(top.parts[1].a.value), [3.0, 4.0])


class Simple(Module):
    def __init__(self, name):
        super().__init__(name)
        self.a = Param("a", [1.0, 2.0], shape=(2,), dynamic=True)

    @forward
    def total(self, a):
        return a.sum()


class Holder(Module):
    def __init__(self, name, child):
        super().__init__(name)
        self.hierarchical_link("child", child)
        self.b = Param("b", 3.0, shape=(), dynamic=True)

    @forward
    def total(self, b):
        child_params, child_dims = self.child.get_values("list"), self.child.get_dims()
        each = backend.module.vmap(self.child.total, in_dims=(child_dims,))(child_params)
        return b + each.sum()


@pytest.mark.parametrize(
    "name", ["active", "active_galaxy", "worker_active", "semi_active", "run", "user"]
)
def test_names_never_change_state(name):
    # Names that look like the words caskade uses for its own state
    module = Simple(name)
    assert not module.active and not module.online
    module.set_values(2 * module.get_values())
    assert np.allclose(backend.to_numpy(module.a.value), [2.0, 4.0])
    assert np.allclose(backend.to_numpy(module.total(module.get_values())), 6.0)

    module.a = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    top = Holder("top", module)
    assert tuple(top.get_values().shape) == (7,)
    assert top.find_param(3) == (module.a, (1, 1))
    if backend.backend == "torch":
        assert np.allclose(backend.to_numpy(top.total(top.get_values())), 24.0)
    top.set_values(top.get_values())
    assert tuple(module.a.value.shape) == (3, 2)


@pytest.mark.parametrize("text", ["active", "active learning round 1", "top_active", "a|b"])
def test_user_memo_leaves_params_writable(text):
    module = Simple("module")
    top = Holder("top", module)
    module.a = np.ones((3, 2))
    with Memo(top, text):
        module.a = 2 * np.ones((3, 2))  # not blocked as if a simulation were running
        assert tuple(top.get_values().shape) == (7,)
        top.set_values(top.get_values())
        if backend.backend == "torch":
            assert np.allclose(backend.to_numpy(top.total(top.get_values())), 15.0)
    assert not module.a.memos


def test_graphviz_boxes_nodes_at_their_shallowest_level():
    pytest.importorskip("graphviz")
    outer, mid, sub = two_levels()
    shared = Param("shared", 1.0)
    sub.shared = shared

    def box_of(dot, node):
        """Depth of the innermost box that draws the node."""
        source = dot.source
        at = source.index(f"\t{id(node)} [label=")
        return source[:at].count("subgraph") - source[:at].count("}")

    dot = outer.graphviz()
    assert [box_of(dot, n) for n in (outer, mid, sub, shared)] == [0, 1, 2, 2]

    mid.shared = shared  # now also used one level up
    assert box_of(outer.graphviz(), shared) == 1
    outer.shared = shared  # and at the top
    dot = outer.graphviz()
    assert [box_of(dot, n) for n in (outer, mid, sub, shared)] == [0, 1, 2, 0]
