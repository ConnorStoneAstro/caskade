"""Stress tests for parameter group functionality.

Tests parameter groups across many caskade features including hierarchical
models, batch dimensions, get/set values, finders, valid transforms,
forward decorator, active context, active cache, collections, pointer
params, dynamic/static transitions, and override params.
"""

import numpy as np
import pytest

from caskade import (
    Module,
    Param,
    forward,
    active_cache,
    ActiveContext,
    ValidContext,
    OverrideParam,
    ActiveStateError,
    ParamConfigurationError,
    NodeList,
    NodeTuple,
    backend,
)


# ──────────────────────────────────────────────────────────────────────
# Helper modules used across multiple tests
# ──────────────────────────────────────────────────────────────────────
class Inner(Module):
    def __init__(self, name=None):
        super().__init__(name)
        self.a = Param("a", 1.0, shape=())
        self.b = Param("b", [2.0, 3.0], shape=(2,))

    @forward
    def compute(self, x, a, b):
        return backend.sum(x + a + b)


class Outer(Module):
    def __init__(self, inner, name=None):
        super().__init__(name)
        self.inner = inner
        self.c = Param("c", 4.0, shape=())
        self.d = Param("d", [5.0, 6.0], shape=(2,))

    @forward
    def run(self, x, c, d):
        return backend.sum(x + c + d) + self.inner.compute(x)


def _make_grouped_outer():
    """Create an Outer model with all params dynamic and group split."""
    inner = Inner()
    outer = Outer(inner, name="outer")
    outer.to_dynamic(False)
    outer.c.group = 1
    outer.d.group = 1
    return outer


# ──────────────────────────────────────────────────────────────────────
# 1. Basic group creation and graph tracking
# ──────────────────────────────────────────────────────────────────────
class TestGroupBasics:

    def test_default_group(self):
        """All dynamic params default to group 0."""
        m = Outer(Inner())
        m.to_dynamic(False)
        assert all(p.group == 0 for p in m.dynamic_params)
        assert m.dynamic_param_groups == (0,)

    def test_assign_group_at_creation(self):
        """Groups assigned at Param creation."""
        p = Param("p", 1.0, group=2)
        assert p.group == 2

    def test_assign_group_later(self):
        """Groups can be reassigned after creation."""
        m = Outer(Inner())
        m.to_dynamic(False)
        m.c.group = 1
        assert m.c.group == 1
        assert 1 in m.dynamic_param_groups

    def test_multiple_groups(self):
        """Multiple distinct groups tracked correctly."""
        m = Outer(Inner())
        m.to_dynamic(False)
        m.inner.a.group = 0
        m.inner.b.group = 1
        m.c.group = 2
        m.d.group = 3
        assert m.dynamic_param_groups == (0, 1, 2, 3)

    def test_group_must_be_int(self):
        """Group must be an integer."""
        p = Param("p", 1.0)
        with pytest.raises(AssertionError):
            p.group = 1.5
        with pytest.raises(AssertionError):
            p.group = "bad"

    def test_reassign_group_updates_graph(self):
        """Changing group triggers update_graph on parents."""
        m = Outer(Inner())
        m.to_dynamic(False)
        assert m.dynamic_param_groups == (0,)
        m.c.group = 5
        assert 5 in m.dynamic_param_groups
        m.c.group = 0
        assert m.dynamic_param_groups == (0,)

    def test_param_order_by_group(self):
        """param_order returns params organized by group."""
        m = _make_grouped_outer()
        order = m.param_order()
        assert isinstance(order, str)
        lines = order.strip().split("\n")
        assert len(lines) == 2


# ──────────────────────────────────────────────────────────────────────
# 2. Groups in hierarchical models
# ──────────────────────────────────────────────────────────────────────
class TestGroupsHierarchical:

    def test_groups_at_top_level(self):
        """Groups assigned at top-level module propagate correctly."""
        outer = _make_grouped_outer()
        assert outer.dynamic_param_groups == (0, 1)
        group0_params = [p for p in outer.dynamic_params if p.group == 0]
        group1_params = [p for p in outer.dynamic_params if p.group == 1]
        assert len(group0_params) == 2  # inner.a, inner.b
        assert len(group1_params) == 2  # outer.c, outer.d

    def test_groups_within_hierarchy(self):
        """Groups assigned to params within sub-modules."""
        inner = Inner()
        outer = Outer(inner, name="outer")
        outer.to_dynamic(False)
        inner.a.group = 1
        assert outer.dynamic_param_groups == (0, 1)
        group1 = [p for p in outer.dynamic_params if p.group == 1]
        assert len(group1) == 1
        assert group1[0] is inner.a

    def test_groups_mixed_hierarchy(self):
        """Groups assigned across different levels of hierarchy."""
        inner = Inner()
        outer = Outer(inner, name="outer")
        outer.to_dynamic(False)
        inner.a.group = 1
        outer.c.group = 1
        assert outer.dynamic_param_groups == (0, 1)
        group0 = [p for p in outer.dynamic_params if p.group == 0]
        group1 = [p for p in outer.dynamic_params if p.group == 1]
        assert len(group0) == 2  # inner.b, outer.d
        assert len(group1) == 2  # inner.a, outer.c

    def test_deep_hierarchy(self):
        """Groups work with deeply nested modules."""

        class Deep(Module):
            def __init__(self, child=None, name=None):
                super().__init__(name)
                self.p = Param("p", 1.0, shape=())
                if child is not None:
                    self.child = child

            @forward
            def go(self, p):
                if hasattr(self, "child"):
                    return p + self.child.go()
                return p

        level3 = Deep(name="level3")
        level2 = Deep(level3, name="level2")
        level1 = Deep(level2, name="level1")
        level1.to_dynamic(False)

        level1.p.group = 0
        level2.p.group = 1
        level3.p.group = 2

        assert level1.dynamic_param_groups == (0, 1, 2)

        p0 = level1.get_values()
        assert len(p0) == 3
        result = level1.go(p0)
        assert backend.module.allclose(result, backend.make_array(3.0))


# ──────────────────────────────────────────────────────────────────────
# 3. Groups with get_values / set_values (all schemes)
# ──────────────────────────────────────────────────────────────────────
class TestGroupsGetSetValues:

    @pytest.fixture
    def grouped_model(self):
        return _make_grouped_outer()

    @pytest.mark.parametrize("scheme", ["array", "list", "dict"])
    def test_get_values_multi_group(self, grouped_model, scheme):
        """get_values returns list of per-group values when multiple groups."""
        vals = grouped_model.get_values(scheme)
        assert isinstance(vals, list)
        assert len(vals) == 2

    @pytest.mark.parametrize("scheme", ["array", "list", "dict"])
    def test_get_values_single_group(self, scheme):
        """get_values returns single value when only one group."""
        m = Outer(Inner())
        m.to_dynamic(False)
        vals = m.get_values(scheme)
        assert not isinstance(vals, list) or scheme == "list"

    def test_get_values_specific_group(self, grouped_model):
        """get_values with group= returns only that group's values."""
        v0 = grouped_model.get_values("array", group=0)
        v1 = grouped_model.get_values("array", group=1)
        assert isinstance(v0, backend.array_type)
        assert isinstance(v1, backend.array_type)
        # Group 0 has inner.a (1) + inner.b (2) = 3 elements
        assert v0.shape[-1] == 3
        # Group 1 has outer.c (1) + outer.d (2) = 3 elements
        assert v1.shape[-1] == 3

    def test_set_values_multi_group_array(self, grouped_model):
        """set_values with multi-group using array scheme."""
        vals = grouped_model.get_values("array")
        grouped_model.set_values(vals)
        vals2 = grouped_model.get_values("array")
        for v, v2 in zip(vals, vals2):
            assert backend.module.allclose(v, v2)

    def test_set_values_multi_group_list(self, grouped_model):
        """set_values with multi-group using list scheme."""
        vals = grouped_model.get_values("list")
        grouped_model.set_values(vals)
        vals2 = grouped_model.get_values("list")
        assert len(vals) == len(vals2)

    def test_set_values_multi_group_dict(self, grouped_model):
        """set_values with multi-group using dict scheme."""
        vals = grouped_model.get_values("dict")
        grouped_model.set_values(vals)
        vals2 = grouped_model.get_values("dict")
        assert len(vals) == len(vals2)

    def test_round_trip_values_multi_group(self, grouped_model):
        """Set then get values should produce identical results."""
        original = grouped_model.get_values("array")
        scaled = [v * 2 for v in original]
        grouped_model.set_values(scaled)
        retrieved = grouped_model.get_values("array")
        for s, r in zip(scaled, retrieved):
            assert backend.module.allclose(s, r)
        grouped_model.set_values(original)
        restored = grouped_model.get_values("array")
        for o, r in zip(original, restored):
            assert backend.module.allclose(o, r)


# ──────────────────────────────────────────────────────────────────────
# 4. Groups with varying batch dimensions
# ──────────────────────────────────────────────────────────────────────
class TestGroupsBatchDims:

    def test_batch_dims_independent_per_group(self):
        """Batch dimensions of separate groups are independent."""
        outer = _make_grouped_outer()

        outer.inner.a = np.ones((3,))
        outer.inner.b = np.ones((3, 2))

        vals = outer.get_values("array")
        assert len(vals) == 2
        assert vals[0].shape == (3, 3)
        assert vals[1].shape == (3,)

    def test_batch_dims_both_groups_different_sizes(self):
        """Both groups batched but with different batch sizes."""
        outer = _make_grouped_outer()

        outer.inner.a = np.ones((4,))
        outer.inner.b = np.ones((4, 2))

        outer.c = np.ones((7,))
        outer.d = np.ones((7, 2))

        vals = outer.get_values("array")
        assert len(vals) == 2
        assert vals[0].shape == (4, 3)
        assert vals[1].shape == (7, 3)

    def test_batch_dims_multi_dim_batch(self):
        """Multi-dimensional batch shapes per group."""
        outer = _make_grouped_outer()

        outer.inner.a = np.ones((2, 3))
        outer.inner.b = np.ones((2, 3, 2))

        vals = outer.get_values("array")
        assert len(vals) == 2
        assert vals[0].shape == (2, 3, 3)
        assert vals[1].shape == (3,)


# ──────────────────────────────────────────────────────────────────────
# 5. Groups with find_param / find_index
# ──────────────────────────────────────────────────────────────────────
class TestGroupsFinders:

    @pytest.fixture
    def grouped_model(self):
        return _make_grouped_outer()

    def test_find_param_with_group(self, grouped_model):
        """find_param with explicit group argument."""
        p0, idx0 = grouped_model.find_param(0, group=0)
        assert p0 is grouped_model.inner.a
        p1, idx1 = grouped_model.find_param(0, group=1)
        assert p1 is grouped_model.c

    def test_find_index_multi_group(self, grouped_model):
        """find_index returns (group, index) tuple when multiple groups."""
        idx = grouped_model.find_index(grouped_model.inner.a)
        assert isinstance(idx, tuple)
        assert idx[0] == 0  # group
        assert isinstance(idx[1], (int, slice))

    def test_find_index_different_groups(self, grouped_model):
        """find_index returns correct group for each param."""
        idx_a = grouped_model.find_index(grouped_model.inner.a)
        idx_c = grouped_model.find_index(grouped_model.c)
        assert idx_a[0] == 0
        assert idx_c[0] == 1

    def test_find_index_list_scheme(self, grouped_model):
        """find_index with list scheme returns (group, list_idx)."""
        idx = grouped_model.find_index(grouped_model.inner.a, scheme="list")
        assert isinstance(idx, tuple)
        assert idx[0] == 0

    def test_find_param_out_of_range(self, grouped_model):
        """find_param raises IndexError for out-of-range index."""
        with pytest.raises(IndexError):
            grouped_model.find_param(100, group=0)

    def test_find_index_unknown_param(self, grouped_model):
        """find_index raises ValueError for param not in model."""
        with pytest.raises(ValueError):
            grouped_model.find_index(Param("unknown", 1.0))


# ──────────────────────────────────────────────────────────────────────
# 6. Groups with to_valid / from_valid and ValidContext
# ──────────────────────────────────────────────────────────────────────
class TestGroupsValid:

    @pytest.fixture
    def bounded_grouped_model(self):
        """Model with valid bounds and multiple groups."""
        inner = Inner()
        inner.a.valid = (0, 10)
        inner.b.valid = (0, 10)
        outer = Outer(inner, name="outer")
        outer.to_dynamic(False)
        outer.c.valid = (0, 10)
        outer.d.valid = (0, 10)
        outer.c.group = 1
        outer.d.group = 1
        return outer

    def test_to_valid_multi_group(self, bounded_grouped_model):
        """to_valid returns list of valid params per group."""
        params = bounded_grouped_model.get_values()
        valid = bounded_grouped_model.to_valid(params)
        assert isinstance(valid, list)
        assert len(valid) == 2

    def test_from_valid_multi_group(self, bounded_grouped_model):
        """from_valid round-trips multi-group params."""
        params = bounded_grouped_model.get_values()
        valid = bounded_grouped_model.to_valid(params)
        recovered = bounded_grouped_model.from_valid(valid)
        for p, r in zip(params, recovered):
            assert backend.module.allclose(p, r)

    @pytest.mark.parametrize("scheme", ["array", "list", "dict"])
    def test_valid_context_multi_group(self, bounded_grouped_model, scheme):
        """ValidContext with multi-group get/set round-trips."""
        init_params = bounded_grouped_model.get_values()
        with ValidContext(bounded_grouped_model):
            params = bounded_grouped_model.get_values(scheme)
            bounded_grouped_model.set_values(params)
        final_params = bounded_grouped_model.get_values()
        for ip, fp in zip(init_params, final_params):
            assert backend.module.allclose(ip, fp)

    def test_to_valid_with_group_arg(self, bounded_grouped_model):
        """to_valid with explicit group argument."""
        params = bounded_grouped_model.get_values()
        v0 = bounded_grouped_model.to_valid(params[0], group=0)
        v1 = bounded_grouped_model.to_valid(params[1], group=1)
        r0 = bounded_grouped_model.from_valid(v0, group=0)
        r1 = bounded_grouped_model.from_valid(v1, group=1)
        assert backend.module.allclose(params[0], r0)
        assert backend.module.allclose(params[1], r1)


# ──────────────────────────────────────────────────────────────────────
# 7. Groups with @forward decorator and fill_params
# ──────────────────────────────────────────────────────────────────────
class TestGroupsForward:

    def test_forward_with_multi_group_params(self):
        """@forward method works with multi-group params passed as arg."""
        outer = _make_grouped_outer()
        params = outer.get_values()
        result = outer.run(10.0, params)
        assert result is not None

    def test_forward_with_params_kwarg(self):
        """@forward with params= kwarg and multi-group."""
        outer = _make_grouped_outer()
        params = outer.get_values()
        result = outer.run(10.0, params=params)
        assert result is not None

    def test_forward_consistency(self):
        """Results should be same whether single or multi-group (same values)."""
        inner1 = Inner()
        outer1 = Outer(inner1, name="outer1")
        outer1.to_dynamic(False)
        result_single = outer1.run(10.0, outer1.get_values())

        outer2 = _make_grouped_outer()
        result_multi = outer2.run(10.0, outer2.get_values())

        assert backend.module.allclose(result_single, result_multi)

    def test_fill_params_multi_group(self):
        """fill_params works correctly with multi-group params."""
        outer = _make_grouped_outer()
        params = outer.get_values()
        with ActiveContext(outer):
            outer.fill_params(params)
            assert outer.inner.a._value is not None
            assert outer.c._value is not None


# ──────────────────────────────────────────────────────────────────────
# 8. Groups with ActiveContext
# ──────────────────────────────────────────────────────────────────────
class TestGroupsActiveContext:

    def test_active_context_multi_group(self):
        """ActiveContext manages state correctly with multi-group."""
        outer = _make_grouped_outer()
        params = outer.get_values()
        with ActiveContext(outer):
            outer.fill_params(params)
            for p in outer.dynamic_params:
                assert p._value is not None
        for p in outer.dynamic_params:
            assert p._value is None

    def test_set_values_blocked_in_active_context(self):
        """set_values raises when module is active."""
        outer = _make_grouped_outer()
        with ActiveContext(outer):
            with pytest.raises(ActiveStateError):
                outer.set_values(outer.get_values())

    def test_nested_active_context_multi_group(self):
        """Nested ActiveContext on same module raises error."""
        outer = _make_grouped_outer()
        with ActiveContext(outer):
            with pytest.raises(ActiveStateError):
                with ActiveContext(outer):
                    pass


# ──────────────────────────────────────────────────────────────────────
# 9. Groups with active_cache
# ──────────────────────────────────────────────────────────────────────
class TestGroupsActiveCache:

    def test_active_cache_with_multi_group(self):
        """active_cache works correctly in multi-group scenarios."""

        class CachedModule(Module):
            def __init__(self):
                super().__init__()
                self.a = Param("a", 1.0, shape=())
                self.b = Param("b", 2.0, shape=())
                self.counter = 0

            @active_cache
            def cached_compute(self, x):
                self.counter += 1
                return x * 2

            @forward
            def run(self, x, a, b):
                c1 = self.cached_compute(x)
                c2 = self.cached_compute(x)  # Should use cache
                return a + b + c1 + c2

        m = CachedModule()
        m.to_dynamic(False)
        m.b.group = 1
        params = m.get_values()
        m.counter = 0
        result = m.run(3.0, params)
        assert m.counter == 1  # cached_compute called only once


# ──────────────────────────────────────────────────────────────────────
# 10. Groups with collections (NodeList / NodeTuple)
# ──────────────────────────────────────────────────────────────────────
class TestGroupsCollections:

    def test_groups_with_node_list(self):
        """Parameter groups work with NodeList containers."""

        class Listed(Module):
            def __init__(self, workers, name=None):
                super().__init__(name)
                self.workers = workers
                self.p = Param("p", 1.0, shape=())

            @forward
            def run(self, x, p):
                return p + sum(w.compute(x) for w in self.workers)

        w1 = Inner(name="w1")
        w2 = Inner(name="w2")
        listed = Listed([w1, w2], name="listed")
        listed.to_dynamic(False)
        w1.a.group = 1
        assert listed.dynamic_param_groups == (0, 1)
        params = listed.get_values()
        assert isinstance(params, list) and len(params) == 2
        result = listed.run(1.0, params)
        assert result is not None

    def test_groups_with_node_tuple(self):
        """Parameter groups work with NodeTuple containers."""
        m = Module("m")
        p1 = Param("p1", 1.0, dynamic=True, group=0)
        p2 = Param("p2", 2.0, dynamic=True, group=1)
        m.tup = NodeTuple((p1, p2), name="tup")
        assert m.dynamic_param_groups == (0, 1)

    def test_collection_dynamic_param_groups(self):
        """NodeList/NodeTuple expose dynamic_param_groups."""
        w1 = Inner(name="w1")
        w2 = Inner(name="w2")
        w1.to_dynamic(False)
        w2.to_dynamic(False)
        w1.a.group = 1
        nl = NodeList([w1, w2], name="nl")
        assert 1 in nl.dynamic_param_groups


# ──────────────────────────────────────────────────────────────────────
# 11. Groups with pointer params
# ──────────────────────────────────────────────────────────────────────
class TestGroupsPointerParams:

    def test_pointer_param_group(self):
        """Pointer and dynamic params tracked correctly with groups."""

        class PtrModule(Module):
            def __init__(self, name=None):
                super().__init__(name)
                self.a = Param("a", 1.0, shape=())
                self.b = Param("b", 2.0, shape=())

            @forward
            def run(self, a, b):
                return a + b

        m1 = PtrModule("m1")
        m2 = PtrModule("m2")

        class Top(Module):
            def __init__(self, m1, m2, name=None):
                super().__init__(name)
                self.m1 = m1
                self.m2 = m2

            @forward
            def run(self, x):
                return self.m1.run() + self.m2.run() + x

        m2.a = m1.a
        top = Top(m1, m2, name="top")
        top.to_dynamic(False)
        m1.b.group = 1

        assert m1.a in top.pointer_params or m1.a in top.dynamic_params
        params = top.get_values()
        result = top.run(0.0, params)
        assert result is not None


# ──────────────────────────────────────────────────────────────────────
# 12. Groups with dynamic/static transitions
# ──────────────────────────────────────────────────────────────────────
class TestGroupsDynamicStatic:

    def test_static_params_not_in_groups(self):
        """Static params are excluded from dynamic_param_groups."""
        outer = _make_grouped_outer()
        assert outer.dynamic_param_groups == (0, 1)
        outer.c.to_static()
        outer.d.to_static()
        assert outer.dynamic_param_groups == (0,)

    def test_to_dynamic_restores_groups(self):
        """Converting back to dynamic preserves group membership."""
        outer = _make_grouped_outer()
        outer.c.to_static()
        assert 1 in outer.dynamic_param_groups  # d still in group 1
        outer.c.to_dynamic()
        assert outer.c.group == 1

    def test_to_static_all_then_dynamic(self):
        """to_static(False) then to_dynamic restores groups."""
        inner = Inner()
        outer = Outer(inner, name="outer")
        outer.to_dynamic(False)
        inner.a.group = 1
        outer.c.group = 2

        outer.to_static(False)
        assert len(outer.dynamic_params) == 0
        assert len(outer.dynamic_param_groups) == 0

        outer.to_dynamic(False)
        assert inner.a.group == 1
        assert outer.c.group == 2
        assert outer.dynamic_param_groups == (0, 1, 2)

    def test_get_values_after_all_static(self):
        """get_values returns empty when all params static."""
        outer = _make_grouped_outer()
        outer.to_static(False)
        vals = outer.get_values()
        assert len(vals) == 0


# ──────────────────────────────────────────────────────────────────────
# 13. Groups with OverrideParam
# ──────────────────────────────────────────────────────────────────────
class TestGroupsOverrideParam:

    def test_override_in_multi_group(self):
        """OverrideParam works correctly when param is in a group."""

        class TestSim(Module):
            def __init__(self):
                super().__init__()
                self.a = Param("a", 3.0)
                self.b = Param("b", 5.0)

            @forward
            def run(self, a, b):
                return a + b

        m = TestSim()
        m.to_dynamic(False)
        m.b.group = 1

        params = m.get_values()
        with ActiveContext(m):
            m.fill_params(params)
            assert m.a._value is not None
            with OverrideParam(m.a, backend.make_array(10.0)):
                assert m.a._value.item() == 10.0
            assert m.a._value.item() == 3.0


# ──────────────────────────────────────────────────────────────────────
# 14. Edge cases
# ──────────────────────────────────────────────────────────────────────
class TestGroupsEdgeCases:

    def test_many_groups(self):
        """Large number of groups works correctly."""

        class ManyParams(Module):
            def __init__(self, n):
                super().__init__()
                for i in range(n):
                    setattr(
                        self, f"p{i}", Param(f"p{i}", float(i), shape=(), dynamic=True, group=i)
                    )

        m = ManyParams(10)
        assert m.dynamic_param_groups == tuple(range(10))
        params = m.get_values()
        assert len(params) == 10

    def test_non_contiguous_groups(self):
        """Groups with gaps (e.g. 0, 5, 10) work correctly."""
        m = Module("m")
        m.a = Param("a", 1.0, dynamic=True, group=0)
        m.b = Param("b", 2.0, dynamic=True, group=5)
        m.c = Param("c", 3.0, dynamic=True, group=10)
        assert m.dynamic_param_groups == (0, 5, 10)
        params = m.get_values()
        assert len(params) == 3

    def test_reassign_all_to_same_group(self):
        """Reassigning all params to same group collapses to single-group."""
        outer = _make_grouped_outer()
        assert len(outer.dynamic_param_groups) == 2
        outer.c.group = 0
        outer.d.group = 0
        assert outer.dynamic_param_groups == (0,)
        vals = outer.get_values("array")
        assert isinstance(vals, backend.array_type)

    def test_empty_group_after_static(self):
        """Group that becomes empty after to_static is removed from groups."""
        outer = _make_grouped_outer()
        assert 1 in outer.dynamic_param_groups
        outer.c.to_static()
        outer.d.to_static()
        assert 1 not in outer.dynamic_param_groups

    def test_single_param_per_group(self):
        """Each group with a single param."""
        m = Module("m")
        m.a = Param("a", 1.0, dynamic=True, group=0)
        m.b = Param("b", 2.0, dynamic=True, group=1)
        m.c = Param("c", 3.0, dynamic=True, group=2)
        params = m.get_values("array")
        assert len(params) == 3
        for p in params:
            assert p.shape == (1,)

    def test_forward_no_params_multi_group(self):
        """Forward with no dynamic params after all static, multi-group."""
        outer = _make_grouped_outer()
        outer.to_static(False)
        result = outer.run(10.0)
        assert result is not None

    def test_group_with_none_value_param_raises(self):
        """Param with None value in a group raises on get_values."""
        m = Module("m")
        m.a = Param("a", 1.0, dynamic=True, group=0)
        m.b = Param("b", None, dynamic=True, group=1)
        with pytest.raises(ParamConfigurationError, match="b"):
            m.get_values()


# ──────────────────────────────────────────────────────────────────────
# 15. Groups with ValidContext in @forward
# ──────────────────────────────────────────────────────────────────────
class TestGroupsValidForward:

    def test_valid_context_forward_multi_group(self):
        """ValidContext with @forward and multi-group params."""

        class BoundedSim(Module):
            def __init__(self):
                super().__init__()
                self.a = Param("a", 1.0, shape=(), valid=(0, 10))
                self.b = Param("b", 2.0, shape=(), valid=(0, 10))

            @forward
            def run(self, x, a, b):
                return x + a + b

        m = BoundedSim()
        m.to_dynamic(False)
        m.b.group = 1

        init_params = m.get_values()
        valid = m.to_valid(init_params)
        recovered = m.from_valid(valid)
        for ip, rp in zip(init_params, recovered):
            assert backend.module.allclose(ip, rp)

        with ValidContext(m):
            valid_params = m.get_values()
            result = m.run(5.0, valid_params)
        assert result is not None

    def test_valid_context_preserves_state(self):
        """ValidContext doesn't leak state after exit."""
        outer = _make_grouped_outer()
        outer.inner.a.valid = (0, 10)
        outer.inner.b.valid = (0, 10)
        outer.c.valid = (0, 10)
        outer.d.valid = (0, 10)

        assert not outer.valid_context
        with ValidContext(outer):
            assert outer.valid_context
        assert not outer.valid_context


# ──────────────────────────────────────────────────────────────────────
# 16. Groups with string representation
# ──────────────────────────────────────────────────────────────────────
class TestGroupsStr:

    def test_str_multi_group(self):
        """String representation includes all params across groups."""
        outer = _make_grouped_outer()
        result = str(outer)
        for node in outer.topological_ordering():
            assert node.name in result

    def test_param_order_str(self):
        """param_order produces readable multi-group output."""
        outer = _make_grouped_outer()
        order = outer.param_order()
        assert "c" in order
        assert "d" in order
        assert "a" in order
        assert "b" in order
