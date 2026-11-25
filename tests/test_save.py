from caskade import Module, Param, GraphError, SaveStateWarning, backend
import numpy as np
import gc
import h5py
import sys
from unittest import mock
import pytest


def _build_test_module(blank=False):
    main = Module("main")
    main.meta.extra_info = "This is some extra info"
    main.saveattrs.add("meta.extra_info")
    m1 = Module("m1")
    m2 = Module("m2")
    m3 = Module("m3")

    main.m1 = m1
    main.m2 = m2

    m1.m3 = m3
    m2.m3 = m3

    p1 = Param("p1", None if blank else 1.0, valid=(0.0, 2.0), units="arcsec")
    p2 = Param("p2", None if blank else (2.0, 2.5), shape=(2,), valid=(None, (5, 6)))
    p2.basic_data = np.array([1, 2, 3])
    p2.saveattrs.add("basic_data")
    p3 = Param(
        "p3",
        None if blank else np.ones((2, 3, 4)),
        shape=(2, 3, 4),
        cyclic=True,
        valid=(np.zeros((2, 3, 4)), 2 * np.ones((2, 3, 4))),
        units="m",
    )
    p4 = Param("p4", p2)
    p5 = Param("p5")
    p5.test_none = None
    p5.saveattrs.add("test_none")

    m1.p1 = p1
    m1.p2 = p2
    m2.p3 = p3
    m2.link("p4_param", p4)
    m2.p5_param = p5
    m3.p4 = p4
    m3.p5 = p5
    m3.link(Param("badpointer", lambda p: p.oops.value))

    return main


def _make_files_and_test(usefileobject=False):
    main = _build_test_module()

    # Save not appendable
    with pytest.warns(SaveStateWarning):
        main.m1.meta.bad_meta = lambda x: x + 1
        main.m1.saveattrs.add("meta.bad_meta")
        if usefileobject:
            with h5py.File("test_save_notappend.h5", "w") as f:
                main.save_state(f, appendable=False)
        else:
            main.save_state("test_save_notappend.h5", appendable=False)

    # bad file
    with pytest.raises(NotImplementedError):
        main.save_state("test_save_bad.png")
    with pytest.raises(NotImplementedError):
        main.load_state("test_save_bad.png")
    with pytest.raises(NotImplementedError):
        main.append_state("test_save_bad.png")

    with pytest.raises(IOError):
        main.append_state("test_save_notappend.h5")

    # Save and append
    with pytest.warns(SaveStateWarning):
        main.m1.p1.very_bad_meta = np.array(["hello", "wor\0ld"], dtype=object)
        main.m1.p1.saveattrs.add("very_bad_meta")
        if usefileobject:
            with h5py.File("test_save_append.h5", "w") as f:
                main.save_state(f, appendable=True)
        else:
            main.save_state("test_save_append.h5", appendable=True)
    main.m1.p1.value = 2.0
    main.m1.p2.value = (3.0, 3.5)
    if usefileobject:
        with h5py.File("test_save_append.h5", "a") as f:
            main.append_state(f)
            assert not hasattr(main.m1.p2, "appended")
            main.append_state(f)
    else:
        main.append_state("test_save_append.h5")
        assert not hasattr(main.m1.p2, "appended")
        main.append_state("test_save_append.h5")


def _load_not_appendable_and_test(usefileobject=False):
    gc.collect()
    main = _build_test_module(blank=True)
    main.meta.extra_info = "The wrong extra info"
    main.saveattrs.add("meta.extra_info")
    if usefileobject:
        with h5py.File("test_save_notappend.h5", "r") as f:
            main.load_state(f)
    else:
        main.load_state("test_save_notappend.h5")
    assert main.meta.extra_info == "This is some extra info"
    assert main.m1.p2.basic_data[2] == 3
    assert main.m1.p1.value.item() == 1.0
    assert main.m1.p1.units == "arcsec"
    assert main.m2.p4_param.value[1].item() == 2.5


def _load_appendable_and_test(usefileobject=False):
    gc.collect()
    main = _build_test_module(blank=True)
    if usefileobject:
        with h5py.File("test_save_append.h5", "r") as f:
            main.load_state(f)
    else:
        main.load_state("test_save_append.h5")
    assert main.m1.p1.value.item() == 2.0
    assert main.m1.p2.value[0].item() == 3.0
    assert main.m1.p2.value[1].item() == 3.5

    with h5py.File("test_save_append.h5", "r") as f:
        assert "main/m1/p1" in f
        assert "main/m1/p2" in f
        assert "main/m1/p3" not in f
        assert "main/m2/p4" not in f
        assert "main/m2/p5" not in f
        assert len(f["main/m1/p2/value"][()]) == 3


def _change_graph_fail_test():
    gc.collect()
    main = _build_test_module()
    main.link(Module("bad"))
    with pytest.raises(GraphError):
        main.append_state("test_save_append.h5")
    with pytest.raises(GraphError):
        main.load_state("test_save_append.h5")


@pytest.mark.parametrize("usefileobject", [False, True])
def test_save_append_load(usefileobject):

    _make_files_and_test(usefileobject=usefileobject)

    # Load not appendable
    _load_not_appendable_and_test(usefileobject=usefileobject)

    # Load appendable
    _load_appendable_and_test(usefileobject=usefileobject)

    # different graph
    _change_graph_fail_test()


def test_save_append_load_batched_param():
    module = Module("main")
    param = Param("batched", np.arange(12).reshape(2, 2, 3), shape=(2, 3))
    module.batched = param

    module.save_state("test_save_batched_append.h5", appendable=True)
    param.value = np.arange(12, 24).reshape(2, 2, 3)
    module.append_state("test_save_batched_append.h5")

    with h5py.File("test_save_batched_append.h5", "r") as h5file:
        assert h5file["main/batched/value"].shape == (2, 2, 2, 3)

    loaded_module = Module("main")
    loaded_module.batched = Param("batched", None, shape=(2, 3))
    loaded_module.load_state("test_save_batched_append.h5")
    np.testing.assert_array_equal(loaded_module.batched.value, np.arange(12, 24).reshape(2, 2, 3))


def test_missing_h5py():
    import importlib

    module = importlib.import_module("caskade.base")
    original = dict(vars(module))
    try:
        with mock.patch.dict(sys.modules, {"h5py": None}):
            # Reload the module to re-trigger the import logic
            importlib.reload(module)

            assert module.h5py is None
    finally:
        # The reload rebuilt the classes of the module, put the originals back
        vars(module).update(original)


def test_missing_h5py_error(monkeypatch):
    monkeypatch.setattr("caskade.base.h5py", None)
    main = Module("main")
    main.p = Param("p", 1.0)

    with pytest.raises(ImportError, match="pip install h5py"):
        main.save_state("test_missing_h5py.h5")
    with pytest.raises(ImportError, match="pip install h5py"):
        main.append_state("test_missing_h5py.h5")
    with pytest.raises(ImportError, match="pip install h5py"):
        main.load_state("test_missing_h5py.h5")


# One pass over the graph, nothing left on the nodes
#################################################################
def numpy(value):
    return backend.to_numpy(value)


class Leaf(Module):
    def __init__(self, name, **children):
        super().__init__(name)
        self.a = Param("a", [1.0, 2.0], shape=(2,), dynamic=True)
        self.b = Param("b", 3.0, shape=(), units="kpc")
        for key, child in children.items():
            setattr(self, key, child)


def layered(depth=6):
    """Layers of two modules that each hold both modules of the next layer, the
    number of paths through the graph doubles with every layer."""
    layer = [Leaf(f"L{depth}_{j}") for j in range(2)]
    for d in range(depth - 1, -1, -1):
        layer = [Leaf(f"L{d}_{j}", **{c.name: c for c in layer}) for j in range(2)]
    return Leaf("top", **{c.name: c for c in layer})


def test_every_node_is_written_and_read_once(tmp_path, monkeypatch):
    top = layered()
    params = [n for n in top.topological_ordering() if isinstance(n, Param)]
    assert len(params) == 30
    calls = {"save": 0, "read": 0, "append": 0}

    def count(name, key):
        real = getattr(Param, name)

        def counted(self, *args):
            calls[key] += 1
            return real(self, *args)

        monkeypatch.setattr(Param, name, counted)

    count("_save_hdf5", "save")
    count("_read_hdf5", "read")
    count("_append_hdf5", "append")

    path = tmp_path / "layers.h5"  # a path object is taken as well as a string
    top.save_state(path, appendable=True)
    for p in params:
        p.value = 2 * p.value
    top.append_state(path)
    expect = [numpy(p.value) for p in params]
    for p in params:
        p.value = 0 * p.value
    top.load_state(path)
    assert calls == {"save": 30, "read": 30, "append": 30}
    assert all(np.allclose(numpy(p.value), e) for p, e in zip(params, expect))
    top.load_state(path, 0)
    assert all(np.allclose(numpy(p.value), e / 2) for p, e in zip(params, expect))

    # the file still shows a shared node at every path that leads to it
    with h5py.File(path, "r") as f:
        assert f["top/L0_0/L1_0/L2_1/a"] == f["top/L0_1/L1_1/L2_1/a"]
        assert f["top/L0_0/L1_0/L2_1/a/value"].shape == (2, 2)
    # and nothing is kept on the nodes
    for node in top.topological_ordering():
        assert not hasattr(node, "_h5group") and not hasattr(node, "appended")


def test_failed_save_and_append_leave_nothing_behind(tmp_path, monkeypatch):
    top = Leaf("top", sub=Leaf("sub"))
    path = str(tmp_path / "fail.h5")
    top.save_state(path, appendable=True)

    real = backend.to_numpy
    count = [0]

    def flaky(value):
        count[0] += 1
        if count[0] == 3:
            raise RuntimeError("no more")
        return real(value)

    for action in (
        lambda: top.append_state(path),
        lambda: top.save_state(str(tmp_path / "other.h5")),
    ):
        count[0] = 0
        monkeypatch.setattr(backend, "to_numpy", flaky)
        with pytest.raises(RuntimeError, match="no more"):
            action()
        monkeypatch.setattr(backend, "to_numpy", real)
    # every value is collected before any is written, so the file is as it was
    with h5py.File(path, "r") as f:
        assert {f[k].shape[0] for k in ("top/a/value", "top/b/value", "top/sub/a/value")} == {1}
    top.a = [5.0, 6.0]
    top.append_state(path)
    top.save_state(str(tmp_path / "other.h5"))
    with h5py.File(path, "r") as f:
        assert np.allclose(f["top/a/value"][()], [[1.0, 2.0], [5.0, 6.0]])
        assert f["top"].attrs["caskade_samples"] == 2


def test_append_keeps_every_param_at_the_same_number_of_states(tmp_path):
    top = Leaf("top")
    top.late = Param("late", None, shape=(2,), dynamic=True)  # no value yet
    path = str(tmp_path / "rows.h5")
    top.save_state(path, appendable=True)
    top.append_state(path)
    top.late = [7.0, 8.0]
    top.append_state(path)
    top.a = None  # and one that loses its value
    top.append_state(path)
    with h5py.File(path, "r") as f:
        late, a = f["top/late/value"][()], f["top/a/value"][()]
        assert late.shape == a.shape == (4, 2)
        assert np.isnan(late[:2]).all() and np.allclose(late[2:], [7.0, 8.0])
        assert np.allclose(a[:3], [1.0, 2.0]) and np.isnan(a[3]).all()
        assert f["top/late/value"].attrs["node_type"] == "dynamic"

        # an appending that was cut short is put right by the next one
    with h5py.File(path, "a") as f:
        f["top/b/value"].resize(5, axis=0)
    top.a = [1.0, 2.0]
    top.append_state(path)
    with h5py.File(path, "r") as f:
        assert {f[f"top/{k}/value"].shape[0] for k in ("a", "b", "late")} == {5}

    other = Leaf("top")
    other.late = Param("late", None, shape=(2,), dynamic=True)
    other.load_state(path, 2)
    assert np.allclose(numpy(other.late.value), [7.0, 8.0])

    free = Leaf("top")
    free.late = Param("late", [7.0, 8.0, 9.0], dynamic=True)  # no longer fits the file
    with pytest.raises(GraphError, match="states of shape"):
        free.append_state(path)
    with h5py.File(path, "r") as f:  # caught before any writing
        assert {f[f"top/{k}/value"].shape[0] for k in ("a", "b", "late")} == {5}


def test_values_are_converted_for_appending(tmp_path):
    if backend.backend != "torch":
        pytest.skip("only torch values can need converting")
    top = Leaf("top")
    path = str(tmp_path / "grad.h5")
    top.save_state(path, appendable=True)
    top.a.value.requires_grad_(True)
    top.append_state(path)
    with h5py.File(path, "r") as f:
        assert f["top/a/value"].shape == (2, 2)


def test_load_changes_nothing_unless_all_of_it_fits(tmp_path):
    top = Leaf("top", sub=Leaf("sub"))
    path = str(tmp_path / "fit.h5")
    top.save_state(path)

    sub = Module("sub")
    sub.a = Param("a", [0.0, 0.0], shape=(2,), dynamic=True)
    sub.b = Param("b", [0.0, 0.0, 0.0], shape=(3,))  # read after ``top.a``, does not fit
    other = Leaf("top", sub=sub)
    other.a = [7.0, 7.0]
    with pytest.raises(GraphError, match=r"shape \(\), which does not fit the shape \(3,\)"):
        other.load_state(path)
    assert np.allclose(numpy(other.a.value), [7.0, 7.0])

    with pytest.raises(GraphError, match=r"named 'elsewhere' was found, the file holds \['top'\]"):
        Leaf("elsewhere").load_state(path)
    for action in (Leaf("elsewhere").append_state, Leaf("top", more=Leaf("more")).load_state):
        with pytest.raises(GraphError):
            action(path)

    small = Module("top")  # the file holds more than this graph has
    small.a = Param("a", None, shape=(2,), dynamic=True)
    with pytest.warns(SaveStateWarning, match="2 entries that the graph of 'top' does not have"):
        small.load_state(path)
    assert np.allclose(numpy(small.a.value), [1.0, 2.0])


def test_only_listed_attributes_and_units_are_loaded(tmp_path):
    top = Leaf("top")
    top.note = "kept"
    top.saveattrs.add("note")
    path = str(tmp_path / "attrs.h5")
    top.save_state(path)
    with h5py.File(path, "a") as f:
        f["top"].attrs["_name"] = "renamed"
        f["top"].attrs["other"] = 5

    other = Leaf("top")
    other.note = "not yet"
    other.saveattrs.add("note")
    other.a.units = "wrong"
    other.load_state(path)
    assert other.note == "kept" and other.name == "top" and not hasattr(other, "other")
    assert other.a.units is None and other.b.units == "kpc"
    assert not hasattr(other, "caskade_format")


def test_files_from_before_the_format_was_stamped(tmp_path):
    # Written by hand as caskade 1.1 wrote them: no ``caskade_`` attributes,
    # node types under their old names, a param that had no value
    path = str(tmp_path / "old.h5")
    with h5py.File(path, "w") as f:
        for name, value, kind in (
            ("a", [[1.0, 2.0], [3.0, 4.0]], "dynamic value"),
            ("b", [3.0, 4.0], "static"),
        ):
            data = np.array(value)
            d = f.create_dataset(
                f"top/{name}/value", data=data, maxshape=(None,) + data.shape[1:], chunks=True
            )
            d.attrs.update(node_type=kind, appendable=True, cyclic=False, units="None")
        d = f.create_dataset("top/late/value", data="None")
        d.attrs.update(node_type="dynamic", appendable=True, cyclic=False, units="None")

    top = Leaf("top")
    top.late = Param("late", [0.0], dynamic=True)
    top.b.units = "wrong"
    top.load_state(path)
    assert np.allclose(numpy(top.a.value), [3.0, 4.0]) and top.a.dynamic
    assert np.allclose(numpy(top.b.value), 4.0) and top.b.static and top.b.units is None
    assert top.late.value is None
    top.load_state(path, 0)
    assert np.allclose(numpy(top.a.value), [1.0, 2.0])

    top.append_state(path)
    with h5py.File(path, "r") as f:
        assert f["top/a/value"].shape == (3, 2) and f["top/b/value"].shape == (3,)
    top.late = [1.0]
    with pytest.raises(IOError, match="older version"):
        top.append_state(path)


# The fast layout
#################################################################
def fast_graph(count=(3, 4, 5)):
    shared = Leaf("shared")
    top = Leaf("top", one=Leaf("one", shared=shared), two=Leaf("two", shared=shared))
    top.count = Param("count", np.array(count), shape=(len(count),))  # whole numbers
    top.empty = Param("empty", None, shape=(2,), dynamic=True)
    top.live = Param("live", None)  # static, set during a simulation
    top.twice = Param("twice", lambda p: 2 * p.a.value)
    top.twice.link("a", top.a)
    return top


@pytest.mark.parametrize("usefileobject", [False, True])
def test_fast_layout(tmp_path, usefileobject):
    path = str(tmp_path / "fast.h5")

    def run(action, mode, *args, **kwargs):
        if not usefileobject:
            return action(path, *args, **kwargs)
        with h5py.File(path, mode) as f:
            return action(f, *args, **kwargs)

    top = fast_graph()
    run(top.save_state, "w", appendable=True, fast=True)
    first = numpy(top.one.a.value).copy()
    top.one.a = [10.0, 20.0]
    top.count = np.array([6, 7, 8])
    run(top.append_state, "a")

    with h5py.File(path, "r") as f:
        assert set(f["top"].keys()) == {"dynamic", "static", "index"}  # whatever the graph
        assert f["top/dynamic"].shape == (2, 8) and f["top/static"].shape == (2, 7)
        assert f["top"].attrs["caskade_layout"] == "fast"

    other = fast_graph()
    for p in other.dynamic_params + other.static_params:
        if p.value is not None:
            p.value = 0 * p.value
    other.b.to_dynamic()
    other.one.shared.a.to_static()  # the file says what is dynamic and what is static
    run(other.load_state, "r")
    assert np.allclose(numpy(other.one.a.value), [10.0, 20.0])
    assert np.allclose(numpy(other.one.shared.b.value), 3.0) and other.b.static
    assert other.one.shared.a.dynamic and other.one.shared is other.two.shared
    assert numpy(other.count.value).tolist() == [6, 7, 8]
    assert numpy(other.count.value).dtype.kind == "i"
    assert other.empty.value is None and other.live.value is None
    assert other.twice.pointer and np.allclose(numpy(other.twice.value), [2.0, 4.0])
    assert other.b.units == "kpc"  # not stored in this layout, so not touched
    run(other.load_state, "r", 0)
    assert np.allclose(numpy(other.one.a.value), first)
    assert numpy(other.count.value).tolist() == [3, 4, 5]

    # not appendable: the arrays are the state itself
    run(top.save_state, "w", fast=True)
    with h5py.File(path, "r") as f:
        assert f["top/dynamic"].shape == (8,)
    with pytest.raises(IOError, match="not appendable"):
        run(top.append_state, "a")
    run(other.load_state, "r")
    assert numpy(other.count.value).tolist() == [6, 7, 8]


def test_fast_layout_holds_what_get_values_returns(tmp_path):
    top = Leaf("top", sub=Leaf("sub"))
    path = str(tmp_path / "chain.h5")
    top.save_state(path, appendable=True, fast=True)
    rows = [numpy(top.get_values())]
    for i in range(3):
        top.set_values(top.get_values() + i)
        top.append_state(path)
        rows.append(numpy(top.get_values()))
    with h5py.File(path, "r") as f:
        assert np.allclose(f["top/dynamic"][()], np.stack(rows))

    # a batch of simulations keeps the value that they share as a single value
    top.a = np.arange(6.0).reshape(3, 2)
    top.save_state(path, fast=True)
    other = Leaf("top", sub=Leaf("sub"))
    other.load_state(path)
    assert other.a.batch_shape == (3,) and other.sub.a.batch_shape == ()
    assert np.allclose(numpy(other.get_values()), numpy(top.get_values()))


def test_fast_layout_only_takes_the_graph_it_was_saved_from(tmp_path):
    top = fast_graph()
    path = str(tmp_path / "strict.h5")
    top.save_state(path, appendable=True, fast=True)

    more = fast_graph()
    more.extra = Param("extra", 1.0)
    with pytest.raises(GraphError, match=r"no value for \['top/extra'\]"):
        more.load_state(path)
    with pytest.raises(GraphError, match="not the ones the file was saved with"):
        more.append_state(path)

    less = Leaf("top")
    less.a = [7.0, 7.0]
    with pytest.raises(GraphError, match="which is not a param of 'top'"):
        less.load_state(path)
    assert np.allclose(numpy(less.a.value), [7.0, 7.0])

    top.empty = [1.0, 2.0]  # gained a value, the rows would no longer line up
    with pytest.raises(GraphError, match="not the ones the file was saved with"):
        top.append_state(path)

    reshaped = fast_graph(count=(0, 0, 0, 0))
    with pytest.raises(GraphError, match="does not fit the shape"):
        reshaped.load_state(path)

    pointer = fast_graph()  # a param that is now worked out from another one stays that way
    pointer.one.a = lambda p: p.b.value * np.ones(2)
    pointer.one.a.link("b", pointer.one.b)
    pointer.load_state(path)
    assert pointer.one.a.pointer and np.allclose(numpy(pointer.count.value), [3, 4, 5])


def test_fast_layout_with_only_one_kind_of_param(tmp_path):
    top = Leaf("top")
    top.to_static()
    path = str(tmp_path / "static_only.h5")
    top.save_state(path, appendable=True, fast=True)
    top.b = 5.0
    top.append_state(path)
    with h5py.File(path, "r") as f:
        assert set(f["top"].keys()) == {"static", "index"}  # there is no dynamic array
        assert f["top/static"].shape == (2, 3)
    other = Leaf("top")
    other.to_static()
    other.load_state(path, 0)
    assert np.allclose(numpy(other.b.value), 3.0)


def test_value_saved_from_a_pointer_is_not_loaded(tmp_path):
    top = Leaf("top")
    top.b = lambda p: p.a.value.sum()
    top.b.link("a", top.a)
    path = str(tmp_path / "pointer.h5")
    top.save_state(path)
    with h5py.File(path, "r") as f:  # the value is in the file for anyone to look at
        assert (
            np.allclose(f["top/b/value"][()], 3.0)
            and f["top/b/value"].attrs["node_type"] == "pointer"
        )

    other = Leaf("top")
    other.b.link("a", other.a)
    other.b = 10.0  # here it is a value of its own, which the file has nothing to say about
    other.load_state(path)
    assert other.b.static and np.allclose(numpy(other.b.value), 10.0)


def test_text_in_place_of_a_value(tmp_path):
    top = Leaf("top")
    path = str(tmp_path / "text.h5")
    top.save_state(path)
    with h5py.File(path, "a") as f:
        attrs = dict(f["top/b/value"].attrs)
        del f["top/b/value"]
        f["top/b"].create_dataset("value", data="none").attrs.update(attrs)
    top.a = [7.0, 7.0]
    with pytest.raises(ValueError, match="Unrecognized value saved for param b"):
        top.load_state(path)
    assert np.allclose(numpy(top.a.value), [7.0, 7.0])
