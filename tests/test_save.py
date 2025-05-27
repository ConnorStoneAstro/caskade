from caskade import Module, Param, GraphError, SaveStateWarning, backend, BackendError
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

    m1.p1 = p1
    m1.p2 = p2
    m2.p3 = p3
    m2.link("p4_param", p4)
    m2.p5_param = p5
    m3.p4 = p4
    m3.p5 = p5

    return main


def _make_files_and_test(usefileobject=False):
    main = _build_test_module()

    # Save not appendable
    with pytest.warns(SaveStateWarning):
        main.m1.meta.bad_meta = None
        main.m1.saveattrs.add("meta.bad_meta")
        if usefileobject:
            with h5py.File("test_save_notappend.h5", "w") as f:
                main.save_state(f, appendable=False)
        else:
            main.save_state("test_save_notappend.h5", appendable=False)
    if backend.backend == "object":
        with pytest.raises(BackendError):
            main.save_state("test_save_badappend.h5", appendable=True)
        with pytest.raises(BackendError):
            main.append_state("test_save_badappend.h5")
        return

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

    if backend.backend == "object":
        return

    # Load appendable
    _load_appendable_and_test(usefileobject=usefileobject)

    # different graph
    _change_graph_fail_test()


def test_missing_h5py():
    with mock.patch.dict(sys.modules, {"h5py": None}):
        import importlib

        # Reload the module to re-trigger the import logic
        module = importlib.import_module("caskade.base")
        importlib.reload(module)

        assert module.h5py is None
