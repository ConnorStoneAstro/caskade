from caskade import Module, Param, GraphError
import numpy as np
import gc

import pytest


def _build_test_module():
    main = Module("main")
    m1 = Module("m1")
    m2 = Module("m2")
    m3 = Module("m3")

    main.m1 = m1
    main.m2 = m2

    m1.m3 = m3
    m2.m3 = m3

    p1 = Param("p1", 1.0, valid=(0.0, 2.0), units="arcsec")
    p2 = Param("p2", (2.0, 2.5), valid=(None, (5, 6)))
    p3 = Param(
        "p3",
        np.ones((2, 3, 4)),
        cyclic=True,
        valid=(np.zeros((2, 3, 4)), 2 * np.ones((2, 3, 4))),
        units="m",
    )
    p4 = Param("p4", p2)
    p5 = Param("p5")

    m1.p1 = p1
    m1.p2 = p2
    m2.p3 = p3
    m3.p4 = p4
    m3.p5 = p5

    return main


def _make_files_and_test():
    main = _build_test_module()

    # bad file
    with pytest.raises(NotImplementedError):
        main.save_state("test_save_bad.png")
    with pytest.raises(NotImplementedError):
        main.load_state("test_save_bad.png")
    with pytest.raises(NotImplementedError):
        main.append_state("test_save_bad.png")

    # Save not appendable
    main.save_state("test_save_notappend.h5", appendable=False)
    with pytest.raises(IOError):
        main.append_state("test_save_notappend.h5")

    # Save and append
    main.save_state("test_save_append.h5", appendable=True)
    main.m1.p1.value = 2.0
    main.m1.p2.value = (3.0, 3.5)
    main.append_state("test_save_append.h5")
    main.append_state("test_save_append.h5")


def _load_not_appendable_and_test():
    gc.collect()
    main = _build_test_module()
    main.load_state("test_save_notappend.h5")
    assert main.m1.p1.value.item() == 1.0
    assert main.m1.p1.units == "arcsec"


def _load_appendable_and_test():
    gc.collect()
    main = _build_test_module()
    main.load_state("test_save_append.h5")
    assert main.m1.p1.value.item() == 2.0
    assert main.m1.p2.value[0].item() == 3.0
    assert main.m1.p2.value[1].item() == 3.5


def _change_graph_fail_test():
    gc.collect()
    main = _build_test_module()
    main.link(Module("bad"))
    with pytest.raises(GraphError):
        main.append_state("test_save_append.h5")
    with pytest.raises(GraphError):
        main.load_state("test_save_append.h5")


def test_save_append_load():

    _make_files_and_test()

    # Load not appendable
    _load_not_appendable_and_test()

    # Load appendable
    _load_appendable_and_test()

    # different graph
    _change_graph_fail_test()
