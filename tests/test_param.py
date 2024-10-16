import pytest
import torch

from caskade import Param, LiveParam


def test_live_param():
    lp1 = LiveParam()
    lp2 = LiveParam()

    assert lp1 is lp2, "LiveParam is not a singleton"


def test_param_creation():

    p1 = Param("test")
    assert p1.name == "test"
    assert p1.dynamic
    assert not p1.live
    assert p1.value is None

    p2 = Param("test", 1.0)
    assert p2.name == "test"
    assert p2.value.item() == 1.0
    p3 = Param("test", shape=(1, 2, 3))
    with pytest.raises(ValueError):
        p3.value = 1.0

    with pytest.raises(AssertionError):
        p4 = Param("test", 1.0, shape=(1, 2, 3))


def test_value_setter():

    # dynamic
    p = Param("test")
    assert p._type == "dynamic"

    # value
    p.value = 1.0
    assert p._type == "value"
    assert p.value.item() == 1.0

    p = Param("testshape", shape=(2,))
    p.value = [1.0, 2.0]

    # pointer
    other = Param("testother", 2.0)
    p.value = other
    assert p._type == "pointer"
    assert p.shape is None

    # function
    p.value = lambda p: p.children["other"].value * 2
    p.link("other", other)
    assert p._type == "function"
    assert p.value.item() == 4.0

    # live
    p.shape = (2,)
    p.value = LiveParam()
    assert p._type == "live"
