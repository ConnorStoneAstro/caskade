import pytest
import torch

from caskade import Param


def test_param_creation():

    # Minimal creation
    p1 = Param("test")
    assert p1.name == "test"
    assert p1.dynamic
    assert p1.value is None

    # Name and value
    p2 = Param("test", 1.0)
    assert p2.name == "test"
    assert p2.value.item() == 1.0
    p3 = Param("test", torch.ones((1, 2, 3)))

    # Cant update value when active
    with pytest.raises(RuntimeError):
        p3.active = True
        p3.value = 1.0

    # Missmatch value and shape
    with pytest.raises(AssertionError):
        p4 = Param("test", 1.0, shape=(1, 2, 3))

    # Cant set shape of pointer or function
    p5 = Param("test", p3)
    with pytest.raises(RuntimeError):
        p5.shape = (1, 2, 3)
    with pytest.raises(ValueError):
        p5.to_valid(1.0)
    with pytest.raises(ValueError):
        p5.from_valid(1.0)

    # Function parameter
    p6 = Param("test", lambda p: p["other"].value * 2)
    p6.link("other", p2)
    with pytest.raises(RuntimeError):
        p6.shape = (1, 2, 3)

    # Missing value and shape
    with pytest.raises(ValueError):
        p7 = Param("test", None, None)

    # Shape is not a tuple
    with pytest.raises(ValueError):
        p8 = Param("test", None, 7)


def test_param_to():
    p = Param("test", 1.0, valid=(0, 2))
    p = p.to(dtype=torch.float64, device="cpu")


def test_value_setter():

    # dynamic
    p = Param("test")
    assert p._type == "dynamic"

    # static
    p.value = 1.0
    assert p._type == "static"
    assert p.value.item() == 1.0

    p = Param("testshape", shape=(2,))
    p.value = [1.0, 2.0]

    # pointer
    other = Param("testother", 2.0)
    p.value = other
    assert p._type == "pointer"
    assert p.shape is None

    # function
    p.value = lambda p: p["other"].value * 2
    p.link("other", other)
    assert p._type == "pointer"
    assert p.value is None


def test_units():
    p = Param("test", units="m")
    assert p.units == "m"


def test_valid():
    p = Param("test", valid=None)
    v = torch.tensor(0.5)
    assert p.to_valid(v) == v, "valid value should not change"
    assert p.from_valid(v) == v, "valid value should not change"

    p.valid = (0, None)
    assert p.to_valid(v) != v, "valid value should change"
    assert p.from_valid(v) != v, "valid value should change"
    assert torch.all(
        p.from_valid(torch.linspace(-1e4, 1e4, 101)) >= 0
    ), "from_valid should map to valid range"

    p.valid = (None, 0)
    assert p.to_valid(v) != v, "valid value should change"
    assert p.from_valid(v) != v, "valid value should change"
    assert torch.all(
        p.from_valid(torch.linspace(-1e4, 1e4, 101)) <= 0
    ), "from_valid should map to valid range"

    p.valid = (0, 1)
    assert p.to_valid(v) != v, "valid value should change"
    assert p.from_valid(v) != v, "valid value should change"
    assert torch.all(
        p.from_valid(torch.linspace(-1e4, 1e4, 101)) >= 0
    ), "from_valid should map to valid range"
    assert torch.all(
        p.from_valid(torch.linspace(-1e4, 1e4, 101)) <= 1
    ), "from_valid should map to valid range"

    p.cyclic = True
    assert p.to_valid(v) == v, "valid cyclic value should not change"
    assert p.from_valid(v) == v, "valid cyclic value should not change"
    assert torch.all(
        p.from_valid(torch.linspace(-1e4, 1e4, 101)) >= 0
    ), "from_valid should map to valid range"
    assert torch.all(
        p.from_valid(torch.linspace(-1e4, 1e4, 101)) <= 1
    ), "from_valid should map to valid range"
