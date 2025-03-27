import pytest
import torch

from caskade import (
    Param,
    ActiveStateError,
    ParamConfigurationError,
    ParamTypeError,
    InvalidValueWarning,
    AttributeCollisionWarning,
    dynamic,
)


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
    p33 = Param("test", dynamic_value=torch.ones((1, 2, 3)))
    assert torch.all(p3.value == p33.value)
    p33v2 = Param("test", dynamic(torch.ones((3, 2, 1))))
    assert p33v2.dynamic
    assert p33v2.value.shape == (3, 2, 1)
    p33v3 = Param("test", dynamic_value=dynamic(torch.ones((3, 2, 1))))
    assert p33v3.dynamic
    assert p33v3.value.shape == (3, 2, 1)

    # Cant update value when active
    with pytest.raises(ActiveStateError):
        p3.active = True
        p3.value = 1.0
    with pytest.raises(ActiveStateError):
        p33.active = True
        p33.dynamic_value = 1.0

    # Missmatch value and shape
    with pytest.raises(ParamConfigurationError):
        p4 = Param("test", 1.0, shape=(1, 2, 3))
    with pytest.raises(ParamConfigurationError):
        p44 = Param("test", dynamic_value=1.0, shape=(1, 2, 3))

    # Cant set shape of pointer or function
    p5 = Param("test", p3)
    with pytest.raises(ParamTypeError):
        p5.shape = (1, 2, 3)
    with pytest.raises(ParamTypeError):
        p5.to_valid(1.0)
    with pytest.raises(ParamTypeError):
        p5.from_valid(1.0)

    # Function parameter
    p6 = Param("test", lambda p: p["other"].value * 2)
    p6.link("other", p2)
    with pytest.raises(ParamTypeError):
        p6.shape = (1, 2, 3)

    # Missing value and shape
    with pytest.raises(ParamConfigurationError):
        p7 = Param("test", None, None)

    # Shape is not a tuple
    with pytest.raises(ParamConfigurationError):
        p8 = Param("test", None, 7)

    # Attempt link with attribute name
    with pytest.warns(AttributeCollisionWarning):
        p6.link("link", p5)

    # Metadata
    p9 = Param("test", 1.0, units="none", cyclic=True, valid=(0, 1))
    assert p9.units == "none"
    assert p9.cyclic
    assert p9.valid[0].item() == 0
    assert p9.valid[1].item() == 1

    # Invalid dynamic value
    with pytest.raises(ParamTypeError):
        p10 = Param("test", dynamic_value=p9)
    with pytest.raises(ParamTypeError):
        p11 = Param("test", dynamic_value=lambda p: p.other.value * 2)
    with pytest.raises(ParamConfigurationError):
        p12 = Param("test", value=1.0, dynamic_value=1.0)

    # Set dynamic from other states
    p13 = Param("test", 1.0)  # static
    p13.dynamic_value = 2.0
    assert p13.value.item() == 2.0
    assert p13.dynamic
    p14 = Param("test")  # dynamic
    p14.dynamic_value = 1.0
    assert p14.value.item() == 1.0
    p15 = Param("test", p14)  # pointer
    p15.dynamic_value = 2.0
    assert p15.value.item() == 2.0
    p16 = Param("test", 1.0)  # static
    p16.value = None
    assert p16.dynamic
    assert p16.dynamic_value.item() == 1.0
    p16.dynamic = False
    assert p16.static
    p16.dynamic = True
    assert p16.dynamic
    p16.static = True
    assert p16.static
    p16.static = False
    assert p16.dynamic


def test_param_to():
    # static
    p = Param("test", 1.0, valid=(0, 2))
    p = p.to(dtype=torch.float64, device="cpu")
    # dynamic value
    p = Param("test", dynamic_value=1.0, valid=(0, 2))
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
    p.value = lambda p: p.other.value * 2
    p.link("other", other)
    assert p._type == "pointer"
    assert p.value.item() == 4.0


def test_to_dynamic_static():

    other = Param("other", 3.0)

    # dynamic
    p = Param("test")
    p.to_dynamic()  # from dynamic
    assert p.dynamic
    p.dynamic_value = 1.0
    assert p.dynamic
    p.to_dynamic()  # from dynamic with dynamic value
    assert p.dynamic
    p.value = 2.0
    p.to_dynamic()  # from static
    assert p.dynamic
    assert p.value.item() == 2.0
    p.value = lambda p: p["other"].value * 2
    p.to_dynamic()  # from pointer, fails
    assert p.dynamic
    assert p.value is None
    p.value = lambda p: p["other"].value * 2
    p.link("other", other)
    p.to_dynamic()  # from pointer, succeeds
    assert p.dynamic
    assert p.value.item() == 6.0

    # static
    p = Param("test", 1.0)
    p.to_static()  # from static
    assert p.static
    p = Param("test")
    with pytest.raises(ParamTypeError):
        p.to_static()  # from dynamic, fails
    p.dynamic_value = 2.0
    p.to_static()  # from dynamic with dynamic value
    assert p.static
    assert p.value.item() == 2.0
    p.value = lambda p: p["other"].value * 2
    with pytest.raises(ParamTypeError):
        p.to_static()  # from pointer, fails
    p.link("other", other)
    p.to_static()  # from pointer, succeeds
    assert p.static
    assert p.value.item() == 6.0


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

    p.value = 0.5

    with pytest.raises(ParamConfigurationError):
        p.valid = None
    with pytest.raises(ParamConfigurationError):
        p.valid = (1, None)
    with pytest.raises(ParamConfigurationError):
        p.valid = (None, 1)
    p.cyclic = False
    with pytest.raises(ParamConfigurationError):
        p.valid = (1, 0)
    with pytest.raises(ParamConfigurationError):
        p.valid = (0, 1, 2)
    with pytest.raises(ParamConfigurationError):
        p.valid = [0, 1]
    with pytest.warns(InvalidValueWarning):
        p.value = -1
    with pytest.warns(InvalidValueWarning):
        p.valid = (0, None)
    with pytest.warns(InvalidValueWarning):
        p.valid = (None, -2)
