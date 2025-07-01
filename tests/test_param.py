import pytest
import numpy as np

from caskade import (
    Param,
    ActiveStateError,
    ParamConfigurationError,
    ParamTypeError,
    GraphError,
    InvalidValueWarning,
    LinkToAttributeError,
    dynamic,
    backend,
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
    if backend.backend == "object":
        with pytest.raises(ParamTypeError):
            p2.shape = (1, 2, 3)
        assert p2.shape is None
        p2 = p2.to()
        return
    assert p2.value.item() == 1.0
    p3 = Param("test", backend.module.ones((1, 2, 3)))
    p33 = Param("test", dynamic_value=backend.module.ones((1, 2, 3)))
    assert backend.all(p3.value == p33.value)
    p33v2 = Param("test", dynamic(backend.module.ones((3, 2, 1))))
    assert p33v2.dynamic
    assert p33v2.value.shape == (3, 2, 1)
    p33v3 = Param("test", dynamic_value=dynamic(backend.module.ones((3, 2, 1))))
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
    with pytest.raises(LinkToAttributeError):
        p6.link("link", p5)

    # Attempt link with existing name
    with pytest.raises(GraphError):
        p6.link("other", p5)

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
    if backend.backend == "object":
        return
    if backend.backend == "jax":
        device = backend.jax.devices()[0]
        backend.jax.config.update("jax_enable_x64", True)
    else:
        device = "cpu"

    # static
    p = Param("test", 1.0, valid=(0, 2))
    p = p.to(dtype=backend.module.float64, device=device)
    # dynamic value
    p = Param("test", dynamic_value=1.0, valid=(0, 2))
    p = p.to(dtype=backend.module.float64, device=device)


def test_params_sticky_to():
    if backend.backend == "object":
        return
    if backend.backend == "jax":
        device = backend.jax.devices()[0]
        backend.jax.config.update("jax_enable_x64", True)
    else:
        device = "cpu"
    # static
    p = Param("test", 1.0, valid=(0, 2))
    p = p.to(dtype=backend.module.float64, device=device)
    p.value = 2.0  # value cast to float64
    assert p.value.dtype == backend.module.float64
    # dynamic value
    p = Param("test", dynamic_value=1.0, dtype=backend.module.float32)
    assert p.value.dtype == backend.module.float32
    p = p.to(dtype=backend.module.float64, device=device)
    assert p.value.dtype == backend.module.float64
    p.dynamic_value = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    assert p.value.dtype == backend.module.float64
    # neither dtype or value set
    p = Param("test", valid=(0, 2))
    assert p.dtype is None
    assert p.device is None
    p = p.to(dtype=backend.module.float64, device=device)
    assert p.dtype == backend.module.float64
    assert p.device == device
    p = p.to()
    p.value = 1.0
    assert p.dtype == backend.module.float64
    assert p.device == device


def test_check_npvalue():
    p = Param("test", [1.0, 2.0, 3.0, 4.0])
    assert np.all(np.array([1.0, 2.0, 3.0, 4.0]) == p.npvalue)


def test_value_setter():

    # dynamic
    p = Param("test")
    assert p._type == "dynamic"

    # static
    p.value = 1.0
    assert p._type == "static"
    if backend.backend == "object":
        return
    assert p.value.item() == 1.0

    p = Param("testshape", shape=(2,))
    p.value = [1.0, 2.0]

    # pointer
    other = Param("testother", 2.0)
    p.value = other
    assert p._type == "pointer"
    assert p.shape == other.shape

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
    if backend.backend == "object":
        return
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
    if backend.backend == "object":
        return

    v = backend.make_array(0.5)
    assert p.to_valid(v) == v, "valid value should not change"
    assert p.from_valid(v) == v, "valid value should not change"

    p.valid = (0, None)
    assert p.to_valid(v) != v, "valid value should change"
    assert p.from_valid(v) != v, "valid value should change"
    assert backend.all(
        p.from_valid(backend.module.linspace(-1e4, 1e4, 101)) >= 0
    ), "from_valid should map to valid range"

    p.valid = (None, 0)
    assert p.to_valid(v) != v, "valid value should change"
    assert p.from_valid(v) != v, "valid value should change"
    assert backend.all(
        p.from_valid(backend.module.linspace(-1e4, 1e4, 101)) <= 0
    ), "from_valid should map to valid range"

    p.valid = (0, 1)
    assert p.to_valid(v) != v, "valid value should change"
    assert p.from_valid(v) != v, "valid value should change"
    assert backend.all(
        p.from_valid(backend.module.linspace(-1e4, 1e4, 101)) >= 0
    ), "from_valid should map to valid range"
    assert backend.all(
        p.from_valid(backend.module.linspace(-1e4, 1e4, 101)) <= 1
    ), "from_valid should map to valid range"

    p.cyclic = True
    assert p.to_valid(v) == v, "valid cyclic value should not change"
    assert p.from_valid(v) == v, "valid cyclic value should not change"
    assert backend.all(
        p.from_valid(backend.module.linspace(-1e4, 1e4, 101)) >= 0
    ), "from_valid should map to valid range"
    assert backend.all(
        p.from_valid(backend.module.linspace(-1e4, 1e4, 101)) <= 1
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

    print(p.valid)
    with pytest.warns(InvalidValueWarning):
        p.value = -1
    with pytest.warns(InvalidValueWarning):
        p.valid = (0, None)
    with pytest.warns(InvalidValueWarning):
        p.valid = (None, -2)
