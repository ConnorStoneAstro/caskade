import pytest
import numpy as np

from caskade import (
    Param,
    Module,
    ActiveStateError,
    ParamConfigurationError,
    ParamTypeError,
    InvalidValueWarning,
    ActiveContext,
    backend,
)


def test_param_creation(many_param, capsys):
    p, name, value, shape, cyclic, valid, units, dynamic, group = many_param

    if name is None:
        assert p.name == "Param"
    else:
        assert p.name == name

    if value is None:
        assert p.value is None
    else:
        assert np.allclose(p.npvalue, np.array(value))

    if shape == () and value == [1.0, 2.0]:
        assert p.batched
        assert p.batch_shape == (2,)
    else:
        assert not p.batched
        assert p.batch_shape == ()

    if shape is not None:
        assert p.shape == shape

    assert p.cyclic == cyclic

    assert p.is_valid(p.value)
    if value is not None:
        assert np.allclose(p.npvalue, backend.to_numpy(p.from_valid(p.to_valid(p.value))))

    assert p.units == units

    if dynamic is not None:
        assert p.dynamic is dynamic
        assert p.static is not dynamic
        assert p.pointer is False

    assert p.group == group

    assert p.name in p.node_str
    assert p.name in str(p)
    assert p.name in repr(p)

    # Ensure no spurious output
    captured = capsys.readouterr()
    assert captured.out == ""


def test_none_shape_param(capsys):
    p = Param("p", np.ones((3, 4, 5)), shape=(None, 5))
    assert p.shape == (4, 5)
    assert p.batch_shape == (3,)
    p.value = np.ones((2, 3, 5))
    assert p.shape == (3, 5)
    assert p.batch_shape == (2,)

    p.value = None

    assert p.shape == (None, 5)

    with pytest.raises(ParamConfigurationError):
        p.value = np.ones(5)
    with pytest.raises(ParamConfigurationError):
        p.value = np.ones((2, 2))

    p.value = np.ones((4, 5))

    with pytest.raises(ValueError):
        p.shape = (4, 2)
    with pytest.raises(ValueError):
        p.shape = (3, 4, 5)

    # Ensure no spurious output
    captured = capsys.readouterr()
    assert captured.out == ""


@pytest.mark.parametrize("value", [None, 1, [1, 2]])
@pytest.mark.parametrize("dynamic", [True, False])
def test_active_state(value, dynamic):
    p = Param("p", value, dynamic=dynamic)
    M = Module()
    M.p = p

    with ActiveContext(M):
        if not (p.static and p.value is None):
            with pytest.raises(ActiveStateError):
                p.value = 1
        with pytest.raises(ActiveStateError):
            p.to_dynamic()
        with pytest.raises(ActiveStateError):
            p.to_static()
        with pytest.raises(ActiveStateError):
            p.to_pointer(lambda p: p.o.value)

    # Live param
    if p.static and p.value is None:
        with ActiveContext(M):
            p.value = 1
            assert p.value == 1
        assert p.value is None


def test_bad_init():
    # Missmatch value and shape
    with pytest.raises(ParamConfigurationError):
        Param("test", 1.0, shape=(1, 2, 3))
    with pytest.raises(ParamConfigurationError):
        Param("test", np.ones((3, 3)), shape=(2, 3))
    with pytest.raises(ParamConfigurationError):
        Param("test", np.ones((3, 3)), shape=(3, 2, 3))

    # Shape is not a tuple
    with pytest.raises(ParamConfigurationError):
        Param("test", None, 7)

    # cyclic without full valid
    with pytest.raises(ParamConfigurationError):
        Param("test", cyclic=True)
    with pytest.raises(ParamConfigurationError):
        Param("test", cyclic=True, valid=(0, None))
    with pytest.raises(ParamConfigurationError):
        Param("test", cyclic=True, valid=(None, 1))

    # Bad valid
    with pytest.raises(ParamConfigurationError):
        Param("test", valid=[0, 1])  # not tuple
    with pytest.raises(ParamConfigurationError):
        Param("test", valid=(1, 0))  # switch high/low
    with pytest.raises(ParamConfigurationError):
        Param("test", valid=(0, 1, 2))  # not length 2

    # Bad pointer init
    p = Param("p")
    point = Param("test", p, shape=(2,))  # check does not raise
    with pytest.raises(ParamTypeError):
        point.shape = ()
    Param("test", p, dynamic=True)
    Param("test", p, dynamic=False)
    with pytest.raises(ParamTypeError):
        Param("test", p, batch_shape=(2,))


@pytest.mark.parametrize("value", [None, 1, (1, 2)])
@pytest.mark.parametrize("dynamic", [None, True, False])
def test_change_type(value, dynamic, capsys):
    p = Param("test", value, dynamic=dynamic)

    if value is not None:
        assert np.allclose(p.npvalue, np.array(value))
    else:
        assert p.value is None

    p.to_dynamic()
    assert p.dynamic
    if value is not None:
        assert np.allclose(p.npvalue, np.array(value))
    else:
        assert p.value is None

    p.to_static()
    assert p.static
    if value is not None:
        assert np.allclose(p.npvalue, np.array(value))
    else:
        assert p.value is None

    p.to_dynamic(3)
    assert p.dynamic
    assert p.value == 3
    p.to_static(4)
    assert p.static
    assert p.value == 4

    p2 = Param("pointme", 5)
    p.to_pointer(p2)
    assert p.pointer
    assert p.value == 5

    if dynamic:
        p.to_dynamic()
        assert p.dynamic
        assert p.value == 5
    else:
        p.to_static()
        assert p.static
        assert p.value == 5

    p.to_pointer(lambda a: a.p2.value)

    if dynamic:
        p.to_dynamic()
        assert p.dynamic
        assert p.value is None
    else:
        p.to_static()
        assert p.static
        assert p.value is None

    # Ensure no spurious output
    captured = capsys.readouterr()
    assert captured.out == ""

    # Invalid change type
    with pytest.raises(ParamTypeError):
        p.to_pointer(1.0)
    with pytest.raises(ParamTypeError):
        p.to_static(p2)
    with pytest.raises(ParamTypeError):
        p.to_static(lambda p: p.other.value)
    with pytest.raises(ParamTypeError):
        p.to_dynamic(p2)
    with pytest.raises(ParamTypeError):
        p.to_dynamic(lambda p: p.other.value)


@pytest.mark.parametrize("value", [None, 1, (1, 2)])
@pytest.mark.parametrize("valid", [None, (0, None), (None, 3), (0, 3)])
def test_param_to(value, valid):
    if backend.backend == "jax":
        device = backend.jax.devices()[0]
        backend.jax.config.update("jax_enable_x64", True)
    else:
        device = "cpu"

    p = Param("test", value, valid=valid)
    p = p.to()
    p = p.to(dtype=backend.module.float64, device=device)


def test_to_pointer():
    p = Param("p")
    o = Param("o", 1)

    def pointfunc(P):
        return P.o.value

    pointfunc.params = o

    p.to_pointer(pointfunc)

    assert np.allclose(p.npvalue, 1)

    def badpointfunc(P):
        return P.O.value

    p.to_pointer(badpointfunc)

    assert p.batch_shape == ()


def test_param_shape():
    p = Param("p", [1, 2], shape=(2,))
    assert p.shape == (2,)

    p.batch_shape = (4,)
    assert p.batch_shape == (4,)

    with pytest.raises(ValueError):
        p.shape = (3, 2)

    with pytest.raises(ParamConfigurationError):
        p.value = np.ones((3, 2))
    with pytest.raises(ParamConfigurationError):
        p.to_dynamic(np.ones((3, 2)))

    p.batch_shape = None  # Reset to now follow value

    with pytest.raises(ParamConfigurationError):
        p.to_dynamic(np.ones((3, 3)))

    p.value = np.ones((3, 2))

    with pytest.raises(ValueError):
        p.shape = (3,)
    p.shape = (2,)
    assert p.batch_shape == (3,)

    p.value = lambda p: p.other.value
    p.link("other", Param("other"))
    assert p.shape == ()


def test_valid():
    p = Param("test", valid=None)

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


def test_node_str():
    p = Param("p", 1.0)
    assert p.node_str == "p|static: 1"
    p = Param("p", [1.0, 2.0])
    assert p.node_str == "p|static: [1, 2]"
    p = Param("p", [1.0, 2.0, 3.0, 4.0, 5.0])
    assert p.node_str == "p|static: (5,)"
