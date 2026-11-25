"""``static_cache``: a result that is made outside a simulation and kept for
every simulation after it, until the method is called outside one again."""

import copy
import pickle
import warnings

import numpy as np
import pytest

from caskade import (
    ActiveStateError,
    Module,
    Param,
    StaticCacheWarning,
    backend,
    forward,
    static_cache,
)


def number(value):
    return float(backend.to_numpy(value))


class Blur(Module):
    def __init__(self, name="blur"):
        super().__init__(name)
        self.sigma = Param("sigma", 2.0, shape=())
        self.other = Param("other", 5.0, shape=())  # static, the kernel does not use it
        self.amp = Param("amp", shape=(), dynamic=True)
        self.calls = 0

    @static_cache
    @forward
    def kernel(self, sigma, scale=1.0):
        self.calls += 1
        return scale * sigma * 10

    @forward
    def image(self, amp):
        return amp * self.kernel()

    @forward
    def scaled(self, amp):
        return amp * self.kernel(scale=100.0)

    @static_cache
    @forward
    def uses_amp(self, sigma, amp):
        return sigma * amp

    @static_cache
    def reads_amp(self):
        return 2 * self.amp.value

    @static_cache
    @forward
    def twice(self):  # made from another stored result
        return 2 * self.kernel()

    @forward
    def image_twice(self, amp):
        return amp * self.twice()


def quiet(func, *args, **kwargs):
    """The result of a call that must not give a warning."""
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        return func(*args, **kwargs)


def amp(value):
    return backend.as_array([value])


def test_result_is_kept_from_one_simulation_to_the_next():
    blur = Blur()
    assert number(quiet(blur.kernel)) == 20.0 and blur.calls == 1
    assert number(quiet(blur.image, amp(3.0))) == 60.0
    assert number(quiet(blur.image, amp(4.0))) == 80.0
    assert blur.calls == 1  # not run in a simulation
    assert not blur.active

    # Arguments make no difference in a simulation
    assert number(quiet(blur.scaled, amp(3.0))) == 60.0 and blur.calls == 1
    # Outside a simulation every call runs the method and stores what it gives
    assert number(quiet(blur.kernel, scale=2.0)) == 40.0 and blur.calls == 2
    assert number(quiet(blur.image, amp(3.0))) == 120.0

    blur.reset_static_cache()
    with pytest.raises(ActiveStateError, match="'kernel' of 'blur'.*no stored result"):
        blur.image(amp(3.0))
    assert not blur.active and blur.calls == 2


def test_used_after_a_param_it_was_made_from_has_changed():
    blur = Blur()
    blur.kernel()

    # A param that the result was not made from
    blur.other = 6.0
    assert number(quiet(blur.image, amp(1.0))) == 20.0

    # Nothing is said when the value is changed, only when the result is used
    quiet(setattr, blur, "sigma", 3.0)
    with pytest.warns(StaticCacheWarning, match=r"'kernel' of 'blur'.*\['sigma'\] were changed"):
        result = blur.image(amp(1.0))
    assert number(result) == 20.0 and blur.calls == 1  # it is not made again either

    assert number(quiet(blur.kernel)) == 30.0
    assert number(quiet(blur.image, amp(1.0))) == 30.0

    # Values that a simulation is given are not a change
    static = blur.get_values(dynamic=False)
    assert number(quiet(blur.image, amp(1.0), static_params=2 * static)) == 30.0

    blur.sigma.to_dynamic()  # same value, but now it is the simulation that sets it
    with pytest.warns(StaticCacheWarning, match="'sigma'"):
        blur.image(backend.as_array([3.0, 1.0]))


def test_dynamic_value_in_the_result():
    blur = Blur()
    blur.amp = 3.0  # still dynamic
    with pytest.warns(StaticCacheWarning, match=r"'uses_amp' of 'blur'.*dynamic.*\['amp'\]"):
        assert number(blur.uses_amp()) == 6.0
    with pytest.warns(StaticCacheWarning, match=r"'uses_amp'.*\['amp'\]"):
        assert number(blur.uses_amp(amp(4.0))) == 8.0
    with pytest.warns(StaticCacheWarning, match=r"'reads_amp'.*\['amp'\]"):
        assert number(blur.reads_amp()) == 6.0

    # The module has a dynamic param, but this result is not made from it
    assert number(quiet(blur.kernel)) == 20.0
    assert number(quiet(blur.kernel, amp(4.0))) == 20.0

    # A new value for the dynamic param is what a simulation is for
    blur.amp = 5.0
    assert number(quiet(blur.image, amp(1.0))) == 20.0


class Chain(Module):
    def __init__(self):
        super().__init__("chain")
        self.a = Param("a", 1.0, shape=())
        self.b = Param("b", shape=(), dynamic=True)
        self.double = Param("double", lambda p: 2 * p.a.value, link=self.a)
        self.shifted = Param("shifted", lambda p: 1 + p.b.value, link=self.b)

    @static_cache
    @forward
    def table(self, double):
        return 10 * double

    @static_cache
    @forward
    def moving(self, shifted):
        return shifted

    @forward
    def run(self, b):
        return b + self.table()


def test_params_reached_through_a_pointer():
    chain = Chain()
    chain.b = 1.0
    assert number(quiet(chain.table)) == 20.0
    chain.a = 2.0
    with pytest.warns(StaticCacheWarning, match=r"\['a'\] were changed"):
        chain.run(backend.as_array([1.0]))
    chain.table()
    chain.double = lambda p: 3 * p.a.value
    chain.double.link(chain.a)
    with pytest.warns(StaticCacheWarning, match=r"\['double'\] were changed"):
        chain.run(backend.as_array([1.0]))
    assert number(quiet(chain.table)) == 60.0

    with pytest.warns(StaticCacheWarning, match=r"dynamic.*\['b'\]"):
        chain.moving()


def test_result_made_from_another_stored_result():
    blur = Blur()
    with pytest.raises(ActiveStateError, match="'kernel' of 'blur'"):
        blur.twice()  # the one it is made from comes first
    blur.kernel()
    assert number(quiet(blur.twice)) == 40.0
    assert number(quiet(blur.image_twice, amp(1.0))) == 40.0
    blur.sigma = 3.0  # it depends on what the other one was made from
    with pytest.warns(StaticCacheWarning, match=r"'twice' of 'blur'.*\['sigma'\]"):
        blur.image_twice(amp(1.0))


class Shapes(Module):
    def __init__(self):
        super().__init__("shapes")
        self.a = Param("a", [1.0, 2.0], shape=(2,))
        self.b = Param("b", shape=(), dynamic=True)
        self.calls = 0
        self.double = Param("double", self.count, link=self.a)

    def count(self, p):
        self.calls += 1
        return 2 * p.a.value

    @static_cache
    @forward
    def size(self):
        # the shape of a value is not a use of the value, and neither is what
        # caskade looks at to fill the params
        return self.b.shape, self.double.shape, self.double.value.sum()


def test_looking_at_a_shape_is_not_a_use_of_the_value():
    shapes = Shapes()
    b_shape, double_shape, total = quiet(shapes.size, backend.as_array([3.0]))
    assert b_shape == () and double_shape == (2,) and number(total) == 6.0
    assert shapes.calls == 1  # a pointer is still worked out once in a simulation
    made_from = {p.name for p, _, _ in shapes._static_cache[type(shapes).size][1]}
    assert made_from == {"a", "double"}


class Table(Module):
    """Methods that are not ``forward`` methods, they are never in a simulation
    unless one is called from a simulation."""

    def __init__(self, name, inner=None):
        super().__init__(name)
        self.step = Param("step", 2.0, shape=())
        self.inner = inner

    @static_cache
    def grid(self):
        values = 3 * self.step.value
        if self.inner is not None:
            values = values + self.inner.grid()  # runs, and is stored, as well
        return values

    @forward
    def use(self):
        return self.grid()


def test_plain_methods():
    inner = Table("inner")
    outer = Table("outer", inner)
    assert number(quiet(outer.grid)) == 12.0
    assert number(quiet(outer.use)) == 12.0 and number(quiet(inner.use)) == 6.0
    inner.step = 5.0  # the outer result was made from the params of both
    with pytest.warns(StaticCacheWarning, match=r"'grid' of 'outer'.*\['step'\]"):
        assert number(outer.use()) == 12.0
    assert number(quiet(outer.grid)) == 21.0


class Stored(Module):  # pickle needs the class to be defined at the top of a file
    def __init__(self):
        super().__init__("stored")
        self.a = Param("a", 4.0, shape=())

    @static_cache
    @forward
    def table(self, a):
        return 10 * a

    @forward
    def run(self):
        return self.table()


@pytest.mark.parametrize("how", ["pickle", "deepcopy"])
def test_stored_result_goes_with_a_copy(how):
    sim = Stored()
    sim.table()
    other = pickle.loads(pickle.dumps(sim)) if how == "pickle" else copy.deepcopy(sim)
    assert number(quiet(other.run)) == 40.0
    other.a = 5.0
    with pytest.warns(StaticCacheWarning, match=r"\['a'\]"):
        other.run()
    assert number(quiet(sim.run)) == 40.0
    assert np.allclose(number(quiet(other.table)), 50.0)
