import numpy as np
import pytest

from caskade import Module, active_cache, backend, forward, Param


def test_active_cache():
    class TestModule(Module):
        def __init__(self):
            super().__init__()
            self.counter = 0
            self.p = Param("p", 10.0)

        @active_cache
        def expensive_computation(self, x):
            self.counter += 1
            return x * 2

        @active_cache
        @forward
        def another_computation(self, x, p=None):
            x2 = self.expensive_computation(3 * x)  # Should use cache
            self.counter += 1
            return x2 + p

        @forward
        def base_calculation(self, x):
            a = self.expensive_computation(x)
            b = self.another_computation(x)
            res = a + b
            a2 = self.expensive_computation(2 * x)  # Should use cache
            b2 = self.another_computation(2 * x)  # Should use cache
            res2 = a2 + b2
            return res + res2

    m = TestModule()
    m.counter = 0
    assert m.expensive_computation(2) == 4
    assert m.counter == 1, "Should compute the result on first call"

    # Call again with same argument, should use cache
    m.counter = 0
    assert m.another_computation(2) == 22
    assert m.counter == 2, "Should only call computations twice"

    # Call with different argument, should compute again
    m.counter = 0
    assert m.base_calculation(3) == 44  # 6, 16, 6, 16
    assert m.counter == 2, "Should only call computations twice due to caching"

    assert not m._active_cache, "Cache should be cleared after forward call"

    class InheritTestModule(TestModule):
        # Check overriding works and cache is separate for each method
        @active_cache
        def expensive_computation(self, x):
            """This is the docstring for the overridden expensive_computation method."""
            self.counter += 1
            return x * 3

        @active_cache
        @forward
        def evenmore_computation(self, x, p=None):
            x2 = self.expensive_computation(4 * x)  # Should use cache
            self.counter += 1
            return x2 + p

        @forward
        def big_calculation(self, x):
            res = self.base_calculation(x)

            res2 = self.evenmore_computation(x)
            return res + res2

    m2 = InheritTestModule()
    m2.counter = 0
    assert m2.expensive_computation(2) == 6
    assert m2.counter == 1, "Should compute the result on first call"

    m2.counter = 0
    assert m2.evenmore_computation(2) == 34  # 24 + 10
    assert m2.counter == 2, "Should only call computations twice"

    m2.counter = 0
    assert m2.big_calculation(3) == 75  # 9, 19, 9, 19, 19
    assert m2.counter == 3, "Should only call computations three times due to caching"

    assert (
        InheritTestModule.expensive_computation.__doc__
        == "This is the docstring for the overridden expensive_computation method."
    )

    assert not m2._active_cache, "Cache should be cleared after forward call"


def test_active_cache_any_position():
    class TestModule(Module):
        def __init__(self):
            super().__init__()
            self.counter = 0
            self.p = Param("p", 2.0)

        @forward
        @active_cache  # below another decorator
        def inner(self, x, p=None):
            self.counter += 1
            return x * p

        @forward
        def outer(self, x):
            return self.inner(x) + self.inner(x)

    m = TestModule()
    assert m.outer(3.0) == 12
    assert m.counter == 1, "Should only compute once"
    assert not m._active_cache, "Cache should be cleared after forward call"


def test_active_cache_override_is_separate():
    class Base(Module):
        @active_cache
        def compute(self):
            return 1

        @forward
        def run(self):
            return self.compute() + self.compute()

    class Child(Base):
        @active_cache
        def compute(self):
            # The parent method has its own cached result
            return 10 + super().compute() + super().compute()

    assert Base().run() == 2
    assert Child().run() == 24


def test_active_cache_reset():
    class TestModule(Module):
        def __init__(self):
            super().__init__()
            self.counter = 0

        @active_cache
        def compute(self):
            self.counter += 1
            return self.counter

        @forward
        def run(self):
            first = self.compute()
            self.reset_active_cache()
            return first + self.compute()

    assert TestModule().run() == 3


def test_active_cache_cleared_in_subgraph():
    class Child(Module):
        def __init__(self):
            super().__init__("child")
            self.counter = 0

        @active_cache
        def compute(self):
            self.counter += 1
            return self.counter

    class Parent(Module):
        def __init__(self, child):
            super().__init__("parent")
            self.hierarchical_link("child", child)

        @forward
        def run(self):
            return self.child.compute() + self.child.compute()

    child = Child()
    parent = Parent(child)
    assert parent.run() == 2
    assert not child._active_cache, "Cache should be cleared after forward call"
    assert parent.run() == 4, "Should compute again in the next simulation"


def test_active_cache_in_a_compiled_function():
    """Under ``jax.jit`` a cached method is worked out once in each call of the
    compiled function, from the values of that call. Nothing is kept from the
    call in which the function was compiled."""
    if backend.backend != "jax":
        pytest.skip("needs jit")
    import jax
    import jax.numpy as jnp

    class Sim(Module):
        def __init__(self):
            super().__init__("sim")
            self.a = Param("a", shape=(), dynamic=True)
            self.s = Param("s", 10.0, shape=())
            self.calls = 0

        @active_cache
        @forward
        def heavy(self, a, s):
            self.calls += 1
            return jnp.sin(a) + s

        @forward
        def run(self, a):
            return self.heavy() + self.heavy() * a + self.heavy()

    def expected(a, s=10.0):
        return (np.sin(a) + s) * (2 + a)

    sim = Sim()
    run = jax.jit(sim.run)
    first = jnp.array([0.5])
    assert np.allclose(run(first), expected(0.5))
    assert sim.calls == 1  # the three uses share one result
    # the result made while compiling is gone, it belongs to that call
    assert not sim._active_cache and not sim.active

    # A new value goes through the cached method, with no python call
    assert np.allclose(run(jnp.array([1.5])), expected(1.5))
    assert sim.calls == 1 and not sim._active_cache

    # The compiled function works the cached method out once
    assert str(jax.make_jaxpr(sim.run)(first)).count("sin") == 1
    assert not sim._active_cache

    # Other transforms go through it as well
    grad = jax.grad(run)(first)
    assert np.allclose(grad, np.cos(0.5) * 2.5 + np.sin(0.5) + 10.0)
    batch = jax.vmap(run)(jnp.array([[0.5], [1.5]]))
    assert np.allclose(batch, [expected(0.5), expected(1.5)])

    # A static value is given at call time to be followed, as for any compiled method
    run_with = jax.jit(lambda p, s: sim.run(p, static_params=s))
    assert np.allclose(run_with(first, jnp.array([10.0])), expected(0.5))
    assert np.allclose(run_with(first, jnp.array([20.0])), expected(0.5, 20.0))
    assert not sim._active_cache and not sim.active
