from caskade import Module, active_cache, forward, Param


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
