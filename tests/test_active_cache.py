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

    assert not hasattr(
        m, "_active_cache_expensive_computation"
    ), "Cache should be cleared after forward call"
    assert not hasattr(
        m, "_active_cache_another_computation"
    ), "Cache should be cleared after forward call"

    assert (
        "_active_cache_expensive_computation" in m._cache_attrs
    ), "module should track cache attributes"
    assert (
        "_active_cache_another_computation" in m._cache_attrs
    ), "module should track cache attributes"
    assert len(m._cache_attrs) == 2, "module should track all cache attributes"
