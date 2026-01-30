import sys
import pytest


# This forces the test to run in a separate subprocess
@pytest.mark.forked
def test_plot_missing_library(monkeypatch):
    monkeypatch.setitem(sys.modules, "jax", None)

    for key in list(sys.modules.keys()):
        if key.startswith("caskade"):
            del sys.modules[key]

    import caskade as ck

    assert ck.utils.jnp is None
