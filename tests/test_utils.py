import sys
import pytest


# This forces the test to run in a separate subprocess
@pytest.mark.filterwarnings("ignore:os.fork()")
@pytest.mark.filterwarnings("ignore:This Process")
@pytest.mark.forked
def test_missing_jax(monkeypatch):
    monkeypatch.setitem(sys.modules, "jax", None)

    for key in list(sys.modules.keys()):
        if key.startswith("caskade"):
            del sys.modules[key]

    import caskade as ck

    assert ck.utils.jnp is None
