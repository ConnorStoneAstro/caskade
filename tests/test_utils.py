import os
import sys
import pytest


# This forces the test to run in a separate subprocess
@pytest.mark.skipif("win" in sys.platform.lower(), reason="does not run on windows")
@pytest.mark.filterwarnings("ignore:os.fork()")
@pytest.mark.filterwarnings("ignore:This Process")
@pytest.mark.forked
def test_missing_jax(monkeypatch):
    if os.environ.get("CASKADE_BACKEND", "torch") == "jax":
        pytest.skip()

    monkeypatch.setitem(sys.modules, "jax.numpy", None)

    for key in list(sys.modules.keys()):
        if key.startswith("caskade"):
            del sys.modules[key]

    import caskade as ck

    assert ck.utils.jnp is None
