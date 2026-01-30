import os
import sys
import pytest
from caskade import backend
import numpy as np

if sys.platform.startswith("win"):
    pytest.skip(reason="Windows cant run these tests", allow_module_level=True)


# This forces the test to run in a separate subprocess
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


def test_broadcast_cat():
    with pytest.raises(ValueError):
        backend.broadcast_cat(())

    arr1 = backend.as_array(1.0)
    arr2 = backend.as_array(np.ones((2, 2)))
    arr3 = backend.as_array(np.ones((3, 2, 3)))
    assert backend.broadcast_cat((arr1, arr2, arr3)).shape == (3, 2, 6)

    with pytest.raises(ValueError):
        backend.broadcast_cat((arr1, arr2, arr3), dim=10)
