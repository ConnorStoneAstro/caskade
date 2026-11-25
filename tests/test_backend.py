import sys

from caskade import backend, Param
from numpy import ndarray

import pytest


def test_backend():

    init_backend = backend.backend

    # Change the backend
    if backend.backend == "torch":
        from torch import Tensor

        p = Param("p", 1.0)
        assert isinstance(p.value, Tensor)
    if backend.backend == "jax":
        from jax import Array

        p = Param("p", 1.0)
        assert isinstance(p.value, Array)

    backend.backend = "numpy"
    p = Param("p", 1.0)
    assert isinstance(p.value, ndarray)
    backend.backend = init_backend

    with pytest.raises(ValueError):
        backend.backend = "invalid_backend"


def test_auto_set_backend_torch(monkeypatch):
    if backend.backend != "torch":
        pytest.skip("Skipping test because backend is not torch")

    monkeypatch.delenv("CASKADE_BACKEND", raising=False)
    monkeypatch.setitem(sys.modules, "jax", None)
    backend.backend = None

    assert backend.backend == "torch"


def test_auto_set_backend_jax(monkeypatch):
    if backend.backend != "jax":
        pytest.skip("Skipping test because backend is not jax")

    monkeypatch.delenv("CASKADE_BACKEND", raising=False)
    monkeypatch.setitem(sys.modules, "torch", None)
    backend.backend = None

    assert backend.backend == "jax"


def test_auto_set_backend_numpy(monkeypatch):
    if backend.backend != "numpy":
        pytest.skip("Skipping test because backend is not numpy")

    monkeypatch.delenv("CASKADE_BACKEND", raising=False)
    monkeypatch.setitem(sys.modules, "torch", None)
    monkeypatch.setitem(sys.modules, "jax", None)
    backend.backend = None

    assert backend.backend == "numpy"


def test_array_backends():
    import numpy as np
    from caskade.backend import ARRAY_BACKENDS, ArrayBackend, Backend

    assert set(ARRAY_BACKENDS) == {"torch", "jax", "numpy"}
    with pytest.raises(TypeError):
        ArrayBackend()  # only the subclass for a library can be used

    numpy_backend = Backend("numpy")
    assert numpy_backend.backend == "numpy"
    assert numpy_backend.module is np
    with pytest.raises(AttributeError):
        numpy_backend.jax

    # A failed switch leaves the backend as it was
    with pytest.raises(ValueError):
        numpy_backend.backend = "invalid_backend"
    assert numpy_backend.backend == "numpy"
    assert numpy_backend.module is np

    x = numpy_backend.make_array([1.0, 2.0])
    assert isinstance(x, numpy_backend.array_type)
    assert isinstance(numpy_backend.as_array([1.0, 2.0]), np.ndarray)
    assert numpy_backend.tolist(x) == [1.0, 2.0]
    assert numpy_backend.view(x, (2, 1)).shape == (2, 1)
    assert numpy_backend.detach(x) is x
    assert numpy_backend.to(x, dtype=np.float32).dtype == np.float32
    assert numpy_backend.to_numpy(x) is x
    assert numpy_backend.concatenate((x, x)).shape == (4,)
    assert numpy_backend.broadcast_cat((x, np.ones((3, 2)))).shape == (3, 4)
    assert numpy_backend.tolist(numpy_backend.ste_clip(x, 1.5, 1.8)) == [1.5, 1.8]
    assert numpy_backend.any(x > 1.5) and not numpy_backend.all(x > 1.5)
    assert np.allclose(numpy_backend.exp(numpy_backend.log(x)), x)
    assert numpy_backend.sum(x) == 3.0
