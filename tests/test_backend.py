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
