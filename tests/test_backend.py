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
