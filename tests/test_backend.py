from caskade import backend, Param
from torch import Tensor
from numpy import ndarray
from jax import Array

import pytest


def test_backend():

    init_backend = backend.backend

    # Change the backend
    backend.backend = "torch"
    p = Param("p", 1.0)
    assert isinstance(p.value, Tensor)
    backend.backend = "numpy"
    p = Param("p", 1.0)
    assert isinstance(p.value, ndarray)
    backend.backend = "jax"
    p = Param("p", 1.0)
    assert isinstance(p.value, Array)
    backend.backend = init_backend

    with pytest.raises(ValueError):
        backend.backend = "invalid_backend"
