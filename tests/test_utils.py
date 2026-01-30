import pytest
from caskade import backend
import numpy as np


def test_broadcast_cat():
    with pytest.raises(ValueError):
        backend.broadcast_cat(())

    arr1 = backend.as_array(1.0)
    arr2 = backend.as_array(np.ones((2, 2)))
    arr3 = backend.as_array(np.ones((3, 2, 3)))
    assert backend.broadcast_cat((arr1, arr2, arr3)).shape == (3, 2, 6)

    with pytest.raises(ValueError):
        backend.broadcast_cat((arr1, arr2, arr3), dim=10)
