import os
import importlib
from typing import Annotated

from torch import Tensor
import numpy as np

ArrayLike = Annotated[
    Tensor,
    "One of: torch.Tensor, numpy.ndarray, jax.numpy.ndarray depending on the chosen backend.",
]


class Backend:
    def __init__(self, backend=None):
        self.backend = backend

    @property
    def backend(self):
        return self._backend

    @backend.setter
    def backend(self, backend):
        if backend is None:
            backend = os.getenv("CASKADE_BACKEND", "torch")
        self.module = self._load_backend(backend)
        self._backend = backend

    def _load_backend(self, backend):
        if backend == "torch":
            self.setup_torch()
            return importlib.import_module("torch")
        elif backend == "jax":
            self.setup_jax()
            return importlib.import_module("jax.numpy")
        elif backend == "numpy":
            self.setup_numpy()
            return importlib.import_module("numpy")
        elif backend == "object":
            self.setup_object()
            return None
        else:
            raise ValueError(f"Unsupported backend: {backend}")

    def setup_torch(self):
        self.make_array = self._make_array_torch
        self._array_type = self._array_type_torch
        self.concatenate = self._concatenate_torch
        self.copy = self._copy_torch
        self.tolist = self._tolist_torch
        self.view = self._view_torch
        self.as_array = self._as_array_torch
        self.to = self._to_torch
        self.to_numpy = self._to_numpy_torch
        self.atan = self._atan_torch

    def setup_jax(self):
        self.jax = importlib.import_module("jax")
        self.make_array = self._make_array_jax
        self._array_type = self._array_type_jax
        self.concatenate = self._concatenate_jax
        self.copy = self._copy_jax
        self.tolist = self._tolist_jax
        self.view = self._view_jax
        self.as_array = self._as_array_jax
        self.to = self._to_jax
        self.to_numpy = self._to_numpy_jax
        self.atan = self._atan_jax

    def setup_numpy(self):
        self.make_array = self._make_array_numpy
        self._array_type = self._array_type_numpy
        self.concatenate = self._concatenate_numpy
        self.copy = self._copy_numpy
        self.tolist = self._tolist_numpy
        self.view = self._view_numpy
        self.as_array = self._as_array_numpy
        self.to = self._to_numpy
        self.to_numpy = self._to_numpy_numpy
        self.atan = self._atan_numpy

    def setup_object(self):
        self.make_array = self._make_array_object
        self._array_type = self._array_type_object
        self.concatenate = None
        self.copy = None
        self.detach = None
        self.tolist = None
        self.view = None
        self.as_array = self._as_array_object
        self.to = None
        self.to_numpy = self._to_numpy_object

    @property
    def array_type(self):
        return self._array_type()

    def _make_array_torch(self, array, dtype=None, device=None):
        return self.module.tensor(array, dtype=dtype, device=device)

    def _make_array_jax(self, array, dtype=None, **kwargs):
        return self.module.array(array, dtype=dtype)

    def _make_array_numpy(self, array, dtype=None, **kwargs):
        return self.module.array(array, dtype=dtype)

    def _make_array_object(self, array, **kwargs):
        return array

    def _array_type_torch(self):
        return self.module.Tensor

    def _array_type_jax(self):
        return self.module.ndarray

    def _array_type_numpy(self):
        return self.module.ndarray

    def _array_type_object(self):
        return object

    def _concatenate_torch(self, arrays, axis=0):
        return self.module.cat(arrays, dim=axis)

    def _concatenate_jax(self, arrays, axis=0):
        return self.module.concatenate(arrays, axis=axis)

    def _concatenate_numpy(self, arrays, axis=0):
        return self.module.concatenate(arrays, axis=axis)

    def _copy_torch(self, array):
        return array.detach().clone()

    def _copy_jax(self, array):
        return self.module.copy(array)

    def _copy_numpy(self, array):
        return self.module.copy(array)

    def _tolist_torch(self, array):
        return array.detach().cpu().tolist()

    def _tolist_jax(self, array):
        return array.block_until_ready().tolist()

    def _tolist_numpy(self, array):
        return array.tolist()

    def _view_torch(self, array, shape):
        return array.view(shape)

    def _view_jax(self, array, shape):
        return array.reshape(shape)

    def _view_numpy(self, array, shape):
        return array.reshape(shape)

    def _as_array_torch(self, array, dtype=None, device=None):
        return self.module.as_tensor(array, dtype=dtype, device=device)

    def _as_array_jax(self, array, dtype=None, **kwargs):
        return self.module.asarray(array, dtype=dtype)

    def _as_array_numpy(self, array, dtype=None, **kwargs):
        return self.module.asarray(array, dtype=dtype)

    def _as_array_object(self, array, **kwargs):
        return array

    def _to_torch(self, array, dtype=None, device=None):
        return array.to(dtype=dtype, device=device)

    def _to_jax(self, array, dtype=None, device=None):
        return self.jax.device_put(array.astype(dtype), device=device)

    def _to_numpy(self, array, dtype=None, **kwargs):
        return array.astype(dtype)

    def _to_numpy_torch(self, array):
        return array.detach().cpu().numpy()

    def _to_numpy_jax(self, array):
        return np.array(array.block_until_ready())

    def _to_numpy_numpy(self, array):
        return array

    def _to_numpy_object(self, array):
        return np.array(array)

    def any(self, array):
        return self.module.any(array)

    def all(self, array):
        return self.module.all(array)

    def sum(self, array, axis=None):
        return self.module.sum(array, axis=axis)

    def tan(self, array):
        return self.module.tan(array)

    def _atan_torch(self, array):
        return self.module.atan(array)

    def _atan_jax(self, array):
        return self.module.arctan(array)

    def _atan_numpy(self, array):
        return self.module.arctan(array)

    def sqrt(self, array):
        return self.module.sqrt(array)


backend = Backend()
