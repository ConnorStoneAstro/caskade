"""Backend abstraction for array operations.

Provides a unified :class:`Backend` class that delegates array creation and
manipulation to one of three libraries: **torch**, **jax**, or **numpy**.
A module-level :data:`backend` instance is created on import and serves as the
primary interface for users.
"""
import os
import importlib
from typing import Annotated

from torch import Tensor
import numpy as np
from . import utils

#: Type alias for array types across backends.
#: Resolves to ``torch.Tensor``, ``numpy.ndarray``, or ``jax.numpy.ndarray``
#: depending on the active backend.
ArrayLike = Annotated[
    Tensor,
    "One of: torch.Tensor, numpy.ndarray, jax.numpy.ndarray depending on the chosen backend.",
]


class Backend:
    """Unified interface for array operations across torch, jax, and numpy.

    Provides a single API for creating and manipulating arrays regardless of
    the underlying library. Methods such as ``make_array``, ``concatenate``,
    ``to``, ``sigmoid``, and ``logit`` are dynamically bound when the backend
    is set, delegating to the appropriate library-specific implementation.

    Parameters
    ----------
    backend : str, optional
        Backend name: ``"torch"``, ``"jax"``, or ``"numpy"``. If ``None``,
        reads from the ``CASKADE_BACKEND`` environment variable, defaulting
        to ``"torch"``.

    Examples
    --------
    Use the module-level ``backend`` instance to switch backends::

        from caskade import backend
        backend.backend = "numpy"
        arr = backend.make_array([1.0, 2.0, 3.0])
    """

    def __init__(self, backend=None):
        """Initialize the backend.

        Parameters
        ----------
        backend : str, optional
            Backend name: ``"torch"``, ``"jax"``, or ``"numpy"``. If ``None``,
            reads from the ``CASKADE_BACKEND`` environment variable, defaulting
            to ``"torch"``.
        """
        self.backend = backend

    @property
    def backend(self):
        """str : Name of the active backend (``"torch"``, ``"jax"``, or ``"numpy"``)."""
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
        else:
            raise ValueError(f"Unsupported backend: {backend}")

    def setup_torch(self):
        self.make_array = self._make_array_torch
        self._array_type = self._array_type_torch
        self.concatenate = self._concatenate_torch
        self.broadcast_cat = utils.broadcast_cat_torch
        self.tolist = self._tolist_torch
        self.view = self._view_torch
        self.detach = self._detach_torch
        self.as_array = self._as_array_torch
        self.to = self._to_torch
        self.to_numpy = self._to_numpy_torch
        self.logit = self._logit_torch
        self.sigmoid = self._sigmoid_torch

    def setup_jax(self):
        self.jax = importlib.import_module("jax")
        self.make_array = self._make_array_jax
        self._array_type = self._array_type_jax
        self.concatenate = self._concatenate_jax
        self.broadcast_cat = utils.broadcast_cat_jax
        self.tolist = self._tolist_jax
        self.view = self._view_jax
        self.detach = self._detach_jax
        self.as_array = self._as_array_jax
        self.to = self._to_jax
        self.to_numpy = self._to_numpy_jax
        self.logit = self._logit_jax
        self.sigmoid = self._sigmoid_jax

    def setup_numpy(self):
        self.make_array = self._make_array_numpy
        self._array_type = self._array_type_numpy
        self.concatenate = self._concatenate_numpy
        self.broadcast_cat = utils.broadcast_cat_numpy
        self.tolist = self._tolist_numpy
        self.view = self._view_numpy
        self.detach = self._detach_numpy
        self.as_array = self._as_array_numpy
        self.to = self._to_numpy
        self.to_numpy = self._to_numpy_numpy
        self.logit = self._logit_numpy
        self.sigmoid = self._sigmoid_numpy

    @property
    def array_type(self):
        """type : The array class for the active backend.

        Returns ``torch.Tensor``, ``jax.numpy.ndarray``, or ``numpy.ndarray``
        depending on the current backend. Useful for ``isinstance`` checks.

        Returns
        -------
        type
            The array class used by the active backend.

        Examples
        --------
        ::

            isinstance(my_array, backend.array_type)
        """
        return self._array_type()

    def _make_array_torch(self, array, dtype=None, device=None):
        return self.module.tensor(array, dtype=dtype, device=device)

    def _make_array_jax(self, array, dtype=None, **kwargs):
        return self.module.array(array, dtype=dtype)

    def _make_array_numpy(self, array, dtype=None, **kwargs):
        return self.module.array(array, dtype=dtype)

    def _array_type_torch(self):
        return self.module.Tensor

    def _array_type_jax(self):
        return self.module.ndarray

    def _array_type_numpy(self):
        return self.module.ndarray

    def _concatenate_torch(self, arrays, axis=0):
        return self.module.cat(arrays, dim=axis)

    def _concatenate_jax(self, arrays, axis=0):
        return self.module.concatenate(arrays, axis=axis)

    def _concatenate_numpy(self, arrays, axis=0):
        return self.module.concatenate(arrays, axis=axis)

    def _detach_torch(self, array):
        return array.detach()

    def _detach_jax(self, array):
        return array

    def _detach_numpy(self, array):
        return array

    def _tolist_torch(self, array):
        return array.detach().cpu().tolist()

    def _tolist_jax(self, array):
        return array.block_until_ready().tolist()

    def _tolist_numpy(self, array):
        return array.tolist()

    def _view_torch(self, array, shape):
        return array.reshape(shape)

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

    def any(self, array):
        """Test whether any element evaluates to ``True``.

        Parameters
        ----------
        array : ArrayLike
            Input array.

        Returns
        -------
        ArrayLike
            Scalar result; ``True`` if any element is non-zero.
        """
        return self.module.any(array)

    def all(self, array):
        """Test whether all elements evaluate to ``True``.

        Parameters
        ----------
        array : ArrayLike
            Input array.

        Returns
        -------
        ArrayLike
            Scalar result; ``True`` if every element is non-zero.
        """
        return self.module.all(array)

    def log(self, array):
        """Compute the natural logarithm element-wise.

        Parameters
        ----------
        array : ArrayLike
            Input array.

        Returns
        -------
        ArrayLike
            Element-wise natural logarithm of the input.
        """
        return self.module.log(array)

    def exp(self, array):
        """Compute the exponential element-wise.

        Parameters
        ----------
        array : ArrayLike
            Input array.

        Returns
        -------
        ArrayLike
            Element-wise exponential of the input.
        """
        return self.module.exp(array)

    def sum(self, array, axis=None):
        """Sum array elements over a given axis.

        Parameters
        ----------
        array : ArrayLike
            Input array.
        axis : int or None, optional
            Axis along which to sum. If ``None``, sums all elements.

        Returns
        -------
        ArrayLike
            Sum of elements.
        """
        return self.module.sum(array, axis=axis)

    def _sigmoid_torch(self, array):
        return self.module.sigmoid(array)

    def _sigmoid_jax(self, array):
        return self.jax.nn.sigmoid(array)

    def _sigmoid_numpy(self, array):
        return 1 / (1 + self.module.exp(-array))

    def _logit_torch(self, array):
        return self.module.logit(array)

    def _logit_jax(self, array):
        return self.jax.scipy.special.logit(array)

    def _logit_numpy(self, array):
        return np.log(array / (1 - array))


#: Module-level :class:`Backend` instance used as the default entry point.
#: Import and configure this object to switch backends globally::
#:
#:     from caskade import backend
#:     backend.backend = "numpy"
backend = Backend()
