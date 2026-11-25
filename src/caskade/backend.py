"""Backend abstraction for array operations.

Each supported array library (**torch**, **jax**, and **numpy**) has its own
:class:`ArrayBackend` subclass holding the library-specific implementation of
every array operation that ``caskade`` needs. The :class:`Backend` class
selects one of them by name and delegates to it. A module-level
:data:`backend` instance is created on import and serves as the primary
interface for users.
"""

import os
import importlib
import importlib.util
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Optional, TypeAlias, Union

import numpy as np
from . import utils

if TYPE_CHECKING:
    # Only type checkers follow these imports, so neither library needs to be
    # installed to use caskade. A type checker that cannot find one of them
    # treats that member of the union as unknown.
    import torch
    import jax

    ArrayLike: TypeAlias = Union[np.ndarray, "torch.Tensor", "jax.Array"]
else:
    #: An array of any supported backend. Type checkers see
    #: ``numpy.ndarray | torch.Tensor | jax.Array``.
    ArrayLike = Any


class ArrayBackend(ABC):
    """Array operations implemented with a single array library.

    There is one subclass per supported library. Operations that are spelled
    the same way in every library are implemented here, the rest are
    implemented by each subclass.
    """

    #: Name used to select this backend
    name: str
    #: Name of the module that holds the array functions of the library
    module_name: str

    def __init__(self):
        self.module = importlib.import_module(self.module_name)

    @property
    @abstractmethod
    def array_type(self) -> type:
        """The array class of the library."""

    @abstractmethod
    def make_array(self, array, dtype=None, device=None):
        """Create a new array, copying the input."""

    @abstractmethod
    def as_array(self, array, dtype=None, device=None):
        """Convert the input to an array, avoiding a copy where possible."""

    @abstractmethod
    def concatenate(self, arrays, axis=0):
        """Join a sequence of arrays along an existing axis."""

    @abstractmethod
    def broadcast_cat(self, arrays, dim=-1):
        """Concatenate arrays along ``dim``, broadcasting all other dimensions."""

    @abstractmethod
    def tolist(self, array) -> list:
        """Convert an array to a (nested) python list."""

    @abstractmethod
    def detach(self, array):
        """Detach an array from any gradient tracking."""

    @abstractmethod
    def to(self, array, dtype=None, device=None):
        """Move and/or cast an array to a device and/or dtype."""

    @abstractmethod
    def to_numpy(self, array) -> np.ndarray:
        """Convert an array to a ``numpy.ndarray``."""

    @abstractmethod
    def ste_clip(self, array, min_val, max_val):
        """Clip an array to a range, preserving gradients at the boundaries
        where the library tracks them (straight-through estimator)."""

    @abstractmethod
    def vmap(self, func, in_dims):
        """Vectorise ``func``, a function of one list of values, over the
        first dimension of each value whose entry in ``in_dims`` is ``0``.
        Values whose entry is ``None`` are passed to every call unchanged."""

    def view(self, array, shape):
        return array.reshape(shape)

    def broadcast_to(self, array, shape):
        return self.module.broadcast_to(array, shape)

    def any(self, array):
        return self.module.any(array)

    def all(self, array):
        return self.module.all(array)

    def log(self, array):
        return self.module.log(array)

    def exp(self, array):
        return self.module.exp(array)

    def sum(self, array, axis=None):
        return self.module.sum(array, axis=axis)


class TorchBackend(ArrayBackend):
    """Array operations implemented with ``torch``."""

    name = "torch"
    module_name = "torch"

    @property
    def array_type(self) -> type:
        return self.module.Tensor

    def make_array(self, array, dtype=None, device=None):
        return self.module.tensor(array, dtype=dtype, device=device)

    def as_array(self, array, dtype=None, device=None):
        return self.module.as_tensor(array, dtype=dtype, device=device)

    def concatenate(self, arrays, axis=0):
        return self.module.cat(arrays, dim=axis)

    def broadcast_cat(self, arrays, dim=-1):
        return utils.broadcast_cat_torch(arrays, dim=dim)

    def tolist(self, array) -> list:
        return array.detach().cpu().tolist()

    def detach(self, array):
        return array.detach()

    def vmap(self, func, in_dims):
        return self.module.vmap(func, in_dims=(list(in_dims),))

    def to(self, array, dtype=None, device=None):
        return array.to(dtype=dtype, device=device)

    def to_numpy(self, array) -> np.ndarray:
        return array.detach().cpu().numpy()

    def ste_clip(self, array, min_val, max_val):
        clipped = self.module.clamp(array, min=min_val, max=max_val)
        return clipped.detach() + (array - array.detach())


class JaxBackend(ArrayBackend):
    """Array operations implemented with ``jax``."""

    name = "jax"
    module_name = "jax.numpy"

    def __init__(self):
        super().__init__()
        self.jax = importlib.import_module("jax")

    @property
    def array_type(self) -> type:
        return self.module.ndarray

    def make_array(self, array, dtype=None, device=None):
        return self.module.array(array, dtype=dtype)

    def as_array(self, array, dtype=None, device=None):
        return self.module.asarray(array, dtype=dtype)

    def concatenate(self, arrays, axis=0):
        return self.module.concatenate(arrays, axis=axis)

    def broadcast_cat(self, arrays, dim=-1):
        return utils.broadcast_cat_jax(arrays, dim=dim)

    def tolist(self, array) -> list:
        return array.block_until_ready().tolist()

    def detach(self, array):
        return array

    def vmap(self, func, in_dims):
        return self.jax.vmap(func, in_axes=(list(in_dims),))

    def to(self, array, dtype=None, device=None):
        return self.jax.device_put(array.astype(dtype), device=device)

    def to_numpy(self, array) -> np.ndarray:
        return np.array(array.block_until_ready())

    def ste_clip(self, array, min_val, max_val):
        clipped = self.module.clip(array, min=min_val, max=max_val)
        stop_gradient = self.jax.lax.stop_gradient
        return stop_gradient(clipped) + (array - stop_gradient(array))


class NumpyBackend(ArrayBackend):
    """Array operations implemented with ``numpy``."""

    name = "numpy"
    module_name = "numpy"

    @property
    def array_type(self) -> type:
        return self.module.ndarray

    def make_array(self, array, dtype=None, device=None):
        return self.module.array(array, dtype=dtype)

    def as_array(self, array, dtype=None, device=None):
        return self.module.asarray(array, dtype=dtype)

    def concatenate(self, arrays, axis=0):
        return self.module.concatenate(arrays, axis=axis)

    def broadcast_cat(self, arrays, dim=-1):
        return utils.broadcast_cat_numpy(arrays, dim=dim)

    def tolist(self, array) -> list:
        return array.tolist()

    def detach(self, array):
        return array

    def vmap(self, func, in_dims):
        # numpy has no vmap, so the calls are made one at a time and stacked
        def looped(values):
            size = next(len(v) for v, dim in zip(values, in_dims) if dim is not None)
            return _stack(
                [
                    func([v if dim is None else v[i] for v, dim in zip(values, in_dims)])
                    for i in range(size)
                ]
            )

        return looped

    def to(self, array, dtype=None, device=None):
        return array.astype(dtype)

    def to_numpy(self, array) -> np.ndarray:
        return array

    def ste_clip(self, array, min_val, max_val):
        # numpy does not track gradients, so this is a standard clip
        return np.clip(array, a_min=min_val, a_max=max_val)


def _stack(results: list):
    """Stack the results of repeated calls along a new first dimension, going
    into tuples, lists and dictionaries to stack the arrays inside them."""
    first = results[0]
    if isinstance(first, (tuple, list)):
        return type(first)(_stack([r[i] for r in results]) for i in range(len(first)))
    if isinstance(first, dict):
        return {key: _stack([r[key] for r in results]) for key in first}
    if first is None:
        return None
    return np.stack(results)


#: The available backends by name.
ARRAY_BACKENDS: dict[str, type[ArrayBackend]] = {
    cls.name: cls for cls in (TorchBackend, JaxBackend, NumpyBackend)
}


class Backend:
    """Unified interface for array operations across torch, jax, and numpy.

    Provides a single API for creating and manipulating arrays regardless of
    the underlying library. Every method delegates to the
    :class:`ArrayBackend` of the selected library, which may be switched at
    any time by setting the ``backend`` attribute.

    Parameters
    ----------
    backend : str, optional
        Backend name: ``"torch"``, ``"jax"``, or ``"numpy"``. If ``None``,
        reads from the ``CASKADE_BACKEND`` environment variable. If that is
        not set either, the first of torch, jax, and numpy that is installed
        is used.

    Examples
    --------
    Use the module-level ``backend`` instance to switch backends::

        from caskade import backend
        backend.backend = "numpy"
        arr = backend.make_array([1.0, 2.0, 3.0])
    """

    def __init__(self, backend: Optional[str] = None):
        """Initialize the backend.

        Parameters
        ----------
        backend : str, optional
            Backend name: ``"torch"``, ``"jax"``, or ``"numpy"``. If ``None``,
            reads from the ``CASKADE_BACKEND`` environment variable. If that
            is not set either, the first of torch, jax, and numpy that is
            installed is used.
        """
        self.backend = backend

    @property
    def backend(self) -> str:
        """str : Name of the active backend (``"torch"``, ``"jax"``, or ``"numpy"``)."""
        return self._backend

    @backend.setter
    def backend(self, backend: Optional[str]):
        if backend is None:
            backend = os.getenv("CASKADE_BACKEND", "none").lower()
            if backend == "none":  # Try to find available backend
                if importlib.util.find_spec("torch") is not None:
                    backend = "torch"
                elif importlib.util.find_spec("jax") is not None:
                    backend = "jax"
                else:
                    backend = "numpy"

        if backend not in ARRAY_BACKENDS:
            raise ValueError(f"Unsupported backend: {backend}")
        # Only switch once the library has been imported successfully
        self._implementation = ARRAY_BACKENDS[backend]()
        self._backend = backend

    @property
    def module(self):
        """module : The array module of the active backend (``torch``,
        ``jax.numpy``, or ``numpy``)."""
        return self._implementation.module

    @property
    def jax(self):
        """module : The top-level ``jax`` module. Only available with the jax backend."""
        if not isinstance(self._implementation, JaxBackend):
            raise AttributeError(f"The '{self.backend}' backend has no attribute 'jax'")
        return self._implementation.jax

    @property
    def array_type(self) -> type:
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
        return self._implementation.array_type

    def make_array(self, array, dtype=None, device=None) -> ArrayLike:
        """Create a new array, copying the input.

        Parameters
        ----------
        array : array_like
            Input data, such as a number, a (nested) sequence, or an array.
        dtype : optional
            Desired data type. Inferred from the input by default.
        device : optional
            Device to place the array on. Only used by the torch backend.

        Returns
        -------
        ArrayLike
            A new array of the active backend.
        """
        return self._implementation.make_array(array, dtype=dtype, device=device)

    def as_array(self, array, dtype=None, device=None) -> ArrayLike:
        """Convert the input to an array, avoiding a copy where possible.

        Parameters
        ----------
        array : array_like
            Input data, such as a number, a (nested) sequence, or an array.
        dtype : optional
            Desired data type. Inferred from the input by default.
        device : optional
            Device to place the array on. Only used by the torch backend.

        Returns
        -------
        ArrayLike
            An array of the active backend.
        """
        return self._implementation.as_array(array, dtype=dtype, device=device)

    def concatenate(self, arrays, axis: int = 0) -> ArrayLike:
        """Join a sequence of arrays along an existing axis.

        Parameters
        ----------
        arrays : sequence of ArrayLike
            Arrays to join. Their shapes must match except along ``axis``.
        axis : int, optional
            Axis along which to join. Defaults to ``0``.

        Returns
        -------
        ArrayLike
            The concatenated array.
        """
        return self._implementation.concatenate(arrays, axis=axis)

    def broadcast_cat(self, arrays, dim: int = -1) -> ArrayLike:
        """Concatenate arrays along ``dim``, broadcasting all other dimensions.

        Parameters
        ----------
        arrays : sequence of ArrayLike
            Arrays to join.
        dim : int, optional
            Dimension along which to join. Defaults to ``-1``.

        Returns
        -------
        ArrayLike
            The concatenated array.
        """
        return self._implementation.broadcast_cat(arrays, dim=dim)

    def tolist(self, array: ArrayLike) -> list:
        """Convert an array to a (nested) python list.

        Parameters
        ----------
        array : ArrayLike
            Input array.

        Returns
        -------
        list
            The values of the array.
        """
        return self._implementation.tolist(array)

    def view(self, array: ArrayLike, shape) -> ArrayLike:
        """Give a new shape to an array without changing its data.

        Parameters
        ----------
        array : ArrayLike
            Input array.
        shape : tuple of int
            The new shape.

        Returns
        -------
        ArrayLike
            The reshaped array.
        """
        return self._implementation.view(array, shape)

    def broadcast_to(self, array: ArrayLike, shape) -> ArrayLike:
        """Broadcast an array to a new shape.

        Parameters
        ----------
        array : ArrayLike
            Input array.
        shape : tuple of int
            The shape to broadcast to.

        Returns
        -------
        ArrayLike
            The broadcast array, a view of the input where the library allows.
        """
        return self._implementation.broadcast_to(array, shape)

    def vmap(self, func, in_dims):
        """Vectorise a function of one list of values over a batch dimension.

        Parameters
        ----------
        func : Callable
            A function that takes a single list of values.
        in_dims : sequence of int or None
            One entry for each value. ``0`` to go over the first dimension of
            that value, ``None`` to pass the same value to every call.

        Returns
        -------
        Callable
            A function that takes the list of batched values and returns the
            results of ``func`` with a new first dimension. This is
            ``torch.vmap`` or ``jax.vmap`` for those backends. ``numpy`` has
            no such transform, there the calls are made one at a time.
        """
        return self._implementation.vmap(func, in_dims)

    def detach(self, array: ArrayLike) -> ArrayLike:
        """Detach an array from any gradient tracking.

        Parameters
        ----------
        array : ArrayLike
            Input array.

        Returns
        -------
        ArrayLike
            The detached array. Only the torch backend changes anything.
        """
        return self._implementation.detach(array)

    def to(self, array: ArrayLike, dtype=None, device=None) -> ArrayLike:
        """Move and/or cast an array to a device and/or dtype.

        Parameters
        ----------
        array : ArrayLike
            Input array.
        dtype : optional
            Desired data type.
        device : optional
            Desired device. Ignored by the numpy backend.

        Returns
        -------
        ArrayLike
            The array with the requested dtype, on the requested device.
        """
        return self._implementation.to(array, dtype=dtype, device=device)

    def to_numpy(self, array: ArrayLike) -> np.ndarray:
        """Convert an array to a ``numpy.ndarray``.

        Parameters
        ----------
        array : ArrayLike
            Input array.

        Returns
        -------
        numpy.ndarray
            The values of the array.
        """
        return self._implementation.to_numpy(array)

    def ste_clip(self, array: ArrayLike, min_val, max_val) -> ArrayLike:
        """Clip an array to a range such that gradients are preserved at the
        boundaries (straight-through estimator).

        Parameters
        ----------
        array : ArrayLike
            Input array.
        min_val, max_val : ArrayLike or float
            Lower and upper bounds of the range.

        Returns
        -------
        ArrayLike
            The clipped array. With the numpy backend this is a standard clip.
        """
        return self._implementation.ste_clip(array, min_val, max_val)

    def any(self, array: ArrayLike):
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
        return self._implementation.any(array)

    def all(self, array: ArrayLike):
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
        return self._implementation.all(array)

    def log(self, array: ArrayLike) -> ArrayLike:
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
        return self._implementation.log(array)

    def exp(self, array: ArrayLike) -> ArrayLike:
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
        return self._implementation.exp(array)

    def sum(self, array: ArrayLike, axis=None) -> ArrayLike:
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
        return self._implementation.sum(array, axis=axis)


#: Module-level :class:`Backend` instance used as the default entry point.
#: Import and configure this object to switch backends globally::
#:
#:     from caskade import backend
#:     backend.backend = "numpy"
backend = Backend()
