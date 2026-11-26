from contextlib import contextmanager
from typing import Callable, Iterator, Optional, Union, Any, Iterable
from warnings import warn
from math import prod
from numbers import Integral

from numpy import ndarray
import numpy as np

from .backend import backend, ArrayLike
from .base import Node, NodeType, _create_rows
from .errors import ParamConfigurationError, ParamTypeError, ActiveStateError, GraphError
from .warnings import InvalidValueWarning


def valid_shape(batch_shape, shape, value_shape):
    """Check whether a value's shape is compatible with a parameter's shape.

    Validates that ``value_shape`` is consistent with the declared ``shape``
    and optional ``batch_shape``. Dimensions set to ``None`` in ``shape``
    act as wildcards and match any size.

    Parameters
    ----------
    batch_shape : tuple of int or None
        Leading batch dimensions, or ``None`` if the parameter is not batched.
    shape : tuple of int or None, or None
        Expected event dimensions. Individual entries may be ``None``
        (wildcard). If the entire argument is ``None``, any shape is accepted.
    value_shape : tuple of int
        The actual shape of the value to validate.

    Returns
    -------
    bool
        ``True`` if the shapes are compatible, ``False`` otherwise.
    """
    # No shape to compare
    if shape is None:
        return True

    # Determine what to compare
    if batch_shape is None:
        value_shape = value_shape[len(value_shape) - len(shape) :]
    else:
        shape = batch_shape + shape

    # Definitely dont match, wrong lengths
    if len(value_shape) != len(shape):
        return False

    # Check for match or None
    return all(s is None or v == s for v, s in zip(value_shape, shape))


NULL = object()


class _ParamValue:
    """The function of a pointer that takes the value of another ``Param``.

    Called with the pointer ``Param``, it returns the value of the child linked
    under ``key``. This is a class at the top level of the module, rather than a
    function made inside ``Param.to_pointer``, so that a graph with pointers can
    be pickled.
    """

    def __init__(self, key: str):
        self.key = key

    def __call__(self, p: "Param"):
        target: Any = p[self.key]  # the ``Param`` being pointed to
        return target.value


#: The params whose ``value`` has been read, while someone is asking
#: (``_record_reads``). ``None`` the rest of the time.
_READS: Optional[set["Param"]] = None


@contextmanager
def _record_reads() -> Iterator[set["Param"]]:
    """Collect the params whose ``value`` is read while the block runs. This is
    how a ``static_cache`` result knows which params it was made from."""
    global _READS
    outer = _READS
    reads: set[Param] = set()
    _READS = reads
    try:
        yield reads
    finally:
        _READS = outer
        if outer is not None:  # what an inner block read, the outer one read as well
            outer.update(reads)


class Param(Node):
    """
    Node to represent a parameter in the graph.

    The ``Param`` object is used to represent a parameter in the graph. During
    runtime this will represent a value which can be used in various
    calculations. The ``Param`` object can be set to a constant value
    (``static``); ``None`` meaning the value is to be provided at runtime
    (``dynamic``); another ``Param`` object meaning it will take on that  value
    at runtime (``pointer``); or a function of other ``Param`` objects to be
    computed at runtime (also ``pointer``, see user guides). These options allow
    users to flexibly set the behavior of the simulator.

    Examples
    --------
    Example making some ``Param`` objects::

        p1 = Param("test", (1.0, 2.0)) # constant value, length 2 vector
        p2 =Param("p2", None, (2,2)) # dynamic 2x2 matrix value
        p3 = Param("p3", p1) # pointer to another parameter
        p4 = Param("p4", lambda p: p.children["other"].value * 2) # arbitrary function of another parameter
        p5 = Param("p5", valid=(0.0,2*pi), units="radians", cyclic=True) # parameter with metadata

    Parameters
    ----------
    name: (str)
        The name of the parameter.
    value: (Optional[Union[ArrayLike, float, int]], optional)
        The value of the parameter. Defaults to None meaning dynamic.
    shape: (Optional[tuple[int, ...]], optional)
        The shape of the parameter. Defaults to () meaning scalar.
    cyclic: (bool, optional)
        Whether the parameter is cyclic, imposing periodic boundary conditions.
        Such as a rotation from 0 to 2pi. Defaults to False.
    valid: (Optional[tuple[Union[ArrayLike, float, int, None]]], optional)
        The valid range of the parameter. Defaults to None meaning all of -inf
        to inf is valid.
    units: (Optional[str], optional)
        The units of the parameter. Defaults to None.
    dynamic: (bool, optional)
        Force param to be dynamic if True. If a value is provided and param is dynamic
        then it has a default value at call time.
    batched (bool, optional):
        If True, the param is assumed batched and the shape may now take the form
        (*B, *D) where *D is the shape of the value.
    dtype: (Optional[Any], optional)
        The data type of the parameter. Defaults to None meaning the data type
        will be inferred from the value.
    device: (Optional[Any], optional)
        The device of the parameter. Defaults to None meaning the device will be
        inferred from the value.
    """

    def __init__(
        self,
        name: str,
        value: Optional[Union[ArrayLike, float, int]] = None,
        shape: Optional[tuple[int, ...]] = None,
        cyclic: bool = False,
        valid: Optional[tuple[Union[ArrayLike, float, int, None], ...]] = None,
        units: Optional[str] = None,
        dynamic: Optional[bool] = None,
        group: int = 0,
        batch_shape: Optional[tuple[int, ...]] = None,
        dtype: Optional[Any] = None,
        device: Optional[Any] = None,
        **kwargs,
    ):
        self._node_type = NodeType.NODE
        super().__init__(name=name, **kwargs)
        self._shape: Optional[tuple[Optional[int], ...]] = None
        self._batch_shape: Optional[tuple[int, ...]] = None
        # The value used while a simulation is running
        self._active_value: Any = None
        # The value held between simulations, or the function of a pointer
        self._stored_value: Any = None
        self._valid: tuple[Optional[ArrayLike], Optional[ArrayLike]] = (None, None)
        self._group = 0
        self._dtype = dtype
        self._device = device
        self._cyclic = cyclic

        self.shape = shape
        if dynamic or (dynamic is None and value is None):
            self.to_dynamic()
        else:
            self.to_static()
        if isinstance(value, (Param, Callable)):
            self.to_pointer(value, clear_links=False)
        else:
            self.value = value
        self.group = group
        self.valid = valid
        self.units = units
        if batch_shape is not None:
            self.batch_shape = batch_shape

    @property
    def dynamic(self) -> bool:
        """Whether this parameter is dynamic.

        Returns
        -------
        bool
            ``True`` if the parameter's value is provided at runtime.
        """
        return self._node_type is NodeType.DYNAMIC

    @property
    def pointer(self) -> bool:
        """Whether this parameter is a pointer.

        Returns
        -------
        bool
            ``True`` if the parameter points to another ``Param`` or a
            callable that is evaluated at runtime.
        """
        return self._node_type is NodeType.POINTER

    @property
    def static(self) -> bool:
        """Whether this parameter is static.

        Returns
        -------
        bool
            ``True`` if the parameter holds a fixed value that does not
            change at runtime.
        """
        return self._node_type is NodeType.STATIC

    @property
    def graphviz_style(self):
        if self.pointer:
            return {
                "style": "filled",
                "color": "lightgrey",
                "fillcolor": "lightgrey",
                "shape": "rarrow",
            }
        elif self.dynamic:
            return {
                "style": "solid",
                "color": "black",
                "fillcolor": "white",
                "shape": "box",
            }
        else:
            if self._stored_value is None:
                return {
                    "style": "filled",
                    "color": "black",
                    "fillcolor": "grey90",
                    "shape": "box",
                }
            return {
                "style": "filled",
                "color": "lightgrey",
                "fillcolor": "lightgrey",
                "shape": "box",
            }

    @property
    def node_type(self):
        """The current type of this parameter node.

        Returns
        -------
        NodeType
            One of ``NodeType.STATIC``, ``NodeType.DYNAMIC``, or
            ``NodeType.POINTER``. These are equal to the strings ``"static"``,
            ``"dynamic"`` and ``"pointer"``.
        """
        return self._node_type

    @node_type.setter
    def node_type(self, value):
        value = NodeType(value)  # a string that is not one of the node types is an error
        pre_type = self._node_type
        self._node_type = value
        if pre_type is not value:
            self.update_graph()

    def to_dynamic(self, value=NULL):
        """Change this parameter to a dynamic parameter.

        If a value is provided, it is stored as the default dynamic value.
        When called without arguments the existing value (if any) is kept.

        Parameters
        ----------
        value : ArrayLike, float, int, None, or sentinel, optional
            The default value for the dynamic parameter. Must not be a
            ``Param`` or callable. By default the current value is retained.

        Raises
        ------
        ActiveStateError
            If the parameter is currently active.
        ParamTypeError
            If *value* is a ``Param`` or callable.
        ParamConfigurationError
            If the value shape does not match the declared shape.
        """
        # While active no value can be set
        if self.active:
            raise ActiveStateError(f"Cannot set parameter {self.name} dynamic value while active.")

        # Catch cases where input is invalid
        if isinstance(value, Param) or callable(value):
            raise ParamTypeError(f"Cannot set dynamic value to pointer ({self.name}).")

        if value is NULL:
            if self.pointer:
                try:
                    self._stored_value = self._stored_value(self)
                except Exception:  # pointer cannot be evaluated yet, so no value to keep
                    self._stored_value = None
            self.node_type = NodeType.DYNAMIC
            return

        if value is not None:
            value = backend.as_array(value, dtype=self._dtype, device=self._device)
            if not valid_shape(self._batch_shape, self._shape, tuple(value.shape)):
                if self.batched:
                    shape = f"{self._shape} with batch dims {self._batch_shape}"
                else:
                    shape = str(self._shape)
                raise ParamConfigurationError(
                    f"Value shape {value.shape} does not match param shape {shape}! Cannot update value. ({self.name})"
                )
        self._stored_value = value
        self.node_type = NodeType.DYNAMIC
        self.is_valid()

    def to_static(self, value=NULL):
        """Change this parameter to a static parameter.

        If a value is provided, it is stored as the fixed static value.
        When called without arguments the existing value (if any) is kept.

        Parameters
        ----------
        value : ArrayLike, float, int, None, or sentinel, optional
            The constant value for the static parameter. Must not be a
            ``Param`` or callable. By default the current value is retained.

        Raises
        ------
        ActiveStateError
            If the parameter is currently active.
        ParamTypeError
            If *value* is a ``Param`` or callable.
        ParamConfigurationError
            If the value shape does not match the declared shape.
        """
        # While active no value can be set
        if self.active:
            raise ActiveStateError(f"Cannot set parameter {self.name} static value while active.")

        # Catch cases where input is invalid
        if isinstance(value, Param) or callable(value):
            raise ParamTypeError(f"Cannot set static value to pointer ({self.name}).")

        if value is NULL:
            if self.pointer:
                try:
                    self._stored_value = self._stored_value(self)
                except Exception:  # pointer cannot be evaluated yet, so no value to keep
                    self._stored_value = None
            self.node_type = NodeType.STATIC
            return

        if value is not None:
            value = backend.as_array(value, dtype=self._dtype, device=self._device)
            if not valid_shape(self._batch_shape, self._shape, tuple(value.shape)):
                if self.batched:
                    shape = f"{self._shape} with batch dims {self._batch_shape}"
                else:
                    shape = str(self._shape)
                raise ParamConfigurationError(
                    f"Value shape {value.shape} does not match param shape {shape}! Cannot update value. ({self.name})"
                )

        self._stored_value = value
        self.node_type = NodeType.STATIC
        self.is_valid()

    def to_pointer(
        self,
        value: "Param" | Callable | None,
        link: Node | tuple[Node, ...] = (),
        clear_links: bool = True,
    ):
        """Change this parameter to a pointer parameter.

        The parameter's value will be computed at runtime by dereferencing
        another ``Param`` or by calling a user-supplied function.

        Parameters
        ----------
        value : Param or callable
            A ``Param`` whose value will be mirrored, or a callable
            ``f(param) -> ArrayLike`` evaluated at runtime.
        link : Node or tuple of Node, optional
            Additional nodes to link into the graph when creating the
            pointer. Defaults to an empty tuple.
        clear_links : bool, optional
            Whether to clear existing links before setting the pointer.
            Defaults to True.

        Raises
        ------
        ActiveStateError
            If the parameter is currently active.
        ParamTypeError
            If *value* is not a ``Param`` or callable.
        """
        # While active no value can be set
        if self.active:
            raise ActiveStateError(f"Cannot set parameter {self.name} to pointer while active")

        if clear_links:
            self.unlink()
        if isinstance(value, Param):
            self.link(value)
            value = _ParamValue(value.name)
        elif value is not None and not callable(value):
            raise ParamTypeError(f"Pointer function must be a Param or callable ({self.name})")
        elif hasattr(value, "params"):
            self.link(value.params)

        self.link(link)
        self._stored_value = value
        self._shape = None
        self.node_type = NodeType.POINTER

    @property
    def shape(self) -> tuple[Optional[int], ...]:
        """The event (non-batch) shape of the parameter value.

        Wildcard dimensions (``None``) in the declared shape are resolved
        using the current value. If no shape was declared, the shape of the
        current value is returned directly.

        Returns
        -------
        tuple of int
            The resolved shape of the parameter.
        """
        value = self._peek()
        # 1. Handle cases where no shape template is defined
        if self._shape is None:
            return tuple(value.shape) if value is not None else ()

        # 2. If value is missing, return the template as-is
        if value is None:
            return self._shape

        # 3. Fill wildcards (None) in _shape using the trailing dimensions of value
        # Negative indexing handles the alignment automatically
        n = len(self._shape)
        return tuple(v if s is None else s for s, v in zip(self._shape, value.shape[-n:]))

    @shape.setter
    def shape(self, shape: Optional[Iterable]):
        if self.pointer:
            raise ParamTypeError(
                f"Cannot set shape of parameter {self.name} with node type 'pointer'"
            )
        if shape is None:
            self._shape = None
            return
        value = self._peek()
        try:
            shape = tuple(shape)
        except TypeError:
            raise ParamConfigurationError(
                f"Param shape must be iterable of ints/None, not: {type(shape)}. ({self.name})"
            ) from None
        if value is None or valid_shape(self._batch_shape, shape, tuple(value.shape)):
            self._shape = shape
            return

        raise ValueError(
            f"Shape {shape} does not match the shape of the value {value.shape}! Unable to set shape. ({self.name})"
        )

    @property
    def batched(self) -> bool:
        """Whether this parameter carries batch dimensions.

        Returns
        -------
        bool
            ``True`` if ``batch_shape`` is non-empty.
        """
        return len(self.batch_shape) > 0

    @property
    def batch_shape(self) -> tuple[int, ...]:
        """The batch dimensions of the parameter value.

        Batch dimensions are the leading dimensions of the value that
        precede the event ``shape``. If an explicit batch shape was set it
        is returned directly; otherwise it is inferred from the value.

        Returns
        -------
        tuple of int
            The batch shape, or ``()`` if the parameter is not batched.
        """
        if self._batch_shape is not None:
            return self._batch_shape
        try:
            value = self._peek()
        except Exception:  # pointer cannot be evaluated yet
            value = None
        if value is None:
            return ()
        return tuple(value.shape[: len(value.shape) - len(self.shape)])

    @batch_shape.setter
    def batch_shape(self, batch_shape: tuple[int, ...]):
        if self.pointer:
            raise ParamTypeError(
                f"Cannot set batch_shape of parameter {self.name} with node type 'pointer'"
            )
        self._batch_shape = batch_shape

    @property
    def group(self) -> int:
        """The group index of this parameter.

        Parameters that share the same group index are collected together
        into a single ``params`` object when calling a simulator's
        ``@forward`` method, as well as when using ``get_values`` or
        ``set_values``. The dynamic params are split by the groups that the
        dynamic params are in, and the static params (``static_params``,
        ``get_values(dynamic=False)``) by the groups that the static params
        are in.

        Returns
        -------
        int
            The group index (default ``0``).
        """
        return self._group

    @group.setter
    def group(self, group: int):
        if isinstance(group, bool) or not isinstance(group, Integral):
            raise ParamConfigurationError(f"Group must be an integer ({self.name})")
        pregroup = self._group
        self._group = int(group)  # a plain int, also from a numpy integer
        if pregroup != self._group:
            self.update_graph()

    @property
    def dtype(self) -> Optional[Any]:
        """The data type of the parameter value.

        If no explicit dtype was set, the dtype is inferred from the
        current value.

        Returns
        -------
        dtype or None
            The data type, or ``None`` if unknown.
        """
        if self._dtype is None:
            try:
                value = self._peek()
                if value is not None:
                    return value.dtype
            except AttributeError:
                pass
        return self._dtype

    @property
    def device(self) -> Optional[Any]:
        """The device on which the parameter value resides.

        If no explicit device was set, the device is inferred from the
        current value.

        Returns
        -------
        device or None
            The device, or ``None`` if unknown.
        """
        if self._device is None:
            try:
                value = self._peek()
                if value is not None:
                    return value.device
            except AttributeError:
                pass
        return self._device

    @property
    def value(self) -> Union[ArrayLike, None]:
        """The current value of the parameter.

        For static and dynamic parameters the stored value is returned.
        For pointer parameters the linked callable is evaluated. During an
        active simulation the result is cached.

        Returns
        -------
        ArrayLike or None
            The parameter value, or ``None`` if no value has been set.
        """
        if _READS is not None:
            _READS.add(self)
        if self._active_value is not None:
            return self._active_value
        if self.pointer:
            value = self._stored_value(self)
            if self.active:
                self._active_value = value
            return value
        return self._stored_value

    @value.setter
    def value(self, value):
        # While active no value can be set
        if self.active:
            if self.static and self._stored_value is None:  # static value set live in sim
                self._active_value = value
                return
            raise ActiveStateError(f"Cannot set value of parameter {self.name} while active")

        if isinstance(value, Param) or callable(value):
            self.to_pointer(value)
        elif self.dynamic:
            self.to_dynamic(value)
        else:
            self.to_static(value)

    def _peek(self) -> Union[ArrayLike, None]:
        """The current value, for ``caskade`` to look at its shape or type. It
        does not count as a use of the value (see ``_record_reads``)."""
        if self._active_value is not None:
            return self._active_value
        if self.pointer:
            value = self._stored_value(self)
            if self.active:
                self._active_value = value
            return value
        return self._stored_value

    @property
    def npvalue(self) -> ndarray:
        """The current value converted to a NumPy array.

        Returns
        -------
        numpy.ndarray
            The value as a NumPy ``ndarray``.
        """
        return backend.to_numpy(self.value)

    def to(self, device=None, dtype=None) -> "Param":
        """
        Moves and/or casts the values of the parameter.

        Parameters
        ----------
        device: (optional)
            The device to move the values to. Defaults to None.
        dtype: (optional)
            The desired data type. Defaults to None.
        """
        if device is not None:
            self._device = device
        else:
            device = self.device
        if dtype is not None:
            self._dtype = dtype
        else:
            dtype = self.dtype
        super().to(device=device, dtype=dtype)
        if not self.pointer and self._stored_value is not None:
            self._stored_value = backend.to(self._stored_value, device=device, dtype=dtype)
        valid = self.valid
        if valid[0] is not None:
            valid = (backend.to(valid[0], device=device, dtype=dtype), valid[1])
        if valid[1] is not None:
            valid = (valid[0], backend.to(valid[1], device=device, dtype=dtype))
        self.valid = valid

        return self

    @property
    def cyclic(self) -> bool:
        """Whether the parameter has cyclic (periodic) boundary conditions.

        When ``True``, values wrap around the ``valid`` range (e.g. an
        angle from 0 to 2π).

        Returns
        -------
        bool
            ``True`` if the parameter is cyclic.
        """
        return self._cyclic

    @cyclic.setter
    def cyclic(self, cyclic: bool):
        self._cyclic = cyclic
        self.valid = self.valid

    # Saving and loading
    #################################################################
    _hdf5_own = ("value",)

    def _value_or_none(self):
        try:
            return self.value
        except Exception:  # pointer cannot be evaluated yet
            return None

    def _save_hdf5(self, h5group, appendable: bool):
        super()._save_hdf5(h5group, appendable)
        value = self._value_or_none()
        if value is None:
            dataset = h5group.create_dataset("value", data="None")
        elif appendable:
            dataset = _create_rows(h5group, "value", backend.to_numpy(value))
        else:
            dataset = h5group.create_dataset("value", data=backend.to_numpy(value))
        attrs = dataset.attrs
        # a plain string in the file, as before ``NodeType`` existed
        attrs["node_type"] = self.node_type.value
        attrs["appendable"] = appendable
        attrs["cyclic"] = self.cyclic
        if self.valid[0] is not None:
            attrs["valid_left"] = backend.to_numpy(self.valid[0])
        if self.valid[1] is not None:
            attrs["valid_right"] = backend.to_numpy(self.valid[1])
        attrs["units"] = self.units if self.units is not None else "None"

    def _append_hdf5(self, h5group):
        dataset = h5group["value"]
        blank = dataset.dtype.kind in "SO"  # the text "None", no state so far has a value
        if not (dataset.attrs["appendable"] if blank else dataset.maxshape[:1] == (None,)):
            raise IOError(
                f"{self.name} is not appendable. Need to save the HDF5 file with `appendable=True`."
            )
        value = self._value_or_none()
        if value is None:
            if blank:
                return None
        else:
            value = backend.to_numpy(value)
            if not blank and tuple(value.shape) != dataset.shape[1:]:
                raise GraphError(
                    f"The value of param '{self.name}' has shape {tuple(value.shape)}, the "
                    f"file holds states of shape {dataset.shape[1:]} for it at '{h5group.name}'."
                )

        def write(samples: Optional[int]):
            if not blank:
                # without a value the new row stays blank
                dataset.resize(dataset.shape[0] + 1 if samples is None else samples, axis=0)
                if value is not None:
                    dataset[-1] = value
                return
            if samples is None:
                raise IOError(
                    f"Param '{self.name}' had no value when the file was saved by an older "
                    "version of caskade, which kept no count of the states. Its value "
                    "cannot be appended, save a new file."
                )
            # the first value of this param, the states before it are blank rows
            attrs = dict(dataset.attrs)
            del h5group["value"]
            _create_rows(h5group, "value", value, samples).attrs.update(attrs)

        return write

    def _plan_value(self, node_type: str, value, where: str):
        """Check that ``value`` (a numpy array or ``None``) read from ``where``
        in a file fits this param, and return the function that takes it on
        with the type (``"static"`` or ``"dynamic"``) it was saved with, or
        ``None`` if it was saved as a pointer (so its value was worked out
        from other params). Not to be called on a param that is now a pointer,
        which keeps being one."""
        if value is not None and not valid_shape(
            self._batch_shape, self._shape, tuple(value.shape)
        ):
            raise GraphError(
                f"The value saved for param '{self.name}' at '{where}' has shape "
                f"{tuple(value.shape)}, which does not fit the shape {self._shape} of the param"
                + (f" with batch shape {self._batch_shape}." if self._batch_shape else ".")
            )
        # Matched as text within the saved string: files written by old versions
        # of caskade can hold names that are no longer node types ("dynamic value")
        if "static" in node_type:
            return lambda: self.to_static(value)
        if "dynamic" in node_type:
            return lambda: self.to_dynamic(value)
        return None

    def _read_hdf5(self, h5group, index: int):
        node_state = super()._read_hdf5(h5group, index)
        dataset = h5group["value"]
        attrs = dict(dataset.attrs)
        value_state = None
        if not self.pointer:
            if dataset.dtype.kind in "SO":  # the text "None"
                if dataset.shape != () or dataset[()] != b"None":
                    raise ValueError(f"Unrecognized value saved for param {self.name}")
                value = None
            elif attrs["appendable"]:
                value = dataset[index]
            else:
                value = dataset[()]
            value_state = self._plan_value(attrs["node_type"], value, h5group.name)

        def take_on():
            if node_state is not None:
                node_state()
            self.cyclic = False
            self.valid = None
            if value_state is not None:
                value_state()
            units = attrs["units"]
            self.units = None if units == "None" else units  # no units are saved as "None"
            self.valid = (attrs.get("valid_left"), attrs.get("valid_right"))
            self.cyclic = attrs["cyclic"]

        return take_on

    @property
    def valid(self) -> tuple[Optional[ArrayLike], Optional[ArrayLike]]:
        """The valid range of the parameter value.

        Returns
        -------
        tuple of (ArrayLike or None, ArrayLike or None)
            ``(lower_bound, upper_bound)``. Either bound may be ``None``
            indicating no constraint on that side.
        """
        return self._valid

    @valid.setter
    def valid(self, valid: Optional[tuple[Union[ArrayLike, float, int, None], ...]]):
        if valid is None:
            valid = (None, None)

        if not isinstance(valid, tuple):
            raise ParamConfigurationError(f"Valid must be a tuple ({self.name})")
        if len(valid) != 2:
            raise ParamConfigurationError(f"Valid must be a tuple of length 2 ({self.name})")
        if self.cyclic and (valid[0] is None or valid[1] is None):
            raise ParamConfigurationError(f"valid must be set for cyclic parameter ({self.name})")

        left, right = valid
        if left is not None:
            left = backend.as_array(left, dtype=self.dtype, device=self.device)
        if right is not None:
            right = backend.as_array(right, dtype=self.dtype, device=self.device)
        if left is not None and right is not None and backend.any(left >= right):
            raise ParamConfigurationError(
                f"Valid range (valid[1] - valid[0]) must be strictly positive ({self.name})"
            )

        self._valid = (left, right)
        self.is_valid()

    def is_valid(self, value=None) -> bool:
        """Check whether a value lies within the allowed range.

        Parameters
        ----------
        value : ArrayLike or None, optional
            The value to check. If ``None`` (default), the parameter's
            current value is used.

        Returns
        -------
        bool
            ``True`` if the value is within the valid range or if no
            constraints are set. ``False`` otherwise; a warning is also
            emitted.
        """
        if self.cyclic or self.pointer:
            return True
        if value is None:
            value = self.value
        if value is None:
            return True
        if self.valid[0] is not None and self.valid[1] is not None:
            if backend.any(value < self.valid[0]) or backend.any(value > self.valid[1]):
                warn(InvalidValueWarning(self.name, value, self.valid))
                return False
        elif self.valid[0] is not None and backend.any(value <= self.valid[0]):
            warn(InvalidValueWarning(self.name, value, self.valid))
            return False
        elif self.valid[1] is not None and backend.any(value >= self.valid[1]):
            warn(InvalidValueWarning(self.name, value, self.valid))
            return False
        return True

    def to_valid(self, value: ArrayLike) -> ArrayLike:
        """Map a value from the parameter's ``valid`` range to the unconstrained
        range ``(-inf, inf)``.

        A range bounded on one side is mapped with a logarithm. A cyclic
        parameter is wrapped into its range. A range bounded on both sides is
        left unchanged here and enforced by ``from_valid``.

        Parameters
        ----------
        value : ArrayLike
            A value in the valid range of the parameter.

        Returns
        -------
        ArrayLike
            The corresponding unconstrained value.
        """
        left, right = self._valid
        if left is None and right is None:
            return value
        if left is None:
            return backend.log(right - value)
        if right is None:
            return backend.log(value - left)
        if self.cyclic:
            return ((value - left) % (right - left)) + left
        return value

    def from_valid(self, value: ArrayLike) -> ArrayLike:
        """Map an unconstrained value back into the parameter's ``valid`` range.

        This is the inverse of ``to_valid``. A value outside a range bounded on
        both sides is clipped to it (preserving gradients).

        Parameters
        ----------
        value : ArrayLike
            An unconstrained value.

        Returns
        -------
        ArrayLike
            The corresponding value in the valid range of the parameter.
        """
        left, right = self._valid
        if left is None and right is None:
            return value
        if left is None:
            return right - backend.exp(value)
        if right is None:
            return backend.exp(value) + left
        if self.cyclic:
            return ((value - left) % (right - left)) + left
        return backend.ste_clip(value, left, right)

    @property
    def node_str(self) -> str:
        """
        Returns a string representation of the node for graph visualization.
        """
        try:
            value = self._peek()
        except Exception:  # pointer cannot be evaluated yet
            value = None
        if value is not None:
            value = backend.to_numpy(value)

            if max(1, prod(value.shape)) == 1:
                return f"{self.name}|{self.node_type}: {value.item():.3g}"
            elif prod(value.shape) <= 4:
                value = str(np.char.mod("%.3g", value).tolist()).replace("'", "")
                return f"{self.name}|{self.node_type}: {value}"
            else:
                return f"{self.name}|{self.node_type}: {self.shape}"
        elif self.static:
            return f"{self.name}|{self.node_type}: live"
        return f"{self.name}|{self.node_type}"

    def __repr__(self) -> str:
        return self.name
