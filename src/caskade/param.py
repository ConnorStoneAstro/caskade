from typing import Optional, Union, Callable, Any, Iterable
from warnings import warn
from math import prod

from numpy import ndarray
import numpy as np

from .backend import backend, ArrayLike
from .base import Node
from .errors import ParamConfigurationError, ParamTypeError, ActiveStateError
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
        valid: Optional[tuple[Union[ArrayLike, float, int, None]]] = None,
        units: Optional[str] = None,
        dynamic: Optional[bool] = None,
        group: int = 0,
        batch_shape: Optional[tuple[int]] = None,
        dtype: Optional[Any] = None,
        device: Optional[Any] = None,
        **kwargs,
    ):
        self._node_type = "node"
        super().__init__(name=name, **kwargs)
        self._shape = None
        self._batch_shape = None
        self._value = None
        self.__value = None
        self._valid = (None, None)
        self._group = 0
        self._dtype = dtype
        self._device = device
        self._cyclic = cyclic

        self.shape = shape
        if dynamic or (dynamic is None and value is None):
            self.to_dynamic()
        else:
            self.to_static()
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
        return "dynamic" in self.node_type

    @property
    def pointer(self) -> bool:
        """Whether this parameter is a pointer.

        Returns
        -------
        bool
            ``True`` if the parameter points to another ``Param`` or a
            callable that is evaluated at runtime.
        """
        return "pointer" in self.node_type

    @property
    def static(self) -> bool:
        """Whether this parameter is static.

        Returns
        -------
        bool
            ``True`` if the parameter holds a fixed value that does not
            change at runtime.
        """
        return "static" in self.node_type

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
            if self.__value is None:
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
        str
            One of ``"static"``, ``"dynamic"``, or ``"pointer"``.
        """
        return self._node_type

    @node_type.setter
    def node_type(self, value):
        pre_type = self.node_type
        self._node_type = value
        if pre_type != self.node_type:
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
                    self.__value = self.__value(self)
                except:
                    self.__value = None
            self.node_type = "dynamic"
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
        self.__value = value
        self.node_type = "dynamic"
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
                    self.__value = self.__value(self)
                except:
                    self.__value = None
            self.node_type = "static"
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

        self.__value = value
        self.is_valid()
        self.node_type = "static"

    def to_pointer(self, value, link=()):
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

        if isinstance(value, Param):
            self.link(value)
            p_name = value.name
            value = lambda p: p[p_name].value
        elif value is not None and not callable(value):
            raise ParamTypeError(f"Pointer function must be a Param or callable ({self.name})")
        elif hasattr(value, "params"):
            self.link(value.params)
        self.link(link)
        self.__value = value
        self._shape = None
        self.node_type = "pointer"

    @property
    def shape(self) -> tuple[int, ...]:
        """The event (non-batch) shape of the parameter value.

        Wildcard dimensions (``None``) in the declared shape are resolved
        using the current value. If no shape was declared, the shape of the
        current value is returned directly.

        Returns
        -------
        tuple of int
            The resolved shape of the parameter.
        """
        value = self.value
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
        value = self.value
        try:
            shape = tuple(shape)
        except TypeError:
            raise ParamConfigurationError(
                f"Param shape must be iterable of ints/None, not: {type(shape)}. ({self.name})"
            )
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
            value = self.value
        except:
            value = None
        if value is None:
            return ()
        return tuple(value.shape[: len(value.shape) - len(self.shape)])

    @batch_shape.setter
    def batch_shape(self, batch_shape: tuple[int]):
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
        ``set_values``.

        Returns
        -------
        int
            The group index (default ``0``).
        """
        return self._group

    @group.setter
    def group(self, group: int):
        assert isinstance(group, int), f"Group must be an integer ({self.name})"
        pregroup = self._group
        self._group = group
        if pregroup != self._group:
            self.update_graph()

    @property
    def dtype(self) -> Optional[str]:
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
                return self.value.dtype
            except AttributeError:
                pass
        return self._dtype

    @property
    def device(self) -> Optional[str]:
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
                return self.value.device
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
        if self._value is not None:
            return self._value
        if self.pointer:
            value = self.__value(self)
            if self.active:
                self._value = value
            return value
        return self.__value

    @value.setter
    def value(self, value):
        # While active no value can be set
        if self.active:
            if self.static and self.__value is None:  # static value set live in sim
                self._value = value
                return
            raise ActiveStateError(f"Cannot set value of parameter {self.name} while active")

        if isinstance(value, Param) or callable(value):
            self.to_pointer(value)
        elif self.dynamic:
            self.to_dynamic(value)
        else:
            self.to_static(value)

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
        device: (Optional[torch.device], optional)
            The device to move the values to. Defaults to None.
        dtype: (Optional[torch.dtype], optional)
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
        if not self.pointer and self.__value is not None:
            self.__value = backend.to(self.__value, device=device, dtype=dtype)
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

    def _save_state_hdf5(self, h5group, appendable: bool = False, _done_save: set = None):
        super()._save_state_hdf5(h5group, appendable=appendable, _done_save=_done_save)
        if "value" not in self._h5group:
            try:
                value = self.value
            except:
                value = None

            if value is None:
                value = "None"
            elif appendable:
                value = backend.to_numpy(value.reshape(1, *value.shape))
            else:
                value = backend.to_numpy(value)
            if appendable:
                self._h5group.create_dataset(
                    "value",
                    data=value,
                    chunks=False if isinstance(value, str) else True,
                    maxshape=None if isinstance(value, str) else (None,) + self.shape,
                    compression=None if isinstance(value, str) else "gzip",
                )
            else:
                self._h5group.create_dataset(
                    "value",
                    data=value,
                )
            self._h5group["value"].attrs["node_type"] = self.node_type
            self._h5group["value"].attrs["appendable"] = appendable
            self._h5group["value"].attrs["cyclic"] = self.cyclic
            if self.valid[0] is not None:
                self._h5group["value"].attrs["valid_left"] = backend.to_numpy(self.valid[0])
            if self.valid[1] is not None:
                self._h5group["value"].attrs["valid_right"] = backend.to_numpy(self.valid[1])
            self._h5group["value"].attrs["units"] = self.units if self.units is not None else "None"

    def _check_append_state_hdf5(self, h5group):
        super()._check_append_state_hdf5(h5group)
        if not h5group["value"].attrs["appendable"]:
            raise IOError(
                f"{self.name} is not appendable. Need to save the HDF5 file with `appendable=True`."
            )

    def _append_state_cleanup(self):
        super()._append_state_cleanup()
        del self.appended

    def _append_state_hdf5(self, h5group):
        super()._append_state_hdf5(h5group)
        if not hasattr(self, "appended"):
            self.appended = True
            try:
                value = self.value
            except:
                value = None
            if value is not None:
                h5group["value"].resize((h5group["value"].shape[0] + 1,) + self.shape)
                h5group["value"][-1] = self.value

    def _load_state_hdf5(self, h5group, index: int = -1, _done_load: set = None):
        super()._load_state_hdf5(h5group, index=index, _done_load=_done_load)
        self.cyclic = False
        self.valid = None
        if not self.pointer:
            if isinstance(h5group["value"][()], bytes):
                assert h5group["value"][()] == b"None"
                value = None
            elif h5group["value"].attrs["appendable"]:
                value = h5group["value"][index]
            else:
                value = h5group["value"][()]

            if "static" in h5group["value"].attrs["node_type"]:
                self.to_static(value)
            elif "dynamic" in h5group["value"].attrs["node_type"]:
                self.to_dynamic(value)
        self.units = h5group["value"].attrs["units"]
        if "valid_left" in h5group["value"].attrs:
            self.valid = (
                h5group["value"].attrs["valid_left"],
                self.valid[1],
            )
        if "valid_right" in h5group["value"].attrs:
            self.valid = (
                self.valid[0],
                h5group["value"].attrs["valid_right"],
            )
        self.cyclic = h5group["value"].attrs["cyclic"]

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
    def valid(self, valid: tuple[Union[ArrayLike, float, int, None]]):
        if valid is None:
            valid = (None, None)

        if not isinstance(valid, tuple):
            raise ParamConfigurationError(f"Valid must be a tuple ({self.name})")
        if len(valid) != 2:
            raise ParamConfigurationError(f"Valid must be a tuple of length 2 ({self.name})")
        if self.cyclic and (valid[0] is None or valid[1] is None):
            raise ParamConfigurationError(f"valid must be set for cyclic parameter ({self.name})")

        if valid[0] is None and valid[1] is None:
            self.to_valid = self._to_valid_base
            self.from_valid = self._from_valid_base
        elif valid[0] is None:
            self.to_valid = self._to_valid_rightvalid
            self.from_valid = self._from_valid_rightvalid
            valid = (None, backend.as_array(valid[1], dtype=self.dtype, device=self.device))
        elif valid[1] is None:
            self.to_valid = self._to_valid_leftvalid
            self.from_valid = self._from_valid_leftvalid
            valid = (backend.as_array(valid[0], dtype=self.dtype, device=self.device), None)
        else:
            if self.cyclic:
                self.to_valid = self._to_valid_cyclic
                self.from_valid = self._from_valid_cyclic
            else:
                self.to_valid = self._to_valid_fullvalid
                self.from_valid = self._from_valid_fullvalid
            valid = (
                backend.as_array(valid[0], dtype=self.dtype, device=self.device),
                backend.as_array(valid[1], dtype=self.dtype, device=self.device),
            )
            if backend.any(valid[0] >= valid[1]):
                raise ParamConfigurationError(
                    f"Valid range (valid[1] - valid[0]) must be strictly positive ({self.name})"
                )

        self._valid = valid
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
        if self.valid[0] is not None and backend.any(value < self.valid[0]):
            warn(InvalidValueWarning(self.name, value, self.valid))
            return False
        elif self.valid[1] is not None and backend.any(value > self.valid[1]):
            warn(InvalidValueWarning(self.name, value, self.valid))
            return False
        return True

    def _to_valid_base(self, value: ArrayLike) -> ArrayLike:
        return value

    def _to_valid_fullvalid(self, value: ArrayLike) -> ArrayLike:
        value = (
            backend.logit((value - self.valid[0]) / (self.valid[1] - self.valid[0])) + self.valid[0]
        )
        return value

    def _to_valid_cyclic(self, value: ArrayLike) -> ArrayLike:
        return ((value - self.valid[0]) % (self.valid[1] - self.valid[0])) + self.valid[0]

    def _to_valid_leftvalid(self, value: ArrayLike) -> ArrayLike:
        return backend.log(value - self.valid[0])

    def _to_valid_rightvalid(self, value: ArrayLike) -> ArrayLike:
        return backend.log(self.valid[1] - value)

    def _from_valid_base(self, value: ArrayLike) -> ArrayLike:
        return value

    def _from_valid_fullvalid(self, value: ArrayLike) -> ArrayLike:
        value = (
            backend.sigmoid(value - self.valid[0]) * (self.valid[1] - self.valid[0]) + self.valid[0]
        )
        return value

    def _from_valid_cyclic(self, value: ArrayLike) -> ArrayLike:
        value = ((value - self.valid[0]) % (self.valid[1] - self.valid[0])) + self.valid[0]
        return value

    def _from_valid_leftvalid(self, value: ArrayLike) -> ArrayLike:
        value = backend.exp(value) + self.valid[0]
        return value

    def _from_valid_rightvalid(self, value: ArrayLike) -> ArrayLike:
        value = self.valid[1] - backend.exp(value)
        return value

    @property
    def node_str(self) -> str:
        """
        Returns a string representation of the node for graph visualization.
        """
        try:
            value = self.value
        except:
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
