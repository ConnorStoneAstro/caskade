from typing import Optional, Union, Callable, Any
from warnings import warn
import traceback
from dataclasses import dataclass
from math import prod

from numpy import ndarray, pi

from .backend import backend, ArrayLike
from .base import Node
from .errors import ParamConfigurationError, ParamTypeError, ActiveStateError
from .warnings import InvalidValueWarning


@dataclass
class dynamic:
    """Basic wrapper for an input to a ``Param`` object to indicate that the
    value should be placed as a dynamic_value so that the ``Param`` is dynamic
    instead of static.

    Usage: ``dynamic(value)``

    Example:

    .. code-block:: python

        class Test(Module):
            def __init__(self, a):
                self.a = Param("a", a)

        t = Test(dynamic(1.0))
        print(t.a.dynamic) # True
    """

    value: Any = None


class Param(Node):
    """
    Node to represent a parameter in the graph.

    The ``Param`` object is used to represent a parameter in the graph. During
    runtime this will represent a value which can be used in various
    calculations. The ``Param`` object can be set to a constant value (``static``);
    ``None`` meaning the value is to be provided at runtime (``dynamic``); another
    ``Param`` object meaning it will take on that  value at runtime (``pointer``);
    or a function of other ``Param`` objects to be computed at runtime (also
    ``pointer``, see user guides). These options allow users to flexibly set the
    behavior of the simulator.

    Examples
    --------
    Example making some ``Param`` objects::

        p1 = Param("test", (1.0, 2.0)) # constant value, length 2 vector
        p2 = Param("p2", None, (2,2)) # dynamic 2x2 matrix value
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
        Whether the parameter is cyclic, such as a rotation from 0 to 2pi.
        Defaults to False.
    valid: (Optional[tuple[Union[ArrayLike, float, int, None]]], optional)
        The valid range of the parameter. Defaults to None meaning all of -inf
        to inf is valid.
    units: (Optional[str], optional)
        The units of the parameter. Defaults to None.
    dynamic_value: (Optional[Union[ArrayLike, float, int]], optional)
        Allows the parameter to store a value while still dynamic (think of it
        as a default value).
    dtype: (Optional[Any], optional)
        The data type of the parameter. Defaults to None meaning the data type
        will be inferred from the value.
    device: (Optional[Any], optional)
        The device of the parameter. Defaults to None meaning the device will
        be inferred from the value.
    """

    graphviz_types = {
        "static": {"style": "filled", "color": "lightgrey", "shape": "box"},
        "dynamic": {"style": "solid", "color": "black", "shape": "box"},
        "dynamic value": {"style": "solid", "color": "#333333", "shape": "box"},
        "pointer": {"style": "filled", "color": "lightgrey", "shape": "rarrow"},
    }

    def __init__(
        self,
        name: str,
        value: Optional[Union[ArrayLike, float, int]] = None,
        shape: Optional[tuple[int, ...]] = (),
        cyclic: bool = False,
        valid: Optional[tuple[Union[ArrayLike, float, int, None]]] = None,
        units: Optional[str] = None,
        dynamic_value: Optional[Union[ArrayLike, float, int]] = None,
        dtype: Optional[Any] = None,
        device: Optional[Any] = None,
    ):
        super().__init__(name=name)
        if value is not None and dynamic_value is not None:
            raise ParamConfigurationError("Cannot set both value and dynamic value")
        if isinstance(value, dynamic):
            dynamic_value = value.value
            value = None
        elif isinstance(dynamic_value, dynamic):
            dynamic_value = dynamic_value.value
        elif value is None and dynamic_value is None and backend.backend != "object":
            if shape is None:
                raise ParamConfigurationError("Either value or shape must be provided")
            if not isinstance(shape, (tuple, list)):
                raise ParamConfigurationError("Shape must be a tuple")
            self.shape = tuple(shape)
        elif (
            not isinstance(value, (Param, Callable))
            and value is not None
            and backend.backend != "object"
        ):
            value = backend.as_array(value, dtype=dtype, device=device)
            if not (shape == () or shape is None or shape == value.shape):
                raise ParamConfigurationError(
                    f"Shape {shape} does not match value shape {value.shape}"
                )
        elif (
            not isinstance(dynamic_value, (Param, Callable))
            and dynamic_value is not None
            and backend.backend != "object"
        ):
            dynamic_value = backend.as_array(dynamic_value, dtype=dtype, device=device)
            if not (shape == () or shape is None or shape == dynamic_value.shape):
                raise ParamConfigurationError(
                    f"Shape {shape} does not match dynamic value shape {dynamic_value.shape}"
                )
        self._type = "null"
        self._dtype = dtype
        self._device = device
        self.value = value
        if not hasattr(self, "_dynamic_value"):
            self.dynamic_value = dynamic_value
        self.cyclic = cyclic
        self.valid = valid
        self.units = units

    @property
    def dynamic(self) -> bool:
        return "dynamic" in self._type

    @dynamic.setter
    def dynamic(self, dynamic: bool):
        if dynamic:
            self.to_dynamic()
        else:
            self.to_static()

    @property
    def pointer(self) -> bool:
        return "pointer" in self._type

    @property
    def static(self) -> bool:
        return "static" in self._type

    @static.setter
    def static(self, static: bool):
        if static:
            self.to_static()
        else:
            self.to_dynamic()

    def to_dynamic(self, **kwargs):
        """Change this parameter to a dynamic parameter. If the parameter has a
        value, this will be stored in the ``dynamic_value`` attribute."""
        if self.dynamic:
            return
        if self.pointer:
            try:
                eval_pointer = self._pointer_func(self)
                self.dynamic_value = eval_pointer
            except Exception as e:
                self.value = None
            return
        self.dynamic_value = self.value

    def to_static(self, **kwargs):
        """Change this parameter to a static parameter. This only works if the
        parameter has a ``dynamic_value`` set, or if the pointer can be
        evaluated."""
        if self.static:
            return
        if self.pointer:
            try:
                eval_pointer = self._pointer_func(self)
                self.value = eval_pointer
            except Exception as e:
                raise ParamTypeError(
                    f"Cannot set pointer parameter {self.name} to static with `to_static`. Pointer could not be evaluated because of: \n"
                    + traceback.format_exc()
                )

            return
        if self.dynamic_value is None:
            raise ParamTypeError(
                f"Cannot set dynamic parameter {self.name} to static when no `dynamic_value` is set"
            )
        self.value = self.dynamic_value

    @property
    def shape(self) -> Optional[tuple[int, ...]]:
        if backend.backend == "object":
            return None
        if self.pointer and self.value is not None:
            return self.value.shape
        return self._shape

    @shape.setter
    def shape(self, shape):
        if backend.backend == "object":
            raise ParamTypeError("Cannot set shape of parameter with backend 'object'")
        if self.pointer:
            raise ParamTypeError(f"Cannot set shape of parameter {self.name} with type 'pointer'")
        self._shape = shape

    @property
    def dtype(self) -> Optional[str]:
        if self._dtype is None:
            try:
                return self.value.dtype
            except AttributeError:
                pass
        return self._dtype

    @property
    def device(self) -> Optional[str]:
        if self._device is None:
            try:
                return self.value.device
            except AttributeError:
                pass
        return self._device

    @property
    def dynamic_value(self) -> Union[ArrayLike, None]:
        return self._dynamic_value

    @dynamic_value.setter
    def dynamic_value(self, value):
        # While active no value can be set
        if self.active:
            raise ActiveStateError(
                f"Cannot set dynamic value of parameter {self.name} while active"
            )

        # No dynamic value
        if value is None:
            self._dynamic_value = None
            return

        # Catch cases where input is invalid
        if isinstance(value, Param) or callable(value):
            raise ParamTypeError(f"Cannot set dynamic value to pointer ({self.name})")

        # unlink if pointer, dynamic_value cannot be a pointer
        if self.pointer:
            for child in tuple(self.children.values()):
                self.unlink(child)

        # Set to dynamic value
        self._type = "dynamic value"
        self._pointer_func = None
        value = backend.as_array(value, dtype=self._dtype, device=self._device)
        self._shape = value.shape if backend.backend != "object" else None
        self._dynamic_value = value
        self._value = None
        try:
            self.valid = self._valid  # re-check valid range
        except AttributeError:
            pass

        self.update_graph()

    @property
    def value(self) -> Union[ArrayLike, None]:
        if self.pointer and self._value is None:
            if self.active:
                self._value = self._pointer_func(self)
            else:
                return self._pointer_func(self)
        if self._value is None:
            return self._dynamic_value
        return self._value

    @value.setter
    def value(self, value):
        # While active no value can be set
        if self.active:
            raise ActiveStateError(f"Cannot set value of parameter {self.name} while active")

        # unlink if pointer to avoid floating references
        if self.pointer:
            for child in tuple(self.children.values()):
                self.unlink(child)

        if value is None:
            if hasattr(self, "_value") and self._value is not None:
                self.dynamic_value = self._value
                return
            self._type = "dynamic"
            self._pointer_func = None
            self._value = None
        elif isinstance(value, Param):
            self._type = "pointer"
            self.link(value)
            self._pointer_func = lambda p: p[value.name].value
            self._shape = None
            self._value = None
            self._dynamic_value = None
        elif callable(value):
            self._type = "pointer"
            self._shape = None
            self._pointer_func = value
            self._value = None
            self._dynamic_value = None
        else:
            self._type = "static"
            value = backend.as_array(value, dtype=self._dtype, device=self._device)
            self._shape = value.shape if backend.backend != "object" else None
            self._value = value
            self._dynamic_value = None
            try:
                self.valid = self._valid  # re-check valid range
            except AttributeError:
                pass

        self.update_graph()

    @property
    def npvalue(self) -> ndarray:
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
        if backend.backend == "object":
            return self
        if device is not None:
            self._device = device
        else:
            device = self.device
        if dtype is not None:
            self._dtype = dtype
        else:
            dtype = self.dtype
        super().to(device=device, dtype=dtype)
        if self.static:
            self._value = backend.to(self._value, device=device, dtype=dtype)
        if self._dynamic_value is not None:
            self._dynamic_value = backend.to(self._dynamic_value, device=device, dtype=dtype)
        valid = self.valid
        if valid[0] is not None:
            valid = (backend.to(valid[0], device=device, dtype=dtype), valid[1])
        if valid[1] is not None:
            valid = (valid[0], backend.to(valid[1], device=device, dtype=dtype))
        self.valid = valid

        return self

    @property
    def cyclic(self) -> bool:
        return self._cyclic

    @cyclic.setter
    def cyclic(self, cyclic: bool):
        self._cyclic = cyclic
        try:
            self.valid = self._valid
        except AttributeError:
            pass

    def _save_state_hdf5(self, h5group, appendable: bool = False, _done_save: set = None):
        super()._save_state_hdf5(h5group, appendable=appendable, _done_save=_done_save)
        if "value" not in self._h5group:
            if self.value is None:
                value = "None"
            elif appendable:
                value = backend.to_numpy(self.value.reshape(1, *self.value.shape))
            else:
                value = backend.to_numpy(self.value)
            if appendable:
                self._h5group.create_dataset(
                    "value",
                    data=value,
                    chunks=True if self.value is not None else False,
                    maxshape=(None,) + tuple(self.shape) if self.value is not None else None,
                    compression="gzip" if self.value is not None else None,
                )
            else:
                self._h5group.create_dataset(
                    "value",
                    data=value,
                )
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
            if self.value is not None:
                h5group["value"].resize((h5group["value"].shape[0] + 1,) + tuple(self.shape))
                h5group["value"][-1] = self.value

    def _load_state_hdf5(self, h5group, index: int = -1, _done_load: set = None):
        super()._load_state_hdf5(h5group, index=index, _done_load=_done_load)
        self.cyclic = False
        self.valid = None
        if not self.pointer:
            if isinstance(h5group["value"][()], bytes):
                assert h5group["value"][()] == b"None"
                self.value = None
            elif h5group["value"].attrs["appendable"]:
                self.value = h5group["value"][index]
            else:
                self.value = h5group["value"][()]
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
        return self._valid

    @valid.setter
    def valid(self, valid: tuple[Union[ArrayLike, float, int, None]]):

        if backend.backend == "object":
            self._valid = (None, None)
            return

        if valid is None:
            valid = (None, None)

        if not isinstance(valid, tuple):
            raise ParamConfigurationError(f"Valid must be a tuple ({self.name})")
        if len(valid) != 2:
            raise ParamConfigurationError(f"Valid must be a tuple of length 2 ({self.name})")

        if valid[0] is None and valid[1] is None:
            if self.cyclic:
                raise ParamConfigurationError(
                    f"Cannot set valid to None for cyclic parameter ({self.name})"
                )
            self.to_valid = self._to_valid_base
            self.from_valid = self._from_valid_base
        elif valid[0] is None:
            if self.cyclic:
                raise ParamConfigurationError(
                    f"Cannot set left valid to None for cyclic parameter ({self.name})"
                )
            self.to_valid = self._to_valid_rightvalid
            self.from_valid = self._from_valid_rightvalid
            valid = (None, backend.as_array(valid[1], dtype=self.dtype, device=self.device))
            if self.value is not None and backend.any(self.value > valid[1]):
                warn(InvalidValueWarning(self.name, self.value, valid))
        elif valid[1] is None:
            if self.cyclic:
                raise ParamConfigurationError(
                    f"Cannot set right valid to None for cyclic parameter ({self.name})"
                )
            self.to_valid = self._to_valid_leftvalid
            self.from_valid = self._from_valid_leftvalid
            valid = (backend.as_array(valid[0], dtype=self.dtype, device=self.device), None)
            if self.value is not None and backend.any(self.value < valid[0]):
                warn(InvalidValueWarning(self.name, self.value, valid))
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
                    f"Valid range (valid[1] - valid[0]) must be positive ({self.name})"
                )
            if (
                self.value is not None
                and not self.cyclic
                and (backend.any(self.value < valid[0]) or backend.any(self.value > valid[1]))
            ):
                warn(InvalidValueWarning(self.name, self.value, valid))

        self._valid = valid

    def _to_valid_base(self, value: ArrayLike) -> ArrayLike:
        if self.pointer:
            raise ParamTypeError(
                f"Cannot apply valid transformation to pointer parameter ({self.name})"
            )
        return value

    def _to_valid_fullvalid(self, value: ArrayLike) -> ArrayLike:
        value = self._to_valid_base(value)
        return backend.tan((value - self.valid[0]) * pi / (self.valid[1] - self.valid[0]) - pi / 2)

    def _to_valid_cyclic(self, value: ArrayLike) -> ArrayLike:
        value = self._to_valid_base(value)
        return (value - self.valid[0]) % (self.valid[1] - self.valid[0]) + self.valid[0]

    def _to_valid_leftvalid(self, value: ArrayLike) -> ArrayLike:
        value = self._to_valid_base(value)
        return value - 1.0 / (value - self.valid[0])

    def _to_valid_rightvalid(self, value: ArrayLike) -> ArrayLike:
        value = self._to_valid_base(value)
        return value + 1.0 / (self.valid[1] - value)

    def _from_valid_base(self, value: ArrayLike) -> ArrayLike:
        if self.pointer:
            raise ParamTypeError(
                f"Cannot apply valid transformation to pointer parameter ({self.name})"
            )
        return value

    def _from_valid_fullvalid(self, value: ArrayLike) -> ArrayLike:
        value = self._from_valid_base(value)
        value = (backend.atan(value) + pi / 2) * (self.valid[1] - self.valid[0]) / pi + self.valid[
            0
        ]
        return value

    def _from_valid_cyclic(self, value: ArrayLike) -> ArrayLike:
        value = self._from_valid_base(value)
        value = (value - self.valid[0]) % (self.valid[1] - self.valid[0]) + self.valid[0]
        return value

    def _from_valid_leftvalid(self, value: ArrayLike) -> ArrayLike:
        value = self._from_valid_base(value)
        value = (value + self.valid[0] + backend.sqrt((value - self.valid[0]) ** 2 + 4)) / 2
        return value

    def _from_valid_rightvalid(self, value: ArrayLike) -> ArrayLike:
        value = self._from_valid_base(value)
        value = (value + self.valid[1] - backend.sqrt((value - self.valid[1]) ** 2 + 4)) / 2
        return value

    @property
    def node_str(self) -> str:
        """
        Returns a string representation of the node for graph visualization.
        """
        if (self.static or self._type == "dynamic value") and backend.backend != "object":
            if max(1, prod(self.value.shape)) == 1:
                return f"{self.name}|{self._type}: {self.npvalue:.3g}"
            else:
                return f"{self.name}|{self._type}: {tuple(self.npvalue.shape)}"
        return f"{self.name}|{self._type}"

    def __repr__(self) -> str:
        return self.name
