from typing import Optional, Union, Callable, Any
from warnings import warn
import traceback
from math import prod

from numpy import ndarray
import numpy as np

from .backend import backend, ArrayLike
from .base import Node
from .errors import ParamConfigurationError, ParamTypeError, ActiveStateError
from .warnings import InvalidValueWarning


def valid_shape(shape, value_shape, batched):
    if shape is None:
        return True
    if value_shape == shape:
        return True
    if batched and len(shape) == 0:
        return True
    if batched and len(value_shape) > 1 and value_shape[-len(shape) :] == shape:
        return True
    return False


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
    dynamic: (Optional[bool], optional)
        Force param to be dynamic if True. If a value is provided and param is dynamic
        then it has a default value if none are provided.
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
        shape: Optional[tuple[int, ...]] = None,
        cyclic: bool = False,
        valid: Optional[tuple[Union[ArrayLike, float, int, None]]] = None,
        units: Optional[str] = None,
        dynamic: Optional[bool] = None,
        batched: bool = False,
        dtype: Optional[Any] = None,
        device: Optional[Any] = None,
        **kwargs,
    ):
        self._node_type = "node"
        super().__init__(name=name, **kwargs)
        self._shape = None
        self._value = None
        self.__value = None
        self._valid = (None, None)
        if value is None and backend.backend != "object":
            if shape is None:
                shape = ()
            if not isinstance(shape, (tuple, list)):
                raise ParamConfigurationError("Shape must be a tuple")
            self.shape = tuple(shape)
        elif (
            not isinstance(value, (Param, Callable))
            and value is not None
            and backend.backend != "object"
        ):
            value = backend.as_array(value, dtype=dtype, device=device)
            if not valid_shape(shape, value.shape, batched):
                raise ParamConfigurationError(
                    f"Shape {shape} does not match value shape {value.shape}"
                )
        self._dtype = dtype
        self._device = device
        self._cyclic = cyclic
        self.batched = batched
        self.shape = shape
        self.value = value
        self.dynamic = dynamic
        self.valid = valid
        self.units = units

    @property
    def dynamic(self) -> bool:
        return "dynamic" in self.node_type

    @dynamic.setter
    def dynamic(self, dynamic: Optional[bool]):
        if dynamic is None:
            return
        elif dynamic:
            self.to_dynamic()
        else:
            self.to_static()

    @property
    def pointer(self) -> bool:
        return "pointer" in self.node_type

    @property
    def static(self) -> bool:
        return "static" in self.node_type

    @static.setter
    def static(self, static: bool):
        if static:
            self.to_static()
        else:
            self.to_dynamic()

    @property
    def node_type(self):
        if self._node_type == "dynamic" and self.__value is not None:
            return "dynamic value"
        return self._node_type

    @node_type.setter
    def node_type(self, value):
        pre_type = self._node_type
        if value == "dynamic value":
            value = "dynamic"
        self._node_type = value
        if pre_type != self._node_type:
            self.update_graph()

    def to_dynamic(self, **kwargs):
        """Change this parameter to a dynamic parameter. If the parameter has a
        value, this will become a "dynamic value" parameter."""
        if self.pointer:
            try:
                self.__value = self.__value(self)
            except Exception:
                self.__value = None
        self.node_type = "dynamic"

    def to_static(self, **kwargs):
        """Change this parameter to a static parameter. This only works if the
        parameter has a ``dynamic_value`` set, or if the pointer can be
        evaluated."""
        if self.static:
            return
        if self.pointer:
            try:
                self.__value = self.__value(self)
            except Exception as e:
                raise ParamTypeError(
                    f"Cannot set pointer parameter {self.name} to static with `to_static`. Pointer could not be evaluated because of: \n"
                    + traceback.format_exc()
                )
        if self.__value is None:
            raise ParamTypeError(
                f"Cannot set dynamic parameter {self.name} to static when no `dynamic_value` is set. Try using `static_value(value)` to provide a value and set to static."
            )
        self.node_type = "static"

    @property
    def shape(self) -> Optional[tuple[int, ...]]:
        if backend.backend == "object":
            return None
        value = self.value
        if self.pointer and value is not None:
            return tuple(value.shape)
        if self._shape is None and value is not None:
            return value.shape
        return self._shape

    @shape.setter
    def shape(self, shape):
        if backend.backend == "object":
            raise ParamTypeError("Cannot set shape of parameter with backend 'object'")
        if self.pointer:
            raise ParamTypeError(f"Cannot set shape of parameter {self.name} with type 'pointer'")
        if shape is None:
            self._shape = None
            return
        shape = tuple(shape)
        value = self.value
        if value is not None and not valid_shape(shape, value.shape, self.batched):
            raise ValueError(f"Shape {shape} does not match the shape of the value {value.shape}")
        self._shape = shape

    def _shape_from_value(self, value_shape):
        if self._shape is None:
            self._shape = value_shape
        if not valid_shape(self._shape, value_shape, self.batched):
            self._shape = value_shape

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

    def static_value(self, value):
        # While active no value can be set
        if self.active:
            raise ActiveStateError(
                f"Cannot set static value of parameter {self.name} while active."
            )

        # Catch cases where input is invalid
        if value is None:
            raise ParamTypeError("Cannot set to static with value of None")
        if isinstance(value, Param) or callable(value):
            raise ParamTypeError(
                f"Cannot set static value to pointer ({self.name}). Try setting `pointer_func(func)` or `pointer_func(param)` to create a pointer."
            )

        if backend.backend != "object":
            value = backend.as_array(value, dtype=self._dtype, device=self._device)
        self.__value = value
        self.node_type = "static"
        if backend.backend != "object":
            self._shape_from_value(tuple(value.shape))
        self.is_valid()

    def dynamic_value(self, value):
        # While active no value can be set
        if self.active:
            raise ActiveStateError(
                f"Cannot set dynamic value of parameter {self.name} while active."
            )

        # No dynamic value
        if value is None:
            self.__value = None
            self.node_type = "dynamic"
            return

        # Catch cases where input is invalid
        if isinstance(value, Param) or callable(value):
            raise ParamTypeError(f"Cannot set dynamic value to pointer ({self.name})")

        # Set to dynamic value
        if backend.backend != "object":
            value = backend.as_array(value, dtype=self._dtype, device=self._device)
        self.__value = value
        self.node_type = "dynamic"
        if backend.backend != "object":
            self._shape_from_value(tuple(value.shape))
        self.is_valid()

    def pointer_func(self, value: Union["Param", Callable]):
        # While active no value can be set
        if self.active:
            raise ActiveStateError(
                f"Cannot set pointer function of parameter {self.name} while active"
            )

        if isinstance(value, Param):
            self.link(value)
            p_name = value.name
            value = lambda p: p[p_name].value
        elif not callable(value):
            raise ParamTypeError(f"Pointer function must be a Param or callable ({self.name})")
        elif hasattr(value, "params"):
            self.link(value.params)
        self.__value = value
        self.node_type = "pointer"

    @property
    def value(self) -> Union[ArrayLike, None]:
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
            raise ActiveStateError(f"Cannot set value of parameter {self.name} while active")

        if value is None:
            self.dynamic_value(None)
        elif isinstance(value, Param) or callable(value):
            self.pointer_func(value)
        elif self.dynamic:
            self.dynamic_value(value)
        else:
            self.static_value(value)

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
        return self._cyclic

    @cyclic.setter
    def cyclic(self, cyclic: bool):
        self._cyclic = cyclic
        self.is_valid()

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
                    maxshape=(None,) + self.shape if self.value is not None else None,
                    compression="gzip" if self.value is not None else None,
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
            if self.value is not None:
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
                self.static_value(value)
            elif "dynamic" in h5group["value"].attrs["node_type"]:
                self.dynamic_value(value)
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
        if backend.backend == "object" or self.cyclic or self.pointer:
            return True
        if value is None:
            value = self.value
        if value is None:
            return True
        if self.valid[0] is not None and backend.any(self.value < self.valid[0]):
            warn(InvalidValueWarning(self.name, value, self.valid))
            return False
        elif self.valid[1] is not None and backend.any(self.value > self.valid[1]):
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
        if self.__value is not None and backend.backend != "object":
            if max(1, prod(self.value.shape)) == 1:
                return f"{self.name}|{self.node_type}: {self.npvalue.item():.3g}"
            elif prod(self.value.shape) <= 4:
                value = str(np.char.mod("%.3g", self.npvalue).tolist()).replace("'", "")
                return f"{self.name}|{self.node_type}: {value}"
            else:
                return f"{self.name}|{self.node_type}: {self.shape}"
        return f"{self.name}|{self.node_type}"

    def __repr__(self) -> str:
        return self.name
