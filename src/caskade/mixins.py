from typing import Optional, Mapping, Sequence, Union
from math import prod
import numpy as np

from .param import Param
from .errors import (
    FillParamsArrayError,
    FillParamsMappingError,
    FillParamsSequenceError,
    ActiveStateError,
    ParamConfigurationError,
)
from .backend import backend, ArrayLike
from .base import Node, Memo


class GetSetValues:
    """Mixin providing methods for getting and setting parameter values.

    Provides array, list, and dict interfaces for reading and writing
    dynamic (or static) parameter values on a ``Module`` or
    ``NodeCollection``.  Also includes helpers for locating parameters
    by index and for converting between raw and valid (transformed)
    parameter spaces.
    """

    @property
    def valid_context(self) -> bool:
        """Return True if the module is in a valid context."""
        try:
            return self._valid_context
        except AttributeError:
            return False

    @valid_context.setter
    def valid_context(self, value: bool):
        """Set the valid context of the module."""
        self._valid_context = value
        for node in self.topological_ordering():
            if isinstance(node, GetSetValues):
                node._valid_context = value

    # Set Values
    #################################################################
    def _set_values_dict(self, node, params, params_list, attribute="_value"):
        for key in params:
            if key in node.children and isinstance(params[key], dict):
                self._set_values_dict(node[key], params[key], params_list, attribute=attribute)
            elif key in node.children and isinstance(node[key], Param):
                setattr(node[key], attribute, params[key])
            elif key in node.children:
                sublist = tuple(p for p in params_list if p in node[key].all_params)
                node[key]._set_values(params[key], sublist, attribute=attribute)
            else:
                raise FillParamsMappingError(self.name, self.children, missing_key=key)

    def _set_values(
        self,
        params: Union[ArrayLike, Sequence, Mapping],
        param_list: tuple[Param],
        attribute="_value",
    ):
        """
        Fill the dynamic parameters of the module with the input values from
        params.

        Parameters
        ----------
        params: (Union[ArrayLike, Sequence, Mapping])
            The input values to fill the dynamic parameters with. The input can
            be an ArrayLike, a Sequence, or a Mapping. If the input is
            array-like, the values are filled in order of the dynamic
            parameters. `params` should be a flattened array-like object with
            all parameters concatenated in the order of the dynamic parameters.
            If `len(params.shape)>1` then all dimensions but the last one are
            considered batch dimensions. If the input is a Sequence, the values
            are filled in order of the dynamic parameters. If the input is a
            Mapping, the values are filled by matching the keys of the Mapping
            to the names of the dynamic parameters. Note that the system does
            not check for missing keys in the dictionary, but you will get an
            error eventually if a value is missing.
        """

        if isinstance(params, backend.array_type):
            if params.shape[-1] == 0:
                return  # No parameters to fill
            # check for batch dimension
            batch = len(params.shape) > 1
            B = tuple(params.shape[:-1]) if batch else ()
            pos = 0
            for param in param_list:
                if param.online:
                    shape = param.shape
                else:
                    depth = max(memo.count("|") for memo in param.memos)
                    shape = param.batch_shape[-depth:] + param.shape
                # Handle scalar parameters
                size = max(1, prod(shape))
                try:
                    val = backend.view(params[..., pos : pos + size], B + shape)
                    setattr(param, attribute, val)
                except (RuntimeError, IndexError, ValueError, TypeError):
                    raise FillParamsArrayError(self.name, params, param_list)

                pos += size
            if pos != params.shape[-1]:
                raise FillParamsArrayError(self.name, params, param_list)
        elif isinstance(params, Sequence):
            if len(params) == 0:
                return
            elif len(params) == len(param_list):
                for param, value in zip(param_list, params):
                    setattr(param, attribute, value)
            else:
                raise FillParamsSequenceError(self.name, params, param_list)
        elif isinstance(params, Mapping):
            self._set_values_dict(self, params, param_list, attribute=attribute)
        else:
            raise TypeError(
                f"Input params type {type(params)} not supported. Should be {backend.array_type.__name__}, Sequence, or Mapping."
            )

    def set_values(
        self, params: Union[ArrayLike, Sequence, Mapping], dynamic=True, attribute="value"
    ):
        """Fill parameter values of the module from the provided data.

        Parameters
        ----------
        params : Union[ArrayLike, Sequence, Mapping]
            Values to assign to the parameters.  Accepted formats:

            * **ArrayLike** – a flat (or batched) array whose last dimension
              is concatenated parameter values in topological order.
            * **Sequence** – one element per parameter, matched by position.
            * **Mapping** – keys matching child names, values being the
              parameter data (may be nested).

            When multiple dynamic parameter groups exist, ``params`` should
            be a sequence of per-group containers.
        dynamic : bool, optional
            If ``True`` (default), sets dynamic parameters; otherwise sets
            static parameters.
        attribute : str, optional
            The ``Param`` attribute to write to, by default ``"value"``.

        Raises
        ------
        ActiveStateError
            If the module is currently in an active (tracing) state.
        """
        if self.active:
            raise ActiveStateError(f"Cannot fill dynamic values when Module {self.name} is active")

        param_list = self.dynamic_params if dynamic else self.static_params

        with Memo(self, self.name + ":semi_set_active"):
            if len(self.dynamic_param_groups) > 1:
                for group, params_g in zip(self.dynamic_param_groups, params):
                    param_list_g = tuple(p for p in param_list if p.group == group)
                    if self.valid_context:
                        params_g = self.from_valid(params_g, param_list_g, group=group)
                    self._set_values(params_g, param_list_g, attribute=attribute)
            else:
                if self.valid_context:
                    params = self.from_valid(params, param_list)
                self._set_values(params, param_list, attribute=attribute)

    # Get Values
    #################################################################
    def _check_values(self, param_list, scheme):
        """Check if all dynamic values are set."""
        bad_params = []
        for param in param_list:
            if param.value is None:
                bad_params.append(param.name)
        if len(bad_params) > 0:
            raise ParamConfigurationError(
                f"{self.name} Param(s) {bad_params} have no value, so the params {scheme} cannot be built. Set their value to use this feature."
            )

    def get_values(
        self, scheme="array", dynamic=True, attribute="value", group: Optional[int] = None
    ) -> Union[ArrayLike, list[ArrayLike], dict[str, Union[dict, ArrayLike]]]:
        """Retrieve parameter values from the module.

        Parameters
        ----------
        scheme : str, optional
            Output format, one of ``"array"`` (default), ``"list"``, or
            ``"dict"``.

            * ``"array"`` / ``"tensor"`` – returns a single flat array with
              all parameter values concatenated along the last axis.
            * ``"list"`` – returns a list of raw parameter values.
            * ``"dict"`` – returns a nested dictionary mirroring the graph
              structure.
        dynamic : bool, optional
            If ``True`` (default), retrieves dynamic parameters; otherwise
            retrieves static parameters.
        attribute : str, optional
            The ``Param`` attribute to read from, by default ``"value"``.
        group : int or None, optional
            Restrict to a specific parameter group.  When ``None`` (default)
            and multiple groups exist, returns a list of per-group results.

        Returns
        -------
        Union[ArrayLike, list[ArrayLike], dict[str, Union[dict, ArrayLike]]]
            Parameter values in the format specified by *scheme*.  When
            multiple groups exist and *group* is ``None``, a list of
            per-group results is returned.
        """
        if len(self.dynamic_param_groups) > 1 and group is None:
            values = []
            for g in self.dynamic_param_groups:
                values.append(
                    self.get_values(scheme=scheme, dynamic=dynamic, attribute=attribute, group=g)
                )
            return values
        param_list = self.dynamic_params if dynamic else self.static_params
        param_list = tuple(p for p in param_list if (group is None or p.group == group))

        self._check_values(param_list, scheme)
        x = []
        if scheme.lower() in ["array", "tensor"]:
            with Memo(self, self.name + ":semi_get_active"):
                for param in param_list:
                    if param.online:
                        B = param.batch_shape
                    else:
                        depth = max(memo.count("|") for memo in param.memos)
                        B = param.batch_shape[:-depth]
                    x.append(getattr(param, attribute).reshape(B + (-1,)))
            if len(x) == 0:
                return backend.make_array([])
            x = backend.detach(backend.broadcast_cat(x, dim=-1))
        elif scheme.lower() == "list":
            for param in param_list:
                x.append(getattr(param, attribute))
        elif scheme.lower() == "dict":
            unique_params = set()
            x = self._recursive_build_params_dict(
                self, unique_params=unique_params, param_list=param_list, attribute=attribute
            )
        if self.valid_context:
            x = self.to_valid(x, group=group)
        return x

    def _recursive_build_params_dict(
        self, node: Node, unique_params: set, param_list, attribute="value"
    ):
        params = {}
        for link, child in node.children.items():
            if isinstance(child, Param) and child in param_list and child not in unique_params:
                unique_params.add(child)
                params[link] = getattr(child, attribute)
            else:
                params[link] = self._recursive_build_params_dict(
                    child, unique_params=unique_params, param_list=param_list, attribute=attribute
                )
                if len(params[link]) == 0:
                    del params[link]
        return params

    def _array_inspection(self, group: Optional[int] = None):
        param_list = self.dynamic_params
        param_list = tuple(p for p in param_list if (group is None or p.group == group))
        self._check_values(param_list, "array")

        x = []
        with Memo(self, self.name + ":semi_findidx_active"):
            for param in param_list:
                if param.online:
                    shape = param.shape
                else:
                    depth = max(memo.count("|") for memo in param.memos)
                    shape = param.batch_shape[-depth:] + param.shape
                if shape == ():
                    x.append((param, ()))
                else:
                    for i in range(prod(shape)):
                        x.append((param, tuple(itm.item() for itm in np.unravel_index(i, shape))))
        return x

    # Finders
    #################################################################
    def find_param(
        self, idx: Union[int, tuple[int]], group: Optional[int] = None, scheme: str = "array"
    ) -> tuple[Param, tuple[int]]:
        """
        Identify which param is associated with the provided index in the
        dynamic params array.

        Parameters
        ----------
        idx: Union[int, tuple[int]]
            The index in the params array at which we wish to find the
            associated param.
        group: Optional[int]
            If the dynamic params have multiple group values, then this argument
            specifies which group to check.
        scheme: str
            Whether to search the array (default) params or list version of
            params. dict is currently unsupported.

        Returns
        -------
        param_info: tuple[Param, tuple[int]]
            A tuple with the Param object and the index within the Param value
            associated with idx (empty tuple if scalar). If idx is a tuple then
            the result is a tuple of these results.
        """
        if not isinstance(idx, int):
            return tuple(self.find_param(i, group, scheme) for i in idx)

        if scheme == "array":
            x = self._array_inspection(group)
            return x[idx]
        elif scheme == "list":
            param_list = tuple(p for p in self.dynamic_params if group is None or p.group == group)
            return param_list[idx]
        elif scheme == "dict":
            raise NotImplementedError(
                "find_param is not implemented for the dict scheme. The dict has the same structure as the graph and so may be inspected in a variety of other ways."
            )
        else:
            raise ValueError(f"unrecognized scheme: {scheme}")

    def find_index(
        self, param: Union[Param, tuple[Param], "Module"], scheme: str = "array"
    ) -> Union[int, slice]:
        """
        Identify what index is associated with a param in the dynamic params
        array.

        Parameters
        ----------
        param: Union[Param, tuple[Param], Module]
            The param for which to find the associated index.
        scheme: str
            Whether to search the array (default) params or list version of
            params. dict is currently unsupported.

        Returns
        -------
        param_info: Union[int, slice]
            A int giving the index associated with the provided Param object. If
            the param is multi-dimensional then the result will be a slice over
            all indices associated with that param.
        """
        # 1. Handle recursive structures
        if isinstance(param, (list, tuple)):
            return tuple(self.find_index(p, scheme) for p in param)
        if isinstance(param, GetSetValues):
            return tuple(
                self.find_index(c, scheme)
                for c in param.children.values()
                if isinstance(c, Param) and c.dynamic
            )

        groups = self.dynamic_param_groups if len(self.dynamic_param_groups) > 1 else [None]

        for group in groups:
            if scheme in ["array", "tensor"]:
                inspection = self._array_inspection(group)
                matches = [i for i, item in enumerate(inspection) if item[0] is param]

                if not matches:
                    continue
                idx = matches[0] if len(matches) == 1 else slice(min(matches), max(matches) + 1)

            elif scheme == "list":
                param_list = [p for p in self.dynamic_params if group is None or p.group == group]
                if param not in param_list:
                    continue
                idx = param_list.index(param)
            elif scheme == "dict":
                raise NotImplementedError("find_index is not implemented for the dict scheme.")
            else:
                raise ValueError(f"unrecognized scheme: {scheme}")

            # Return with group prefix if we are in multi-group mode
            return (group, idx) if len(self.dynamic_param_groups) > 1 else idx

        raise ValueError(f"Param {param.name} could not be found in dynamic params.")

    # To/From Valid
    #################################################################
    def _transform_params(self, node, init_params, param_list, transform_attr):
        if isinstance(init_params, backend.array_type):
            trans_params = []
            batch = len(init_params.shape) > 1
            B = tuple(init_params.shape[:-1]) if batch else ()
            pos = 0
            with Memo(self, self.name + ":semi_trans_active"):
                for param in param_list:
                    if param.online:
                        shape = param.shape
                    else:
                        depth = max(memo.count("|") for memo in param.memos)
                        shape = param.batch_shape[-depth:] + param.shape
                    size = max(1, prod(shape))  # Handle scalar parameters
                    return_shape = (*B, size)
                    val = getattr(param, transform_attr)(
                        backend.view(init_params[..., pos : pos + size], B + shape)
                    )
                    trans_params.append(backend.view(val, return_shape))
                    pos += size
            trans_params = backend.concatenate(trans_params, axis=-1)
        elif isinstance(init_params, Sequence):
            trans_params = []
            if len(init_params) == len(param_list):
                for param, value in zip(param_list, init_params):
                    trans_params.append(getattr(param, transform_attr)(value))
            else:
                raise FillParamsSequenceError(self.name, init_params, param_list)
        elif isinstance(init_params, Mapping):
            trans_params = {}
            for key in init_params:
                if key in node.children and isinstance(node[key], Param):
                    trans_params[key] = getattr(node[key], transform_attr)(init_params[key])
                elif key in node.children:
                    sublist = tuple(p for p in param_list if p in node[key].children.values())
                    trans_params[key] = self._transform_params(
                        node[key], init_params[key], sublist, transform_attr
                    )
                else:
                    raise FillParamsMappingError(self.name, self.children, missing_key=key)
        else:
            raise TypeError(
                f"Input params type {type(init_params)} not supported. Should be {backend.array_type.__name__}, Sequence, or Mapping."
            )
        return trans_params

    def to_valid(
        self, params: Union[ArrayLike, Sequence, Mapping], param_list=None, group=None
    ) -> Union[ArrayLike, Sequence, Mapping]:
        """Map parameter values from their natural range to an unconstrained space.

        Takes parameter values that lie within each parameter's valid range
        (e.g. 0–1 for an axis ratio) and maps them into the unconstrained
        domain ``(-inf, inf)``.  The inverse mapping :meth:`from_valid` will
        map any value in ``(-inf, inf)`` back into the original valid range.
        This is useful for interfacing with samplers and optimizers that
        require unconstrained parameters.

        Parameters
        ----------
        params : Union[ArrayLike, Sequence, Mapping]
            Raw parameter values in any supported format (array, sequence,
            or mapping).
        param_list : tuple of Param or None, optional
            Subset of parameters to transform.  Defaults to all dynamic
            parameters.
        group : int or None, optional
            Restrict to a specific parameter group.  When ``None`` and
            multiple groups exist, all groups are transformed.

        Returns
        -------
        Union[ArrayLike, Sequence, Mapping]
            Transformed values in the same format as *params*.
        """
        if param_list is None:
            param_list = self.dynamic_params
        if len(self.dynamic_param_groups) > 1:
            if group is None:
                valid_params = []
                for g, params_g in zip(self.dynamic_param_groups, params):
                    param_list_g = tuple(p for p in param_list if p.group == g)
                    valid_params.append(
                        self._transform_params(self, params_g, param_list_g, "to_valid")
                    )
                return valid_params
            else:
                param_list = tuple(p for p in param_list if p.group == group)
        return self._transform_params(self, params, param_list, "to_valid")

    def from_valid(
        self, valid_params: Union[ArrayLike, Sequence, Mapping], param_list=None, group=None
    ) -> Union[ArrayLike, Sequence, Mapping]:
        """Map parameter values from the unconstrained space back to their natural range.

        Takes values in the unconstrained domain ``(-inf, inf)`` (as
        produced by :meth:`to_valid` or proposed by an optimizer/sampler)
        and maps them back into each parameter's original valid range
        (e.g. 0–1 for an axis ratio).

        Parameters
        ----------
        valid_params : Union[ArrayLike, Sequence, Mapping]
            Parameter values in the unconstrained ``(-inf, inf)`` domain,
            in any supported format (array, sequence, or mapping).
        param_list : tuple of Param or None, optional
            Subset of parameters to transform.  Defaults to all dynamic
            parameters.
        group : int or None, optional
            Restrict to a specific parameter group.  When ``None`` and
            multiple groups exist, all groups are transformed.

        Returns
        -------
        Union[ArrayLike, Sequence, Mapping]
            Inverse-transformed values in the same format as *valid_params*.
        """
        if param_list is None:
            param_list = self.dynamic_params
        if len(self.dynamic_param_groups) > 1:
            if group is None:
                params = []
                for g, valid_params_g in zip(self.dynamic_param_groups, valid_params):
                    param_list_g = tuple(p for p in param_list if p.group == g)
                    params.append(
                        self._transform_params(self, valid_params_g, param_list_g, "from_valid")
                    )
                return params
            else:
                param_list = tuple(p for p in param_list if p.group == group)
        return self._transform_params(self, valid_params, param_list, "from_valid")
