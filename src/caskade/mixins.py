from typing import TYPE_CHECKING, Any, Optional, Mapping, Sequence, Union, Callable
from math import prod
import numpy as np

from .param import Param
from .errors import (
    FillParamsError,
    FillParamsArrayError,
    FillParamsMappingError,
    FillParamsSequenceError,
    ActiveStateError,
    ParamConfigurationError,
)
from .backend import backend, ArrayLike
from .base import Node

if TYPE_CHECKING:
    from .module import Module


def _split_batch_shape(
    viewer: Node, param: Param, batch_shape: Optional[tuple[int, ...]] = None
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Split the batch dimensions of ``param`` as they are seen from ``viewer``.

    Returns ``(batch, links)``. ``links`` holds the trailing dimensions that
    belong to the hierarchical links between ``viewer`` and the param, one per
    link and nearest link first, these are part of a single simulation of
    ``viewer``. ``batch`` holds the dimensions before them, which are a batch
    of whole simulations of ``viewer``. A param with no batch dimensions has
    one value that every copy shares, so both are empty. See
    ``Node.hierarchical_link`` for the full rule. ``batch_shape`` may be given
    when it is already known, otherwise it is asked of the param.

    Raises
    ------
    ParamConfigurationError
        If the param has batch dimensions, but fewer than there are links.
        There is no telling which of the links those dimensions are for.
    """
    if batch_shape is None:
        batch_shape = param.batch_shape
    depth = viewer._hierarchy_depths().get(param, 0)
    if depth == 0 or not batch_shape:
        return batch_shape, ()
    cut = len(batch_shape) - depth
    if cut < 0:
        raise ParamConfigurationError(
            f"Param '{param.name}' is below {depth} hierarchical links from '{viewer.name}', "
            f"so its batch shape needs at least {depth} dimensions (one for each link, "
            f"after any batch dimensions of '{viewer.name}'), or none at all for a single "
            f"value shared by every copy. Its batch shape is {batch_shape}."
        )
    return batch_shape[:cut], batch_shape[cut:]


def _link_shape(viewer: Node, param: Param) -> tuple[int, ...]:
    """The ``links`` part of ``_split_batch_shape`` on its own.

    This runs for every param each time a simulation is given its values, and
    most params have no hierarchical link above them. Their batch shape is not
    needed to answer, so it is not worked out.
    """
    if viewer._hierarchy_depths().get(param, 0) == 0:
        return ()
    return _split_batch_shape(viewer, param)[1]


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

    # Groups
    #################################################################
    def _split_groups(
        self, params, param_list: tuple[Param, ...], groups: tuple[int, ...]
    ) -> list[tuple[Any, tuple[Param, ...]]]:
        """Pair each entry of ``params`` with the params of its group.

        Raises
        ------
        FillParamsError
            If ``params`` is not a sequence with one entry for each group.
        """
        if isinstance(params, (backend.array_type, str)) or not isinstance(params, Sequence):
            given = f"a {type(params).__name__}"
        elif len(params) != len(groups):
            given = f"{len(params)} entries"
        else:
            return [
                (params_g, tuple(p for p in param_list if p.group == group))
                for group, params_g in zip(groups, params)
            ]
        raise FillParamsError(
            f"The params of '{self.name}' are in {len(groups)} groups {groups}, so their values "
            "are given as a list with one entry for each group, the way `get_values` returns "
            f"them. Got {given}."
        )

    # Set Values
    #################################################################
    def _set_values_dict(self, node, params, params_list, attribute="_active_value"):
        for key in params:
            if key in node.children and isinstance(params[key], dict):
                self._set_values_dict(node[key], params[key], params_list, attribute=attribute)
            elif key in node.children and isinstance(node[key], Param):
                setattr(node[key], attribute, params[key])
            elif key in node.children:
                # An array or list in the mapping holds the values of every param below
                # this child. It is a piece of the array of ``self``, the module that
                # was given the mapping, so ``self`` says which dimensions are link ones.
                below = node[key]._hierarchy_depths()  # every node at or below the child
                sublist = tuple(p for p in params_list if p in below)
                node[key]._set_values(params[key], sublist, attribute=attribute, viewer=self)
            else:
                raise FillParamsMappingError(self.name, self.children, missing_key=key)

    def _set_values(
        self,
        params: Union[ArrayLike, Sequence, Mapping],
        param_list: tuple[Param, ...],
        attribute: str = "_active_value",
        viewer: Optional[Node] = None,
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
        viewer: (Optional[Node])
            The module whose array ``params`` is (a piece of), this decides
            which batch dimensions of a param are flattened into the array.
            It is ``self`` unless ``params`` was found inside a Mapping that was
            given to a module higher in the graph.
        """

        if isinstance(params, backend.array_type):
            if params.shape[-1] == 0:
                return  # No parameters to fill
            if viewer is None:
                viewer = self
            # check for batch dimension
            batch = len(params.shape) > 1
            B = tuple(params.shape[:-1]) if batch else ()
            shapes = tuple(_link_shape(viewer, param) + param.shape for param in param_list)
            pos = 0
            for param, shape in zip(param_list, shapes):
                # Handle scalar parameters
                size = max(1, prod(shape))
                try:
                    val = backend.view(params[..., pos : pos + size], B + shape)
                    setattr(param, attribute, val)
                except (RuntimeError, IndexError, ValueError, TypeError) as err:
                    raise FillParamsArrayError(self.name, params, param_list, shapes) from err

                pos += size
            if pos != params.shape[-1]:
                raise FillParamsArrayError(self.name, params, param_list, shapes)
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
        self,
        params: Union[ArrayLike, Sequence, Mapping],
        dynamic: bool = True,
        attribute: str = "value",
        respect_valid: bool = True,
    ):
        """Fill parameter values of the module from the provided data.

        Parameters
        ----------
        params : Union[ArrayLike, Sequence, Mapping]
            Values to assign to the parameters.  Accepted formats:

            * **ArrayLike** - a flat (or batched) array whose last dimension is
              concatenated parameter values in topological order.
            * **Sequence** - one element per parameter, matched by position.
            * **Mapping** - keys matching child names, values being the
              parameter data (may be nested).

            When multiple dynamic parameter groups exist, ``params`` should be a
            sequence of per-group containers.
        dynamic : bool, optional
            If ``True`` (default), sets dynamic parameters; otherwise sets
            static parameters.
        attribute : str, optional
            The ``Param`` attribute to write to, by default ``"value"``.
        respect_valid : bool, optional
            If ``True`` (default), will apply from_valid transformations before
            assignment.

        Raises
        ------
        ActiveStateError
            If the module is currently in an active (tracing) state.
        """
        if self.active:
            raise ActiveStateError(f"Cannot fill dynamic values when Module {self.name} is active")

        param_list = self.dynamic_params if dynamic else self.static_params
        groups = self.dynamic_param_groups if dynamic else self.static_param_groups

        if len(groups) > 1:
            for params_g, param_list_g in self._split_groups(params, param_list, groups):
                if self.valid_context and respect_valid:
                    params_g = self._transform_params(self, params_g, param_list_g, "from_valid")
                self._set_values(params_g, param_list_g, attribute=attribute)
        else:
            if self.valid_context and respect_valid:
                params = self._transform_params(self, params, param_list, "from_valid")
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

    @staticmethod
    def _getattr_or_callable(obj, attribute):
        """Get an attribute or call a callable on an object."""
        if callable(attribute):
            return attribute(obj)
        return getattr(obj, attribute)

    def _recursive_build_params_dict(
        self, node: Node, unique_params: set, param_list, attribute="value"
    ):
        params = {}
        for link, child in node.children.items():
            if isinstance(child, Param) and child in param_list and child not in unique_params:
                unique_params.add(child)
                params[link] = self._getattr_or_callable(child, attribute)
            else:
                params[link] = self._recursive_build_params_dict(
                    child, unique_params=unique_params, param_list=param_list, attribute=attribute
                )
                if len(params[link]) == 0:
                    del params[link]
        return params

    def get_values(
        self,
        scheme: str = "array",
        dynamic: bool = True,
        attribute: Union[str, Callable] = "value",
        group: Optional[int] = None,
        respect_valid: bool = True,
    ) -> Union[ArrayLike, list[ArrayLike], dict[str, Union[dict, ArrayLike]]]:
        """Retrieve parameter values from the module.

        Parameters
        ----------
        scheme : str, optional
            Output format, one of ``"array"`` (default), ``"list"``, or
            ``"dict"``.

            * ``"array"`` / ``"tensor"`` - returns a single flat array with
              all parameter values concatenated along the last axis.
            * ``"list"`` - returns a list of raw parameter values.
            * ``"dict"`` - returns a nested dictionary mirroring the graph
              structure.
        dynamic : bool, optional
            If ``True`` (default), retrieves dynamic parameters; otherwise
            retrieves static parameters.
        attribute : Union[str, Callable], optional
            The ``Param`` attribute to read from, by default ``"value"``.
        group : int or None, optional
            Restrict to a specific parameter group.  When ``None`` (default)
            and multiple groups exist, returns a list of per-group results.
        respect_valid : bool, optional
            If ``True`` (default), will apply to_valid transformation.

        Returns
        -------
        Union[ArrayLike, list[ArrayLike], dict[str, Union[dict, ArrayLike]]]
            Parameter values in the format specified by *scheme*.  When
            multiple groups exist and *group* is ``None``, a list of
            per-group results is returned.
        """
        groups = self.dynamic_param_groups if dynamic else self.static_param_groups
        if len(groups) > 1 and group is None:
            values = []
            for g in groups:
                values.append(
                    self.get_values(
                        scheme=scheme,
                        dynamic=dynamic,
                        attribute=attribute,
                        group=g,
                        respect_valid=respect_valid,
                    )
                )
            return values
        param_list = self.dynamic_params if dynamic else self.static_params
        param_list = tuple(p for p in param_list if (group is None or p.group == group))

        self._check_values(param_list, scheme)
        x = []
        if scheme.lower() in ["array", "tensor"]:
            for param in param_list:
                B = _split_batch_shape(self, param)[0]
                x.append(self._getattr_or_callable(param, attribute).reshape(B + (-1,)))
            if len(x) == 0:
                return backend.make_array([])
            x = backend.broadcast_cat(x, dim=-1)
            if not self.active:
                # Stored values are handed out cut off from their history. The
                # values of a running simulation are part of its calculation.
                x = backend.detach(x)
        elif scheme.lower() == "list":
            for param in param_list:
                x.append(self._getattr_or_callable(param, attribute))
        elif scheme.lower() == "dict":
            unique_params = set()
            x = self._recursive_build_params_dict(
                self, unique_params=unique_params, param_list=param_list, attribute=attribute
            )
        if self.valid_context and respect_valid:
            x = self._transform_params(self, x, param_list, "to_valid")
        return x

    def get_dims(
        self, scheme: str = "list", dynamic: bool = True, group: Optional[int] = None
    ) -> Union[Optional[int], list[Optional[int]], dict[str, Union[dict, Optional[int]]]]:
        """The dimension to ``vmap`` over for each value that ``get_values``
        returns, to run one simulation for each entry of a batch.

        The entry for a value is ``0`` when it has batch dimensions (they are
        in front) and ``None`` when it has a single value that every simulation
        shares. The result has the layout of ``get_values`` with the same
        arguments, so the two go together::

            values = module.get_values("list")
            dims = module.get_dims()
            results = vmap(module.method, in_dims=(None, dims))(x, values)

        Parameters
        ----------
        scheme : str, optional
            ``"list"`` (default) or ``"dict"`` for one entry per param. With
            ``"array"`` the answer is a single entry, since the array holds
            every param and has the batch dimensions when any param has them.
        dynamic : bool, optional
            If ``True`` (default), for the dynamic parameters; otherwise for
            the static parameters.
        group : int or None, optional
            Restrict to a specific parameter group, as in ``get_values``.
        """

        def dim(param):
            return 0 if _split_batch_shape(self, param)[0] else None

        if scheme.lower() in ["array", "tensor"]:
            dims = self.get_dims("list", dynamic=dynamic, group=group)
            groups = self.dynamic_param_groups if dynamic else self.static_param_groups
            if len(groups) > 1 and group is None:
                return [0 if 0 in d else None for d in dims]
            return 0 if 0 in dims else None
        return self.get_values(
            scheme, dynamic=dynamic, attribute=dim, group=group, respect_valid=False
        )

    def _array_inspection(self, group: Optional[int] = None):
        param_list = self.dynamic_params
        param_list = tuple(p for p in param_list if (group is None or p.group == group))
        self._check_values(param_list, "array")

        x = []
        for param in param_list:
            shape = _link_shape(self, param) + param.shape
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
        if group is None and len(self.dynamic_param_groups) > 1:
            raise ValueError(
                f"The dynamic params of '{self.name}' are in the groups "
                f"{self.dynamic_param_groups}, each with params of its own. Give the `group` "
                "that the index is for."
            )

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
        self, param: Union[Param, tuple[Param, ...], "Module"], scheme: str = "array"
    ) -> Union[int, slice]:
        """
        Identify what index is associated with a param in the dynamic params
        array.

        Parameters
        ----------
        param: Union[Param, tuple[Param, ...], Module]
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
    def _transform_params(
        self,
        node: Node,
        init_params: Union[ArrayLike, Sequence, Mapping],
        param_list: tuple[Param, ...],
        transform_attr: str,
    ):
        if isinstance(init_params, backend.array_type):
            trans_params = []
            batch = len(init_params.shape) > 1
            B = tuple(init_params.shape[:-1]) if batch else ()
            pos = 0
            for param in param_list:
                # The array is (a piece of) the array of ``self``, as in ``_set_values``
                shape = _link_shape(self, param) + param.shape
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
                    below = node[key]._hierarchy_depths()  # every node at or below this child
                    sublist = tuple(p for p in param_list if p in below)
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

    def _transform_groups(self, params, param_list, group, transform_attr: str):
        """``to_valid`` or ``from_valid`` of values in the layout of ``get_values``."""
        if param_list is None:
            param_list = self.dynamic_params
            groups = self.dynamic_param_groups
        else:
            groups = tuple(sorted(set(p.group for p in param_list)))
        if group is not None:
            param_list = tuple(p for p in param_list if p.group == group)
        elif len(groups) > 1:
            return [
                self._transform_params(self, params_g, param_list_g, transform_attr)
                for params_g, param_list_g in self._split_groups(params, param_list, groups)
            ]
        return self._transform_params(self, params, param_list, transform_attr)

    def to_valid(
        self,
        params: Union[ArrayLike, Sequence, Mapping],
        param_list: Optional[tuple[Param, ...]] = None,
        group: Optional[int] = None,
    ) -> Union[ArrayLike, Sequence, Mapping]:
        """Map parameter values from their natural range to an unconstrained space.

        Takes parameter values that lie within each parameter's valid range
        (e.g. 0-1 for an axis ratio) and maps them into the unconstrained
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
        return self._transform_groups(params, param_list, group, "to_valid")

    def from_valid(
        self,
        valid_params: Union[ArrayLike, Sequence, Mapping],
        param_list: Optional[tuple[Param, ...]] = None,
        group: Optional[int] = None,
    ) -> Union[ArrayLike, Sequence, Mapping]:
        """Map parameter values from the unconstrained space back to their natural range.

        Takes values in the unconstrained domain ``(-inf, inf)`` (as
        produced by :meth:`to_valid` or proposed by an optimizer/sampler)
        and maps them back into each parameter's original valid range
        (e.g. 0-1 for an axis ratio).

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
        return self._transform_groups(valid_params, param_list, group, "from_valid")
