from typing import Optional, Union
import yaml
from ruamel.yaml import YAML
from ruamel.yaml.comments import CommentedMap
import torch
import io
import re
from inspect import signature
import inspect
import uuid
from typing import Annotated, get_origin, get_args

from .errors import GraphError, NodeConfigurationError
import numpy as np

class Node:
    """
    Base graph node class for ``caskade`` objects.

    The ``Node`` object is the base class for all ``caskade`` objects. It is used to
    construct the directed acyclic graph (DAG). The primary function of the
    ``Node`` object is to manage the parent-child relationships between nodes in
    the graph. There is limited functionality for the ``Node`` object, though it
    implements the base versions of the ``active`` state and ``to`` /
    ``update_graph`` methods. The ``active`` state is used to communicate
    through the graph that the simulator is currently running. The ``to`` method
    is used to move and/or cast the values of the parameter. The ``update_graph``
    method is used signal all parents that the graph below them has changed.

    Examples
    --------

    Example making some ``Node`` objects and then linking/unlinking them::

       n1 = Node()
       n2 = Node()
       n1.link("subnode", n2) # link n2 as a child of n1, may use any str as the key
       n1.unlink("subnode") # alternately n1.unlink(n2) to unlink by object
    """

    graphviz_types = {"node": {"style": "solid", "color": "black", "shape": "circle"}}

    # TODO - use path instead of id
    id_set = dict()

    def __init__(self, name: Optional[str] = None, uid: Optional[str] = None):
        if name is None:
            name = self.__class__.__name__

        if uid is not None:
            if uuid in Node.id_set:
                raise GraphError(f"ID {uid} already exists")
        else:
            uid = str(uuid.uuid4())

        Node.id_set[uid] = self

        if not isinstance(name, str):
            raise NodeConfigurationError(f"{self.__class__.__name__} name must be a string")
        if "|" in name:
            raise NodeConfigurationError(f"{self.__class__.__name__} cannot contain '|'")

        self._name = name
        self._children = {}
        self._parents = set()
        self._active = False
        self._type = "node"
        self.uid = uid

    @property
    def name(self) -> str:
        return self._name

    @property
    def children(self) -> dict:
        return self._children

    @property
    def parents(self) -> set:
        return self._parents

    def link(self, key: Union[str, "Node"], child: Optional["Node"] = None):
        """Link the current ``Node`` object to another ``Node`` object as a child.

        Parameters
        ----------
        key: (Union[str, Node])
            The key to link the child node with.
        child: (Optional[Node], optional)
            The child ``Node`` object to link to. Defaults to None in which
            case the key is used as the child.

        Examples
        --------

        Example making some ``Node`` objects and then linking/unlinking them. demonstrating multiple ways to link/unlink::

            n1 = Node()
            n2 = Node()

            n1.link("subnode", n2) # may use any str as the key
            n1.unlink("subnode")

            # Alternately, link by object
            n1.link(n2)
            n1.unlink(n2)
        """
        if self.active:
            raise GraphError("Cannot link/unlink nodes while the graph is active")
        if child is None:
            child = key
            key = child.name
        # Avoid double linking to the same object
        if key in self.children:
            raise GraphError(f"Child key {key} already linked to parent {self.name}")
        if child in self.children.values():
            raise GraphError(f"Child {child.name} already linked to parent {self.name}")
        # avoid cycles
        if self in child.topological_ordering():
            raise GraphError(
                f"Linking {child.name} to {self.name} would create a cycle in the graph"
            )

        self._children[key] = child
        child._parents.add(self)
        self.update_graph()

    def unlink(self, key: Union[str, "Node"]):
        """Unlink the current ``Node`` object from another ``Node`` object which is a child."""
        if self.active:
            raise GraphError("Cannot link/unlink nodes while the graph is active")
        if isinstance(key, Node):
            node_key = None
            for node in self.children:
                if self.children[node] == key:
                    node_key = node
                    break
            if node_key is None:
                raise GraphError(f"Child {key.name} not linked to parent {self.name}")
            else:
                key = node_key
        self._children[key]._parents.remove(self)
        self._children[key].update_graph()
        del self._children[key]
        self.update_graph()

    def topological_ordering(
        self, with_type: Optional[str] = None, with_isinstance: Optional[object] = None
    ) -> tuple["Node"]:
        """Return a topological ordering of the graph below the current node."""
        ordering = [self]
        for node in self.children.values():
            for subnode in node.topological_ordering():
                if subnode not in ordering:
                    ordering.append(subnode)
        if with_type is not None:
            ordering = filter(lambda n: with_type in n._type, ordering)
        if with_isinstance is not None:
            ordering = filter(lambda n: isinstance(n, with_isinstance), ordering)
        return tuple(ordering)

    def update_graph(self):
        """Triggers a call to all parents that the graph below them has been
        updated. The base ``Node`` object does nothing with this information, but
        other node types may use this to update internal state."""
        for parent in self.parents:
            parent.update_graph()

    @property
    def active(self) -> bool:
        return self._active

    @active.setter
    def active(self, value: bool):
        # Avoid unnecessary updates
        if self._active == value:
            return

        # Set self active level
        self._active = value

        # Propagate active level to children
        for child in self._children.values():
            child.active = value

    def to(self, device=None, dtype=None):
        """
        Moves and/or casts the PyTorch values of the ``Node``.

        Parameters
        ----------
        device: (Optional[torch.device], optional)
            The device to move the values to. Defaults to None.
        dtype: (Optional[torch.dtype], optional)
            The desired data type. Defaults to None.
        """

        for child in self.children.values():
            child.to(device=device, dtype=dtype)

        return self

    def graphviz(self, top_down=True) -> "graphviz.Digraph":
        """Return a graphviz object representing the graph below the current
        node in the DAG.

        Parameters
        ----------
        top_down: (bool, optional)
            Whether to draw the graph top-down (current node at top) or
            bottom-up (current node at bottom). Defaults to True.
        """
        import graphviz  # noqa

        components = set()

        def add_node(node, dot):
            if node in components:
                return
            dot.attr("node", **node.graphviz_types[node._type])
            dot.node(node.uid, repr(node))
            components.add(node)

            for child in node.children.values():
                add_node(child, dot)
                if top_down:
                    dot.edge(node.uid, child.uid)
                else:
                    dot.edge(child.uid, node.uid)

        dot = graphviz.Digraph(strict=True)
        add_node(self, dot)
        return dot

    def graph_dict(self) -> dict[str, dict]:
        """Return a dictionary representation of the graph below the current
        node."""
        graph = {
            f"{self.name}|{self._type}": {},
        }
        for node in self.children.values():
            graph[f"{self.name}|{self._type}"].update(node.graph_dict())
        return graph

    def graph_print(self, dag: dict, depth: int = 0, indent: int = 4, result: str = "") -> str:
        """Print the graph dictionary in a human-readable format."""
        for key in dag:
            result = f"{result}{' ' * indent * depth}{key}\n"
            result = self.graph_print(dag[key], depth + 1, indent, result) + "\n"
        if result:  # remove trailing newline
            result = result[:-1]
        return result

    def _unique_name(self):
        return f"{self.name}_{id(self)}"

    def _anchor_name(self):
        return f"{self.name}_anchor_{id(self)}"

    def __str__(self) -> str:
        return self.graph_print(self.graph_dict())

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.name})"

    def __getitem__(self, key: str) -> "Node":
        return self.children[key]

    def __eq__(self, other: "Node") -> bool:
        return self is other

    def __hash__(self) -> int:
        return hash(self.uid)

    def __del__(self):
        Node._names.discard(self.name)

    def process_non_node(self, c, data, params, new_full_path):
        if isinstance(c, Node):
            return c._traverse(data, params, new_full_path)
        # Convert torch tensors and Variables to numpy array
        if isinstance(c, (torch.Tensor, torch.autograd.Variable)):
            return c.detach().cpu().numpy().tolist()
        # Convert torch.Size to list
        if isinstance(c, torch.Size):
            return list(c)
        # Convert numpy arrays to lists
        if isinstance(c, np.ndarray):
            return c.tolist()
        # Convert numpy scalars (e.g. np.int64, np.float32) to native Python scalars
        if isinstance(c, np.generic):
            return c.item()
        # TODO - fix this
        # Process tuple or list recursively
        if isinstance(c, (tuple, list)): 
            return [self.process_non_node(cc, data, params, new_full_path) for cc in c]
        # If none of the above, return as is
        return c

    def _traverse(self, data, params, full_path=""):

        meta = CommentedMap()
        meta.yaml_set_anchor(self._anchor_name())
        meta['kind'] = self.__class__.__name__
        meta['name'] = self.name
        meta['uid'] = self.uid

        tree = CommentedMap()

        child_list = []

        later_params = {}

        def process_child(c, key, c_type = None):
            if not isinstance(c, Node):
                return self.process_non_node(c, data, params, f'{full_path}.{key}')
            else:
                # can't check if instance of Param because can't import Param, hacky workaround
                # TODO - find a better way to check if it's a Param
                if c._type not in ['module', 'node']:
                    c._traverse(data, params, f'{full_path}.{key}')

                    # If the child is a Param, then the parent object probably expects the original tensor/float/etc saved, NOT the param object
                    # This checks what type the init_kwarg should be, and tries to match that value
                    # TODO - this is not compatible with the actual Param dict that is returned
                    if c_type != None:
                        if check_value_type(c, c_type):
                            return c
                        elif c.get('value') and check_value_type(c.get('value'), c_type):
                            return c.get('value')
                        elif c.get('dynamic_value') and check_value_type(c.get('dynamic_value'), c_type):
                            return c.get('dynamic_value')
                        elif check_value_type(None, c_type):
                            return None
                        else:
                            return cast_value(c, c_type)

                    return c

                # TODO - ensure it is properly loaded as a node collection
                elif isinstance(c, (list, tuple)):
                    new_list = []
                    for n in c:
                        if data.get(n._unique_name()) is None:
                            data[n._unique_name()] = n._traverse(data, params, f'{full_path}.{key}')

                        new_list.append(data[n._unique_name()])

                    return new_list

                else:
                    if data.get(c._unique_name()) is None:
                        data[c._unique_name()] = c._traverse(data, params, f'{full_path}.{key}')

                    return data[c._unique_name()]

        for key, v in inspect.signature(self.__init__).parameters.items():
            try:
                c = getattr(self, key)
                child_list.append(c)

                annot = v.annotation
                c_type = None
                if not annot is inspect.Parameter.empty:
                    if get_origin(annot) is Annotated:
                        # If it's Annotated, get the underlying type (first argument).
                        c_type = get_args(annot)[0]
                    else:
                        c_type = annot

                tree[key] = process_child(c, key, c_type)

            except AttributeError:
                pass

        for key, c in list(self.__dict__.items()) + list(self.children.items()):
            # all init params should be ignored since they will be properly set
            # TODO - do not use vars from module__dict__, node__dict__ or param__dict__, these are already handled
            if key not in ['dynamic_params', 'pointer_params', 'local_dynamic_params', 'dynamic_modules']:
                # if key starts with _, it's a private variable and should be ignored
                if key.startswith('_'):
                    continue
                #if c in child_list:
                #    continue
                if isinstance(c, Node):
                    child_list.append(c)
                    later_params[key] = c.uid
                    c._traverse(data, params, f'{full_path}.{key}')
                else:
                    later_params[key] = self.process_non_node(c, data, params, f'{full_path}.{key}')

        meta["init_kwargs"] = tree
        later_params['path'] = full_path

        params[self.uid] = later_params

        return meta

    def save(self, path: str = 'data.yml'):
        data = CommentedMap()
        params = CommentedMap()
        data[self._unique_name()] = self._traverse(data, params, self.name)
        data['params_dict'] = params

        yaml = YAML()
        yaml.representer.add_representer(type(None), none_representer)
        stream = io.StringIO()
        yaml.dump(data, stream)
        yaml_string = stream.getvalue()

        # replacing unique names (with ids) by readable names (without ids)
        pattern = r'\b(?!(?:\w+_anchor_))(\w+)_\d{10}'
        replacement = r'\1'
        yaml_string = re.sub(pattern, replacement, yaml_string)

        pattern = r"['\"]?anchor\((.*?)\)['\"]?"
        replacement = r"\1"
        yaml_string = re.sub(pattern, replacement, yaml_string)

        yaml_string = re.sub('_anchor', '', yaml_string)

        # Save the edited YAML string to a file
        with open(path, 'w') as f:
            f.write(yaml_string)

        return data

# Partially taken from caustics
def load(config, rbase):
    def get_attr(base, x):
        if isinstance(base, list):
            return next(item for item in rbase if item.__name__ == x)
        else:
            return getattr(rbase, x)

    if isinstance(config, str):
        with open(config, "r") as f:
            config_dict = yaml.safe_load(f)
    else:
        config_dict = yaml.safe_load(config)

    # These are all the params and nodes that have to be created or updated
    params_dict = config_dict.pop("params_dict", {})

    modules = {}
    for name, obj in config_dict.items():
        kwargs = obj.get("init_kwargs", {})
        for kwarg in kwargs:
            for subname, subobj in config_dict.items():
                if subname == name:  # only look at previous objects
                    break
                if subobj == kwargs[kwarg] and isinstance(kwargs[kwarg], dict):
                    # fill already constructed object
                    kwargs[kwarg] = modules[subname]

        base = rbase

        for part in obj["kind"].split("."):
            base = get_attr(base, part)
        if "name" in signature(base).parameters:  # type: ignore[arg-type]
            kwargs["name"] = name
        # Instantiate the caustics object
        modules[name] = base(**kwargs)  # type: ignore[operator]
        # can the ids be updated with these values?

    res = modules[tuple(modules.keys())[-1]]

    # Once the main object is created, we can create the rest of the objects
    param_base = get_attr(rbase, 'Param')

    actual_params = {}
    other_params = {}

    # TODO - handle this properly in _traverse function
    for key, value in params_dict.items():
        if value.get('param') != None:
            actual_params[key] = value
        else:
            other_params[key] = value

    # The logic: find the node if it exists
    # If the node exists, re-initialize it with the new values in 'param'
    # Otherwise create a new node with the values in 'param'
    for key, value in actual_params.items():
        node1 = Node.id_set.get(key)
        path = value['path']

        value['param']['valid'] = tuple(value['param']['valid'])

        if value['param'].get('shape') != None:
            value['param']['shape'] = torch.Size(value['param']['shape'])

        # it's a pointer but the pointer values haven't been set yet
        if value['param'].get('value') == None and value['param'].get('dynamic_value') == None and value['param'].get('shape') == None:
            if value.get('is_func'):
                func = eval(value['func'])
                value['param']['value'] = func

        # The node with this id doesn't exist
        # But it might still exist in the graph, so try to find it
        # Oherwise create it
        if node1 is None:
            parent = res
            path = value['path']
            node1 = param_base(**value['param'])

            try:
                for p in path.split('.')[1:-1]:
                    parent = getattr(parent, p)

                k = path.split('.')[-1]

                if hasattr(parent, k):
                    child_node = getattr(parent, k)
                    parent.unlink(child_node)
                    setattr(parent, k, node1)
                else:
                    setattr(parent, k, node1)
            except Exception as e:
                print(e)
                continue
        else:
            # TODO - test this
            # re-init
            node1.__init__(**value['param'])

        # TODO - function may be set twice
        if value.get('is_func'):
            func = eval(value['func'])
            node1._pointer_func = func

        # find the pointers and link them
        pointers = value.get('pointers')
        for pointer in pointers:
            param2 = Node.id_set.get(pointer)

            if param2 is not None:
                if node1.children.get(param2.uid) == None:
                    node1.link(param2)


    # These are Nodes and Modules
    for key, value in other_params.items():
        node1 = Node.id_set.get(key)
        path = value['path']

        if node1 is None:
            node1 = res
            for p in path.split('.')[1:]:
                if hasattr(node1, p):
                    node1 = getattr(node1, p)
                else:
                    node1 = None

        if node1 is None:
            print('Node not found:', value)
            continue

        # TODO - list case is not handled
        for k, v in value.items():
            # this is the node's uid, so it has to be ignored or else it will be attached itself
            if k == 'uid' or k == 'path':
                continue
            if is_uuid(v):
                new_node = Node.id_set.get(v)
                if new_node is None:
                    node_dict = config_dict.get(v)
                    if node_dict is None:
                        continue
                    else:
                        node_path = node_dict['path']
                        new_node = res
                        for p in node_path.split('.')[1:]:
                            if hasattr(new_node, p):
                                new_node = getattr(node1, p)
                            else:
                                continue

                if hasattr(node1, k):
                    old_node = getattr(node1, k)
                    if new_node != old_node:
                        if not isinstance(old_node, param_base):
                            setattr(node1, k, new_node)
                        else:
                            node1.unlink(old_node)
                            setattr(node1, k, new_node)
                else:
                    setattr(node1, k, new_node)
            else:
                setattr(node1, k, v)

    return res

def none_representer(dumper, value):
    # Represent None as a scalar with the YAML null tag and explicit "null" value.
    return dumper.represent_scalar('tag:yaml.org,2002:null', 'null')

def check_value_type(value, expected_type) -> bool:
    origin = get_origin(expected_type)
    # If the type is an Annotated type, extract the underlying type.
    if origin is Annotated:
        expected_type = get_args(expected_type)[0]
        # After unwrapping, update origin
        origin = get_origin(expected_type)

    # If it's a Union (which is how Optional is implemented), check if value matches any type.
    if origin is Union:
        return any(isinstance(value, t) for t in get_args(expected_type))

    # Otherwise, just use isinstance.
    return isinstance(value, expected_type)


def cast_value(value, annotated_type):
    # If the type is Annotated, extract the underlying type.
    if get_origin(annotated_type) is Annotated:
        underlying_type = get_args(annotated_type)[0]
    else:
        underlying_type = annotated_type
    try:
        return underlying_type(value)
    except Exception as e:
        raise ValueError(f"Failed to cast {value} to {underlying_type}") from e

def is_uuid(value):
    try:
        uuid.UUID(str(value))
        return True
    except ValueError:
        return False