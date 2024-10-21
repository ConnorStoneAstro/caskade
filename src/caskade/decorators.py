import inspect
import functools

from .context import ActiveContext


def forward(method):
    """
    Decorator to define a forward method for a module.

    Parameters
    ----------
    method: (Callable)
        The forward method to be decorated.

    Returns
    -------
    Callable
        The decorated forward method.
    """

    # Get kwargs from function signature
    method_kwargs = []
    for arg in inspect.signature(method).parameters.values():
        if arg.default is not arg.empty:
            method_kwargs.append(arg.name)
    method_kwargs = tuple(method_kwargs)

    @functools.wraps(method)
    def wrapped(self, *args, **kwargs):
        if self.active:
            kwargs.update(self.fill_kwargs(method_kwargs))
            return method(self, *args, **kwargs)

        # Extract params from the arguments
        if len(self.dynamic_params) == 0:
            params = {}
        elif "params" in kwargs:
            params = kwargs.pop("params")
        elif args:
            params = args[-1]
            args = args[:-1]
        else:
            raise ValueError(
                f"Params must be provided for dynamic modules. Expected {len(self.dynamic_params)} params."
            )

        with ActiveContext(self):
            self.fill_params(params)
            kwargs.update(self.fill_kwargs(method_kwargs))
            return method(self, *args, **kwargs)

    return wrapped


def apply_all_descendants(with_type=None):
    """
    Decorator to apply a method to all descendants of the current node.

    Parameters
    ----------
    method: (Callable)
        The method to be decorated.

    Returns
    -------
    Callable
        The decorated method.
    """

    def _apply_all_descendants(method):
        @functools.wraps(method)
        def wrapped(self, *args, **kwargs):
            method(self, *args, **kwargs)
            for node in self.descendants(with_type):
                if hasattr(node, method.__name__):
                    getattr(node, method.__name__)(*args, **kwargs)

        return wrapped

    return _apply_all_descendants


def apply_all_ancestors(with_type=None):
    """
    Decorator to apply a method to all ancestors of the current node.

    Parameters
    ----------
    method: (Callable)
        The method to be decorated.

    Returns
    -------
    Callable
        The decorated method.
    """

    def _apply_all_ancestors(method):
        @functools.wraps(method)
        def wrapped(self, *args, **kwargs):
            method(self, *args, **kwargs)
            for node in self.ancestors(with_type):
                if hasattr(node, method.__name__):
                    getattr(node, method.__name__)(*args, **kwargs)

        return wrapped

    return _apply_all_ancestors
