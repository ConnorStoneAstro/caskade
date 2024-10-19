from ._version import version as VERSION  # noqa

from .base import Node
from .context import ActiveContext
from .decorators import forward
from .module import Module
from .param import Param
from .tests import test


__version__ = VERSION
__author__ = "Connor and Alexandre"

__all__ = ("Node", "Module", "Param", "ActiveContext", "forward", "test")
