from .base import Node


class Module(Node):

    def __init__(self, name):
        super().__init__(name=name)

    @property
    def active(self):
        return self._active

    @active.setter
    def active(self, value):
        self._active = value
        for child in self._children.values():
            child.active = value
