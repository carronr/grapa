"""This module defines class ContainerMetadata to store metadata in Curves and Graph.

@author: Romain Carron
Copyright (c) 2025, Empa, Laboratory for Thin Films and Photovoltaics, Romain Carron
"""

from typing import Optional, Any, Union, TYPE_CHECKING
from copy import deepcopy

if TYPE_CHECKING:
    from grapa.graph import Graph
    from grapa.curve import Curve


class MetadataContainer:
    """
    Container for handling metadata as pairs key-values.
    It behaves similarly to a dict. Main differences:

    - key are str, lower case

    - it is callable (historical reason): obj(key) is equivalent to obj.get(key)

    Note: although convenient, users of grapa may find advantageous to stick to the
    methods defined in classes Curve and Graph (update({...}) and attr_del(key)),
    as these behave consistently for Curve and Graph.
    """

    VALUE_DEFAULT = ""

    def __init__(self, host_curve_or_graph: Union["Graph", "Curve"]):
        self._data = {}
        self.host = host_curve_or_graph

    @staticmethod
    def format(key: str) -> str:
        """Formats the key. key must be a str"""
        return key.lower().strip(" =:\t\r\n").replace("ï»¿", "")

    def __call__(self, key: str, default=VALUE_DEFAULT):
        """Historical reasons. Notation shortcut, to be able to call Curve.attr(key).
        Now this is Curve._attr(key)."""
        return self.get(key, default=default)

    def __str__(self):
        return self._data.__str__()

    def __contains__(self, key: str) -> bool:
        """Returns True / False. Useful to test: if key in container:"""
        return self.has_attr(key)

    def __iter__(self):
        """Return an iterator over the keys of the container."""
        return self._data.__iter__()

    def __len__(self):
        """number of items"""
        return self._data.__len__()

    def __getitem__(self, key: str) -> Any:
        """Short for get(key). Key lower case"""
        return self.get(key)

    def __setitem__(self, key: str, value):
        """Shortcut for update({key: value}). key lower case."""
        return self.update({key: value})

    def __delitem__(self, key: str):
        """Does not raise KeyError"""
        return self.pop(key)

    def clear(self):
        """Remove all items from the container"""
        if self.host.recorder.is_log_active():
            old = deepcopy(self._data)
        self._data.clear()
        if self.host.recorder.is_log_active():
            self.host.recorder.log(self, ("clear", [], {}), ("update", [old], {}))
            self.host.recorder.log(self, ("", [], {}), ("clear", [], {}))

    def items(self):
        """to iterate over items"""
        return self._data.items()

    def get(self, key: str, default=VALUE_DEFAULT):
        """Returns the value associated with key.
        If key not found, returns VALUE_DEFAULT (i.e. not KeyError)

        :param key: a str
        :param default: default return value.
        :return: The value associated with key."""
        k = self.format(key)
        if k in self._data:
            return self._data[k]
        return default

    def values(self, keys_list: Optional[list] = None):
        """Returns a copied dict of all attributes key-values.

        :param keys_list: is provided, the returned dict contains only these keys."""
        if isinstance(keys_list, list):
            out = {}
            for key in keys_list:
                out.update({self.format(key): self.get(key)})
            return out
        return dict(self._data)

    def has_attr(self, key: str) -> bool:
        """Returns if a key has been defined.

        :param key: the key of interest
        :return: True if key is defined and not default, False otherwise"""
        val = self.get(key)
        return not self.is_attr_value_default(val)

    @classmethod
    def is_attr_value_default(cls, value) -> bool:
        """Test if value corresponds to default value.

        :param value: a value
        :return: True if val is default value (e.g. not defined), False otherwise"""
        if isinstance(value, type(cls.VALUE_DEFAULT)) and value == cls.VALUE_DEFAULT:
            return True
        return False

    def update(self, attributes: dict) -> None:
        r"""
        Updates attributes. Some clean-up is performed e.g. key as lower case and strip
        of characters ` =:\\t\\r\\n`.

        :param attributes: a dict, key-value

        """
        for key, value in attributes.items():
            k = self.format(key)
            if not self.is_attr_value_default(value):
                if self.host.recorder.is_log_active():
                    old = deepcopy(self.get(k))
                self._data.update({k: value})
                if self.host.recorder.is_log_active():
                    self.host.recorder.log(
                        self.host,
                        ("update", [{k: value}], {}),
                        ("update", [{k: old}], {}),
                    )
            elif k in self._data:
                if self.host.recorder.is_log_active():
                    old = deepcopy(self.get(k))
                del self._data[k]
                if self.host.recorder.is_log_active():
                    self.host.recorder.log(
                        self, ("pop", [k], {}), ("update", [{k: old}], {})
                    )

    def pop(self, key: str):
        """Removes an existing key-value pair.
        Returns the earlier value, or default if not specified.
        Does not raise KeyError."""
        k = self.format(key)
        if self.host.recorder.is_log_active():
            old = deepcopy(self.get(k))
        out = self._data.pop(k, self.VALUE_DEFAULT)
        if self.host.recorder.is_log_active():
            self.host.recorder.log(self, ("pop", [key], {}), ("update", [{k: old}], {}))
        return out
