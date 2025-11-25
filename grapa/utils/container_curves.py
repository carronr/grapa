"""Container for Curve objects in Graph, behaves similarly as a list.

@author: Romain Carron
Copyright (c) 2025, Empa, Laboratory for Thin Films and Photovoltaics, Romain Carron
"""

import logging
from typing import List, Union, TYPE_CHECKING

if TYPE_CHECKING:
    from grapa.graph import Graph  # for where type hint is needed
    from grapa.curve import Curve  # for where type hint is needed


logger = logging.getLogger(__name__)


class CurveContainer:
    """A container for Curve objects in Graph. It behaves similarly as a list.
    Separate class to allow added behaviors, e.g. ctrl-z/ctrl-y, undo/redo"""

    # in .append: outsource typecheck complexity into CurveContaininer?
    # to test: curve_replace, self.data[idx] = new_curve

    def __init__(self, host_graph) -> None:
        self._data: "List[Curve]" = []
        self.graph: Graph = host_graph

    def _typecheck_raise(self, curve) -> bool:
        """Check if the object is of type Curve. Raise TypeError if not."""
        from grapa.curve import Curve

        if not isinstance(curve, Curve):
            raise TypeError("Only Curve objects can be added to the container.")
        return True

    def __len__(self):
        return len(self._data)

    def __getitem__(self, c):
        return self._data.__getitem__(c)

    def __iter__(self):
        return self._data.__iter__()

    def __delitem__(self, c: int):
        old = self._data[c]
        del self._data[c]
        if old not in self.graph:  # could have been twice in graph
            old.graphs_membersof.unregister_graph(self.graph)
        self._log_if_active(("__delitem__", [c], {}), ("insert", [c, old], {}))

    def __setitem__(self, c: int, new):
        self._typecheck_raise(new)
        already_in = new in self.graph
        old = self._data[c]
        self._data.__setitem__(c, new)
        if old not in self.graph:  # could have been twice in graph, curve swap, set
            old.graphs_membersof.unregister_graph(self.graph)
        if not already_in:  # eg when swap, user set
            self._data[c].graphs_membersof.register_graph(self.graph)
        self._log_if_active(
            ("__setitem__", [c, new], {}), ("__setitem__", [c, old], {})
        )

    def clear(self):
        """Empty the container, remove all its content."""
        old = [curve for curve in self.graph]
        self._data.clear()
        for curve in old:
            if curve not in self.graph:  # could have been twice in graph
                curve.graphs_membersof.unregister_graph(self.graph)
        if self.graph.recorder.is_log_active():
            self.graph.recorder.log(self, ("clear", [], {}), ("", [], {}))
            for curve in reversed(old):
                self.graph.recorder.log(self, ("", [], {}), ("append", [curve], {}))
            self.graph.recorder.log(self, ("", [], {}), ("clear", [], {}))

    def insert(self, c: int, curve):
        """Insert at position c a Curve, a list of Curves, or the Curves in a Graph."""
        from grapa.graph import Graph

        if isinstance(curve, (list, tuple, Graph)):
            lst = [c for c in curve]
            for curv_ in reversed(lst):
                self.insert(c, curv_)
            return

        self._typecheck_raise(curve)
        self._data.insert(c, curve)
        curve.graphs_membersof.register_graph(self.graph)
        self._log_if_active(("insert", [c, curve], {}), ("__delitem__", [c], {}))

    def append(self, curve: Union["Curve", list, tuple, "Graph"]):
        """Append to the container a Curve, a list of Curves, or Curves in a Graph."""
        from grapa.graph import Graph

        if isinstance(curve, (list, tuple, Graph)):
            for curv_ in curve:
                self.append(curv_)
            return

        self._typecheck_raise(curve)
        self._data.append(curve)
        self._data[-1].graphs_membersof.register_graph(self.graph)
        self._log_if_active(
            ("append", [curve], {}), ("__delitem__", [len(self) - 1], {})
        )

    def pop(self, c: int):
        """Pop and return the curve at position c."""
        curve = self._data.pop(c)
        if curve not in self.graph:  # could have been twice in graph
            curve.graphs_membersof.unregister_graph(self.graph)
        self._log_if_active(("pop", [c], {}), ("insert", [c, curve], {}))
        return curve

    def reverse(self):
        """Reverse the order of curves in the container."""
        self._data.reverse()
        self._log_if_active(("reverse", [], {}), ("reverse", [], {}))

    def _log_if_active(self, do, undo):
        if self.graph.recorder.is_log_active():
            self.graph.recorder.log(self, do, undo)
