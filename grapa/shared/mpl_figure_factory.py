# -*- coding: utf-8 -*-
"""
Utilities to dispatch Matplotlib/pyplot usage.

Provides a minimal, backend-agnostic dispatcher for pyplot-managed vs OO figures.
"""

import importlib
from types import ModuleType
from typing import Any, Optional
from matplotlib.figure import Figure


class MplFigureFactory:
    """Dispatch selected calls to pyplot or OO Matplotlib based on a flag.
    Feel free to use figure.add_axes, add_subplot, ax.plot, fig.savefig etc.
    """

    def __init__(self, use_pyplot: bool = True) -> None:
        self.use_pyplot = use_pyplot
        self._pyplot: Optional[ModuleType] = None

    def _require_pyplot(self) -> ModuleType:
        if self._pyplot is not None:
            return self._pyplot

        self._pyplot = importlib.import_module("matplotlib.pyplot")  # may raise Excepti
        return self._pyplot

    def figure(self, *args: Any, **kwargs: Any) -> Figure:
        """Create a figure via pyplot (if enabled) or via Figure() otherwise."""
        if self.use_pyplot:
            plt = self._require_pyplot()
            print("MplFigureFactory PYPLOT figure, figures", plt.get_fignums())
            return plt.figure(*args, **kwargs)

        return Figure(*args, **kwargs)

    def close(self, fig: Optional[Figure] = None) -> None:
        """Close a pyplot-managed figure, or clear the figure if not pyplot"""
        if self.use_pyplot:
            plt = self._require_pyplot()
            print("MplFigureFactory PYPLOT close, figures", plt.get_fignums())
            plt.close(fig)
            return
        if fig is not None and hasattr(fig, "clf"):
            fig.clf()

    # def show(self, *args: Any, **kwargs: Any) -> None:
    #     if self.use_pyplot:
    #         plt = self._require_pyplot()
    #         plt.show(*args, **kwargs)

    # def pause(self, *args: Any, **kwargs: Any) -> None:
    #     if self.use_pyplot:
    #         plt = self._require_pyplot()
    #         plt.pause(*args, **kwargs)
