"""Tests for the MplFigureFactory dispatcher."""
import importlib
from types import ModuleType

import pytest

from matplotlib.figure import Figure

from grapa.shared.mpl_figure_factory import MplFigureFactory


class DummyPyplot(ModuleType):
    """Minimal pyplot stub for dispatcher tests. Avoids atual use of matplotlib/pyplot
    """

    def __init__(self):
        """Initialize the dummy pyplot module."""
        super().__init__("matplotlib.pyplot")
        self.called = []

    def figure(self, *args, **kwargs):
        """Record and return a sentinel figure value."""
        self.called.append(("figure", args, kwargs))
        return "dummy-fig"

    def close(self, fig=None):
        """Record close requests."""
        self.called.append(("close", (fig,), {}))

    def get_fignums(self):
        # self.called.append(("get_fignums", (), {}))
        return []


def test_figure_oo_mode_does_not_import_pyplot(monkeypatch):
    """Figure() should be used when use_pyplot is False."""

    def _fail_import(name):
        raise AssertionError("pyplot should not be imported in OO mode")

    monkeypatch.setattr(importlib, "import_module", _fail_import)
    disp = MplFigureFactory(use_pyplot=False)
    fig = disp.figure()
    assert isinstance(fig, Figure)


def test_figure_pyplot_mode_uses_pyplot(monkeypatch):
    """pyplot.figure should be used when use_pyplot is True."""
    dummy = DummyPyplot()

    def _fake_import(name):
        assert name == "matplotlib.pyplot"
        return dummy

    monkeypatch.setattr(importlib, "import_module", _fake_import)
    disp = MplFigureFactory(use_pyplot=True)
    fig = disp.figure(1, test=2)
    assert fig == "dummy-fig"
    assert dummy.called == [("figure", (1,), {"test": 2})]


def test_close_pyplot_mode_delegates(monkeypatch):
    """close() should delegate to pyplot when enabled."""
    dummy = DummyPyplot()

    def _fake_import(_name):
        return dummy

    monkeypatch.setattr(importlib, "import_module", _fake_import)
    disp = MplFigureFactory(use_pyplot=True)
    disp.close()
    assert dummy.called == [("close", (None,), {})]


def test_pyplot_import_error_propagates(monkeypatch):
    """ImportError from pyplot should propagate in pyplot mode."""

    def _fake_import(name):
        raise ImportError("no pyplot")

    monkeypatch.setattr(importlib, "import_module", _fake_import)
    disp = MplFigureFactory(use_pyplot=True)
    with pytest.raises(ImportError):
        disp.figure()
