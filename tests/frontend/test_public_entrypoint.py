import os
import sys
import tkinter as tk

import pytest

import grapa
import grapa.GUI
from grapa.frontend import gui_main


def _requires_display():
    return sys.platform.startswith("linux") and not (
        os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")
    )


class _FakeRoot:
    def iconbitmap(self, *_args, **_kwargs):
        return None

    def update_idletasks(self):
        return None

    def destroy(self):
        return None


class _FakeConsole:
    def get_console(self):
        return None


class _FakeApplication:
    def __init__(self, master, open_file=None, config_file=None):
        self.master = master
        self.open_file = open_file
        self.config_file = config_file
        self.frames = {"console": _FakeConsole()}

    def print_last_release(self):
        return None

    def mainloop(self):
        self.master.update_idletasks()
        self.master.destroy()
        return None


def test_public_grapa_entrypoint_launches_gui(monkeypatch):
    """Smoke-test that the public `grapa.grapa()` entrypoint opens the GUI."""
    if _requires_display():
        pytest.skip("Tk GUI smoke test requires a display")

    monkeypatch.setattr(gui_main.tk, "Tk", _FakeRoot)
    monkeypatch.setattr(gui_main, "Application", _FakeApplication)
    monkeypatch.setattr(sys, "argv", [sys.argv[0]])

    grapa.grapa()


def test_public_gui_module_entrypoint_launches_gui(monkeypatch):
    """Smoke-test that the `grapa.GUI.build_ui` entrypoint opens the GUI."""
    if _requires_display():
        pytest.skip("Tk GUI smoke test requires a display")

    monkeypatch.setattr(gui_main.tk, "Tk", _FakeRoot)
    monkeypatch.setattr(gui_main, "Application", _FakeApplication)
    monkeypatch.setattr(sys, "argv", [sys.argv[0]])

    grapa.GUI.build_ui()
