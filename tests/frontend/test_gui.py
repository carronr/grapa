import os
import pytest
import tkinter as tk

from grapa.frontend.gui_main import Application

from .. import grapa_folder, Graph, HiddenPrints, GrapaWarning


def test_startfilename(grapa_folder):
    os.chdir(grapa_folder)
    root = tk.Tk()
    root.withdraw()
    try:
        app = Application(master=root)
        file = app.get_file()
        expected = r"c:\_python\_python_packages\grapa\grapa\examples\subplots_examples.txt"
        assert os.path.normcase(os.path.normpath(file)) == os.path.normcase(
            os.path.normpath(expected)
        )
    finally:
        root.destroy()


