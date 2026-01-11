import os
import pytest
import tkinter as tk

from grapa.GUI import Application

from .. import grapa_folder, Graph, HiddenPrints, GrapaWarning


def test_startfilename(grapa_folder):
    os.chdir(grapa_folder)
    root = tk.Tk()
    app = Application(master=root)
    file = app.get_file()
    assert (
        file
        == r"c:\_python\_python_packages\grapa\grapa\examples\subplots_examples.txt"
    )


