import os
import sys
import tkinter as tk


class GrapaOpenbis:
    def __init__(self, app):
        self.app = app
        # self.path = r"C:\_python\_python_packages\openbis_uploader_207\Abt207"
        self.path = r"G:\Limit\Programms\Openbis_uploader207\production\Abt207"

        self.ok = True
        try:
            import pybis  # not in project requirements, intentionally
        except ImportError:
            print("GrapaOpenbis: cannot import pybis, will not show GUI elements.")
            self.ok = False
        if not os.path.exists(self.path):
            print(
                "GrapaOpenbis: cannot find indicated path, will not show GUI elements."
            )
            self.ok = False

    def create_widgets(self, frame):
        if self.ok:
            tk.Label(frame, text="Openbis files").pack(side="left")
            btn = tk.Button(frame, text="Open", command=self.open_files_openbis)
            btn.pack(side="left")
            btn = tk.Button(frame, text="Merge", command=self.merge_files_openbis)
            btn.pack(side="left")

    def open_files_openbis(self, merge=False):
        if self.path not in sys.path:
            sys.path.append(self.path)
        try:
            from downloadfiles_selection import open_gui_select_files
        except ImportError:
            msg = (
                "GrapaOpenbis open: cannot import "
                "downloadfiles_selection.open_gui_select_files, abort."
            )
            print(msg)
            return

        window = tk.Toplevel(self.app.master)
        files = open_gui_select_files(master=window, download=True, destroy=True)
        if len(files) > 0:
            if merge:
                self.app.mergeGraph(files)
            else:
                self.app.openFile(files)

    def merge_files_openbis(self):
        self.open_files_openbis(merge=True)
