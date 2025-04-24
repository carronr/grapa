# -*- coding: utf-8 -*-
"""Graphical user interface GUI of grapa

@author: Romain Carron
Copyright (c) 2025, Empa, Laboratory for Thin Films and Photovoltaics, Romain Carron
"""


import os
import sys
import contextlib
from datetime import datetime
import logging

print("Loading tkinter...")
import tkinter as tk
from dateutil import parser

print("Loading numpy...")
import numpy as np

print("Loading matplotlib...")
import matplotlib.pyplot as plt

print("Loading grapa...")
path = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
if path not in sys.path:
    sys.path.append(path)

from grapa import __version__, logger_handler
from grapa.curve import Curve
from grapa.graph import Graph
from grapa.utils.graphIO import GraphIO
from grapa.gui.observable import Observable
from grapa.gui.GUIMainElements import (
    GUIFrameMenuMain,
    GUIFrameConsole,
    GUIFrameCentral,
    GUIFrameTemplateColorize,
    GUIFrameActionsGeneric,
    GUIFrameTree,
    GUIFramePropertyEditor,
    GUIFrameActionsCurves,
)

logger = logging.getLogger(__name__)


# WISHLIST:
# - config autotemplate for boxplot
# - B+W preview
# - open config file in OS default editor

# TODO
# - LONGTERM check import .graph instead of grapa. Look at standard python packages. Check from GUI, with spyder various versions, and when calling script from external python script (Note: gui Change curve type was failing in early test, when importing in different manner)
# - progressively remove all uses of pyplot, does not mix well with tkagg and stuff
# - Stackplot: transparency Alpha?
# - scatter, add text values for each point. [clear all annotations] [text formatting "{.2g}". Remove only] [kwargs: centered both]


# to be able to print in something else than default console
@contextlib.contextmanager
def _stdout_redirect(where):
    """A context manager to redirect the print() stements to a tkinter widget"""
    cp = sys.stdout
    sys.stdout = where  # standard output for print() statements
    if hasattr(logger_handler, "setStream"):  # added in python 3.7
        logger_handler.setStream(sys.stdout)  # need to update the logger as well
    try:
        yield where
    finally:
        sys.stdout = cp  # sys.stdout = sys.__stdout__#does not work apparently
        if hasattr(logger_handler, "setStream"):  # added in python 3.7
            logger_handler.setStream(sys.stdout)


class Application(tk.Frame):
    """Main Frame for the GUI"""

    DEFAULT_SCREENDPI = 72
    DEFAULT_CANVAS_BACKGROUNDCOLOR = "white"

    def __init__(self, master):
        self.master = master
        super().__init__(master)
        self.master.title("Grapa software v" + __version__)
        self.initiated = False
        self.newgraph_kwargs = {"silent": True}
        try:  # possibly retrieve arguments from command line
            self.newgraph_kwargs.update({"config": sys.argv[1]})
            # argv[1] is the config.txt file to be used in this session
        except Exception:
            pass
        # create observable before UI -> element can register at initialization
        self.observables = {"focusTree": Observable()}
        # handles some GUI changes when changing the selected curve
        # there is 1 other observable, buried in the tabs. See below.
        # start define GUI
        self.pack(fill=tk.BOTH, expand=True)
        self.fonts = {}
        self._init_fonts(self)
        self._create_widgets(self)
        self.get_canvas().draw()  # to resize canvas, for autoDPI to work
        self.master.update()
        # open default Graph
        filename = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "examples",
            "subplots_examples.txt",
        )
        self.open_file(filename)
        self.update_ui()
        # register updates of UI when user changes tabs
        self.get_tabs().register(self.callback_tab_changed)
        # .initiated is a flag to allow update of UI, set True after init of UI
        self.initiated = True
        # other keys are bound to master within menuLeft or elsewhere
        self.master.bind("<Control-r>", lambda e: self.update_ui())

    def _init_fonts(self, frame):
        """Initialize fonts that are used throughout the GUI"""
        a = tk.Label(frame, text="")
        self.fonts["bold"] = tk.font.Font(font=a["font"])
        self.fonts["bold"].configure(weight="bold")
        self.fonts["fg_default"] = a.cget("fg")

    def _create_widgets(self, frame):
        """Create widgets"""
        # right frame
        fr = tk.Frame(frame)
        fr.pack(side="right", fill=tk.Y, pady=2, expand=False)
        self._cw_frame_right(fr)
        # frame left/graph/console
        fr = tk.Frame(frame)
        fr.pack(side="left", anchor="n", fill=tk.BOTH, expand=True)
        self._cw_frame_main(fr)

    def _cw_frame_main(self, frame):
        """Create widgets main frame"""
        # create console, indicators
        self.frame_console = GUIFrameConsole(frame, self)
        self.frame_console.frame.pack(side="bottom", fill=tk.X)
        # create graph and left menu
        fr = tk.Frame(frame)
        fr.pack(side="top", fill=tk.BOTH, expand=True)
        self.menu_left = GUIFrameMenuMain(fr, self)
        self.menu_left.frame.pack(side="left", anchor="n", fill=tk.Y, pady=2)
        self.frame_central = GUIFrameCentral(fr, self, relief="raised", borderwidth=2)
        self.frame_central.frame.pack(
            side="left", anchor="n", fill=tk.BOTH, pady=2, expand=True
        )

    def _cw_frame_right(self, frame):
        """Create widgets from on the right"""
        pady = 5
        # template actions
        self.frame_tpl_col = GUIFrameTemplateColorize(frame, self)
        self.frame_tpl_col.frame.pack(side="top", fill=tk.X, anchor="w", pady=pady)
        # properties
        self.frame_tree = GUIFrameTree(frame, self)
        self.frame_tree.frame.pack(side="top", fill=tk.X, anchor="w", pady=pady)
        # NEW property
        self.frame_prop = GUIFramePropertyEditor(frame, self)
        self.frame_prop.frame.pack(side="top", fill=tk.X, anchor="w", pady=pady)
        # actions on Curves
        self.frame_act_gen = GUIFrameActionsGeneric(frame, self)
        self.frame_act_gen.frame.pack(side="top", fill=tk.X, anchor="w", pady=pady)
        # Actions on curves
        self.frame_act_crv = GUIFrameActionsCurves(frame, self)
        self.frame_act_crv.frame.pack(side="top", fill=tk.X, anchor="w", pady=pady)

    # update_ui function
    def update_ui(self):
        """Update/refresh the GUI"""
        # print('update_ui main')
        # import time
        sectionstoupdate = [
            [self.frame_tpl_col, "section Template & Colorize"],
            [self.frame_tree, "Treeview property box"],
            [self.frame_prop, "Property editor"],
            [self.frame_act_gen, "section Actions of Curves "],
            [self.frame_act_crv, "section Actions specific"],
            [self.frame_central, "central section (plot and editor)"],
            [self.frame_console, "section Console"],
        ]
        for section in sectionstoupdate:
            # t0 = time.perf_counter()
            try:
                section[0].update_ui()
            except Exception:
                msg = "Exception during update of {}."
                logger.critical(msg.format(section[1]), exc_info=True)
            # t1 = time.perf_counter()
            # print('update_ui elapsed time:', t1-t0, section[1])

    # getters and setters
    def get_frame_graph(self):
        """Return the Frame containg the graph"""
        return self.frame_central.get_frame_graph()

    def get_frame_options(self):
        """Return the Frame Options"""
        return self.frame_central.get_frame_options()

    def graph(self, newgraph=None, **kwargs):
        """
        Get current Graph
        newgraph: to change current Graph. Does NOT refresh interface
        """
        tabs = self.get_tabs()
        if newgraph is not None:
            if isinstance(newgraph, Graph):
                # by default, give same dpi for new plot as for previous
                if "dpi" not in kwargs:
                    kwargs["dpi"] = self.DEFAULT_SCREENDPI  # initial value
                tabs.append_new(newgraph, **kwargs)
                tabs.select(-1)
                self.set_auto_screendpi()
            else:
                logger.error("GUI.graph(), newgraph must be a Graph.")
        # return self.back_graph
        return tabs.get_graph()

    def get_ax(self):
        """Return active ax"""
        return self.get_frame_graph().ax

    def get_fig(self):
        """Return the Figure"""
        return self.get_frame_graph().fig

    def get_canvas(self):
        """Return the canvas"""
        return self.get_frame_graph().get_canvas()

    def get_tabs(self):
        """Returns the tabs widget"""
        return self.frame_central.get_tabs()

    def get_tab_properties(self, **kwargs):
        """
        Get properties of current active tab.

        :param kwargs: updates the properties of current active tab
        """
        if len(kwargs) > 0:
            self.get_tabs().update_current(**kwargs)
            # also updates folder
        return self.get_tabs().get_properties()

    def get_folder(self, newvalue=None):
        """
        Get current active folder

        :param newvalue: set newvalue for the current active folder
        """
        if newvalue is not None:
            self.get_tabs().update_current(folder=newvalue)
        return self.get_tabs().get_folder()

    def get_file(self, newvalue=None):
        """
        Get current active file. Also updates folder and title

        :param newvalue: set newvalue for the current active file
        """
        if newvalue is not None:
            self.get_tabs().update_current(filename=newvalue)
            # also updates folder
        return self.get_tabs().get_filename()

    def get_selected_curves(self, multiple=False):
        """Returns a list of unique, sorted indices [idx0, idx1, ...]"""
        curves, keys = self.frame_tree.get_tree_active_curve(multiple=multiple)
        return sorted(list(set(curves)))

    def get_clipboard(self):
        """returns content of clipboard"""
        return self.master.clipboard_get()

    def set_clipboard(self, data):
        """place new data in the clipboard"""
        self.master.clipboard_clear()
        self.master.clipboard_append(data)

    def set_auto_screendpi(self):
        """Sets the displayed DPI to automatic value"""
        self.get_frame_options().set_auto_screendpi()

    # callbacks
    def callback_tab_changed(self, *_args, **_kwargs):
        """React to change in tab selection"""
        # print('Tab changed triggers update_ui')
        if self.initiated:
            self.update_ui()

    def disable_canvas_callbacks(self):
        """Disable the callbacks on the canvas"""
        self.get_frame_graph().disable_canvas_callbacks()

    def enable_canvas_callbacks(self):
        """Enable the callbacks on the canvas"""
        self.get_frame_graph().enable_canvas_callbacks()

    # Other function to handle functionalities of applications
    def store_selected_curves(self):
        """
        Keeps in memory the selected curves. To be called in functions that
        will call update_ui(), before any action that may change the curves
        (esp. order, number, new Curves etc. Changes in attributes should be
        safe)
        """
        self.frame_tree.store_selected_curves()

    @staticmethod
    def args_to_str(*args, **kwargs):
        """Format args and kwargs for printing commands in the console"""

        def to_str(a):
            if isinstance(a, str):
                return "'" + a.replace("\n", "\\n") + "'"
            if isinstance(a, Graph):
                return "graph"
            if isinstance(a, Curve):
                return "curve"
            return str(a)

        p = [to_str(a) for a in args]
        p += [(key + "=" + to_str(kwargs[key])) for key in kwargs]
        return ", ".join(p)

    def call_graph_method(self, method, *args, **kwargs):
        """Execute method on currently active graph, with args and kwargs"""
        out = getattr(self.graph(), method)(*args, **kwargs)
        if self.if_print_commands():
            print("graph." + method + "(" + self.args_to_str(*args, **kwargs) + ")")
        return out

    def call_curve_method(self, curve, method, *args, **kwargs):
        """Execute method on active graph[curve], with args and kwargs"""
        out = getattr(self.graph()[curve], method)(*args, **kwargs)
        if self.if_print_commands():
            msg = "graph[{}].{}({})"
            print(msg.format(str(curve), method, self.args_to_str(*args, **kwargs)))
        return out

    def prompt_file(self, initialdir="", type_="open", multiple=False, **kwargs):
        """Prompts for file open/save"""
        if initialdir == "":
            initialdir = self.get_folder()
        if type_ == "save":
            return tk.filedialog.asksaveasfilename(initialdir=initialdir, **kwargs)
        if multiple:
            out = list(tk.filedialog.askopenfilenames(initialdir=initialdir, **kwargs))
            if len(out) == 0:
                return None
            if len(out) == 1:
                return out[0]
            return out
        # if not multiple
        return tk.filedialog.askopenfilename(initialdir=initialdir, **kwargs)

    def prompt_folder(self, initialdir=""):
        """Prompts for folder open"""
        if initialdir == "":
            initialdir = self.get_folder()
        return tk.filedialog.askdirectory(initialdir=initialdir)

    def if_print_commands(self):
        """Returns if the checkbox Print Commands is ticked"""
        return self.menu_left.if_print_commands()

    def open_file(self, file):
        """Open a file.

        :paramfile: a str, or a list of str to open multiple files, or a Graph
        """
        lbl = file
        if isinstance(file, Graph):
            # do not print anything
            graph = file
            lbl = ""
            if hasattr(graph, "fileexport"):
                lbl = graph.fileexport
            elif hasattr(graph, "filename"):
                lbl = graph.filename
        elif isinstance(file, list):
            print("Open multiple files (first:", lbl[0], ")")
            lbl = file[0]
            graph = Graph(file, **self.newgraph_kwargs)
        else:
            print("Open file:", lbl.replace("/", "\\"))
            graph = Graph(file, **self.newgraph_kwargs)
        if lbl == "":
            lbl = None
        self.graph(newgraph=graph, filename=lbl)
        if self.if_print_commands():
            print("graph = Graph('" + str(file) + "')")
        # updateUI triggerd by apparition (and selection) and new tab
        pass

    def merge_graph(self, graph):
        """graph can be a str (filename), or a Graph"""
        file = ""
        if not isinstance(graph, Graph):
            file = graph
            print("Merge with file:", file)
            graph = Graph(graph)
        else:
            pass  # stays silent
            # print('Merge with graph', file)
        self.graph().merge(graph)
        # change default save folder only if was empty
        if self.get_folder() == "" and file != "":
            self.get_folder(file)
        if self.if_print_commands():
            print("graph.merge(Graph('" + file + "'))")
        self.update_ui()

    def save_graph(self, filesave, fileext="", save_altered=False, if_compact=True):
        """
        Saves Graph into a file. Separated plot() and export() calls.

        :param filesave: str, filename to save the graph
        :param fileext: str, image data format (e.g. .png). If '', retrieve preference.
        :param save_altered: as per Graph.export(). Default False.
        :param if_compact: as per Graph.export(). Default True.
        """
        graph = self.graph()
        if graph.attr("meastype") == "":
            graph.update({"meastype": GraphIO.GRAPHTYPE_GRAPH})
        try:
            fig, ax = graph.plot(
                filesave=filesave,
                img_format=fileext,
                if_export=False,
                fig_ax=[self.get_fig(), None],
            )
            while isinstance(ax, (list, np.ndarray)) and len(ax) > 0:
                ax = ax[0]
        except Exception:
            logger.error("Exception during plotting of the Graph.", exc_info=True)
        if fileext in [".xml"]:
            filesave += fileext
        graph.export(
            filesave=filesave, save_altered=save_altered, if_compact=if_compact
        )
        self.get_file(filesave)  # updates file, folder and tab title
        if self.if_print_commands():
            msg = "graph.plot(filesave='{}', imgFormat='{}', ifExport=False))"
            print(msg.format(filesave, fileext))
            msg = "graph.export(filesave='{}', saveAltered='{}', ifCompact='{}')"
            print(msg.format(filesave, str(save_altered), str(if_compact)))
        self.update_ui()

    def insert_curve_to_graph(self, curve, update_ui=True):
        """Appends 1 Curve, or a list of Curves to the active graph"""
        if update_ui:
            self.store_selected_curves()
        if isinstance(curve, list):
            for c in curve:  # several Curves into variable curve
                self.insert_curve_to_graph(c, update_ui=False)
        elif isinstance(curve, Curve):
            kw = {}
            selected = self.get_selected_curves(multiple=False)
            if len(selected) > 0 and selected[0] >= 0:
                kw.update({"idx": selected[0] + 1})
            self.graph().append(curve, **kw)
        else:
            msg = "insert_curve_to_graph:could not handle class of curve {}."
            logger.error(msg.format(type(curve)))
        if update_ui:
            self.update_ui()

    def blink_widget(
        self, widget, niter, delay=500, property_="background", values="auto"
    ):
        """Make a widget blink a few times (alternating background color)"""
        if values == "auto":
            values = ["", "red"]
        if "" in values:  # if default value - retrieve current setting
            values = list(values)  # work on duplicate
            for i in range(len(values)):
                if values[i] == "":
                    values[i] = widget.cget(property_)

        widget.config(**{property_: values[niter % len(values)]})
        if niter > 0:
            kwargs = {"property_": property_, "values": values, "delay": delay}
            self.master.after(
                delay, lambda: self.blink_widget(widget, niter - 1, **kwargs)
            )

    @staticmethod
    def print_last_release():
        """Retrieve and prints the content of the last release, if recent enough"""

        def read_file_until_line_startwith(file_, string):
            before = []
            after = []
            while True:
                line = file_.readline().strip(" \r\n=*")
                if len(line) == 0:
                    continue
                if not line.startswith(string):
                    before.append(line)
                    continue
                if len(before) > 0:
                    after.append(before.pop(-1))  # actually, last line also in "after"
                after.append(line)
                return before, after
            return before, after

        filename = "version_notes.rst"
        fname = os.path.join(os.path.dirname(os.path.realpath(__file__)), filename)
        with open(fname, "r", encoding="utf-8") as file:
            _, title = read_file_until_line_startwith(file, "Release")
            content, _ = read_file_until_line_startwith(file, "Release")
        date = title[-1].split(" ")
        if len(date) < 2:
            return

        date = date[1]
        try:
            dayssincelast = (datetime.now() - parser.parse(date)).days
        except ValueError as e:
            dayssincelast = 0
            logger.error("print_last_release: Exception, date", date, type(e), e)

        if dayssincelast < 15:
            print("\n".join(title + content))
        return

    def quit(self):
        """Quits application"""
        plt.close(self.get_fig())
        self.master.quit()
        self.master.destroy()


def build_ui():
    """To start the main grapa app"""
    root = tk.Tk()
    folder = os.path.dirname(os.path.abspath(__file__))
    try:
        root.iconbitmap(os.path.join(folder, "gui", "datareading.ico"))
    except Exception:
        pass  # not a real problem if fails, and cannot afford to get stuck here
    app = Application(master=root)

    # starts running programm
    with _stdout_redirect(app.frame_console.get_console()):
        app.print_last_release()
        app.mainloop()


if __name__ == "__main__":
    build_ui()
