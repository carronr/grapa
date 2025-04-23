# -*- coding: utf-8 -*-
"""
Defines the different elements of the GUI.

@author: Romain Carron
Copyright (c) 2025, Empa, Laboratory for Thin Films and Photovoltaics, Romain Carron
"""
import os
import copy
import warnings
import logging
import tkinter as tk
from tkinter import ttk
from tkinter.colorchooser import askcolor
from _tkinter import TclError

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import hex2color, rgb2hex
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


try:
    from matplotlib.backends.backend_tkagg import (
        NavigationToolbar2TkAgg as NavigationToolbar2Tk,
    )
except ImportError:
    from matplotlib.backends.backend_tkagg import NavigationToolbar2Tk

from grapa import KEYWORDS_GRAPH, KEYWORDS_CURVE
from grapa.curve import Curve, get_point_closest_to_xy
from grapa.graph import Graph, ConditionalPropertyApplier
from grapa.mathModule import roundSignificant, is_number
from grapa.colorscale import Colorscale, PhotoImageColorscale, colorscales_from_config

from grapa.utils.funcgui import FuncGUI, AlterListItem
from grapa.utils.graphIO import GraphIO
from grapa.utils.plot_graph import image_to_clipboard
from grapa.utils.string_manipulations import strToVar, varToStr, listToString

from grapa.datatypes.curveCV import CurveCV
from grapa.datatypes.curveJscVoc import CurveJscVoc

from grapa.gui.widgets_tooltip import CreateToolTip
from grapa.gui.widgets_custom import (
    bind_tree,
    TextWriteable,
    FrameTitleContentHide,
    FrameTitleContentHideHorizontal,
    EntryVar,
    OptionMenuVar,
    CheckbuttonVar,
    ComboboxVar,
    LabelVar,
    ButtonVar,
)
from grapa.gui.widgets_graphmanager import GraphsTabManager
from grapa.gui.GUIdataEditor import GuiDataEditor
from grapa.gui.GUITexts import GuiManagerAnnotations
from grapa.gui.interface_openbis import GrapaOpenbis

from grapa.scripts.script_correlations import process_file as corr_process_file
from grapa.scripts.script_JVSummaryToBoxPlots import JVSummaryToBoxPlots
from grapa.scripts.script_processCVfT import script_process_cvft
from grapa.scripts.script_processCVCf import script_processCf, script_processCV
from grapa.scripts.script_processJV import processSampleCellsMap, processJVfolder
from grapa.scripts.script_processJscVoc import script_processJscVoc


logger = logging.getLogger(__name__)


class GUIFrameCanvasGraph:
    """
    Handles the canvas with graph, and the toolbar
    Contains a Frame, to be embedded in a wider application (.frame.pack())
    Some methods will call methods of application
    """

    def __init__(self, master, application, **kwargs):
        self.frame = tk.Frame(master, **kwargs)
        self.app = application
        # store a list of custom canvas event, to easily remove them
        self.canvas_events = []
        # mechanism to fire events only when mouse is pressed
        self._mouse_pressed = False
        self.callback_notifycanvas_registered = []
        self.tabs = None  # will be created later, here to reserve the name
        self.fig, self.ax = None, None
        self.canvas = None
        self._create_widgets(self.frame)
        self.app.master.bind("<Control-w>", lambda e: self.close_tab())

    def update_ui(self):
        """Update plot on canvas"""
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Attempt to set non-positive")
            self.fig.clear()
            self.canvas.draw()
        # background color
        self.set_canvas_background()
        # screen DPI: to update before the update_ui() call
        self.fig.set_dpi(self.app.get_tab_properties()["dpi"])
        fig, ax = Graph.plot(self.app.graph(), fig_ax=[self.fig, None])
        while isinstance(ax, (list, np.ndarray)) and len(ax) > 0:
            ax = ax[-1]
        self.fig, self.ax = fig, ax
        # draw canvas
        try:
            self.canvas.show()
        except AttributeError:
            # FigureCanvasTkAgg has no attribute show in later versions of matplotlib
            pass
        self.canvas.draw()

    def get_canvas(self):
        """Returns the canvas"""
        return self.canvas

    def get_tabs(self):
        """Returns the tabs element"""
        return self.tabs

    def registercallback_notifycanvas(self, func):
        """Register functions to callback when the canvas gets notified of an event"""
        self.callback_notifycanvas_registered.append(func)

    def _create_widgets(self, frame):
        """Create widgets"""
        self._cw_graph_selector(frame)
        self._cw_canvas(frame)

    def _cw_graph_selector(self, frame):
        """frame for selection of the graph"""
        fr = tk.Frame(frame)
        fr.pack(side="top", fill=tk.X)
        # button close
        fr1 = tk.Frame(fr, width=20, height=20)
        fr1.propagate(False)
        fr1.pack(side="right", anchor="n")
        btn = tk.Button(fr1, text="\u2715", command=self.close_tab)  # 2A2F
        btn.pack(side="left", anchor="n", fill=tk.BOTH, expand=True)
        CreateToolTip(btn, "Close selected tab. Ctrl+W")
        # tabs
        defdict = {
            "dpi": self.app.DEFAULT_SCREENDPI,
            "backgroundcolor": self.app.DEFAULT_CANVAS_BACKGROUNDCOLOR,
        }
        self.tabs = GraphsTabManager(fr, width=200, height=0, defdict=defdict)
        self.tabs.pack(side="left", fill=tk.X, expand=True)

    def _cw_canvas(self, frame):
        """Create widget canvas for graph"""
        dpi = self.app.DEFAULT_SCREENDPI
        defsize = Graph.FIGSIZE_DEFAULT
        figsize = (defsize[0], defsize[1] * 1.15)
        # canvassize = [1.03*defsize[0]*dpi, 1.03*defsize[1]*dpi]
        # plt.figure create and attr .number, to be recalled elsewhere. Maybe could
        # use mpl.figure.Figure() but unsure
        self.fig = plt.figure(figsize=figsize, dpi=dpi)
        self.canvas = FigureCanvasTkAgg(self.fig, master=frame)
        # possibly also configure width=canvassize[0], height=canvassize[1]
        self.canvas.get_tk_widget().pack(
            side="top", anchor="w", fill=tk.BOTH, expand=True
        )
        self.canvas.mpl_connect("resize_event", self.update_upon_resize_window)
        try:
            self.canvas.show()
        except AttributeError:
            pass  # changes in later versions of FigureCanvasTkAgg
        self.ax = self.fig.add_subplot(111)

    def set_canvas_background(self):
        """Change background color of the canvas"""
        # check value exists
        color = "white"
        try:
            color = self.app.get_tab_properties()["backgroundcolor"]
        except KeyError:
            self.app.get_tab_properties(backgroundcolor=color)
        # set value
        try:
            self.canvas.get_tk_widget().configure(background=color)
        except TclError:
            msg = (
                "set_canvas_background: illegal color name ({}). Please use either"
                "tkinter color names, or RGB as #90ee90."
            )
            logger.error(msg.format(color), exc_info=True)
            self.canvas.get_tk_widget().configure(background="white")

    def update_upon_resize_window(self, *_args):
        """Updates the frame including the canvas, but only when main app is ready"""
        if self.app.initiated:
            # print('updateUponResizeWindow args', _args)
            self.app.frame_central.update_ui()  # only central part of UI

    def enable_canvas_callbacks(self):
        """restore suitable list of canvas callbacks, for datapicker"""

        def callback_press_canvas(event):
            self._mouse_pressed = True
            callback_notify_canvas(event)

        def callback_release_canvas(_event):
            self._mouse_pressed = False

        def callback_notify_canvas(event):
            if not self._mouse_pressed:
                return
            for func in self.callback_notifycanvas_registered:
                func(event)
            # print('Clicked at', event.xdata, event.ydata, 'curve')

        self.disable_canvas_callbacks()
        self.canvas_events.append(
            self.canvas.mpl_connect("button_press_event", callback_press_canvas)
        )
        self.canvas_events.append(
            self.canvas.mpl_connect("button_release_event", callback_release_canvas)
        )
        self.canvas_events.append(
            self.canvas.mpl_connect("motion_notify_event", callback_notify_canvas)
        )

    def disable_canvas_callbacks(self):
        """disable all registered canvas callbacks"""
        for cid in self.canvas_events:
            self.canvas.mpl_disconnect(cid)
        self.canvas_events.clear()

    def close_tab(self):
        """Closes current tab"""
        self.tabs.pop()


class GUIFrameCentralOptions:
    """
    Handles the options within central panel with data tranform, axis limits
    Contains a Frame, to be embedded in a wider application (.frame.pack())
    Some methods will call methods of application
    """

    def __init__(self, master, application, canvas, **kwargs):
        self.frame = tk.Frame(master, **kwargs)
        self.app = application
        self.toolbar = None  # here to reserve the name, will be created later
        self.alterlistgui = []
        self._widgets = {}
        self._create_widgets(self.frame, canvas)

    def update_ui(self):
        """Updates the GUI"""
        graph = self.app.graph()
        # to be called to update the frame:
        self.alterlistgui = graph.alterListGUI()
        # Data transform, plottype
        [_idx, display, _alter, typeplot] = self.identify_data_transform()
        options = [i[0] for i in self.alterlistgui]
        self._widgets["alter"].reset_values(options, func=self.update_data_transform)
        for a in self.alterlistgui:
            if a[0] == display:
                self._widgets["alter"].set(display)
                break
        self._widgets["typeplot"].set(typeplot)
        # labels, limits
        xlim = graph.attr("xlim", ["", ""])
        ylim = graph.attr("ylim", ["", ""])
        xlim = [(x if not isinstance(x, str) and not np.isnan(x) else "") for x in xlim]
        ylim = [(y if not isinstance(y, str) and not np.isnan(y) else "") for y in ylim]
        self._widgets["xlabel"].set(varToStr(graph.attr("xlabel")))
        self._widgets["ylabel"].set(varToStr(graph.attr("ylabel")))
        self._widgets["xlim0"].set(varToStr(xlim[0]))
        self._widgets["xlim1"].set(varToStr(xlim[1]))
        self._widgets["ylim0"].set(varToStr(ylim[0]))
        self._widgets["ylim1"].set(varToStr(ylim[1]))
        # tabs properties DPI
        tabproperties = self.app.get_tab_properties()
        self._widgets["DPI"].set(tabproperties["dpi"])

    def _create_widgets(self, frame, canvas):
        """Creates the widgets"""
        # toolbar
        fr0 = tk.Frame(frame)
        fr0.pack(side="top", anchor="w", fill=tk.X)
        self._cw_toolbar(fr0, canvas)
        # line 1
        fr1 = tk.Frame(frame)
        fr1.pack(side="top", anchor="w", fill=tk.X)
        self._cw_line1(fr1)
        # line 2
        fr2 = tk.Frame(frame)
        fr2.pack(side="top", anchor="w", fill=tk.X)
        self._cw_line2(fr2)

    def _cw_toolbar(self, frame, canvas):
        """Creates the widgets of the toolbar"""
        btn0 = tk.Button(frame, text="Refresh GUI", command=self.app.update_ui)
        btn0.pack(side="left", anchor="center", padx=5, pady=2)
        CreateToolTip(btn0, "Ctrl+R")
        btn1 = tk.Button(
            frame, text="Save zoom/subplots", command=self._set_limits_subplots
        )
        btn1.pack(side="left", anchor="center", padx=5, pady=2)
        tk.Label(frame, text="   ").pack(side="left")
        self.toolbar = NavigationToolbar2Tk(canvas, frame)
        self.toolbar.update()

    def _cw_line1(self, frame):
        """Create widgets Data transfor, plot type, annotation popup"""
        lbl = tk.Label(frame, text="Data transform")
        lbl.pack(side="left", anchor="n", pady=7)
        # varAlter initially empty, gets updated frequently
        self._widgets["alter"] = OptionMenuVar(frame, ("",), default="")
        self._widgets["alter"].pack(side="left", anchor="center")
        # button change graph type
        lbl = tk.Label(frame, text="   Plot type")
        lbl.pack(side="left", anchor="n", pady=7)
        typeplot_list = AlterListItem.TYPEPLOTS
        self._widgets["typeplot"] = OptionMenuVar(
            frame, typeplot_list, default="", func=self.update_typeplot
        )
        self._widgets["typeplot"].pack(side="left")
        # popup to handle text annotations
        tk.Label(frame, text="     ").pack(side="left")
        btn = tk.Button(
            frame,
            text="Annotations, legend and titles",
            command=self.open_popup_annotations,
        )
        btn.pack(side="left", anchor="n", pady=5)
        # DPI, background color
        fr = tk.Frame(frame)
        fr.pack(side="right", anchor="center")
        self._cw_line1_right(fr)

    def _cw_line1_right(self, frame):
        """DPI, button Save"""
        # screen dpi
        lbl = tk.Label(frame, text="  Screen dpi")
        lbl.pack(side="left")
        self._widgets["DPI"] = EntryVar(
            frame, value=self.app.DEFAULT_SCREENDPI, width=5
        )
        self._widgets["DPI"].pack(side="left", padx=5)
        self._widgets["DPI"].bind(
            "<Return>", lambda event: self.set_screendpi_from_entry()
        )
        # button
        btn = tk.Button(
            frame, text="Save", command=self.set_screendpi_bgcolor_from_entry
        )
        btn.pack(side="left")

    def _cw_line2(self, frame):
        """Create widgets labels, axis limits, data editor"""
        self._widgets["xlabel"] = EntryVar(frame, "", width=20)
        self._widgets["ylabel"] = EntryVar(frame, "", width=20)
        self._widgets["xlim0"] = EntryVar(frame, "", width=8)
        self._widgets["xlim1"] = EntryVar(frame, "", width=8)
        self._widgets["ylim0"] = EntryVar(frame, "", width=8)
        self._widgets["ylim1"] = EntryVar(frame, "", width=8)
        tk.Label(frame, text="xlabel:").pack(side="left", anchor="center")
        self._widgets["xlabel"].pack(side="left", anchor="center")
        tk.Label(frame, text="   ylabel:").pack(side="left", anchor="center")
        self._widgets["ylabel"].pack(side="left", anchor="center")
        tk.Label(frame, text="   xlim:").pack(side="left", anchor="center")
        self._widgets["xlim0"].pack(side="left", anchor="center")
        tk.Label(frame, text="to").pack(side="left", anchor="center")
        self._widgets["xlim1"].pack(side="left", anchor="center")
        tk.Label(frame, text="   ylim:").pack(side="left", anchor="center")
        self._widgets["ylim0"].pack(side="left", anchor="center")
        tk.Label(frame, text="to").pack(side="left", anchor="center")
        self._widgets["ylim1"].pack(side="left", anchor="center")
        tk.Label(frame, text="   ").pack(side="left")
        btn = tk.Button(frame, text="Save", command=self.update_attributes)
        btn.pack(side="left", anchor="center")
        tk.Label(frame, text="   ").pack(side="left")
        self._widgets["xlabel"].bind("<Return>", lambda event: self.update_attributes())
        self._widgets["ylabel"].bind("<Return>", lambda event: self.update_attributes())
        self._widgets["xlim0"].bind("<Return>", lambda event: self.update_attributes())
        self._widgets["xlim1"].bind("<Return>", lambda event: self.update_attributes())
        self._widgets["ylim0"].bind("<Return>", lambda event: self.update_attributes())
        self._widgets["ylim1"].bind("<Return>", lambda event: self.update_attributes())
        bt = tk.Button(frame, text="Data editor", command=self.open_popup_data_editor)
        bt.pack(side="right", anchor="center")

    def _set_limits_subplots(self):
        """Called by Button Save zoom/subplots"""
        self.app.store_selected_curves()  # before modifs, to prepare update_ui
        ax = self.app.get_ax()
        xlim = list(ax.get_xlim())
        ylim = list(ax.get_ylim())
        a = ["left", "bottom", "right", "top", "wspace", "hspace"]
        subplots = [getattr(self.app.get_fig().subplotpars, key) for key in a]
        self.app.call_graph_method(
            "update", {"xlim": xlim, "ylim": ylim, "subplots_adjust": subplots}
        )
        self.app.update_ui()

    def update_typeplot(self, new):
        """handles change of value of data transform"""
        self.app.store_selected_curves()  # before modifs, to prepare update_ui
        self._widgets["typeplot"].set(new)
        self.app.call_graph_method("update", {"typeplot": new})
        self.app.update_ui()

    def set_screendpi_from_entry(self, update_ui=True):
        """Updates the DPI according to the value stored in the GUI Entry"""
        if update_ui:
            self.app.store_selected_curves()  # before modifs, to prepare update_ui
        try:
            self.app.get_tab_properties(dpi=float(self._widgets["DPI"].get()))
            self.check_valid_screendpi()
        except ValueError:  # float() conversion may have failed
            pass
        if update_ui:
            if self.app.initiated:
                self.app.update_ui()

    def check_valid_screendpi(self):
        """Checks that the DPI value stored in tab is reasonable"""
        new = strToVar(self.app.get_tab_properties()["dpi"])
        # print('checkValidScreenDPI', new)
        if new == "":
            new = self.app.DEFAULT_SCREENDPI
        if not is_number(new):
            return False
        fig = self.app.get_fig()
        figsize = np.array(fig.get_size_inches())
        min_max_px = np.array([[10, 10], [1000, 1000]])
        new_min_max = [max(min_max_px[0] / figsize), min(2 * min_max_px[1] / figsize)]
        new = min(max(new, new_min_max[0]), new_min_max[1])
        new = roundSignificant(new, 2)
        # print('Set screen DPI to '+str(new)+'.')
        # print('   ', new)
        self.app.get_tab_properties(dpi=new)
        return True

    def set_auto_screendpi(self):
        """Provides best guess for screen DPI"""
        self.app.master.update_idletasks()
        wh = [
            self.app.get_canvas().get_tk_widget().winfo_width(),
            self.app.get_canvas().get_tk_widget().winfo_height(),
        ]
        figsize = self.app.graph().attr("figsize", Graph.FIGSIZE_DEFAULT)
        dpimax = min([wh[i] / figsize[i] for i in range(2)])
        dpi = self.app.get_tab_properties()["dpi"]
        new = None
        if dpi > dpimax * 1.02:  # shall reduce screen dpi
            new = np.max([10, np.round(2 * (dpimax - 3), -1) / 2])
        elif dpi < dpimax * 0.8 and dpi < 100:  # maybe can zoom in
            new = np.min([180, np.round(2 * (dpimax - 3), -1) / 2])
        # print('   ', new)
        if new is not None and new != dpi:
            self.app.get_tab_properties(dpi=new)
            self.app.blink_widget(self._widgets["DPI"], 5)
            self._widgets["DPI"].set(new)  # a bit useless, done at next update_ui...

    def set_screendpi_bgcolor_from_entry(self):
        """Updates both screen DPI and background color"""
        self.app.store_selected_curves()  # before modifications, to prepare update_ui
        self.set_bgcolor_from_entry(update_ui=False)
        self.set_screendpi_from_entry(update_ui=False)
        if self.app.initiated:
            self.app.update_ui()

    def update_data_transform(self, new):
        """Update widget of Data Transform"""
        [_idx, display, _alter, _] = self.identify_data_transform()
        if display == new:
            return True  # no change
        for a in self.alterlistgui:
            if new == a[0]:
                self.app.store_selected_curves()  # before modifs, to prepare update_ui
                self._widgets["alter"].set(new)
                self._widgets["typeplot"].set(a[2])
                self.app.call_graph_method("update", {"alter": a[1], "typeplot": a[2]})
                self.app.update_ui()
        return False

    def identify_data_transform(self):
        """Retrieve the current value of the Data transform widget"""
        # retrieve current alter
        graph = self.app.graph()
        alter = graph.attr("alter")
        if alter == "":
            alter = self.alterlistgui[0][1]
        typeplot = graph.attr("typeplot")
        # find index in list of allowed alterations
        for i, item in enumerate(self.alterlistgui):
            if alter == item[1]:
                return [i] + item[0:2] + [typeplot]
        return [np.nan, "File-defined", alter, typeplot]

    def update_attributes(self):
        """update the quick modifs, located below the graph"""
        self.app.store_selected_curves()  # before modifs, to prepare update_ui
        xlim = [
            strToVar(self._widgets["xlim0"].get()),
            strToVar(self._widgets["xlim1"].get()),
        ]
        ylim = [
            strToVar(self._widgets["ylim0"].get()),
            strToVar(self._widgets["ylim1"].get()),
        ]
        xlabel = strToVar(self._widgets["xlabel"].get())
        ylabel = strToVar(self._widgets["ylabel"].get())
        self.app.call_graph_method(
            "update", {"xlabel": xlabel, "ylabel": ylabel, "xlim": xlim, "ylim": ylim}
        )
        self.app.update_ui()

    def open_popup_annotations(self):
        """Open window for annotation editor"""
        win = tk.Toplevel(self.app.master)
        GuiManagerAnnotations(win, self.app.graph(), self.catch_annotations)

    def catch_annotations(self, dictupdate):
        """process output of annotation popup editor"""
        self.app.store_selected_curves()  # before modifs, to prepare update_ui
        self.app.graph().update(dictupdate)
        self.app.update_ui()

    def open_popup_data_editor(self):
        """Open window for data edition"""
        # opens manager
        win = tk.Toplevel(self.app.master)
        GuiDataEditor(win, self.app.graph(), self.catch_data_editor)

    def catch_data_editor(self):
        """Modification of the Curve are performed within popup. Little to do"""
        self.app.update_ui()


class GUIFrameDataPicker:
    """
    Handles the data picker
    Contains a Frame, to be embedded in a wider application (.frame.pack())
    Some methods will call methods of application
    """

    def __init__(self, master, application, **kwargs):
        self.frame = tk.Frame(master, **kwargs)
        self.app = application
        self.var_idx = np.nan  # datapoint index on curve
        self.var_curve_previous_labels = []
        self.crosshairx, self.crosshairy = None, None
        self._widgets = {}
        self._create_widgets(self.frame)
        self.app.master.bind("<Control-m>", lambda e: self.save_point())

    def update_ui(self):
        """Updates the GUI elements"""
        # drop-down curve list
        graph = self.app.graph()
        order_lbl = ["label", "sample", "filename"]
        values, labels = [], []
        for i, curve in enumerate(graph):
            values.append(i)
            lbl = str(i) + " (no label)"
            for test in order_lbl:
                tmp = curve.attr(test)
                if tmp != "":
                    lbl = str(i) + " " + str(tmp).replace("'", "'")
                    break
            labels.append(lbl)
        if labels != self.var_curve_previous_labels:
            self.var_curve_previous_labels = labels
            default = self._widgets["curve"].get()
            self._widgets["curve"].reset_values(values, labels=labels, default=default)
        # crosshair
        self.update_crosshair(draw=False)

    def _create_widgets(self, frame):
        """Creates the GUI elements"""
        fr = FrameTitleContentHideHorizontal(
            frame,
            self._cw_title,
            self._cw_datapicker,
            default="hide",
            showHideTitle=True,
        )
        fr.pack(side="top", fill=tk.X, anchor="w")

    def _cw_title(self, frame):
        """Create widgets Title"""
        tk.Label(frame, text="Data picker").pack(side="top")

    def _cw_datapicker(self, frame):
        """Prepare creation datapicker"""
        fr0 = tk.Frame(frame)
        fr0.pack(side="top", anchor="w", fill=tk.X)
        self._cw_datapicker_up(fr0)
        fr1 = tk.Frame(frame)
        fr1.pack(side="top", anchor="w", fill=tk.X)
        self._cw_datapicker_down(fr1)

    def _cw_datapicker_up(self, frame):
        tk.Label(frame, text="Click on graph").pack(side="left", anchor="center")
        tk.Label(frame, text="x").pack(side="left", anchor="w")
        self._widgets["x"] = EntryVar(frame, 0, width=10, varType=tk.DoubleVar)
        self._widgets["x"].pack(side="left", anchor="center")
        tk.Label(frame, text="y").pack(side="left", anchor="center")
        self._widgets["y"] = EntryVar(frame, 0, width=10, varType=tk.DoubleVar)
        self._widgets["y"].pack(side="left", anchor="center")
        self._widgets["restrict"] = CheckbuttonVar(frame, "Restrict to data", False)
        self._widgets["restrict"].pack(side="left", anchor="center")
        tk.Label(frame, text="curve").pack(side="left", anchor="center")
        self._widgets["curve"] = OptionMenuVar(frame, ("0",), "0", varType=tk.IntVar)
        self._widgets["curve"].pack(side="left", anchor="center")
        try:
            self._widgets["curve"].var.trace_add("write", self.select_curve)
        except AttributeError:  # IntVar has no attribute 'trace_add'
            self._widgets["curve"].var.trace("w", self.select_curve)
        self._widgets["crosshair"] = CheckbuttonVar(
            frame, "Crosshair", False, command=self.update_crosshair
        )
        self._widgets["crosshair"].pack(side="left", anchor="center")

    def _cw_datapicker_down(self, frame):
        btn0 = tk.Button(
            frame, text="Create text with coordinates", command=self.create_textbox
        )
        btn0.pack(side="left", anchor="center")
        tk.Label(frame, text=" or ").pack(side="left")
        btn1 = tk.Button(frame, text="Save point", command=self.save_point)
        btn1.pack(side="left", anchor="center")
        CreateToolTip(btn1, "Ctrl+M")
        self._widgets["ifTransform"] = CheckbuttonVar(frame, "screen data", True)
        self._widgets["ifTransform"].pack(side="left", anchor="center")
        self._widgets["ifCurveSpec"] = CheckbuttonVar(frame, "Curve specific", False)
        self._widgets["ifCurveSpec"].pack(side="left", anchor="center")
        # explanatory text for checkbox
        self._widgets["explain"] = LabelVar(frame, "")
        self._widgets["explain"].pack(side="left", anchor="center")

    def select_curve(self, *_args):
        """Called when user selects another Curve in data picker"""
        c = self._widgets["curve"].get()
        lbl = self.app.graph()[c].getDataCustomPickerXY(0, strDescription=True)
        self._widgets["explain"].set(lbl)

    def get_x_y_attrupd(self):
        """retrieve (and transform) data in x and y Entry."""
        # default is data in datapicker textbox
        x = self._widgets["x"].get()
        y = self._widgets["y"].get()
        attrupd = {}
        if self._widgets["restrict"].get():
            # if datapicker was restricted to existing data point
            graph = self.app.graph()
            c = self._widgets["curve"].get()
            print("get_x_y_attrupd c", type(c), c)
            if c >= 0:
                idx = self.var_idx
                alter = graph.get_alter()
                # if user want data transform instead if raw data
                if self._widgets["ifTransform"].get():
                    # raw data is displayed in checkbox, need transform
                    x = graph[c].x_offsets(index=idx, alter=alter[0])[0]
                    y = graph[c].y_offsets(index=idx, alter=alter[1])[0]
                # if user want curve-specific data picker
                if self._widgets["ifCurveSpec"].get():
                    # default will be transformed & offset modified data
                    # maybe the Curve object overrides the default method ?
                    # case for CurveCf at least
                    x, y, attrupd = graph[c].getDataCustomPickerXY(idx, alter=alter)
        return x, y, attrupd

    def create_textbox(self):
        """Create a new text annotation on Graph"""
        self.app.store_selected_curves()  # before modifs, to prepare update_ui
        x, y, _ = self.get_x_y_attrupd()
        text = (
            "x: " + str(roundSignificant(x, 5)) + "\ny: " + str(roundSignificant(y, 5))
        )
        textxy = ""
        textargs = {"textcoords": "data", "xytext": [x, y], "fontsize": 8}
        self.app.call_graph_method("text_add", text, textxy, textargs=textargs)
        if not self.app.if_print_commands():
            print("New text annotation:", text.replace("\n", "\\n"))
        self.app.update_ui()

    def save_point(self):
        """Save point"""
        self.app.store_selected_curves()  # before modifs, to prepare update_ui
        attr = {"linespec": "x", "color": "k", "_dataPicker": True}
        x, y, attrupd = self.get_x_y_attrupd()
        attr.update(attrupd)
        # implement in curve results
        graph = self.app.graph()
        c = graph.curves("_dataPicker", True)
        if len(c) == 0:
            # must create datapicker Curve
            curve = Curve(np.array([[x], [y]]), attr)
            if curve.attr("Curve", None) is not None:
                casted = curve.castCurve(curve.attr("Curve"))
                if casted is not False:
                    curve = casted
            graph.append(curve)
        else:
            # datapicker Curve exists already
            c[0].appendPoints([x], [y])
        self.app.update_ui()

    def update_crosshair(self, draw=True):
        """
        draw the canvas including crosshair, except if called from update_ui
        which handles draw() by itself
        """
        # first, delete existing crosshair
        if self.crosshairx is not None:
            self.crosshairx.remove()
            self.crosshairx = None
        if self.crosshairy is not None:
            self.crosshairy.remove()
            self.crosshairy = None
        # then makes new ones
        if self._widgets["crosshair"].get():
            self.app.enable_canvas_callbacks()
            xdata = self._widgets["x"].get()
            ydata = self._widgets["y"].get()
            curve = self._widgets["curve"].get()
            restrict = self._widgets["restrict"].get()
            idx = self.var_idx
            graph = self.app.graph()
            alter = graph.get_alter()
            if curve >= 0 and curve >= len(graph):
                curve = len(graph) - 1
                self._widgets["curve"].set(curve)
            posx, posy = xdata, ydata
            if restrict and curve >= 0 and not np.isnan(idx):
                posx = graph[curve].x_offsets(index=idx, alter=alter[0])
                posy = graph[curve].y_offsets(index=idx, alter=alter[1])
                # print('crosshair', xdata, ydata, posx, posy)
            ax = self.app.get_ax()
            self.crosshairx = ax.axvline(posx, 0, 1, color=[0.5, 0.5, 0.5])
            self.crosshairy = ax.axhline(posy, 0, 1, color=[0.5, 0.5, 0.5])
        else:
            self.app.disable_canvas_callbacks()
        if draw:
            self.app.get_canvas().draw()

    def event_mouse_motion(self, event):
        """identifies datapoint closest to mouse click"""
        xdata, ydata = event.xdata, event.ydata
        self.var_idx = np.nan
        if (
            event.xdata is not None
            and self._widgets["restrict"].get()
            and self._widgets["curve"].get() > -1
        ):
            graph = self.app.graph()
            curve = graph[self._widgets["curve"].get()]
            if curve is not None:
                xdata, ydata, idx = get_point_closest_to_xy(
                    curve, xdata, ydata, alter=graph.attr("alter")
                )
                self.var_idx = idx
        if xdata is not None:
            self._widgets["x"].set(xdata)
        if ydata is not None:
            self._widgets["y"].set(ydata)
        self.update_crosshair()


class GUIFrameCentral:
    """
    Handles the central panel with graph, axis limits, and datapicker
    Contains a Frame, to be embedded in a wider application (.frame.pack())
    Some methods will call methods of application
    """

    def __init__(self, master, application, **kwargs):
        self.frame = tk.Frame(master, **kwargs)
        self.app = application
        self._widgets = {}
        self._create_widgets(self.frame)
        self._widgets["frameGraph"].registercallback_notifycanvas(
            self._widgets["frameDataPicker"].event_mouse_motion
        )

    def update_ui(self):
        """Update GUI"""
        self._widgets["frameGraph"].update_ui()
        self._widgets["frameOptions"].update_ui()
        self._widgets["frameDataPicker"].update_ui()

    def get_frame_graph(self):
        """Returns: Frame frameGraph"""
        return self._widgets["frameGraph"]

    def get_frame_options(self):
        """Returns the Frame Options"""
        return self._widgets["frameOptions"]

    def get_tabs(self):
        """Returns the tabs"""
        return self.get_frame_graph().get_tabs()

    def _create_widgets(self, frame):
        """Creates widgets"""
        fr = tk.Frame(frame)
        fr.pack(side="bottom", fill=tk.X)
        # Canvas dor Graph
        self._widgets["frameGraph"] = GUIFrameCanvasGraph(frame, self.app)
        self._widgets["frameGraph"].frame.pack(side="top", fill=tk.BOTH, expand=True)
        # note: cannot use self.app.getCanvas(), instantiation of self not finished
        canvas = self._widgets["frameGraph"].get_canvas()
        # Datatransform, annotations, labels, limits
        self._widgets["frameOptions"] = GUIFrameCentralOptions(fr, self.app, canvas)
        self._widgets["frameOptions"].frame.pack(side="top", fill=tk.X)
        # Datapicker
        self._widgets["frameDataPicker"] = GUIFrameDataPicker(fr, self.app)
        self._widgets["frameDataPicker"].frame.pack(side="top", fill=tk.X)


class GUIFrameConsole:
    """
    Handles the display of current file/folder, and the output console
    Contains a Frame, to be embedded in a wider application (.frame.pack())
    Some methods will call methods of application
    """

    def __init__(self, master, application, **kwargs):
        self._consoleheight_roll = [8, 20, 0]
        self._consoleheight = self._consoleheight_roll[0]

        self.app = application
        self.frame = tk.Frame(master, **kwargs)
        self._widgets = {}
        self._create_widgets(self.frame)

    def update_ui(self):
        """Update widgets"""
        self._widgets["file"].set(self._shorten(self.app.get_file()))
        self._widgets["folder"].set(self._shorten(self.app.get_folder()))

    def get_console(self):
        """Returns the Console widget"""
        return self._widgets["console"]

    def _create_widgets(self, frame):
        """Creates the widgets"""
        # fr = FrameTitleContentHide(frame, self.cw_linefile, self.cw_frconsole,
        #                            default='show', horizLineFrame=True,
        #                            contentpackkwargs={'expand': True})
        # fr.pack(side='top', fill=tk.X, anchor='w')
        title = tk.Frame(frame)
        self._cw_linefile(title)
        title.pack(side="top", fill=tk.X, anchor="w")
        self._widgets["framecontent"] = tk.Frame(frame)
        self._cw_frconsole(self._widgets["framecontent"])
        self._widgets["framecontent"].pack(side="top", fill=tk.X, anchor="w")

    def _cw_linefile(self, frame):
        """upper line of widgets"""
        # current file
        self._widgets["file"] = LabelVar(frame, value="")
        self._widgets["file"].pack(side="left", anchor="center")
        # current folder
        self._widgets["folder"] = LabelVar(frame, value="")
        # self._widgets["folder"].pack(side='top', anchor='w')  # DO NOT DISPLAY
        # button, horizontal line
        fr = tk.Frame(frame, width=20, height=20)
        fr.propagate(False)
        btn = tk.Button(fr, text="\u21F3")
        btn.pack(side="left", anchor="n", fill=tk.BOTH, expand=1)
        fr.pack(side="left", anchor="center")
        line = FrameTitleContentHide.frameHline(frame)
        line.pack(side="left", anchor="center", fill=tk.X, expand=1, padx=5)
        bind_tree(frame, "<Button-1>", self._change_text_height)

    def _change_text_height(self, *_args):
        """Toggle height of the console between different values"""
        old = int(self._consoleheight)
        try:
            idx = self._consoleheight_roll.index(self._consoleheight) + 1
            if idx == len(self._consoleheight_roll):
                idx = 0
        except ValueError:
            idx = 0
        self._consoleheight = self._consoleheight_roll[idx]
        self._widgets["console"].configure(height=self._consoleheight)
        if self._consoleheight == 0:
            self._widgets["framecontent"].pack_forget()
        elif old == 0:
            # if was hidden, need to .pack() it again
            self._widgets["framecontent"].pack(side="top", fill=tk.X, anchor="w")
            # expand=True, fill=tk.X)

    def _cw_frconsole(self, frame):
        """Create widgets console"""
        frame.columnconfigure(0, weight=1)
        # console
        console = TextWriteable(frame, wrap="word", height=self._consoleheight)
        scroll_y = tk.Scrollbar(frame, orient="vertical", command=console.yview)
        scroll_y.grid(row=0, column=1, sticky=tk.E + tk.N + tk.S)
        console.grid(row=0, column=0, sticky=tk.W + tk.E)
        console.configure(yscrollcommand=scroll_y.set)
        self._widgets["console"] = console

    @staticmethod
    def _shorten(string):
        """Cuts a long string into a shorter one"""
        if len(string) > 150:
            string = string[:60] + " ... " + string[-60:]
        return string


class GUIFrameMenuMain:
    """
    Handles the main menu (on the left of the application)
    Contains a Frame, to be embedded in a wider application (.frame.pack())
    Some methods will call methods of application
    """

    def __init__(self, master, application, **kwargs):
        self.frame = tk.Frame(master, **kwargs)
        self.app = application
        self.showhide = None  # defined later, here reserve the variable
        self._widgets = {}
        self._create_widgets(self.frame)
        self._bind_keystrokes()

    def _bind_keystrokes(self):
        """Bind keystrokes to the behavior of the main application"""
        frame = self.app.master
        frame.bind("<Control-s>", lambda e: self.save_graph())
        frame.bind("<Control-Shift-S>", lambda e: self.save_graph_as())
        frame.bind("<Control-o>", lambda e: self.open_file())
        frame.bind("<Control-Shift-O>", lambda e: self.merge_file())
        # too larges chances to mess up with that one
        # self.master.bind('<Control-v>', lambda e: self.openClipboard())
        frame.bind("<Control-Shift-V>", lambda e: self.merge_clipboard())
        frame.bind("<Control-Shift-N>", lambda e: self.insert_curve_empty())

    def if_print_commands(self):
        """Returns status of checkbox printCommands"""
        return self._widgets["printCommands"].get()

    def _create_widgets(self, frame):
        """Creates the widgets"""
        self.showhide = FrameTitleContentHide(
            frame,
            None,
            None,
            default="show",
            layout="vertical",
            showHideTitle=True,
            createButtons=False,
            horizLineFrame=False,
        )
        self.showhide.pack(side="top", fill=tk.BOTH, expand=True)
        self._cw_all_hide(self.showhide.get_frame_title())
        self._cw_all_show(self.showhide.get_frame_content())
        self.showhide.get_frame_title().pack_forget()

    def _cw_all_hide(self, frame):
        fr, _ = self.showhide.createButton(frame, symbol="\u25B6", size="auto")
        fr.pack(side="top", anchor="w")
        canvas = tk.Canvas(frame, width=20, height=40)
        canvas.pack(side="top", anchor="w")
        canvas.create_text(
            6, 40, text="Menu", angle=90, anchor="w", font=self.app.fonts["bold"]
        )

    def _cw_all_show(self, frame):
        """Top inner frame"""
        self._cw_open_merge(frame)
        self._cw_save(frame)
        fr = tk.Frame(frame)
        fr.pack(side="bottom", fill="both", expand=True)
        self._cw_scripts(fr)
        self._cw_bottom(fr)

    def _cw_section_title(self, frame, title):
        """Widgets of section title"""
        fr = tk.Frame(frame)
        fr.pack(side="top", fill=tk.X)
        lbl = tk.Label(fr, text=title, font=self.app.fonts["bold"])
        lbl.pack(side="left", anchor="center")
        hline = FrameTitleContentHide.frameHline(fr)
        hline.pack(side="left", anchor="center", fill=tk.X, expand=1, padx=5)
        return fr

    def _cw_open_merge(self, frame):
        """Section open & merge"""
        fr = self._cw_section_title(frame, "Open or merge files")
        fr_, _ = self.showhide.createButton(fr, symbol="\u25C1", size="auto")
        fr_.pack(side="right", anchor="center")
        # buttons
        fr = tk.Frame(frame)
        fr.pack(side="top", fill=tk.X, padx=10)
        tk.Label(fr, text="Open").grid(sticky="w", row=0, column=0)
        of = tk.Button(fr, text="File  ", command=self.open_file)
        oc = tk.Button(fr, text="Clipboard", command=self.open_clipboard)
        of.grid(sticky="w", row=0, column=1, padx=5)
        oc.grid(sticky="w", row=0, column=2)
        tk.Label(fr, text="Merge with").grid(sticky="w", column=0, row=1)
        mf = tk.Button(fr, text="File  ", command=self.merge_file)
        mc = tk.Button(fr, text="Clipboard", command=self.merge_clipboard)
        mf.grid(sticky="w", column=1, row=1, padx=5)
        mc.grid(sticky="w", column=2, row=1)
        CreateToolTip(of, "Ctrl+O")
        CreateToolTip(mf, "Ctrl+Shift+O")
        CreateToolTip(mc, "Ctrl+Shift+V")
        # Misc - open folders, etc
        fr = tk.Frame(frame)
        fr.pack(side="top", fill=tk.X, padx=10)
        u = tk.Button(fr, text="Open all in folder", command=self.open_folder)
        # u.grid(column=1, row=1, sticky='w')
        u.pack(side="left")
        self._widgets["openSubfolders"] = CheckbuttonVar(fr, "subfolders", 0)
        # self._widgets["openSubfolders"].grid(column=2, row=1)
        self._widgets["openSubfolders"].pack(side="left")
        # new curve, close
        fr = tk.Frame(frame)
        fr.pack(side="top", fill=tk.X, padx=10)
        v = tk.Button(fr, text="Insert", command=self.insert_curve_empty)
        # v.grid(column=1, row=2, sticky='w')
        v.pack(side="left")
        tk.Label(fr, text="new empty Curve").pack(side="left")
        explain = "For creating a new subplot, inset, image etc. Ctrl+Shift+N"
        CreateToolTip(u, "Open all files in a given folder")
        CreateToolTip(v, explain)

        # ### Grapa Openbis integration
        fr = tk.Frame(frame)
        fr.pack(side="top", fill=tk.X, padx=10)
        grapaopenbis = GrapaOpenbis(self.app)
        grapaopenbis.create_widgets(fr)
        # ### End of Grapa Openbis integration

    def _cw_save(self, frame):
        """Widgets Save data and graphs"""
        fr = tk.Label(frame, text="")
        fr.pack(side="top")
        # Section Save title
        self._cw_section_title(frame, "Save data & graph")
        # Buttons
        fr = tk.Frame(frame)
        fr.pack(side="top", fill=tk.X, padx=10)
        s_ = tk.Button(fr, text="Save", command=self.save_graph)
        s_.pack(side="left", anchor="n")
        sa = tk.Button(fr, text="Save as...", command=self.save_graph_as)
        sa.pack(side="left", anchor="n", padx=5)
        sc = tk.Button(fr, text="Copy image", command=self.save_image_to_clipboard)
        sc.pack(side="left", anchor="n")
        CreateToolTip(s_, "Ctrl+S")
        CreateToolTip(sa, "Ctrl+Shift+S")
        CreateToolTip(sc, "Copy image to clipboard (Windows only)")
        # options
        fr = tk.Frame(frame)
        fr.pack(side="top", fill=tk.X, padx=10)
        self._widgets["saveScreen"] = CheckbuttonVar(fr, "Screen data (better not)", 0)
        self._widgets["saveScreen"].pack(side="top", anchor="w", padx=5)
        self._widgets["saveSepara"] = CheckbuttonVar(fr, "Keep separated x columns", 0)
        self._widgets["saveSepara"].pack(side="top", anchor="w", padx=5)

    def _cw_scripts(self, frame):
        """Section Scripts"""
        fr = tk.Label(frame, text="")
        fr.pack(side="top")
        self._cw_section_title(frame, "Data processing scripts")
        self._cw_scripts_jvfit(frame)
        self._cw_scripts_jvsummary(frame)
        self._cw_scripts_cv_cf_jscvoc(frame)

    def _cw_scripts_jvfit(self, frame):
        """Section Scripts JV"""
        lbl = tk.Label(frame, text="JV curves (files in 1 folder):")
        lbl.pack(side="top", anchor="w", padx=0, pady=2)
        # diode weights
        fr = tk.Frame(frame)
        fr.pack(side="top", fill=tk.X, padx=10)
        lbl = tk.Label(fr, text="Fit weight diode region")
        lbl.pack(side="left", anchor="n")
        self._widgets["JVDiodeweight"] = EntryVar(fr, "5", width=6)
        self._widgets["JVDiodeweight"].pack(side="left", anchor="n")
        CreateToolTip(lbl, "1 neutral, 10 increased weight")
        CreateToolTip(self._widgets["JVDiodeweight"], "1 neutral, 10 increased weight")
        # buttons
        btn0 = tk.Button(
            frame, text="Fit cell-by-cell", command=self.script_fit_jv_combined
        )
        btn1 = tk.Button(
            frame, text="Fit curves separately", command=self.script_fit_jv_all
        )
        btn0.pack(side="top", anchor="w", padx=10)
        btn1.pack(side="top", anchor="w", padx=10)
        doc = (
            "Process J-V files in a folder, including area correction and graphical "
            "summaries. Considers only 1 dark and 1 illuminated per cell (pixel)."
        )
        CreateToolTip(btn0, doc)
        doc = (
            "Process J-V files in a folder, including area correction. Process each "
            "file independently. Cannot create graphical summaries."
        )
        CreateToolTip(btn1, doc)

    def _cw_scripts_jvsummary(self, frame):
        """JV summaries"""
        lbl = tk.Label(frame, text="Operations on JV summaries:")
        btn0 = tk.Button(
            frame, text="JV sample maps (1 file)", command=self.script_jv_sample_maps
        )
        btn1 = tk.Button(
            frame, text="Boxplots (files in 1 folder)", command=self.script_jv_boxplots
        )
        lbl.pack(side="top", anchor="w", padx=0, pady=2)
        btn0.pack(side="top", anchor="w", padx=10)
        btn1.pack(side="top", anchor="w", padx=10)
        doc = "Create graphical summaries from a summary file of IV measurement."
        CreateToolTip(btn0, doc)
        doc = (
            "Create boxplots from data table files located in a folder. Custom"
            "processing if files are summary of IV measurement."
        )
        CreateToolTip(btn1, doc)

    def _cw_scripts_cv_cf_jscvoc(self, frame):
        """GUI for scripts C-V, C-f and Jsc-Voc"""
        lbl = tk.Label(frame, text="C-V, C-f, Jsc-Voc data processing")
        lbl.pack(side="top", anchor="w", padx=0, pady=2)
        # CV
        fr = tk.Frame(frame)
        fr.pack(side="top", fill=tk.X, padx=10)
        btn = tk.Button(fr, text="C-V (1 folder)", command=self.script_cv)
        btn.pack(side="left", pady=2)
        lbl = tk.Label(fr, text="ROI")
        lbl.pack(side="left", pady=2)
        valuedef_cv = listToString(CurveCV.CST_MottSchottky_Vlim_def)
        self._widgets["CVROI"] = EntryVar(fr, valuedef_cv, width=10)
        self._widgets["CVROI"].pack(side="left", pady=2)
        CreateToolTip(btn, "Parse a folder containing C-V files and process them.")
        # Cf
        btn0 = tk.Button(frame, text="C-f (1 folder)", command=self.script_cf)
        btn0.pack(side="top", anchor="w", padx=10)
        CreateToolTip(btn0, "Parse a folder containing C-f files and process them")
        # CVf-T
        btn = tk.Button(frame, text="CVf-T maps(1 folder)", command=self.script_cvft)
        btn.pack(side="top", anchor="w", padx=10)
        tooltiplbl = (
            "Parse folder and 1 subfolder level for Cf files according to V "
            "and T, computes CVf-T maps as well as CV-T and Cf-T plots"
        )
        CreateToolTip(btn, tooltiplbl)
        # Jsc-Voc
        fr = tk.Frame(frame)
        fr.pack(side="top", fill=tk.X, padx=10)
        btn = tk.Button(fr, text="Jsc-Voc (1 file)", command=self.script_jsc_voc)
        btn.pack(side="left", pady=2)
        lbl = tk.Label(fr, text="Jsc min")
        lbl.pack(side="left", pady=2)
        valuedef_voc_jsc = str(CurveJscVoc.CST_Jsclim0)
        self._widgets["jscVocROI"] = EntryVar(fr, valuedef_voc_jsc, width=6)
        self._widgets["jscVocROI"].pack(side="left", pady=2)
        tooltiplbl = "fit range of interest (min value or range), in mA/cm2"
        CreateToolTip(lbl, tooltiplbl)
        CreateToolTip(self._widgets["jscVocROI"], tooltiplbl)
        doc = "Performs data treatment on a cile containing Jsc-Voc pairs."
        CreateToolTip(btn, doc)
        # Correlation plots - e.g. SCAPS
        lbl = tk.Label(frame, text="Correlations")
        lbl.pack(side="top", anchor="w", padx=0, pady=2)
        fr = tk.Frame(frame)
        fr.pack(side="top", fill=tk.X, padx=10)
        btn = tk.Button(fr, text="(1 file)", command=self.script_correlations)
        btn.pack(side="left", pady=2)
        tk.Label(fr, text="e.g. SCAPS batch result").pack(side="left")
        doc = (
            "Select a file containing e.g. list of experiments according to 2 (or "
            "more) parameter variations."
        )
        CreateToolTip(btn, doc)

    def _cw_bottom(self, frame):
        """Widgets to the bottom of the menu"""
        fr0 = tk.Frame(frame)
        fr0.pack(side="bottom", fill=tk.X)
        # bottom section
        self._cw_section_title(fr0, "")
        # contentkwargs
        fr = tk.Frame(fr0)
        fr.pack(side="top", fill=tk.X)
        self._widgets["printCommands"] = CheckbuttonVar(
            fr,
            "Commands in console",
            False,
            command=lambda: print("Commands printing is not fully tested yet"),
        )
        self._widgets["printCommands"].pack(side="left", anchor="center")
        btn = tk.Button(fr, text="QUIT", fg="red", command=self.quit_main)
        btn.pack(side="right", anchor="n", pady=2, padx=5)

    # methods
    def open_file(self):
        """Open a file chosen by user"""
        file = self.app.prompt_file(multiple=True)
        if file != "" and file is not None:
            self.app.open_file(file)

    def merge_file(self):
        """Merge data with file chosen by user"""
        file = self.app.prompt_file(multiple=True)
        if file != "" and file is not None:
            self.app.merge_graph(file)

    def open_folder(self):
        """Open all files in a folder given by user"""
        folder = self.app.prompt_folder()
        if folder is not None and folder != "":
            files = self._list_files_in_folder(folder)
            self.app.open_file(files)

    def open_clipboard(self):
        """Retrieve the content of clipboard and create a graph with it"""
        folder = self.app.get_folder()
        dpi = self.app.get_tab_properties()["dpi"]
        tmp = self.app.get_clipboard()
        graph = Graph(tmp, {"isfilecontent": True}, **self.app.newgraph_kwargs)
        print("Import data from clipboard (" + str(len(graph)) + " Curves found).")
        self.app.graph(graph, title="from clipboard", folder=folder, dpi=dpi)
        # update_ui will be triggered by change in tabs

    def merge_clipboard(self):
        """Retrieves the content of clipboard, and appends it to the graph"""
        tmp = self.app.get_clipboard()
        graph = Graph(tmp, {"isfilecontent": True})
        print("Import data from clipboard (" + str(len(graph)) + " Curves found).")
        self.app.merge_graph(graph)

    def save_graph(self):
        """Saves the graph (image & data) with same filename as last time"""
        if self.app.graph().attr("meastype") not in ["", GraphIO.GRAPHTYPE_GRAPH]:
            msg = (
                "WARNING saving data: are you sure you are not overwriting a graph "
                "file?\nClick on Save As... to save the current graph (or delete "
                "property 'Misc.'->'meastype')."
            )
            print(msg)
            self.save_graph_as(filesave="")
        else:
            filesave = self.app.get_file()
            self.save_graph_as(filesave=filesave)

    def save_graph_as(self, filesave=""):
        """saves the graph to image + data file, asks for a new filename"""
        defext = ""
        if filesave == "":
            defext = copy.copy(self.app.graph().config("save_imgformat", ".png"))
            if isinstance(defext, list):
                defext = defext[0]
            filesave = self.app.prompt_file(type_="save", defaultextension=defext)
        if filesave is None or filesave == "":
            # asksaveasfile return `None` if dialog closed with "cancel".
            return
        # retrieve info from GUI
        save_altered = self._widgets["saveScreen"].get()
        if_compact = not self._widgets["saveSepara"].get()
        # some checks to avoid erasing something important
        filesave, fileext = os.path.splitext(filesave)
        fileext = fileext.lower()
        forbidden_ext = ["py", "txt", ".py", ".txt"]
        for ext in forbidden_ext:
            fileext = fileext.replace(ext, "")
        if fileext == defext:
            fileext = ""
        self.app.save_graph(
            filesave, fileext=fileext, save_altered=save_altered, if_compact=if_compact
        )

    def _list_files_in_folder(self, folder):
        # returns a list with all files in folder. Look for subfolders status
        subfolders = self._widgets["openSubfolders"].get()
        n_max = 1000
        out = []
        if subfolders:
            for root, _subdirs, files in os.walk(folder):
                for file in files:
                    if len(out) < n_max:
                        out.append(str(os.path.join(root, file)))
        else:  # not subfolders
            for file in os.listdir(folder):
                if os.path.isfile(os.path.join(folder, file)) and len(out) < n_max:
                    out.append(str(os.path.join(folder, file)))
        return out

    def insert_curve_empty(self):
        """Append an empty Curve in current Graph"""
        curve = Curve([[0], [0]], {})
        self.app.insert_curve_to_graph(curve)

    def save_image_to_clipboard(self):
        """Copy an image of current graph into the clipboard"""
        folder = self.app.get_folder()
        return image_to_clipboard(self.app.graph(), folder=folder)

    def script_fit_jv_combined(self, group_cell: bool = True):
        """Script JV process, grouped by cells or not.

        :param group_cell: True to group results by cell, False for independent
               processing for each individual JV file.
        """
        folder = self.app.prompt_folder()
        if folder != "":
            print("... Processing folder ...")
            weight = strToVar(self._widgets["JVDiodeweight"].get())
            ngkw = self.app.newgraph_kwargs
            graph = processJVfolder(
                folder,
                ylim=(-np.inf, np.inf),
                groupCell=group_cell,
                fitDiodeWeight=weight,
                newGraphKwargs=ngkw,
            )
            self.app.open_file(graph)

    def script_fit_jv_all(self):
        """Script JV process, each file indepedently"""
        self.script_fit_jv_combined(group_cell=False)

    def script_jv_sample_maps(self):
        """Script JV sample maps"""
        file = self.app.prompt_file()
        if file is not None and file != "":
            print("...creating sample maps...")
            ngkw = self.app.newgraph_kwargs
            filelist, fileout = processSampleCellsMap(file, newGraphKwargs=ngkw)
            if len(fileout) > 0:
                graph = Graph(fileout[0])
                self.app.open_file(graph)
            elif len(filelist) > 0:
                graph = Graph(filelist[-1])
                self.app.open_file(graph)

    def script_jv_boxplots(self):
        """Script boxplots"""
        folder = self.app.prompt_folder()
        if folder is not None and folder != "":
            print("...creating boxplots...")
            tmp = JVSummaryToBoxPlots(
                folder=folder,
                exportPrefix="boxplots_",
                replace=[],
                silent=True,
                newGraphKwargs=self.app.newgraph_kwargs,
            )
            self.app.open_file(tmp)

    def script_jsc_voc(self):
        """Script JscVoc"""
        file = self.app.prompt_file()
        if file is not None and file != "":
            roi_jsclim = strToVar(self._widgets["jscVocROI"].get())
            ngkw = self.app.newgraph_kwargs
            graph = script_processJscVoc(
                file, ROIJsclim=roi_jsclim, newGraphKwargs=ngkw
            )
            self.app.open_file(graph)

    def script_cv(self):
        """Script CV"""
        folder = self.app.prompt_folder()
        if folder is not None and folder != "":
            roi_fit = strToVar(self._widgets["CVROI"].get())
            ngkw = self.app.newgraph_kwargs
            graph = script_processCV(folder, ROIfit=roi_fit, newGraphKwargs=ngkw)
            self.app.open_file(graph)

    def script_cf(self):
        """Script Cf"""
        folder = self.app.prompt_folder()
        if folder is not None and folder != "":
            ngkw = self.app.newgraph_kwargs
            graph = script_processCf(folder, newGraphKwargs=ngkw)
            self.app.open_file(graph)

    def script_cvft(self):
        """Script C-V-f-T"""
        folder = self.app.prompt_folder()
        if folder is not None and folder != "":
            ngkw = self.app.newgraph_kwargs
            graph = script_process_cvft(folder, newGraphKwargs=ngkw)
            self.app.open_file(graph)

    def script_correlations(self):
        """Script processing of SCAPS output IV, CV, Cf QE files"""
        file = self.app.prompt_file()
        if file is not None and file != "":
            ngkw = self.app.newgraph_kwargs
            graph = corr_process_file(file, newGraphKwargs=ngkw)
            self.app.open_file(graph)

    def quit_main(self):
        """Quits main application"""
        self.app.quit()


class GUIFrameTemplateColorize:
    """
    Handles the templates and colorization features
    Contains a Frame, to be embedded in a wider application (.frame.pack())
    Some methods will call methods of application
    """

    def __init__(self, master, application, **kwargs):
        self.frame = tk.Frame(master, **kwargs)
        self.app = application
        self.colorlist = None  # defined later, here to reserve variable name
        self.photoimages = None  # defined later, here to reserve variable name
        self._widgets = {}
        self._create_widgets(self.frame)

    def update_ui(self):
        """Update widgets"""
        # update conditional formatting
        graph = self.app.graph()
        property_list = []
        for curve in graph:
            for key in curve.get_attributes():
                if key not in property_list:
                    property_list.append(key)
        property_list.sort()
        property_list.insert(0, "linestyle")
        self._widgets["conTestProp"].reset_values(property_list)
        self._widgets["conApplyProp"].reset_values(property_list)
        tab_properties = self.app.get_tab_properties()
        self._widgets["BGC"].set(tab_properties["backgroundcolor"])

    def _create_widgets(self, frame):
        """Create widgets"""
        fr = FrameTitleContentHide(
            frame,
            self._cw_title,
            self._cw_content,
            default="hide",
            contentkwargs={"padx": 5},
        )
        fr.pack(side="top", fill=tk.X, anchor="w")

    def _cw_title(self, frame):
        """Create widgets title"""
        text = "Background, Template, Colorize, Conditional formatting"
        lbl = tk.Label(frame, text=text, font=self.app.fonts["bold"])
        lbl.pack(side="left")

    def _cw_content(self, frame):
        """Create widgets content of show-hide"""
        self._cw_content_background(frame)
        self._cw_content_template(frame)
        self._cw_content_colors(frame)
        self._cw_content_conditional(frame)

    def _cw_content_background(self, frame):
        # background color
        fr = tk.Frame(frame)
        fr.pack(side="top", fill=tk.X, anchor="w", pady=3)
        btn = tk.Button(fr, text="Save", command=self.set_bgcolor_from_entry)
        btn.pack(side="left")
        lbl = tk.Label(fr, text="Background")
        lbl.pack(side="left")
        colorlist = ["white", "black", "grey15", "#90ee90"]
        self._widgets["BGC"] = ComboboxVar(fr, colorlist, "white", width=7)
        self._widgets["BGC"].pack(side="left")
        self._widgets["BGC"].bind(
            "<Return>", lambda event: self.set_bgcolor_from_entry()
        )

    def _cw_content_template(self, frame):
        """Templates"""
        fr = tk.Frame(frame)
        fr.pack(side="top", fill=tk.X, anchor="w")
        btn0 = tk.Button(fr, text="Load & apply template", command=self.load_template)
        btn0.pack(side="left", anchor="n")
        self._widgets["tplCrvProp"] = CheckbuttonVar(fr, "also Curves properties", 1)
        self._widgets["tplCrvProp"].pack(side="left")
        btn1 = tk.Button(fr, text="Save template", command=self.save_template)
        btn1.pack(side="right", anchor="n")

    def _cw_content_colors(self, frame):
        """Create widgets options for Colorscales"""
        frmain = tk.Frame(frame)
        frmain.pack(side="top", fill=tk.X, anchor="w", pady=3)
        # line 1
        fr = tk.Frame(frmain)
        fr.pack(side="top", fill=tk.X, anchor="w")
        btn0 = tk.Button(fr, text="Colorize", command=self.colorize_graph)
        btn0.pack(side="left")
        self._widgets["colEmpty"] = CheckbuttonVar(fr, "repeat if no label  ", False)
        self._widgets["colEmpty"].pack(side="left")
        self._widgets["colInvert"] = CheckbuttonVar(fr, "invert", False)
        self._widgets["colInvert"].pack(side="left")
        self._widgets["colAvoidWhite"] = CheckbuttonVar(fr, "avoid white", False)
        self._widgets["colAvoidWhite"].pack(side="left")
        self._widgets["colCurveSelect"] = CheckbuttonVar(fr, "curve selection", False)
        self._widgets["colCurveSelect"].pack(side="left")
        # line 2
        fr = tk.Frame(frmain)
        fr.pack(side="top", anchor="w")
        self._widgets["colChoice"] = EntryVar(fr, "", width=70)
        self._widgets["colChoice"].pack(side="left")
        # line 3
        fr = tk.Frame(frmain)
        fr.pack(side="top", anchor="w")
        self._cw_content_photoimages(fr)

    def _cw_content_photoimages(self, frame):
        """Create widgets photoimages"""
        n_per_line = 12
        graphtest = Graph(**self.app.newgraph_kwargs)
        self.colorlist = colorscales_from_config(graphtest)
        # need to keep reference of images, otherwise tk.Button loose it
        self.photoimages = []  # None] * len(self.colorList)
        width, height = 30, 15
        j = 0  # for display purpose, number of widgets actually created
        for i, color in enumerate(self.colorlist):
            photoimage = PhotoImageColorscale(width=width, height=height)
            try:
                photoimage.fill_colorscale(color)
            except ValueError:
                # if python does not recognize values (e.g. inferno, viridis)
                continue  # does NOT create widget
            widget = tk.Button(
                frame,
                image=photoimage,
                command=lambda i_=i: self._set_col_choice(i_),
            )
            widget.grid(column=int(j % n_per_line), row=int(np.floor(j / n_per_line)))
            self.photoimages.append(photoimage)
            j += 1

    def _cw_content_conditional(self, frame):
        """Create widgets cnoditionall formatting"""
        fr = tk.Frame(frame)
        fr.pack(side="top", anchor="w")

        modes = ConditionalPropertyApplier.MODES_VALUES
        command = self.conditional_format
        btn0 = tk.Button(fr, text="Save", command=command)
        btn0.pack(side="left")
        tk.Label(fr, text="if").pack(side="left")
        self._widgets["conTestProp"] = ComboboxVar(fr, [], "label", width=9)
        self._widgets["conTestProp"].pack(side="left")
        self._widgets["conTestMode"] = ComboboxVar(fr, modes, "contains", width=8)
        self._widgets["conTestMode"].pack(side="left")
        self._widgets["conTestValue"] = EntryVar(fr, "", width=9)
        self._widgets["conTestValue"].pack(side="left")
        tk.Label(fr, text=", apply").pack(side="left")
        self._widgets["conApplyProp"] = ComboboxVar(fr, [], "color", width=9)
        self._widgets["conApplyProp"].pack(side="left")
        self._widgets["conApplyValue"] = EntryVar(fr, "", width=9)
        self._widgets["conApplyValue"].pack(side="left")
        self._widgets["conTestProp"].bind("<Return>", lambda event: command())
        self._widgets["conTestMode"].bind("<Return>", lambda event: command())
        self._widgets["conTestValue"].bind("<Return>", lambda event: command())
        self._widgets["conApplyProp"].bind("<Return>", lambda event: command())
        self._widgets["conApplyValue"].bind("<Return>", lambda event: command())

    def conditional_format(self):
        """Apply conditional formatting"""
        graph = self.app.graph()
        test_prop = self._widgets["conTestProp"].get()
        test_mode = self._widgets["conTestMode"].get()
        test_value = self._widgets["conTestValue"].get()
        apply_prop = self._widgets["conApplyProp"].get()
        apply_value = self._widgets["conApplyValue"].get()
        if test_mode not in ConditionalPropertyApplier.MODES_BYINPUTTYPE["str"]:
            test_value = strToVar(test_value)
        apply_value = strToVar(apply_value)
        ConditionalPropertyApplier.apply(
            graph, test_prop, test_mode, test_value, apply_prop, apply_value
        )
        self.app.update_ui()

    def load_template(self):
        """load and apply template on current Graph"""
        file = self.app.prompt_file()
        if file != "" and file is not None:
            print("Open template file:", file)
            self.app.store_selected_curves()  # before modifs to prepare update_ui
            template = Graph(file, complement={"label": ""})
            also_curves = self._widgets["tplCrvProp"].get()
            self.app.graph().apply_template(template, also_curves=also_curves)
            self.app.update_ui()

    def save_template(self):
        """save template file from current Graph"""
        file = self.app.prompt_file(type_="save", defaultextension=".txt")
        if file is not None and file != "":
            file, _fileext = os.path.splitext(file)
            # fileext = fileext.replace("py", "")
            self.app.graph().export(filesave=file, if_template=True)
            # no need to refresh UI

    def colorize_graph(self):
        """Colorize the Curves in the Graph"""
        self.app.store_selected_curves()  # before modifs, to prepare update_ui
        col = strToVar(self._widgets["colChoice"].get())
        if len(col) == 0:
            col = self.colorlist[0].get_colorscale()
            self._set_col_choice(0)
        invert = self._widgets["colInvert"].get()
        kwargs = {
            "avoidWhite": self._widgets["colAvoidWhite"].get(),
            "sameIfEmptyLabel": self._widgets["colEmpty"].get(),
        }
        if self._widgets["colCurveSelect"].get():
            curves = self.app.get_selected_curves(multiple=True)
            if len(curves) > 0 and curves[0] >= 0:
                # if no curve is selected, colorize all curves
                kwargs.update({"curvesselection": curves})
        try:
            colorscale = Colorscale(col, invert=invert)
            self.app.graph().colorize(colorscale, **kwargs)
        except ValueError:
            # error to be printed in GUI console, and not hidden in terminal
            logger.error("colorize_graph:", exc_info=True)
        if self.app.if_print_commands():  # to explicit creation of colorscale
            msg = "colorscale = Colorscale({}, invert={})"
            print(msg.format(col, invert))
            msg = "graph.colorize(colorscale, {})"
            arg = ", ".join(["{}={!r}".format(k, v) for k, v in kwargs.items()])
            print(msg.format(arg))
        self.app.update_ui()

    def _set_col_choice(self, i):
        """Reads variable (Colorscale object), extract colors (np.array),
        converts into string"""
        # print('_setColChoice', i, self.colorList[i].getColorScale())
        if isinstance(self.colorlist[i].get_colorscale(), str):
            scale = self.colorlist[i].get_colorscale()  # eg. viridis
        else:
            lst = []
            for elem in self.colorlist[i].get_colorscale():
                if not isinstance(elem, str):
                    tostr = [str(nb if not nb.is_integer() else int(nb)) for nb in elem]
                    lst.append("[" + ",".join(tostr) + "]")
                    # print("_setColChoice, elem,", elem)
                else:
                    lst.append("'" + elem + "'")
            scale = "[" + ", ".join(lst) + "]"
        self._widgets["colChoice"].set(scale)

    def set_bgcolor_from_entry(self, update_ui=True):
        """Set background color of the canvas"""
        if update_ui:
            self.app.store_selected_curves()  # before modifications, prepare update_ui
        value = str(self._widgets["BGC"].get())
        self.app.get_tab_properties(backgroundcolor=value)
        if update_ui:
            if self.app.initiated:
                self.app.update_ui()


class GUIFrameActionsGeneric:
    """
    Handles the panel Actions on Curves
    Contains a Frame, to be embedded in a wider application (.frame.pack())
    Some methods will call methods of application
    """

    CASTCURVERENAMEGUI = {"Fit Arrhenius": "Fit"}

    def __init__(self, master, application, **kwargs):
        self.frame = tk.Frame(master, **kwargs)
        self.app = application
        self._widgets = {}
        self._create_widgets(self.frame)
        # bind keys
        self.app.master.bind("<Control-Delete>", lambda e: self.delete_curve())
        self.app.master.bind(
            "<Control-Shift-C>", lambda e: self.copy_curve_to_clipboard()
        )
        self.app.master.bind("<Control-h>", lambda e: self.show_hide_curve())

    def update_ui(self):
        """Update widgets"""
        # ShowHide: handled by Observable
        # cast curve optionmenu: handled by Observable
        pass

    def _create_widgets(self, frame):
        """Create widgets"""
        fr = FrameTitleContentHide(frame, self._cw_title, self._cw_content)
        fr.pack(side="top", fill=tk.X, anchor="w")

    def _cw_title(self, frame):
        """Create widgets Title"""
        lbl = tk.Label(frame, text="Actions on Curves", font=self.app.fonts["bold"])
        lbl.pack(side="left", anchor="n")

    def _cw_content(self, frame):
        """Create widgets content"""

        def to_grid(row, column, func, **kwargs):
            kw = {"sticky": "w"}
            kw.update(kwargs)
            fr = tk.Frame(frame)
            fr.grid(row=row, column=column, **kw)
            func(fr)

        to_grid(0, 0, self._cw_reorder, padx=5)
        to_grid(1, 0, self._cw_delete, padx=5)
        to_grid(2, 0, self._cw_duplicate, padx=5)
        to_grid(3, 0, self._cw_show_hide, padx=5)
        to_grid(0, 1, self._cw_clipboard)
        to_grid(1, 1, self._cw_cast)
        to_grid(2, 1, self._cw_quick_attr)
        to_grid(3, 1, self._cw_label_replace)

    def _cw_reorder(self, frame):
        """Create widgets reorder"""
        tk.Label(frame, text="Reorder").pack(side="left")
        b0 = tk.Button(frame, text=" \u21E7 ", command=self.shift_curve_top)
        b0.pack(side="left", padx=1)
        b1 = tk.Button(frame, text=" \u21D1 ", command=self.shift_curve_up)
        b1.pack(side="left", padx=1)
        b2 = tk.Button(frame, text=" \u21F5 ", command=self.shift_curve_reverse)
        b2.pack(side="left", padx=1)
        b3 = tk.Button(frame, text=" \u21D3 ", command=self.shift_curve_down)
        b3.pack(side="left", padx=1)
        b4 = tk.Button(frame, text=" \u21E9 ", command=self.shift_curve_bottom)
        b4.pack(side="left", padx=1)
        CreateToolTip(b0, "Selection to Top")
        CreateToolTip(b1, "Selection Up")
        CreateToolTip(b2, "Reverse all Curves")
        CreateToolTip(b3, "Selection Down")
        CreateToolTip(b4, "Selection Bottom")

    def _cw_delete(self, frame):
        """Create widgets Delete"""
        tk.Label(frame, text="Delete Curve").pack(side="left")
        b0 = tk.Button(frame, text="Curve", command=self.delete_curve)
        b0.pack(side="left", padx=3)
        b1 = tk.Button(frame, text="All hidden", command=self.delete_curves_hidden)
        b1.pack(side="left")
        CreateToolTip(b0, "Ctrl+Delete")

    def _cw_duplicate(self, frame):
        """Create widgets Duplicate"""
        tk.Label(frame, text="Duplicate Curve").pack(side="left")
        btn = tk.Button(frame, text="Duplicate", command=self.duplicate_curve)
        btn.pack(side="left")

    def _cw_show_hide(self, frame):
        """Create widgets Show Hide"""

        def update_var_sh(curve, _key):
            if not isinstance(curve, Curve):
                val = "Show Curve"  # -1 if no curve selected
            else:
                val = "Hide Curve" if curve.visible() else "Show Curve"
            self._widgets["showHide"].set(val)

        self.app.observables["focusTree"].register(update_var_sh)
        b0 = ButtonVar(frame, "Show Curve", command=self.show_hide_curve)
        self._widgets["showHide"] = b0
        b0.pack(side="left")
        b1 = tk.Button(frame, text="All", command=self.show_hide_all)
        b1.pack(side="left", padx="5")
        b2 = tk.Button(frame, text="Invert", command=self.show_hide_invert)
        b2.pack(side="left")
        CreateToolTip(b0, "Ctrl+H")

    def _cw_clipboard(self, frame):
        """Create widgets Clipboard"""
        tk.Label(frame, text="Copy to clipboard").pack(side="left")
        b0 = tk.Button(frame, text="Curve", command=self.copy_curve_to_clipboard)
        b0.pack(side="left", padx="5")
        b1 = tk.Button(frame, text="Graph", command=self.copy_graph_to_clipboard)
        b1.pack(side="left")
        vals = ("raw", "with properties", "screen data", "screen data, prop.")
        omv = OptionMenuVar(frame, vals, default=vals[1], width=6)
        self._widgets["clipboardOpts"] = omv
        self._widgets["clipboardOpts"].pack(side="left")
        CreateToolTip(b0, "Ctrl+Shift+C")

    def _cw_cast(self, frame):
        """Create widgets Cast Curve"""

        def update_var_cast(curve, _key):
            # update GUI cast action: empy menu and refill it
            self._widgets["castCurve"]["menu"].delete(0, "end")
            self._widgets["castCurve"].set("")
            if isinstance(curve, Curve):
                cast_list = curve.castCurveListGUI(onlyDifferent=False)
                values = [cast[0] for cast in cast_list]
                default = curve.classNameGUI()
                for key, value in self.CASTCURVERENAMEGUI.items():
                    values = [v.replace(key, value) for v in values]
                    if default == key:
                        default = value
                self._widgets["castCurve"].reset_values(values, default=default)

        self.app.observables["focusTree"].register(update_var_cast)
        tk.Label(frame, text="Change Curve type").pack(side="left")
        self._widgets["castCurve"] = OptionMenuVar(frame, ("",), "")
        self._widgets["castCurve"].pack(side="left", padx="2")
        tk.Button(frame, text="Save", command=self.cast_curve).pack(side="left")

    def _cw_quick_attr(self, frame):
        """Create widgets Quick attr: label, color"""

        def update_qa(curve, _key):
            if isinstance(curve, Curve):
                self._widgets["QALabel"].set(varToStr(curve.attr("label")))
                self._widgets["QAColor"].set(varToStr(curve.attr("color")))

        self.app.observables["focusTree"].register(update_qa)
        tk.Label(frame, text="Label").pack(side="left")
        self._widgets["QALabel"] = EntryVar(frame, "", width=15)
        self._widgets["QALabel"].pack(side="left")
        tk.Label(frame, text="Color").pack(side="left")
        btn0 = tk.Button(frame, text="Pick", command=self.choose_color)
        btn0.pack(side="left")
        self._widgets["QAColor"] = EntryVar(frame, "", width=8)
        self._widgets["QAColor"].pack(side="left")
        btn1 = tk.Button(frame, text="Save", command=self.set_quick_attr)
        btn1.pack(side="left")
        self._widgets["QALabel"].bind("<Return>", lambda event: self.set_quick_attr())
        self._widgets["QAColor"].bind("<Return>", lambda event: self.set_quick_attr())

    def _cw_label_replace(self, frame):
        """Create widgets Replace str in label"""
        tk.Label(frame, text="Replace in labels").pack(side="left")
        self._widgets["labelOld"] = EntryVar(frame, "old string", width=10)
        self._widgets["labelOld"].pack(side="left")
        self._widgets["labelNew"] = EntryVar(frame, "new string", width=10)
        self._widgets["labelNew"].pack(side="left")
        btn = tk.Button(frame, text="Replace", command=self.replace_labels)
        btn.pack(side="left")
        self._widgets["labelOld"].bind("<Return>", lambda event: self.replace_labels())
        self._widgets["labelNew"].bind("<Return>", lambda event: self.replace_labels())

    def shift_curve(self, up_down, relative=True):
        """Chift a Curve within the Graph"""
        curves = self.app.get_selected_curves(multiple=True)
        test = up_down > 0
        curves.sort(reverse=test)
        selected = []
        graph = self.app.graph()
        for curve in curves:
            idx2 = up_down
            if curve == 0:
                idx2 = max(idx2, 0)
            if curve == len(graph) - 1 and relative:
                idx2 = min(idx2, 0)
            if relative:
                self.app.call_graph_method("curves_swap", curve, idx2, relative=True)
                selected.append(curve + idx2)
            else:
                self.app.call_graph_method("curve_move_to_index", curve, idx2)
                selected.append(idx2)
                # print('moveCurve', curve, idx2, upDown)
                if idx2 < curve or (idx2 == curve and curve == 0):
                    up_down += 1
                elif idx2 > curve or (idx2 == curve and curve >= len(graph) - 1):
                    up_down -= 1
        for i in range(len(selected)):
            selected[i] = max(0, min(len(graph) - 1, selected[i]))
        if len(selected) > 0:
            sel = [graph[c] for c in selected]
            keys = [""] * len(sel)
            self.app.get_tab_properties(
                selectionCurvesKeys=(sel, keys), focusCurvesKeys=([sel[0]], [""])
            )
        # print(self.app.getTabProperties())
        self.app.update_ui()

    def shift_curve_down(self):
        """Shift Curve down in the Graph"""
        self.shift_curve(1)

    def shift_curve_up(self):
        """Shift a Curve up in the Graph"""
        self.shift_curve(-1)

    def shift_curve_top(self):
        """Shift Curve in the position in the graph"""
        self.shift_curve(0, relative=False)

    def shift_curve_bottom(self):
        """hift Curve at the end of a Graph"""
        self.shift_curve(len(self.app.graph()) - 1, relative=False)

    def shift_curve_reverse(self):
        """Reverse the order of  the Curves within the Graph"""
        self.app.graph().curves_reverse()
        self.app.update_ui()

    def delete_curve(self):
        """Delete the currently selected curve."""
        self.app.store_selected_curves()  # before modifs, to prepare update_ui
        curves = list(self.app.get_selected_curves(multiple=True))
        curves.sort(reverse=True)
        for curve in curves:
            if not is_number(curve):
                break
                # can happen if someone presses the delete button twice in a row
            if curve > -1:
                self.app.call_graph_method("curve_delete", curve)
        self.app.update_ui()

    def delete_curves_hidden(self):
        """Delete all the hidden curves."""
        self.app.store_selected_curves()  # before modifs, to prepare update_ui
        graph = self.app.graph()
        to_del = []
        for c, curve in enumerate(graph):
            if not curve.visible():
                to_del.append(c)
        to_del.sort(reverse=True)
        for c in to_del:
            self.app.call_graph_method("curve_delete", c)
        self.app.update_ui()

    def duplicate_curve(self):
        """Duplicate the currently selected curve."""
        self.app.store_selected_curves()  # before modifs, to prepare update_ui
        curves = list(self.app.get_selected_curves(multiple=True))
        curves.sort(reverse=True)
        selected = []
        for curve in curves:
            if not is_number(curve):
                # can happen if someone presses the delete button twice in arow
                break
            if curve > -1:
                self.app.call_graph_method("curve_duplicate", curve)
                selected = [s + 1 for s in selected]
                selected.append(curve)
        self.app.update_ui()

    def show_hide_curve(self):
        """Toggle if Curve is displayed or not"""
        self.app.store_selected_curves()
        curves = self.app.get_selected_curves(multiple=True)
        graph = self.app.graph()
        for c in curves:
            if -1 < c < len(graph):
                graph[c].visible(not graph[c].visible())
                if self.app.if_print_commands():
                    print("graph[{}].visible(not graph[{}].visible())".format(c, c))
        self.app.update_ui()

    def show_hide_all(self):
        """Toggle if displayed or not, for all Curves with same value"""
        graph = self.app.graph()
        if len(graph) > 0:
            self.app.store_selected_curves()  # before modifs to prepare update_ui
            new = False if graph[0].visible() else True
            for curve in graph:
                curve.visible(new)
            if self.app.if_print_commands():
                print("for curve in graph:")
                print("    curve.visible({})".format(new))
        self.app.update_ui()

    def show_hide_invert(self):
        """Toggle if displayed or not, for all Curves independently"""
        self.app.store_selected_curves()  # before modifs, to prepare update_ui
        for curve in self.app.graph():
            curve.visible(not curve.visible())
        if self.app.if_print_commands():
            print("for curve in graph:")
            print("    curve.visible(not curve.visible())")
        self.app.update_ui()

    def copy_curve_to_clipboard(self):
        """Cpy the Curve into the clipboard"""
        graph = self.app.graph()
        curves = self.app.get_selected_curves(multiple=True)
        if len(curves) == 0:
            return
        content = ""
        opts = self._widgets["clipboardOpts"].get()
        if_attrs = "prop" in opts
        if_trans = "screen" in opts
        # print("opts", opts, if_attrs, if_trans)
        if not if_attrs:
            labels = [varToStr(graph[c].attr("label")) for c in curves]
            content += "\t" + "\t\t\t".join(labels) + "\n"
        else:
            keys = []
            for c in curves:
                for key in graph[c].get_attributes():
                    if key not in keys:
                        keys.append(key)
            keys.sort()
            for key in keys:
                labels = [varToStr(graph[c].attr(key)) for c in curves]
                content += key + "\t" + "\t\t\t".join(labels) + "\n"
        data = [graph.getCurveData(c, ifAltered=if_trans) for c in curves]
        length = max([d.shape[1] for d in data])
        for le in range(length):
            tmp = []
            for d in data:
                if le < d.shape[1]:
                    tmp.append("\t".join([str(e) for e in d[:, le]]))
                else:
                    tmp.append("\t".join([""] * d.shape[0]))
            content += "\t\t".join(tmp) + "\n"
        self.app.set_clipboard(content)

    def copy_graph_to_clipboard(self):
        """Copy the content of a Graph into the clipboard"""
        opts = self._widgets["clipboardOpts"].get()
        if_attrs = "prop" in opts
        if_trans = "screen" in opts
        graph = self.app.graph()
        data = graph.export(
            if_clipboard_export=True,
            if_only_labels=(not if_attrs),
            save_altered=if_trans,
        )
        self.app.set_clipboard(data)

    def cast_curve(self):
        """Cast (change the type of) a Curve"""
        graph = self.app.graph()
        curves = self.app.get_selected_curves(multiple=True)
        type_new = self._widgets["castCurve"].get()
        for key, value in self.CASTCURVERENAMEGUI.items():
            if type_new == value:
                type_new = key
        selected = []
        for curve in curves:
            if -1 < curve < len(graph):
                test = self.app.call_graph_method("castCurve", type_new, curve)
                selected.append(curve)
                if not test:
                    logger.error("castCurve impossible.")
            else:
                logger.error("castCurve impossible ({}, {})".format(type_new, curve))
        if len(selected) > 0:
            sel = [graph[c] for c in selected]
            keys = [""] * len(sel)
            self.app.get_tab_properties(
                selectionCurvesKeys=(sel, keys), focusCurvesKeys=([sel[0]], [""])
            )
        self.app.update_ui()

    def choose_color(self):
        """Prompts the user to choose a color"""
        curves = self.app.get_selected_curves(multiple=False)
        if curves[0] != -1:
            try:
                colorcurrent = rgb2hex(strToVar(self._widgets["QAColor"].get()))
            except Exception:
                colorcurrent = None
            ans = askcolor(color=colorcurrent)
            if ans[0] is not None:
                self._widgets["QAColor"].set(
                    listToString([np.round(a, 3) for a in hex2color(ans[1])])
                )  # [np.round(val/256,3) for val in ans[0]]

    def set_quick_attr(self):
        """Updates the values of the keywords label, and color"""
        self.app.store_selected_curves()
        curves = self.app.get_selected_curves(multiple=True)
        for c in curves:
            arg = {
                "label": strToVar(self._widgets["QALabel"].get()),
                "color": strToVar(self._widgets["QAColor"].get()),
            }
            self.app.call_curve_method(c, "update", arg)
        self.app.update_ui()

    def replace_labels(self):
        """Replace labels graph.replace_labels(old, new)"""
        self.app.store_selected_curves()  # before modifs, to prepare update_ui
        old = self._widgets["labelOld"].get()
        new = self._widgets["labelNew"].get()
        self.app.call_graph_method("replace_labels", old, new)
        self._widgets["labelOld"].set("")
        self._widgets["labelNew"].set("")
        self.app.update_ui()


class GUIFramePropertyEditor:
    """
    Handles the panel Property edit
    Contains a Frame, to be embedded in a wider application (.frame.pack())
    Some methods will call methods of application
    """

    VARVALUEWIDTH = 30

    def __init__(self, master, application, **kwargs):
        self.frame = tk.Frame(master, **kwargs)
        self.app = application
        self._widgets = {}
        self._create_widgets(self.frame)

    def update_ui(self):
        """Update the widgets"""
        pass

    def _create_widgets(self, frame):
        """Create the widgets"""
        fr = FrameTitleContentHide(frame, self._cw_title, self._cw_content)
        fr.pack(side="top", fill=tk.X, anchor="w")

    def _cw_title(self, frame):
        """Create the widgets Title"""
        lbl = tk.Label(frame, text="Property editor", font=self.app.fonts["bold"])
        lbl.pack(side="top", anchor="w")

    def _cw_content(self, frame):
        """Create the widgets of the frame Content"""
        fr = tk.Frame(frame)
        fr.pack(side="top", anchor="w", fill=tk.X, padx=5)
        self.cw_content_edit(fr)
        tk.Label(frame, text="\n").pack(side="left", anchor="w", padx=5)
        # self.NewPropExample = tk.StringVar()
        # self.NewPropExample.set('')
        self._widgets["example"] = LabelVar(frame, "", justify="left")
        self._widgets["example"].pack(side="left", anchor="w")

    def cw_content_edit(self, frame):
        """Create the widgets to edit"""
        tk.Label(frame, text="Property:").pack(side="left")
        self._widgets["key"] = OptionMenuVar(frame, (), "")
        self._widgets["key"].pack(side="left")
        self._widgets["key"].var.trace("w", self.select_key)
        self.app.observables["focusTree"].register(self.impose_key)
        self._widgets["value"] = ComboboxVar(frame, [], "", width=self.VARVALUEWIDTH)
        self._widgets["value"].pack(side="left")
        self._widgets["value"].bind("<Return>", lambda event: self.save_key_value())

        btn = tk.Button(frame, text="Save", command=self.save_key_value)
        btn.pack(side="right")

    def impose_key(self, curve, key):
        """triggers by observable when user selects a property in the Tree"""
        # print('imposekey curve', curve, 'key', key)
        if curve == -1:
            key_list = KEYWORDS_GRAPH["keys"]
        else:
            key_list = KEYWORDS_CURVE["keys"]
        self._widgets["key"].reset_values(key_list)
        if key != "":
            self._widgets["key"].set(key, force=True)
            # set() triggers .selectKey() that updates varExamples and varValue
        else:
            # keep same key, try to refresh in case user selected another curve
            self._widgets["key"].set(self._widgets["key"].get(), force=True)

    def select_key(self, *_args, **_kwargs):
        """User selects an item on the drop-down menu"""
        # print('selectKey args', _args, 'kwargs', _kwargs)
        key = self._widgets["key"].get()
        curves = self.app.get_selected_curves(multiple=False)
        if len(curves) == 0:
            curve = -1
        else:
            curve = curves[0]
        if curve == -1:
            keywords = KEYWORDS_GRAPH
            current_val = self.app.graph().attr(key)
        else:
            keywords = KEYWORDS_CURVE
            current_val = self.app.graph()[curve].attr(key)
        try:
            # set text with examples, and populate Combobox values field
            i = keywords["keys"].index(key)
            valuesnew = [str(v) for v in keywords["guiexamples"][i]]
            self._widgets["value"]["values"] = valuesnew
            self._widgets["example"].set(keywords["guitexts"][i])
        except ValueError:
            self._widgets["value"]["values"] = []
            self._widgets["example"].set("")
        # set values, cosmetics
        self._widgets["value"].set(varToStr(current_val))
        state = "disabled" if key.startswith("==") else "normal"
        self._widgets["value"].configure(state=state)
        width = self.VARVALUEWIDTH
        if len(key) > 20:  # reduce widget's width to try to prevent window resize
            width = max(
                int(self.VARVALUEWIDTH / 2),
                int(self.VARVALUEWIDTH - (len(key) - 20) * 2 / 3),
            )
        self._widgets["value"].configure(width=width)

    def save_key_value(self):
        """
        New property on current curve: catch values, send to dedicated function
        Handles interface. Calls updateProperty to act on graph
        """
        curves = self.app.get_selected_curves(multiple=True)
        key = self._widgets["key"].get()
        val = self._widgets["value"].get()  # here .get(), strToVar done later
        if key == "['key', value]":
            val = strToVar(val)
            if not isinstance(val, (list, tuple)) or len(val) < 2:
                msg = (
                    "GUI.GUIFramePropertyEditor.save_key_value. Input must be a "
                    "list or a tuple with 2 elements ({}, {})"
                )
                logger.error(msg.format(val, type(val)))
                return
            key, val = val[0], val[1]
        self._update_key(curves, key, val)  # triggers update_ui
        # Tree may be still stuck on another key. Triggers change of varValue
        # and varExample
        self._widgets["key"].set(key, force=True)

    def _update_key(self, curve, key, val, if_update=True, vartype="auto"):
        """
        Input was filtered by setPropValue()
        perform curve(curve).update({key: strToVar(val)})
        curve -1: update() on graph
        ifUpdate: by default update the GUI. if False, does not. Assumes the
            update will be performed only once, later
        varType: changes type of val into varType. Otherwise, try best guess.
        """
        if key == "Property" or key.startswith("--") or key.startswith("=="):
            return
        self.app.store_selected_curves()  # before modifs, to prepare update_ui
        # print ('Property edition: curve',curve,', key', key,', value', val,
        #        '(',type(val),')')
        # possibly force variable type
        if vartype in [str, int, float, list, dict]:
            val = vartype(val)
        else:
            val = strToVar(val)
        # curve identification
        if not isinstance(curve, list):
            curve = [curve]
        for c_ in curve:
            try:
                c = int(c_)
            except (ValueError, TypeError):
                msg = "Cannot edit property: curve {}, key {}, value {}."
                logger.error(msg.format(c_, key, val))
                return
            if c < 0:
                self.app.call_graph_method("update", {key: val})
            else:
                self.app.call_curve_method(c, "update", {key: val})
        if if_update:
            self.app.update_ui()


class GUIFrameTree:
    """
    Handles the panel box with the Tree displaying all properties
    Contains a Frame, to be embedded in a wider application (.frame.pack())
    Some methods will call methods of application
    """

    def __init__(self, master, application, **kwargs):
        self.frame = tk.Frame(master, **kwargs)
        self.app = application
        self._widgets = {}
        self._create_widgets(self.frame)
        # self.prevTreeSelect = [-1]  # to be used upon refresh

    def update_ui(self):
        """Update content of Treeview"""
        graph = self.app.graph()
        tree = self._widgets["tree"]
        select = {"toFocus": [], "toSelect": []}
        try:
            props = self.app.get_tab_properties()
            select["focus"] = props["focusCurvesKeys"]
            select["selec"] = props["selectionCurvesKeys"]
        except KeyError:  # expected at initialisation
            select["focus"] = ([], [])
            select["selec"] = ([], [])
        # print('update_ui prepare selection', select)
        # clear displayed content
        tree.delete(*tree.get_children())
        # tree: update graphinfo
        attr = graph.graphinfo.values()
        idx0 = tree.insert("", "end", text="Graph", tag="-1", values=("",))
        self._update_ui_check_select(idx0, -1, "", select)
        self._update_ui_add_treebranch(idx0, attr, -1, select)
        # tree: update headers
        attr = graph.headers.values()
        if len(attr) > 0:
            idx = tree.insert("", "end", text="Misc.", tag="-1", values=("",))
            self._update_ui_add_treebranch(idx, attr, -1, select)
        # tree & list of curves
        order_lbl = ["label", "sample", "filename"]
        for i, curve in enumerate(graph):
            # decide for label
            lbl = "(no label)"
            for test in order_lbl:
                tmp = curve.attr(test)
                if tmp != "":
                    lbl = varToStr(tmp)
                    break
            # tree
            tx = "Curve " + str(i) + " " + lbl
            idx = tree.insert("", "end", tag=str(i), values=("",), text=tx)
            self._update_ui_check_select(idx, curve, "", select)
            color = self.app.fonts["fg_default"] if curve.visible() else "grey"
            tree.tag_configure(str(i), foreground=color)
            # attributes
            attr = curve.get_attributes()
            self._update_ui_add_treebranch(idx, attr, curve, select)
        # set focus and selection to previously selected element
        if len(select["toSelect"]) > 0:
            # print('Update_ui selection_set', tuple(select['toSelect']))
            tree.selection_set(tuple(select["toSelect"]))
            for iid in select["toSelect"]:
                tree.see(iid)
        if len(select["toFocus"]) > 0:
            # print('Update_ui focus', select['toFocus'][0])
            tree.focus(select["toFocus"][0])
            tree.see(select["toFocus"][0])
        self.forget_selected_curves()

    @staticmethod
    def _update_ui_check_select(id_, curve, key, select):
        """part of widget update"""
        for i in range(len(select["focus"][0])):
            if select["focus"][0][i] == curve and select["focus"][1][i] == key:
                select["toFocus"].append(id_)
        for i in range(len(select["selec"][0])):
            if select["selec"][0][i] == curve and select["selec"][1][i] == key:
                select["toSelect"].append(id_)

    def _update_ui_add_treebranch(self, idx, attr, curve, select):
        """part of widget update, add a branch in the tree widget"""
        tree = self._widgets["tree"]
        key_list = []
        for key in attr:
            key_list.append(key)
        key_list.sort()
        for key in key_list:
            val = varToStr(attr[key])
            # val = varToStr(val)  # echap a second time to go to Treeview
            try:
                id_ = tree.insert(idx, "end", text=key, values=(val,), tag=key)
                self._update_ui_check_select(id_, curve, key, select)
            except Exception:
                msg = "_update_ui_add_treebranch: key {}:\n  {} {}\n  {} {}\n  {} {}"
                msgargs = [key, type(attr[key]), attr[key], type(varToStr(attr[key]))]
                msgargs += [varToStr(attr[key]), type(val), val]
                logger.error(msg.format(*msgargs), exc_info=True)
                # for v in val:
                #     print('   ', v)

    def _create_widgets(self, frame):
        """Create widgets"""
        fr = FrameTitleContentHide(frame, self._cw_title, self._cw_content)
        fr.pack(side="top", fill=tk.X, anchor="w")

    def _cw_title(self, frame):
        """Create widgets Title"""
        lbl = tk.Label(frame, text="List of properties", font=self.app.fonts["bold"])
        lbl.pack(side="top", anchor="w")

    def _cw_content(self, frame):
        """Create widgets of Content"""

        # START OF FIX: because of change in tk. Baseically added the following
        def fixed_map(option):
            # Returns the style map for 'option' with any styles starting with
            # ("!disabled", "!selected", ...) filtered out
            # style.map() returns an empty list for missing options, so this
            # should be future-safe
            return [
                elm
                for elm in style.map("Treeview", query_opt=option)
                if elm[:2] != ("!disabled", "!selected")
            ]

        style = ttk.Style()
        style.map(
            "Treeview",
            foreground=fixed_map("foreground"),
            background=fixed_map("background"),
        )
        # END OF FIX
        tree = ttk.Treeview(frame, columns=("#1",))
        tree.pack(side="left", anchor="n")
        treeysb = ttk.Scrollbar(frame, orient="vertical", command=tree.yview)
        treeysb.pack(side="right", anchor="n", fill=tk.Y)
        tree.configure(yscroll=treeysb.set)
        tree.column("#0", width=170)
        tree.heading("#0", text="Name")
        tree.column("#1", width=290)
        tree.heading("#1", text="Value")
        tree.bind("<<TreeviewSelect>>", self._select_tree_item)
        self._widgets["tree"] = tree
        self._widgets["treeysb"] = treeysb

    def get_tree_active_curve(self, multiple=True):
        """
        Return 2 lists with following structure: [idx0, idx1, ...], [key0, key1, ...]
        Each element corresponds to a selected line in the Treeview.
        idx: curve index, -1 for header
        key: selected attribute

        :param multiple: if True, returns information related to the line returned by
               .selection(). If False, return multiple, according to .focus().
        :return: 2 lists with following structure: [idx0, idx1, ...], [key0, key1, ...]
        """
        interest = self._widgets["tree"].focus()
        if multiple:
            interest = self._widgets["tree"].selection()
        idxs, keys = self._get_tree_item_id_key(interest)
        return idxs, keys

    def _select_tree_item(self, _a):
        idxs, keys = self.get_tree_active_curve(multiple=False)
        #  self.prevTreeSelect = sorted(list(set(idxs)))  # remove duplicates
        # update observable related to selected curve
        if len(idxs) > 0:
            # provides - Curve if possible, index otherwise
            graph = self.app.graph()
            curve = idxs[0]
            if -1 < curve < len(graph):
                curve = graph[curve]
            # print('observable focustree', curve, keys[0])
            self.app.observables["focusTree"].update_observers(curve, keys[0])

    def _get_tree_item_id_key(self, item=None):
        # handle several selected items
        if item is None:
            item = self._widgets["tree"].focus()
        if isinstance(item, tuple):
            idxs, keys = [], []
            for it in item:
                idx, key = self._get_tree_item_id_key(it)
                if len(idx) > 0:
                    idxs.append(idx[0])
                    keys.append(key[0])
            # print('_getTreeItemIdKey multiple', idxs, keys)
            return idxs, keys
        # handle single element
        if len(item) == 0:
            return [], []
        selected = self._widgets["tree"].item(item)
        parent_id = self._widgets["tree"].parent(item)
        if parent_id == "":  # selected a main line (with no parent_id)-> no attr
            idx = selected["tags"]
            key = ""
        else:  # selected a line presenting an attribute
            parent = self._widgets["tree"].item(parent_id)
            idx = parent["tags"]
            key = selected["text"]
            if not parent["open"]:
                # print('wont select that one, parent not open', parent_id)
                return [], []
        if isinstance(key, list):
            key = key[0]
        if isinstance(idx, list):
            idx = idx[0]
        return [idx], [key]

    def store_selected_curves(self):
        """Store selected Curves and attributes to restore upon update_ui"""
        # store Curve instead of indices, because indices can change
        graph = self.app.graph()
        idxs, keys = self.get_tree_active_curve(multiple=False)
        for i in range(len(idxs)):
            # print('storeSelectedCurves false', idxs[i], '.', keys[i], '.')
            if 0 <= idxs[i] < len(graph):
                idxs[i] = graph[idxs[i]]
        self.app.get_tab_properties(focusCurvesKeys=(idxs, keys))
        idxs, keys = self.get_tree_active_curve(multiple=True)
        for i in range(len(idxs)):
            # print('storeSelectedCurves true', idxs[i], '.', keys[i], '.')
            if 0 <= idxs[i] < len(graph):
                idxs[i] = graph[idxs[i]]
        self.app.get_tab_properties(selectionCurvesKeys=(idxs, keys))

    def forget_selected_curves(self):
        """Empty memory wich Curves were actually sold"""
        self.app.get_tab_properties(
            focusCurvesKeys=([], []), selectionCurvesKeys=([], [])
        )


class GUIFrameActionsCurves:
    """
    Handles part with actions specific to curves
    Contains a Frame, to be embedded in a wider application (.frame.pack())
    Some methods will call methods of application
    """

    def __init__(self, master, application, **kwargs):
        self.frame = tk.Frame(master, **kwargs)
        self.app = application
        self.list_callinfos = []
        self.frame_action = None  # define later, here only defines the name
        self._widgets = []
        self._create_widgets(self.frame)
        self.previous = {"curve": None, "type": None, "le": None}
        self.app.observables["focusTree"].register(self._cw_frame_action)

    def update_ui(self):
        """Update widgets"""
        curve = self.app.get_selected_curves(multiple=False)
        if len(curve) > 0 and curve[0] > -1:
            curve = self.app.graph()[curve[0]]
        self._cw_frame_action(curve, force=True)

    def _create_widgets(self, frame):
        """Create widgets: itle frame"""
        fr = tk.Frame(frame)
        fr.pack(side="top", anchor="w", fill=tk.X)
        lbl = tk.Label(
            fr, text="Actions specific to selected Curve", font=self.app.fonts["bold"]
        )
        lbl.pack(side="left", anchor="center")
        hline = FrameTitleContentHide.frameHline(fr)
        hline.pack(side="left", anchor="center", fill=tk.X, expand=1, padx=5)
        # content
        self.frame_action = tk.Frame(frame)
        self.frame_action.pack(side="top", fill=tk.X, padx=5)
        # argsFunc=[-1])

    def _cw_frame_action(self, curve, *_args, force=False):
        # identify index of active curve
        graph = self.app.graph()
        graph_i = None
        for c, curve2 in enumerate(graph):
            if curve2 == curve:
                graph_i = c
                break
        # retrieve list of actions
        func_list = []
        if graph_i is not None:
            try:
                func_list = curve.funcListGUI(graph=graph, graph_i=graph_i)
            except Exception:
                msg = "{}.funcListGUI, graph_i {}, graph {}."
                logger.error(msg.format(type(curve), graph_i, graph), exc_info=True)
                return
        # args: specific attribute selected. Useless here
        if (
            not force
            and curve == self.previous["curve"]
            and isinstance(curve, self.previous["type"])
            and len(func_list) == self.previous["le"]
        ):
            # no need to change anything, do not update anything
            # print('GUIFrameActionsCurves: can think to not update actions')
            pass
        self.previous = {"curve": curve, "type": type(curve), "le": len(func_list)}
        # destroy all existing widgets
        for line in self._widgets:  # widgets
            for widget in line:
                widget.destroy()
        self.list_callinfos = []
        self._widgets = []
        # create new widgets
        for j, line in enumerate(func_list):
            if not isinstance(line, FuncGUI):
                line = FuncGUI(None, None).init_legacy(line)
            # create widgets
            callinfo, widgets = self._define_widgets_funcgui(
                line, self.frame_action, self.call_action, j
            )
            self.list_callinfos.append(callinfo)
            self._widgets.append(widgets)
        if len(self.list_callinfos) == 0:
            widget = tk.Label(self.frame_action, text="No possible action.")
            widget.pack(side="top", anchor="w")
            self._widgets.append([widget])

    @staticmethod
    def _define_widgets_funcgui(funcgui, frame, callback, callbackarg):
        """
        Creates a frame and widgets inside. Returns:
        callinfo: [func, StringVar1, StringVar2, ..., {hiddenvars}]
        widgets: [innerFrame, widget1, widget2, ...]
        frame: where the widgets are created. substructure will be provided
        callback: function to call when user validates his input
        callbackarg: index of guiAction. callback(callbackarg)
        """
        callinfo = {
            "func": funcgui.func,
            "args": [],
            "kwargs": dict(funcgui.hiddenvars),
        }
        widgets = []  # list of widgets, to be able to destroy later
        # create inner frame
        fr = tk.Frame(frame)
        fr.pack(side="top", anchor="w", fill=tk.X)
        widgets.append(fr)
        # validation button
        if funcgui.func is None:
            widget = tk.Label(fr, text=funcgui.textsave)
        else:
            widget = tk.Button(
                fr, text=funcgui.textsave, command=lambda j_=callbackarg: callback(j_)
            )
        widget.pack(side="left", anchor="w")
        widgets.append(widget)
        if len(funcgui.tooltiptext) > 0:
            CreateToolTip(widget, funcgui.tooltiptext)
        # list of widgets
        lookuppreviouswidget = {}
        for field in funcgui.fields:
            options = dict(field["options"])
            widgetclass = field["widgetclass"]

            # first, a Label widget for to help the user
            widget = tk.Label(fr, text=field["label"])
            widget.pack(side="left", anchor="w", fill=tk.X)
            widgets.append(widget)

            if widgetclass is None:
                # do not create any widget; purpose: show the label
                continue
            # widgetname: tranform into reference to class
            try:
                if widgetclass in ["Combobox"]:  # Combobox
                    widgetclass = getattr(ttk, widgetclass)
                else:
                    widgetclass = getattr(tk, widgetclass)
            except Exception as e:
                msg = (
                    "ERROR FuncGUI.create_widgets, cannot create widget of class {"
                    "}. Exception {} {}"
                )
                print(msg.format(widgetclass, type(e), e))
                continue

            # Frame: interpreted as to create a new line
            if widgetclass == tk.Frame:
                fr = tk.Frame(frame)
                fr.pack(side="top", anchor="w", fill=tk.X)
                widgets.append(fr)  # inner Frame
                continue  # stop there, go to next widget

            # create stringvar
            if widgetclass == tk.Checkbutton:
                stringvar = tk.BooleanVar()
                stringvar.set(bool(field["value"]))
            else:
                stringvar = tk.StringVar()
                stringvar.set(str(field["value"]))

            if field["keyword"] is None:
                callinfo["args"].append(stringvar)
            else:
                callinfo["kwargs"].update({field["keyword"]: stringvar})
            # default size if widgetclass is Entry
            if widgetclass == tk.Entry and "width" not in options:
                widthtest = int(
                    (40 - len(field["label"]) / 3 - len(funcgui.textsave) / 2)
                    / len(funcgui.fields)
                )
                width = max(8, widthtest)
                if len(stringvar.get()) < 2:
                    width = int(0.3 * width + 0.7)
                options.update({"width": width})
            # link to StringVar
            if widgetclass == tk.Checkbutton:
                options.update({"variable": stringvar})
            else:
                options.update({"textvariable": stringvar})
            # create widget
            try:
                widget = widgetclass(fr, **options)
            except Exception as e:
                print("Exception", type(e), e)
                msg = "Could not create widget {}, {}, {}"
                print(msg.format(field["label"], widgetclass.__name__, options))
                continue
            widget.pack(side="left", anchor="w")
            widgets.append(widget)
            # bind
            bind = field["bind"]
            if bind is not None:
                if bind == "beforespace":
                    widget.bind(
                        "<<ComboboxSelected>>",
                        lambda event: event.widget.set(
                            event.widget.get().split(" ")[0]
                        ),
                    )
                elif bind == "previouswidgettogglereadonly":
                    # -1 is itfuncgui, -2 its Label, previous widget is -3
                    lookuppreviouswidget.update({stringvar._name: widgets[-3]})
                    stringvar.trace_add(
                        "write",
                        lambda w, _1, _2: lookuppreviouswidget[w].config(
                            state="readonly" if int(frame.getvar(w)) else "normal"
                        ),
                    )
                else:
                    widget.bind("<<ComboboxSelected>>", bind)
            widget.bind("<Return>", lambda event, j_=callbackarg: callback(j_))
        # end of loop, return
        return callinfo, widgets

    def call_action(self, j):
        """when user validates his input for curve action"""
        self.app.store_selected_curves()
        curve = self.app.get_selected_curves(multiple=False)[0]
        graph = self.app.graph()
        info = self.list_callinfos[j]
        # retrieve function to call
        func, args, kwargs = info["func"], info["args"], dict(info["kwargs"])
        # retrieve values of user-adjustable parameters
        args = [strToVar(a.get()) for a in args]
        for key in kwargs:
            if isinstance(kwargs[key], tk.StringVar):
                kwargs[key] = strToVar(kwargs[key].get())
            elif isinstance(kwargs[key], tk.BooleanVar):
                kwargs[key] = strToVar(kwargs[key].get())
            # else: hiddenvars
        # check if func is method of Graph object and not of the Curve
        if not hasattr(func, "__self__"):
            args = [self.app.graph()] + args
            # to get call as Graph.func(graph, *args)

        def execute_func(curve, func, args, kwargs):
            # execute curve action
            graph = self.app.graph()
            try:
                res = func(*args, **kwargs)
            except Exception:
                msg = "While executing function {} with args {} kwargs {}."
                logger.error(msg.format(func, args, kwargs), exc_info=True)
                return
            if self.app.if_print_commands():
                try:
                    subject = func.__module__
                    if "graph" in subject:
                        subject = "graph"
                    elif "curve" in subject:
                        subject = "graph[" + str(curve) + "]"
                    else:
                        msg = "callAction print: subject not determined ({}, {}, {})"
                        logger.warning(msg.format(subject, func.__name__, j))
                    msg = "curve = {}.{}({})"
                    msgformat = msg.format(
                        subject, func.__name__, self.app.args_to_str(*args, **kwargs)
                    )
                    print(msgformat)
                except Exception:
                    pass  # error while doing useless output does not matter
            # where to place new Curves
            idx = curve + 1
            while idx < len(graph):  # increments idx in specific cases
                type_ = graph[idx].attr("type")
                if not type_.startswith("scatter") and not type_.startswith("errorbar"):
                    break
                idx += 1
            # insert newly created Curves
            if isinstance(res, Curve):
                self.app.call_graph_method("append", res, idx=idx)
            elif (
                isinstance(res, list)  # if list of Curves
                and np.array([isinstance(c, Curve) for c in res]).all()
            ):
                self.app.call_graph_method("append", res, idx=idx)
            elif isinstance(res, Graph):  # if returns a Graph, create a new one
                folder = self.app.get_folder()
                dpi = self.app.get_tab_properties()["dpi"]
                self.app.graph(res, title="Curve action output", folder=folder, dpi=dpi)
            elif not isinstance(res, bool) or res is not True:  # don't, only if "True"
                print("Curve action output:")
                print(res)
            # TO CHECK: if want to print anything if True, etc.

        # handling actions on multiple Curves
        to_execute = {curve: func}
        curves = self.app.get_selected_curves(multiple=True)
        if len(curves) > 1:
            func_gui0 = graph[curve].funcListGUI(graph=graph, graph_i=curve)[j]
            if not isinstance(func_gui0, FuncGUI):
                func_gui0 = FuncGUI(None, None).init_legacy(func_gui0)
            for c in curves:
                if c != curve:
                    # check that function is offered by other selected curves
                    list1 = graph[c].funcListGUI(graph=graph, graph_i=c)
                    if len(list1) > j:
                        func_gui1 = list1[j]
                        if not isinstance(func_gui1, FuncGUI):
                            func_gui1 = FuncGUI(None, None).init_legacy(func_gui1)
                        if func_gui0.is_similar(func_gui1):
                            to_execute.update({c: func_gui1.func})
        keys = list(to_execute.keys())
        keys.sort(reverse=True)
        # start execution by last one - to handle new curves
        for c in keys:
            if len(keys) > 1:
                lbl = graph[c].attr("label", "")
                if len(lbl) > 0:
                    lbl = "(" + lbl + ")"
                print("Action on Curve", c, lbl)
            execute_func(c, to_execute[c], args, kwargs)
        # after execution
        self.app.update_ui()
