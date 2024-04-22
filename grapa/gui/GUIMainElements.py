# -*- coding: utf-8 -*-
"""
@author: Romain Carron
Copyright (c) 2024, Empa, Laboratory for Thin Films and Photovoltaics,
Romain Carron
"""
import sys
import os
import copy
import numpy as np
import tkinter as tk
from tkinter import ttk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

try:
    from matplotlib.backends.backend_tkagg import (
        NavigationToolbar2TkAgg as NavigationToolbar2Tk,
    )
except ImportError:
    from matplotlib.backends.backend_tkagg import NavigationToolbar2Tk

path = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
)
if path not in sys.path:
    sys.path.append(path)
from grapa.graph import Graph
from grapa.curve import Curve
from grapa.graphIO import FILEIO_GRAPHTYPE_GRAPH
from grapa.mathModule import (
    strToVar,
    varToStr,
    roundSignificant,
    is_number,
    listToString,
)
from grapa.colorscale import Colorscale, PhotoImageColorscale
from grapa.gui.GUImisc import TextWriteable, bind_tree
from grapa.gui.GUImisc import (
    imageToClipboard,
    FrameTitleContentHide,
    FrameTitleContentHideHorizontal,
)
from grapa.gui.GUImisc import (
    EntryVar,
    OptionMenuVar,
    CheckbuttonVar,
    ComboboxVar,
    LabelVar,
)
from grapa.gui.createToolTip import CreateToolTip
from grapa.gui.GUIgraphManager import GraphsTabManager
from grapa.gui.GUIFuncGUI import FuncGUI
from grapa.interface_openbis import GrapaOpenbis


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
        self.canvasEvents = []
        # mechanism to fire events only when mouse is pressed
        self._mousePressed = False
        self.callback_notifyCanvas_registered = []
        self.createWidgets(self.frame)
        self.app.master.bind("<Control-w>", lambda e: self.closeTab())

    def updateUI(self):
        """Update plot on canvas"""
        self.fig.clear()
        self.canvas.draw()
        # to update before the updateUI() call
        self.fig.set_dpi(self.app.getTabProperties()["dpi"])
        # update graph
        #        try:
        fig, ax = Graph.plot(self.app.graph(), figAx=[self.fig, None])
        while isinstance(ax, (list, np.ndarray)) and len(ax) > 0:
            ax = ax[-1]
        self.fig, self.ax = fig, ax
        # except ValueError as e:
        #     print('Exception ValueError during GUI plot canvas update.')
        #     print('Exception', type(e), e)
        #     pass
        # draw canvas
        try:
            self.canvas.show()
        except AttributeError:
            # handles FigureCanvasTkAgg has no attribute show in later
            # versions of matplotlib
            pass
        self.canvas.draw()

    def createWidgets(self, frame):
        self.cw_graphSelector(frame)
        self.cw_canvas(frame)

    def registerCallback_notifyCanvas(self, func):
        self.callback_notifyCanvas_registered.append(func)

    def cw_graphSelector(self, frame):
        fr = tk.Frame(frame)
        fr.pack(side="top", fill=tk.X)
        # button close
        fr1 = tk.Frame(fr, width=20, height=20)
        fr1.propagate(0)
        fr1.pack(side="right", anchor="n")
        btn = tk.Button(fr1, text="\u2715", command=self.closeTab)  # 2A2F
        btn.pack(side="left", anchor="n", fill=tk.BOTH, expand=True)
        CreateToolTip(btn, "Close selected tab. Ctrl+W")
        # tabs
        defdict = {"dpi": self.app.DEFAULT_SCREENDPI}
        self.tabs = GraphsTabManager(fr, width=200, height=0, defdict=defdict)
        self.tabs.pack(side="left", fill=tk.X, expand=True)

    def cw_canvas(self, frame):
        # canvas for graph
        dpi = self.app.DEFAULT_SCREENDPI
        defsize = Graph.FIGSIZE_DEFAULT
        figsize = [defsize[0], defsize[1] * 1.15]
        # canvassize = [1.03*defsize[0]*dpi, 1.03*defsize[1]*dpi]
        self.fig = plt.figure(figsize=figsize, dpi=dpi)
        self.canvas = FigureCanvasTkAgg(self.fig, master=frame)
        self.canvas.get_tk_widget().configure(background="white")
        # possibly also configure width=canvassize[0], height=canvassize[1]
        self.canvas.get_tk_widget().pack(
            side="top", anchor="w", fill=tk.BOTH, expand=True
        )
        self.canvas.mpl_connect("resize_event", self.updateUponResizeWindow)
        try:
            self.canvas.show()
        except AttributeError:
            pass  # changes in later versions of FigureCanvasTkAgg
        self.ax = self.fig.add_subplot(111)

    def updateUponResizeWindow(self, *args):
        # updates only when main application is ready
        if self.app.initiated:
            # print('updateUponResizeWindow args', args)
            self.app.frameCentral.updateUI()  # only central part of UI

    def enableCanvasCallbacks(self):
        """restore suitable list of canvas callbacks, for datapicker"""

        def callback_pressCanvas(event):
            self._mousePressed = True
            callback_notifyCanvas(event)

        def callback_releaseCanvas(event):
            self._mousePressed = False

        def callback_notifyCanvas(event):
            if not self._mousePressed:
                return
            for func in self.callback_notifyCanvas_registered:
                func(event)
            # print('Clicked at', event.xdata, event.ydata, 'curve')

        self.disableCanvasCallbacks()
        self.canvasEvents.append(
            self.canvas.mpl_connect("button_press_event", callback_pressCanvas)
        )
        self.canvasEvents.append(
            self.canvas.mpl_connect("button_release_event", callback_releaseCanvas)
        )
        self.canvasEvents.append(
            self.canvas.mpl_connect("motion_notify_event", callback_notifyCanvas)
        )

    def disableCanvasCallbacks(self):
        """disable all registered canvas callbacks"""
        for cid in self.canvasEvents:
            self.canvas.mpl_disconnect(cid)
        self.canvasEvents.clear()

    def closeTab(self):
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
        self.createWidgets(self.frame, canvas)

    def updateUI(self):
        graph = self.app.graph()
        # to be called to update the frame:
        self.alterListGUI = graph.alterListGUI()
        # Data transform, plottype
        [idx, display, alter, typePlot] = self.identifyDataTransform()
        options = [i[0] for i in self.alterListGUI]
        self.varAlter.resetValues(options, func=self.updateDataTransform)
        for a in self.alterListGUI:
            if a[0] == display:
                self.varAlter.set(display)
                break
        self.varTypeplot.set(typePlot)
        # labels, limits
        xlim = graph.attr("xlim", ["", ""])
        ylim = graph.attr("ylim", ["", ""])
        xlim = [(x if not isinstance(x, str) and not np.isnan(x) else "") for x in xlim]
        ylim = [(y if not isinstance(y, str) and not np.isnan(y) else "") for y in ylim]
        self.varXlabel.set(varToStr(graph.attr("xlabel")))
        self.varYlabel.set(varToStr(graph.attr("ylabel")))
        self.varXlim0.set(varToStr(xlim[0]))
        self.varXlim1.set(varToStr(xlim[1]))
        self.varYlim0.set(varToStr(ylim[0]))
        self.varYlim1.set(varToStr(ylim[1]))
        self._varDPI.set(self.app.getTabProperties()["dpi"])

    def createWidgets(self, frame, canvas):
        # toolbar
        fr0 = tk.Frame(frame)
        fr0.pack(side="top", anchor="w", fill=tk.X)
        self.cw_toolbar(fr0, canvas)
        # line 1
        fr1 = tk.Frame(frame)
        fr1.pack(side="top", anchor="w", fill=tk.X)
        self.cw_line1(fr1)
        # line 2
        fr2 = tk.Frame(frame)
        fr2.pack(side="top", anchor="w", fill=tk.X)
        self.cw_line2(fr2)

    def cw_toolbar(self, frame, canvas):
        btn0 = tk.Button(frame, text="Refresh GUI", command=self.app.updateUI)
        btn0.pack(side="left", anchor="center", padx=5, pady=2)
        CreateToolTip(btn0, "Ctrl+R")
        btn1 = tk.Button(
            frame, text="Save zoom/subplots", command=self._setLimitsSubplots
        )
        btn1.pack(side="left", anchor="center", padx=5, pady=2)
        tk.Label(frame, text="   ").pack(side="left")
        frame2 = tk.Frame(frame)
        self.toolbar = NavigationToolbar2Tk(canvas, frame)
        self.toolbar.update()

    def cw_line1(self, frame):
        lbl = tk.Label(frame, text="Data transform")
        lbl.pack(side="left", anchor="n", pady=7)
        # varAlter initially empty, gets updated frequently
        self.varAlter = OptionMenuVar(frame, [""], default="")
        self.varAlter.pack(side="left", anchor="center")
        # button change graph type
        lbl = tk.Label(frame, text="   Plot type")
        lbl.pack(side="left", anchor="n", pady=7)
        plotTypeList = [
            "",
            "plot",
            "plot norm.",
            "semilogy",
            "semilogy norm.",
            "semilogx",
            "loglog",
        ]
        self.varTypeplot = OptionMenuVar(
            frame, plotTypeList, default="", func=self.updateTypeplot
        )
        self.varTypeplot.pack(side="left")
        # popup to handle text annotations
        tk.Label(frame, text="     ").pack(side="left")
        btn = tk.Button(
            frame, text="Annotations, legend and titles", command=self.popupAnnotations
        )
        btn.pack(side="left", anchor="n", pady=5)
        # screen dpi
        btn = tk.Button(frame, text="Save", command=self.setScreenDPIFromEntry)
        btn.pack(side="right", anchor="n", pady=5)
        self._varDPI = EntryVar(frame, value=self.app.DEFAULT_SCREENDPI, width=5)
        self._varDPI.pack(side="right", anchor="n", pady=8, padx=5)
        self._varDPI.bind("<Return>", lambda event: self.setScreenDPIFromEntry())
        lbl = tk.Label(frame, text="Screen dpi")
        lbl.pack(side="right", anchor="n", pady=7)

    def cw_line2(self, frame):
        self.varXlabel = EntryVar(frame, "", width=20)
        self.varYlabel = EntryVar(frame, "", width=20)
        self.varXlim0 = EntryVar(frame, "", width=8)
        self.varXlim1 = EntryVar(frame, "", width=8)
        self.varYlim0 = EntryVar(frame, "", width=8)
        self.varYlim1 = EntryVar(frame, "", width=8)
        tk.Label(frame, text="xlabel:").pack(side="left", anchor="c")
        self.varXlabel.pack(side="left", anchor="c")
        tk.Label(frame, text="   ylabel:").pack(side="left", anchor="c")
        self.varYlabel.pack(side="left", anchor="c")
        tk.Label(frame, text="   xlim:").pack(side="left", anchor="c")
        self.varXlim0.pack(side="left", anchor="c")
        tk.Label(frame, text="to").pack(side="left", anchor="c")
        self.varXlim1.pack(side="left", anchor="c")
        tk.Label(frame, text="   ylim:").pack(side="left", anchor="c")
        self.varYlim0.pack(side="left", anchor="c")
        tk.Label(frame, text="to").pack(side="left", anchor="c")
        self.varYlim1.pack(side="left", anchor="c")
        tk.Label(frame, text="   ").pack(side="left")
        btn = tk.Button(frame, text="Save", command=self.updateAttributes)
        btn.pack(side="left", anchor="c")
        tk.Label(frame, text="   ").pack(side="left")
        self.varXlabel.bind("<Return>", lambda event: self.updateAttributes())
        self.varYlabel.bind("<Return>", lambda event: self.updateAttributes())
        self.varXlim0.bind("<Return>", lambda event: self.updateAttributes())
        self.varXlim1.bind("<Return>", lambda event: self.updateAttributes())
        self.varYlim0.bind("<Return>", lambda event: self.updateAttributes())
        self.varYlim1.bind("<Return>", lambda event: self.updateAttributes())
        bt = tk.Button(frame, text="Data editor", command=self.popupDataEditor)
        bt.pack(side="right", anchor="c")

    def _setLimitsSubplots(self):
        self.app.storeSelectedCurves()  # before modifs, to prepare updateUI
        ax = self.app.getAx()
        xlim = list(ax.get_xlim())
        ylim = list(ax.get_ylim())
        a = ["left", "bottom", "right", "top", "wspace", "hspace"]
        subplots = [getattr(self.app.getFig().subplotpars, key) for key in a]
        self.app.callGraphMethod(
            "update", {"xlim": xlim, "ylim": ylim, "subplots_adjust": subplots}
        )
        self.app.updateUI()

    def updateTypeplot(self, new):
        self.app.storeSelectedCurves()  # before modifs, to prepare updateUI
        self.varTypeplot.set(new)
        self.app.callGraphMethod("update", {"typeplot": new})
        self.app.updateUI()

    def setScreenDPIFromEntry(self):
        """Updates the DPI according to the value stored in the GUI Entry"""
        self.app.storeSelectedCurves()  # before modifs, to prepare updateUI
        try:
            self.app.getTabProperties(dpi=float(self._varDPI.get()))
            self.checkValidScreenDPI()
        except Exception:  # float() conversion may have failed
            pass
        if self.app.initiated:
            self.app.updateUI()

    def checkValidScreenDPI(self):
        """Checks that the DPI value stored in tab is reasonable"""
        new = strToVar(self.app.getTabProperties()["dpi"])
        # print('checkValidScreenDPI', new)
        if new == "":
            new = self.app.DEFAULT_SCREENDPI
        if not is_number(new):
            return False
        fig = self.app.getFig()
        figSize = np.array(fig.get_size_inches())
        minMaxPx = np.array([[10, 10], [1000, 1000]])
        newMinMax = np.array(
            [max(minMaxPx[0] / figSize), min(2 * minMaxPx[1] / figSize)]
        )
        new = min(max(new, newMinMax[0]), newMinMax[1])
        new = roundSignificant(new, 2)
        # print('Set screen DPI to '+str(new)+'.')
        # print('   ', new)
        self.app.getTabProperties(dpi=new)

    def setAutoScreenDPI(self):
        """Provides a best guess for screen DPI"""
        self.app.master.update_idletasks()
        wh = [
            self.app.getCanvas().get_tk_widget().winfo_width(),
            self.app.getCanvas().get_tk_widget().winfo_height(),
        ]
        figsize = self.app.graph().attr("figsize", Graph.FIGSIZE_DEFAULT)
        dpimax = min([wh[i] / figsize[i] for i in range(2)])
        dpi = self.app.getTabProperties()["dpi"]
        new = None
        if dpi > dpimax * 1.02:  # shall reduce screen dpi
            new = np.max([10, np.round(2 * (dpimax - 3), -1) / 2])
        elif dpi < dpimax * 0.8 and dpi < 100:  # maybe can zoom in
            new = np.min([180, np.round(2 * (dpimax - 3), -1) / 2])
        # print('   ', new)
        if new is not None and new != dpi:
            self.app.getTabProperties(dpi=new)
            self.app.blinkWidget(self._varDPI, 5)
            self._varDPI.set(new)  # a bit useless, done at next updateUI...

    def updateDataTransform(self, new):
        [idx, display, alter, typePlot] = self.identifyDataTransform()
        if display == new:
            return True  # no change
        for a in self.alterListGUI:
            if new == a[0]:
                self.app.storeSelectedCurves()  # before modifs, to prepare updateUI
                self.varAlter.set(new)
                self.varTypeplot.set(a[2])
                self.app.callGraphMethod("update", {"alter": a[1], "typeplot": a[2]})
                self.app.updateUI()
        return False

    def identifyDataTransform(self):
        # retrieve current alter
        graph = self.app.graph()
        alter = graph.attr("alter")
        if alter == "":
            alter = self.alterListGUI[0][1]
        typePlot = graph.attr("typeplot")
        # find index in list of allowed alterations
        for i in range(len(self.alterListGUI)):
            if alter == self.alterListGUI[i][1]:
                return [i] + self.alterListGUI[i][0:2] + [typePlot]
        return [np.nan, "File-defined", alter, typePlot]

    def updateAttributes(self):
        """update the quick modifs, located below the graph"""
        self.app.storeSelectedCurves()  # before modifs, to prepare updateUI
        xlim = [strToVar(self.varXlim0.get()), strToVar(self.varXlim1.get())]
        ylim = [strToVar(self.varYlim0.get()), strToVar(self.varYlim1.get())]
        xlabel = strToVar(self.varXlabel.get())
        ylabel = strToVar(self.varYlabel.get())
        self.app.callGraphMethod(
            "update", {"xlabel": xlabel, "ylabel": ylabel, "xlim": xlim, "ylim": ylim}
        )
        self.app.updateUI()

    def popupAnnotations(self):
        """Open window for annotation manager"""
        from gui.GUIpopup import GuiManagerAnnotations

        win = tk.Toplevel(self.app.master)
        GuiManagerAnnotations(win, self.app.graph(), self.catchAnnotations)

    def catchAnnotations(self, dictupdate):
        self.app.storeSelectedCurves()  # before modifs, to prepare updateUI
        self.app.graph().update(dictupdate)
        self.app.updateUI()

    def popupDataEditor(self):
        """Open window for data edition"""
        # opens manager
        from gui.GUIdataEditor import GuiDataEditor

        win = tk.Toplevel(self.app.master)
        GuiDataEditor(win, self.app.graph(), self.catchDataEditor)

    def catchDataEditor(self):
        # modification of the Curve are performed within popup. Little to do
        self.app.updateUI()


class GUIFrameDataPicker:
    """
    Handles the data picker
    Contains a Frame, to be embedded in a wider application (.frame.pack())
    Some methods will call methods of application
    """

    def __init__(self, master, application, **kwargs):
        self.frame = tk.Frame(master, **kwargs)
        self.app = application
        self.varIdx = np.nan  # datapoint index on curve
        self.varCurvePreviousLabels = []
        self.createWidgets(self.frame)
        self.app.master.bind("<Control-m>", lambda e: self.savePoint())

    def updateUI(self):
        # drop-down curve list
        graph = self.app.graph()
        orderLbl = ["label", "sample", "filename"]
        values, labels = [], []
        for i in range(len(graph)):
            values.append(i)
            lbl = str(i) + " (no label)"
            for test in orderLbl:
                tmp = graph[i].attr(test)
                if tmp != "":
                    lbl = str(i) + " " + str(tmp).replace("'", "'")
                    break
            labels.append(lbl)
        if labels != self.varCurvePreviousLabels:
            self.varCurvePreviousLabels = labels
            default = self.varCurve.get()
            self.varCurve.resetValues(values, labels=labels, default=default)
        # crosshair
        self.updateCrosshair(draw=False)

    def createWidgets(self, frame):
        fr = FrameTitleContentHideHorizontal(
            frame, self.cw_title, self.cw_datapicker, default="hide", showHideTitle=True
        )
        fr.pack(side="top", fill=tk.X, anchor="w")

    def cw_title(self, frame):
        tk.Label(frame, text="Data picker").pack(side="top")

    def cw_datapicker(self, frame):
        fr0 = tk.Frame(frame)
        fr0.pack(side="top", anchor="w", fill=tk.X)
        self.cw_datapickerUp(fr0)
        fr1 = tk.Frame(frame)
        fr1.pack(side="top", anchor="w", fill=tk.X)
        self.cw_datapickerDown(fr1)

    def cw_datapickerUp(self, frame):
        tk.Label(frame, text="Click on graph").pack(side="left", anchor="c")
        tk.Label(frame, text="x").pack(side="left", anchor="c")
        self.varX = EntryVar(frame, 0, width=10, varType=tk.DoubleVar)
        self.varX.pack(side="left", anchor="c")
        tk.Label(frame, text="y").pack(side="left", anchor="c")
        self.varY = EntryVar(frame, 0, width=10, varType=tk.DoubleVar)
        self.varY.pack(side="left", anchor="c")
        self.varRestrict = CheckbuttonVar(frame, "Restrict to data", False)
        self.varRestrict.pack(side="left", anchor="c")
        tk.Label(frame, text="curve").pack(side="left", anchor="c")
        self.varCurve = OptionMenuVar(frame, [0], 0, varType=tk.IntVar)
        self.varCurve.pack(side="left", anchor="c")
        try:
            self.varCurve.var.trace_add("write", self.selectCurve)
        except AttributeError:  # IntVar has no attribute 'trace_add'
            self.varCurve.var.trace("w", self.selectCurve)
        self.varCrosshair = CheckbuttonVar(
            frame, "Crosshair", False, command=self.updateCrosshair
        )
        self.varCrosshair.pack(side="left", anchor="c")

    def cw_datapickerDown(self, frame):
        btn0 = tk.Button(
            frame, text="Create text with coordinates", command=self.createTextbox
        )
        btn0.pack(side="left", anchor="c")
        tk.Label(frame, text=" or ").pack(side="left")
        btn1 = tk.Button(frame, text="Save point", command=self.savePoint)
        btn1.pack(side="left", anchor="c")
        CreateToolTip(btn1, "Ctrl+M")
        self.varIfTransform = CheckbuttonVar(frame, "screen data", True)
        self.varIfTransform.pack(side="left", anchor="c")
        self.varIfCurveSpec = CheckbuttonVar(frame, "Curve specific", False)
        self.varIfCurveSpec.pack(side="left", anchor="c")
        # explanatory text for checkbox
        self.varExplain = LabelVar(frame, "")
        self.varExplain.pack(side="left", anchor="c")

    def selectCurve(self, *args):
        """Called when user selects another Curve in data picker"""
        c = self.varCurve.get()
        lbl = self.app.graph()[c].getDataCustomPickerXY(0, strDescription=True)
        self.varExplain.set(lbl)

    def getXY(self):
        """retrieve (and transform) data in x and y Entry."""
        # default is data in datapicker textbox
        x = self.varX.get()
        y = self.varY.get()
        attrUpd = {}
        if self.varRestrict.get():
            # if datapicker was restricted to existing data point
            graph = self.app.graph()
            c = self.varCurve.get()
            print("getXY c", type(c), c)
            if c >= 0:
                idx = self.varIdx
                alter = graph.attr("alter")
                # if user want data transform instead if raw data
                if self.varIfTransform.get():
                    # raw data is displayed in checkbox, need transform
                    if isinstance(alter, str):
                        alter = ["", alter]
                    x = graph[c].x_offsets(index=idx, alter=alter[0])[0]
                    y = graph[c].y_offsets(index=idx, alter=alter[1])[0]
                # if user want curve-specific data picker
                if self.varIfCurveSpec.get():
                    # default will be transformed & offset modified data
                    # maybe the Curve object overrides the default method ?
                    # case for CurveCf at least
                    x, y, attrUpd = graph[c].getDataCustomPickerXY(idx, alter=alter)
        return x, y, attrUpd

    def createTextbox(self):
        """Create a new text annotation on Graph"""
        self.app.storeSelectedCurves()  # before modifs, to prepare updateUI
        x, y, attrUpd = self.getXY()
        text = (
            "x: " + str(roundSignificant(x, 5)) + "\ny: " + str(roundSignificant(y, 5))
        )
        textxy = ""
        textargs = {"textcoords": "data", "xytext": [x, y], "fontsize": 8}
        self.app.callGraphMethod("addText", text, textxy, textargs=textargs)
        if not self.app.ifPrintCommands():
            print("New text annotation:", text.replace("\n", "\\n"))
        self.app.updateUI()

    def savePoint(self):
        self.app.storeSelectedCurves()  # before modifs, to prepare updateUI
        attr = {"linespec": "x", "color": "k", "_dataPicker": True}
        x, y, attrUpd = self.getXY()
        attr.update(attrUpd)
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
        self.app.updateUI()

    def updateCrosshair(self, draw=True):
        """
        draw the canvas including crosshair, except if called from updateUI
        which handles draw() by itself
        """
        # first, delete existing crosshair
        try:
            self.crosshairx.remove()
            del self.crosshairx
        except Exception:
            pass
        try:  # first delete existing crosshair
            self.crosshairy.remove()
            del self.crosshairy
        except Exception:
            pass
        # then makes new ones
        if self.varCrosshair.get():
            self.app.enableCanvasCallbacks()
            xdata = self.varX.get()
            ydata = self.varY.get()
            curve = self.varCurve.get()
            restrict = self.varRestrict.get()
            idx = self.varIdx
            graph = self.app.graph()
            alter = graph.attr("alter")
            if curve >= 0 and curve >= len(graph):
                curve = len(graph) - 1
                self.varCurve.set(curve)
            if isinstance(alter, str):
                alter = ["", alter]
            posx, posy = xdata, ydata
            if restrict and curve >= 0 and not np.isnan(idx):
                posx = graph[curve].x_offsets(index=idx, alter=alter[0])
                posy = graph[curve].y_offsets(index=idx, alter=alter[1])
                # print('crosshair', xdata, ydata, posx, posy)
            ax = self.app.getAx()
            self.crosshairx = ax.axvline(posx, 0, 1, color=[0.5, 0.5, 0.5])
            self.crosshairy = ax.axhline(posy, 0, 1, color=[0.5, 0.5, 0.5])
        else:
            self.app.disableCanvasCallbacks()
        if draw:
            self.app.getCanvas().draw()

    def eventMouseMotion(self, event):
        xdata, ydata = event.xdata, event.ydata
        self.varIdx = np.nan
        if (
            event.xdata is not None
            and self.varRestrict.get()
            and self.varCurve.get() > -1
        ):
            graph = self.app.graph()
            curve = graph[self.varCurve.get()]
            if curve is not None:
                xdata, ydata, idx = curve.getPointClosestToXY(
                    xdata, ydata, alter=graph.attr("alter")
                )
                self.varIdx = idx
        if xdata is not None:
            self.varX.set(xdata)
        if ydata is not None:
            self.varY.set(ydata)
        self.updateCrosshair()


class GUIFrameCentral:
    """
    Handles the central panel with graph, axis limits, and datapicker
    Contains a Frame, to be embedded in a wider application (.frame.pack())
    Some methods will call methods of application
    """

    def __init__(self, master, application, **kwargs):
        self.frame = tk.Frame(master, **kwargs)
        self.app = application
        self.createWidgets(self.frame)
        self.frameGraph.registerCallback_notifyCanvas(
            self.frameDataPicker.eventMouseMotion
        )

    def updateUI(self):
        self.frameGraph.updateUI()
        self.frameOptions.updateUI()
        self.frameDataPicker.updateUI()

    def createWidgets(self, frame):
        fr = tk.Frame(frame)
        fr.pack(side="bottom", fill=tk.X)
        # Canvas dor Graph
        self.frameGraph = GUIFrameCanvasGraph(frame, self.app)
        self.frameGraph.frame.pack(side="top", fill=tk.BOTH, expand=True)
        canvas = self.frameGraph.canvas
        # Datatransform, annotations, labels, limits
        self.frameOptions = GUIFrameCentralOptions(fr, self.app, canvas)
        self.frameOptions.frame.pack(side="top", fill=tk.X)
        # Datapicker
        self.frameDataPicker = GUIFrameDataPicker(fr, self.app)
        self.frameDataPicker.frame.pack(side="top", fill=tk.X)


class GUIFrameConsole:
    """
    Handles the display of current file/folder, and the output console
    Contains a Frame, to be embedded in a wider application (.frame.pack())
    Some methods will call methods of application
    """

    def __init__(self, master, application, **kwargs):
        self._consoleHeightRoll = [8, 20, 0]

        self.app = application
        self.frame = tk.Frame(master, **kwargs)
        self.createWidgets(self.frame)

    def updateUI(self):
        self.varFile.set(self._shorten(self.app.getFile()))
        self.varFolder.set(self._shorten(self.app.getFolder()))

    def createWidgets(self, frame):
        # fr = FrameTitleContentHide(frame, self.cw_linefile, self.cw_frconsole,
        #                            default='show', horizLineFrame=True,
        #                            contentpackkwargs={'expand': True})
        # fr.pack(side='top', fill=tk.X, anchor='w')
        title = tk.Frame(frame)
        self.cw_linefile(title)
        title.pack(side="top", fill=tk.X, anchor="w")
        self.content = tk.Frame(frame)
        self.cw_frconsole(self.content)
        self.content.pack(side="top", fill=tk.X, anchor="w")

    def cw_linefile(self, frame):
        # current file
        self.varFile = LabelVar(frame, value="")
        self.varFile.pack(side="left", anchor="center")
        # current folder
        self.varFolder = LabelVar(frame, value="")
        # self.varFolder.pack(side='top', anchor='w')  # DO NOT DISPLAY
        # button, horizontal line
        fr = tk.Frame(frame, width=20, height=20)
        fr.propagate(0)
        btn = tk.Button(fr, text="\u21F3")
        btn.pack(side="left", anchor="n", fill=tk.BOTH, expand=1)
        fr.pack(side="left", anchor="center")
        line = FrameTitleContentHide.frameHline(frame)
        line.pack(side="left", anchor="center", fill=tk.X, expand=1, padx=5)
        bind_tree(frame, "<Button-1>", self._changeTextHeight)

    def _changeTextHeight(self, *args):
        old = int(self._consoleHeight)
        try:
            idx = self._consoleHeightRoll.index(self._consoleHeight) + 1
            if idx == len(self._consoleHeightRoll):
                idx = 0
        except ValueError:
            idx = 0
        self._consoleHeight = self._consoleHeightRoll[idx]
        self.console.configure(height=self._consoleHeight)
        if self._consoleHeight == 0:
            self.content.pack_forget()
        elif old == 0:  # if was hidden, need to .pack() it again
            self.content.pack(
                side="top", fill=tk.X, anchor="w"
            )  # expand=True, fill=tk.X)

    def cw_frconsole(self, frame):
        frame.columnconfigure(0, weight=1)
        # console
        self._consoleHeight = self._consoleHeightRoll[0]
        self.console = TextWriteable(frame, wrap="word", height=self._consoleHeight)
        scroll_y = tk.Scrollbar(frame, orient="vertical", command=self.console.yview)
        scroll_y.grid(row=0, column=1, sticky=tk.E + tk.N + tk.S)
        self.console.grid(row=0, column=0, sticky=tk.W + tk.E)
        self.console.configure(yscrollcommand=scroll_y.set)

    def _shorten(self, string):
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
        self.createWidgets(self.frame)
        self.bindKeystrokes()

    def bindKeystrokes(self):
        """Bind keystrokes to the behavior of the main application"""
        frame = self.app.master
        frame.bind("<Control-s>", lambda e: self.saveGraph())
        frame.bind("<Control-Shift-S>", lambda e: self.saveGraphAs())
        frame.bind("<Control-o>", lambda e: self.openFile())
        frame.bind("<Control-Shift-O>", lambda e: self.mergeFile())
        # too larges chances to mess up with that one
        # self.master.bind('<Control-v>', lambda e: self.openClipboard())
        frame.bind("<Control-Shift-V>", lambda e: self.mergeClipboard())
        frame.bind("<Control-Shift-N>", lambda e: self.insertCurveEmpty())

    def createWidgets(self, frame):
        self.showHide = FrameTitleContentHide(
            frame,
            None,
            None,
            default="show",
            layout="vertical",
            showHideTitle=True,
            createButtons=False,
            horizLineFrame=False,
        )
        self.showHide.pack(side="top", fill=tk.BOTH, expand=True)
        self.cw_allHide(self.showHide.getFrameTitle())
        self.cw_allShow(self.showHide.getFrameContent())
        self.showHide.getFrameTitle().pack_forget()

    def cw_allHide(self, frame):
        fr, _ = self.showHide.createButton(frame, symbol="\u25B6", size="auto")
        fr.pack(side="top", anchor="w")
        canvas = tk.Canvas(frame, width=20, height=40)
        canvas.pack(side="top", anchor="w")
        canvas.create_text(
            6, 40, text="Menu", angle=90, anchor="w", font=self.app.fonts["bold"]
        )

    def cw_allShow(self, frame):
        # top inner frame
        self.cw_openMerge(frame)
        self.cw_save(frame)
        fr = tk.Frame(frame)
        fr.pack(side="bottom", fill="both", expand=True)
        self.cw_scripts(fr)
        self.cw_bottom(fr)

    def cw_sectionTitle(self, frame, title):
        fr = tk.Frame(frame)
        fr.pack(side="top", fill=tk.X)
        lbl = tk.Label(fr, text=title, font=self.app.fonts["bold"])
        lbl.pack(side="left", anchor="center")
        hline = FrameTitleContentHide.frameHline(fr)
        hline.pack(side="left", anchor="center", fill=tk.X, expand=1, padx=5)
        return fr

    def cw_openMerge(self, frame):
        # Section open & merge
        fr = self.cw_sectionTitle(frame, "Open or merge files")
        fr_, _ = self.showHide.createButton(fr, symbol="\u25C1", size="auto")
        fr_.pack(side="right", anchor="center")
        # buttons
        fr = tk.Frame(frame)
        fr.pack(side="top", fill=tk.X, padx=10)
        tk.Label(fr, text="Open").grid(sticky="w", row=0, column=0)
        of = tk.Button(fr, text="File  ", command=self.openFile)
        oc = tk.Button(fr, text="Clipboard", command=self.openClipboard)
        of.grid(sticky="w", row=0, column=1, padx=5)
        oc.grid(sticky="w", row=0, column=2)
        tk.Label(fr, text="Merge with").grid(sticky="w", column=0, row=1)
        mf = tk.Button(fr, text="File  ", command=self.mergeFile)
        mc = tk.Button(fr, text="Clipboard", command=self.mergeClipboard)
        mf.grid(sticky="w", column=1, row=1, padx=5)
        mc.grid(sticky="w", column=2, row=1)
        CreateToolTip(of, "Ctrl+O")
        CreateToolTip(mf, "Ctrl+Shift+O")
        CreateToolTip(mc, "Ctrl+Shift+V")
        # Misc - open folders, etc
        fr = tk.Frame(frame)
        fr.pack(side="top", fill=tk.X, padx=10)
        u = tk.Button(fr, text="Open all in folder", command=self.openFolder)
        # u.grid(column=1, row=1, sticky='w')
        u.pack(side="left")
        self.varOpenSubfolders = CheckbuttonVar(fr, "subfolders", 0)
        # self.varOpenSubfolders.grid(column=2, row=1)
        self.varOpenSubfolders.pack(side="left")
        # new curve, close
        fr = tk.Frame(frame)
        fr.pack(side="top", fill=tk.X, padx=10)
        v = tk.Button(fr, text="Insert", command=self.insertCurveEmpty)
        # v.grid(column=1, row=2, sticky='w')
        v.pack(side="left")
        tk.Label(fr, text="new empty Curve").pack(side="left")
        explain = "For creating a new subplot, inset, image etc. Ctrl+Shift+N"
        CreateToolTip(u, "Open all files in a given folder")
        CreateToolTip(v, explain)

        ### Grapa Openbis integration
        fr = tk.Frame(frame)
        fr.pack(side="top", fill=tk.X, padx=10)
        grapaopenbis = GrapaOpenbis(self.app)
        grapaopenbis.create_widgets(fr)
        ### End of Grapa Openbis integration

    def cw_save(self, frame):
        fr = tk.Label(frame, text="")
        fr.pack(side="top")
        # Section Save title
        self.cw_sectionTitle(frame, "Save data & graph")
        # Buttons
        fr = tk.Frame(frame)
        fr.pack(side="top", fill=tk.X, padx=10)
        s_ = tk.Button(fr, text="Save", command=self.saveGraph)
        s_.pack(side="left", anchor="n")
        sa = tk.Button(fr, text="Save as...", command=self.saveGraphAs)
        sa.pack(side="left", anchor="n", padx=5)
        sc = tk.Button(fr, text="Copy image", command=self.saveImageToClipboard)
        sc.pack(side="left", anchor="n")
        CreateToolTip(s_, "Ctrl+S")
        CreateToolTip(sa, "Ctrl+Shift+S")
        # options
        fr = tk.Frame(frame)
        fr.pack(side="top", fill=tk.X, padx=10)
        self.varSaveScreen = CheckbuttonVar(fr, "Screen data (better not)", 0)
        self.varSaveScreen.pack(side="top", anchor="w", padx=5)
        self.varSaveSepara = CheckbuttonVar(fr, "Keep separated x columns", 0)
        self.varSaveSepara.pack(side="top", anchor="w", padx=5)

    def cw_scripts(self, frame):
        # Section Scripts
        fr = tk.Label(frame, text="")
        fr.pack(side="top")
        self.cw_sectionTitle(frame, "Data processing scripts")
        self.cw_scripts_JVfit(frame)
        self.cw_scripts_JVsummary(frame)
        self.cw_scripts_CVCdJscVoc(frame)

    def cw_scripts_JVfit(self, frame):
        # Section Scripts JV
        lbl = tk.Label(frame, text="JV curves (files in 1 folder):")
        lbl.pack(side="top", anchor="w", padx=0, pady=2)
        # diode weights
        fr = tk.Frame(frame)
        fr.pack(side="top", fill=tk.X, padx=10)
        lbl = tk.Label(fr, text="Fit weight diode region")
        lbl.pack(side="left", anchor="n")
        self.varJVDiodeweight = EntryVar(fr, "5", width=6)
        self.varJVDiodeweight.pack(side="left", anchor="n")
        CreateToolTip(lbl, "1 neutral, 10 increased weight")
        CreateToolTip(self.varJVDiodeweight, "1 neutral, 10 increased weight")
        # buttons
        btn0 = tk.Button(
            frame, text="Fit cell-by-cell", command=self.scriptFitJVCombined
        )
        btn1 = tk.Button(
            frame, text="Fit curves separately", command=self.scriptFitJVAll
        )
        btn0.pack(side="top", anchor="w", padx=10)
        btn1.pack(side="top", anchor="w", padx=10)

    def cw_scripts_JVsummary(self, frame):
        lbl = tk.Label(frame, text="Operations on JV summaries:")
        btn0 = tk.Button(
            frame, text="JV sample maps (1 file)", command=self.scriptJVSampleMaps
        )
        btn1 = tk.Button(
            frame, text="Boxplots (files in 1 folder)", command=self.scriptJVBoxplots
        )
        lbl.pack(side="top", anchor="w", padx=0, pady=2)
        btn0.pack(side="top", anchor="w", padx=10)
        btn1.pack(side="top", anchor="w", padx=10)

    def cw_scripts_CVCdJscVoc(self, frame):
        lbl = tk.Label(frame, text="C-V, C-f, Jsc-Voc data processing")
        lbl.pack(side="top", anchor="w", padx=0, pady=2)
        # CV
        fr = tk.Frame(frame)
        fr.pack(side="top", fill=tk.X, padx=10)
        btn = tk.Button(fr, text="C-V (1 folder)", command=self.scriptCV)
        btn.pack(side="left", pady=2)
        lbl = tk.Label(fr, text="ROI")
        lbl.pack(side="left", pady=2)
        from grapa.datatypes.curveCV import CurveCV

        valuedefCV = listToString(CurveCV.CST_MottSchottky_Vlim_def)
        self.varCVROI = EntryVar(fr, valuedefCV, width=10)
        self.varCVROI.pack(side="left", pady=2)
        # Cf
        btn0 = tk.Button(frame, text="C-f (1 folder)", command=self.scriptCf)
        btn0.pack(side="top", anchor="w", padx=10)
        # Jsc-Voc
        fr = tk.Frame(frame)
        fr.pack(side="top", fill=tk.X, padx=10)
        btn = tk.Button(fr, text="Jsc-Voc (1 file)", command=self.scriptJscVoc)
        btn.pack(side="left", pady=2)
        lbl = tk.Label(fr, text="Jsc min")
        lbl.pack(side="left", pady=2)
        from grapa.datatypes.curveJscVoc import CurveJscVoc

        valuedefVocJsc = str(CurveJscVoc.CST_Jsclim0)
        self.varJscVocROI = EntryVar(fr, valuedefVocJsc, width=6)
        self.varJscVocROI.pack(side="left", pady=2)
        tooltiplbl = "fit range of interest (min value or range), in mA/cm2"
        CreateToolTip(lbl, tooltiplbl)
        CreateToolTip(self.varJscVocROI, tooltiplbl)
        # Correlation plots - e.g. SCAPS
        lbl = tk.Label(frame, text="Correlations")
        lbl.pack(side="top", anchor="w", padx=0, pady=2)
        fr = tk.Frame(frame)
        fr.pack(side="top", fill=tk.X, padx=10)
        btn = tk.Button(fr, text="(1 file)", command=self.scriptCorrelations)
        btn.pack(side="left", pady=2)
        tk.Label(fr, text="e.g. SCAPS batch result").pack(side="left")

    def cw_bottom(self, frame):
        fr0 = tk.Frame(frame)
        fr0.pack(side="bottom", fill=tk.X)
        # bottom section
        self.cw_sectionTitle(fr0, "")
        # contentkwargs
        fr = tk.Frame(fr0)
        fr.pack(side="top", fill=tk.X)
        self.varPrintCommands = CheckbuttonVar(
            fr,
            "Commands in console",
            False,
            command=lambda: print("Commands printing is not fully tested yet"),
        )
        self.varPrintCommands.pack(side="left", anchor="center")
        btn = tk.Button(fr, text="QUIT", fg="red", command=self.quitMain)
        btn.pack(side="right", anchor="n", pady=2, padx=5)

    # methods
    def openFile(self):
        """Open a file chosen by user"""
        file = self.app.promptFile(multiple=True)
        if file != "" and file is not None:
            self.app.openFile(file)

    def mergeFile(self):
        """Merge data with file chosen by user"""
        file = self.app.promptFile(multiple=True)
        if file != "" and file is not None:
            self.app.mergeGraph(file)

    def openFolder(self):
        """Open all files in a folder given by user"""
        folder = self.app.promptFolder()
        if folder is not None and folder != "":
            files = self._listFilesinFolder(folder)
            self.app.openFile(files)

    def openClipboard(self):
        """Retrieve the content of clipboard and create a graph with it"""
        folder = self.app.getFolder()
        dpi = self.app.getTabProperties()["dpi"]
        tmp = self.app.getClipboard()
        graph = Graph(tmp, {"isfilecontent": True}, **self.app.newGraphKwargs)
        print("Import data from clipboard (" + str(len(graph)) + " Curves found).")
        self.app.graph(graph, title="from clipboard", folder=folder, dpi=dpi)
        # updateUI will be triggered by change in tabs

    def mergeClipboard(self):
        """Retrieves the content of clipboard, and appends it to the graph"""
        tmp = self.app.getClipboard()
        graph = Graph(tmp, {"isfilecontent": True})
        print("Import data from clipboard (" + str(len(graph)) + " Curves found).")
        self.app.mergeGraph(graph)

    def saveGraph(self, filename=""):
        """Saves the graph (image & data) with same filename as last time"""
        if self.app.graph().attr("meastype") not in ["", FILEIO_GRAPHTYPE_GRAPH]:
            print(
                "ERROR when saving data: are you sure you are not",
                "overwriting a graph file?",
            )
            print(
                "Click on Save As... to save the current graph (or delete",
                'attribute "meastype").',
            )
            self.saveGraphAs(filesave="")
        else:
            filesave = self.app.getFile()
            self.saveGraphAs(filesave=filesave)

    def saveGraphAs(self, filesave=""):
        """saves the graph to image + data file, asks for a new filename"""
        # TODO
        defext = ""
        if filesave == "":
            defext = copy.copy(self.app.graph().config("save_imgformat", ".png"))
            if isinstance(defext, list):
                defext = defext[0]
            filesave = self.app.promptFile(type="save", defaultextension=defext)
        if filesave is None or filesave == "":
            # asksaveasfile return `None` if dialog closed with "cancel".
            return
        # retrive info from GUI
        saveAltered = self.varSaveScreen.get()
        ifCompact = not self.varSaveSepara.get()
        # some checks to avoid erasig something important
        filesave, fileext = os.path.splitext(filesave)
        fileext = fileext.lower()
        forbiddenExt = ["py", "txt", ".py", ".txt"]
        for ext in forbiddenExt:
            fileext = fileext.replace(ext, "")
        if fileext == defext:
            fileext = ""
        self.app.saveGraph(
            filesave, fileext=fileext, saveAltered=saveAltered, ifCompact=ifCompact
        )

    def _listFilesinFolder(self, folder):
        # returns a list with all files in folder. Look for subfolders status
        subfolders = self.varOpenSubfolders.get()
        nMax = 1000
        out = []
        if subfolders:
            for root, subdirs, files in os.walk(folder):
                for file in files:
                    fileName, fileExt = os.path.splitext(file)
                    if len(out) < nMax:
                        out.append(str(os.path.join(root, file)))
        else:  # not subfolders
            for file in os.listdir(folder):
                fileName, fileExt = os.path.splitext(file)
                if os.path.isfile(os.path.join(folder, file)) and len(out) < nMax:
                    out.append(str(os.path.join(folder, file)))
        return out

    def insertCurveEmpty(self):
        """Append an empty Curve in current Graph"""
        curve = Curve([[0], [np.nan]], {})
        self.app.insertCurveToGraph(curve)

    def saveImageToClipboard(self):
        """Copy an image of current graph into the clipboard"""
        return imageToClipboard(self.app.graph())

    def scriptFitJVCombined(self, groupCell=True):
        """
        Script JV process, grouped by cells
        - groupCell: True to group results by cell, False for independent
          processing for each cell
        """
        from grapa.scripts.script_processJV import processJVfolder

        folder = self.app.promptFolder()
        if folder != "":
            print("... Processing folder ...")
            weight = strToVar(self.varJVDiodeweight.get())
            ngkw = self.app.newGraphKwargs
            graph = processJVfolder(
                folder,
                ylim=(-np.inf, np.inf),
                groupCell=groupCell,
                fitDiodeWeight=weight,
                newGraphKwargs=ngkw,
            )
            self.app.openFile(graph)

    def scriptFitJVAll(self):
        """Script JV process, each file indepedently"""
        self.scriptFitJVCombined(groupCell=False)

    def scriptJVSampleMaps(self):
        """Script JV sample maps"""
        from grapa.scripts.script_processJV import processSampleCellsMap

        file = self.app.promptFile()
        if file is not None and file != "":
            print("...creating sample maps...")
            ngkw = self.app.newGraphKwargs
            filelist = processSampleCellsMap(file, newGraphKwargs=ngkw)
            if len(filelist) > 0:
                graph = Graph(filelist[-1])
                self.app.openFile(graph)

    def scriptJVBoxplots(self):
        """Script boxplots"""
        from grapa.scripts.script_JVSummaryToBoxPlots import JVSummaryToBoxPlots

        folder = self.app.promptFolder()
        if folder is not None and folder != "":
            print("...creating boxplots...")
            tmp = JVSummaryToBoxPlots(
                folder=folder,
                exportPrefix="boxplots_",
                replace=[],
                silent=True,
                newGraphKwargs=self.app.newGraphKwargs,
            )
            self.app.openFile(tmp)

    def scriptJscVoc(self):
        """Script JscVoc"""
        from grapa.scripts.script_processJscVoc import script_processJscVoc

        file = self.app.promptFile()
        if file is not None and file != "":
            ROIJsclim = strToVar(self.varJscVocROI.get())
            ngkw = self.app.newGraphKwargs
            graph = script_processJscVoc(file, ROIJsclim=ROIJsclim, newGraphKwargs=ngkw)
            self.app.openFile(graph)

    def scriptCV(self):
        """Script CV"""
        from grapa.scripts.script_processCVCf import script_processCV

        folder = self.app.promptFolder()
        if folder is not None and folder != "":
            ROIfit = strToVar(self.varCVROI.get())
            graph = script_processCV(
                folder, ROIfit=ROIfit, newGraphKwargs=self.app.newGraphKwargs
            )
            self.app.openFile(graph)

    def scriptCf(self):
        """Script Cf"""
        from grapa.scripts.script_processCVCf import script_processCf

        folder = self.app.promptFolder()
        if folder is not None and folder != "":
            ngkw = self.app.newGraphKwargs
            graph = script_processCf(folder, newGraphKwargs=ngkw)
            self.app.openFile(graph)

    def scriptCorrelations(self):
        """Script processing of SCAPS output IV, CV, Cf QE files"""
        from grapa.scripts.script_correlations import process_file as corr_process_file

        file = self.app.promptFile()
        if file is not None and file != "":
            ngkw = self.app.newGraphKwargs
            graph = corr_process_file(file, newGraphKwargs=ngkw)
            self.app.openFile(graph)

    def quitMain(self):
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
        self.createWidgets(self.frame)

    def updateUI(self):
        pass

    def createWidgets(self, frame):
        fr = FrameTitleContentHide(
            frame,
            self.cw_title,
            self.cw_content,
            default="hide",
            contentkwargs={"padx": 10},
        )
        fr.pack(side="top", fill=tk.X, anchor="w")

    def cw_title(self, frame):
        lbl = tk.Label(frame, text="Template & Colorize", font=self.app.fonts["bold"])
        lbl.pack(side="left")

    def cw_content(self, frame):
        self.cw_content_template(frame)
        self.cw_content_colors(frame)

    def cw_content_template(self, frame):
        # templates
        fr = tk.Frame(frame)
        fr.pack(side="top", fill=tk.X, anchor="w", pady=5)
        btn0 = tk.Button(fr, text="Load & apply template", command=self.loadTemplate)
        btn0.pack(side="left", anchor="n")
        self.varTplCrvProp = CheckbuttonVar(fr, "also Curves properties", 1)
        self.varTplCrvProp.pack(side="left")
        btn1 = tk.Button(fr, text="Save template", command=self.saveTemplate)
        btn1.pack(side="right", anchor="n")

    def cw_content_colors(self, frame):
        # line 1
        fr = tk.Frame(frame)
        fr.pack(side="top", fill=tk.X, anchor="w")
        btn0 = tk.Button(fr, text="Colorize", command=self.colorizeGraph)
        btn0.pack(side="left")
        self.varColEmpty = CheckbuttonVar(fr, "repeat if no label  ", False)
        self.varColEmpty.pack(side="left")
        self.varColInvert = CheckbuttonVar(fr, "invert", False)
        self.varColInvert.pack(side="left")
        self.varColAvoidWhite = CheckbuttonVar(fr, "avoid white", False)
        self.varColAvoidWhite.pack(side="left")
        self.varColCurveSelect = CheckbuttonVar(fr, "curve selection", False)
        self.varColCurveSelect.pack(side="left")
        # line 2
        fr = tk.Frame(frame)
        fr.pack(side="top", anchor="w")
        self.varColChoice = EntryVar(fr, "", width=70)
        self.varColChoice.pack(side="left")
        # line 3
        fr = tk.Frame(frame)
        fr.pack(side="top", anchor="w")
        self.cw_content_colSamples(fr)

    def cw_content_colSamples(self, frame):
        nPerLine = 12
        self.colorList = Colorscale.GUIdefaults(**self.app.newGraphKwargs)
        # need to keep reference of images, otherwise tk.Button loose it
        self._colImgs = [None] * len(self.colorList)
        width, height = 30, 15
        j = 0  # for display purpose, number of widgets actually created
        for i in range(len(self.colorList)):
            self._colImgs[i] = PhotoImageColorscale(width=width, height=height)
            try:
                self._colImgs[i].fillColorscale(self.colorList[i])
            except ValueError:
                # if python does not recognize values (e.g. inferno, viridis)
                continue  # does NOT create widget
            widget = tk.Button(
                frame,
                image=self._colImgs[i],
                command=lambda i_=i: self._setColChoice(i_),
            )
            widget.grid(column=int(j % nPerLine), row=int(np.floor(j / nPerLine)))
            j += 1

    def loadTemplate(self):
        """load and apply template on current Graph"""
        file = self.app.promptFile()
        if file != "" and file is not None:
            print("Open template file:", file)
            self.app.storeSelectedCurves()  # before modifs to prepare updateUI
            template = Graph(file, complement={"label": ""})
            alsoCurves = self.varTplCrvProp.get()
            self.app.graph().applyTemplate(template, alsoCurves=alsoCurves)
            self.app.updateUI()

    def saveTemplate(self):
        """save template file from current Graph"""
        file = self.app.promptFile(type="save", defaultextension=".txt")
        if file is not None and file != "":
            file, fileext = os.path.splitext(file)
            fileext = fileext.replace("py", "")
            self.app.graph().export(filesave=file, ifTemplate=True)
            # no need to refresh UI

    def colorizeGraph(self):
        self.app.storeSelectedCurves()  # before modifs, to prepare updateUI
        col = strToVar(self.varColChoice.get())
        if len(col) == 0:
            col = self.colorList[0].getColorScale()
            self._setColChoice(0)
        invert = self.varColInvert.get()
        kwargs = {
            "avoidWhite": self.varColAvoidWhite.get(),
            "sameIfEmptyLabel": self.varColEmpty.get(),
        }
        if self.varColCurveSelect.get():
            curves = self.app.getSelectedCurves(multiple=True)
            if len(curves) > 0 and curves[0] >= 0:
                # if no curve is selected, colorize all curves
                kwargs.update({"curvesselection": curves})
        try:
            colorscale = Colorscale(col, invert=invert)
            self.app.graph().colorize(colorscale, **kwargs)
        except ValueError as e:
            # error to be printed in GUI console, and not hidden in terminal
            print("ValueError colorizeGraph:", e)
        if self.app.ifPrintCommands():  # to explicit creation of colorscale
            print(
                "colorscale = Colorscale(" + str(col) + ", invert=" + str(invert) + ")"
            )
            print(
                "graph.colorize(colorscale, "
                + ", ".join(["{}={!r}".format(k, v) for k, v in kwargs.items()])
                + ")"
            )
        self.app.updateUI()

    def _setColChoice(self, i):
        # reads variable (Colorscale object), extract colors (np.array),
        # converts into string
        # print('_setColChoice', i, self.colorList[i].getColorScale())
        if isinstance(self.colorList[i].getColorScale(), str):
            scale = self.colorList[i].getColorScale()  # eg. viridis
        else:
            lst = []
            for elem in self.colorList[i].getColorScale():
                if not isinstance(elem, str):
                    toStr = [str(nb if not nb.is_integer() else int(nb)) for nb in elem]
                    lst.append("[" + ",".join(toStr) + "]")
                    print("_setColChoice, elem,", elem)
                else:
                    lst.append("'" + elem + "'")
            scale = "[" + ", ".join(lst) + "]"
        self.varColChoice.set(scale)


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
        self.createWidgets(self.frame)
        # bind keys
        self.app.master.bind("<Control-Delete>", lambda e: self.deleteCurve())
        self.app.master.bind("<Control-Shift-C>", lambda e: self.copyCurveToClipboard())
        self.app.master.bind("<Control-h>", lambda e: self.showHideCurve())

    def updateUI(self):
        # ShowHide: handled by Observable
        # cast curve optionmenu: handled by Observable
        pass

    def createWidgets(self, frame):
        fr = FrameTitleContentHide(frame, self.cw_title, self.cw_content)
        fr.pack(side="top", fill=tk.X, anchor="w")

    def cw_title(self, frame):
        lbl = tk.Label(frame, text="Actions on Curves", font=self.app.fonts["bold"])
        lbl.pack(side="left", anchor="n")

    def cw_content(self, frame):
        def toGrid(row, column, func, **kwargs):
            kw = {"sticky": "w"}
            kw.update(kwargs)
            fr = tk.Frame(frame)
            fr.grid(row=row, column=column, **kw)
            func(fr)

        toGrid(0, 0, self.cw_reorder, padx=5)
        toGrid(1, 0, self.cw_delete, padx=5)
        toGrid(2, 0, self.cw_duplicate, padx=5)
        toGrid(3, 0, self.cw_showHide, padx=5)
        toGrid(0, 1, self.cw_clipboard)
        toGrid(1, 1, self.cw_cast)
        toGrid(2, 1, self.cw_quickAttr)
        toGrid(3, 1, self.cw_labelReplace)

    def cw_reorder(self, frame):
        tk.Label(frame, text="Reorder").pack(side="left")
        b0 = tk.Button(frame, text="\u21E7", command=self.shiftCurveTop)
        b0.pack(side="left", padx=1)
        b1 = tk.Button(frame, text="\u21D1 Up", command=self.shiftCurveUp)
        b1.pack(side="left", padx=1)
        b2 = tk.Button(frame, text="\u21D3 Down", command=self.shiftCurveDown)
        b2.pack(side="left", padx=1)
        b3 = tk.Button(frame, text="\u21E9", command=self.shiftCurveBottom)
        b3.pack(side="left", padx=1)

    def cw_delete(self, frame):
        tk.Label(frame, text="Delete Curve").pack(side="left")
        b0 = tk.Button(frame, text="Curve", command=self.deleteCurve)
        b0.pack(side="left", padx=3)
        b1 = tk.Button(frame, text="All hidden", command=self.deleteCurvesHidd)
        b1.pack(side="left")
        CreateToolTip(b0, "Ctrl+Delete")

    def cw_duplicate(self, frame):
        tk.Label(frame, text="Duplicate Curve").pack(side="left")
        btn = tk.Button(frame, text="Duplicate", command=self.duplicateCurve)
        btn.pack(side="left")

    def cw_showHide(self, frame):
        def updateVarSH(curve, key):
            if not isinstance(curve, Curve):
                val = "Show Curve"  # -1 if no curve selected
            else:
                val = "Show Curve" if curve.isHidden() else "Hide Curve"
            self.varShowHide.set(val)

        self.app.observables["focusTree"].register(updateVarSH)
        self.varShowHide = tk.StringVar()
        self.varShowHide.set("Show Curve")
        b0 = tk.Button(frame, textvariable=self.varShowHide, command=self.showHideCurve)
        b0.pack(side="left")
        b1 = tk.Button(frame, text="All", command=self.showHideAll)
        b1.pack(side="left", padx="5")
        b2 = tk.Button(frame, text="Invert", command=self.showHideInvert)
        b2.pack(side="left")
        CreateToolTip(b0, "Ctrl+H")

    def cw_clipboard(self, frame):
        tk.Label(frame, text="Copy to clipboard").pack(side="left")
        b0 = tk.Button(frame, text="Curve", command=self.copyCurveToClipboard)
        b0.pack(side="left", padx="5")
        b1 = tk.Button(frame, text="Graph", command=self.copyGraphToClipboard)
        b1.pack(side="left")
        vals = ["raw", "with properties", "screen data", "screen data, prop."]
        self.varClipboardOpts = OptionMenuVar(frame, vals, default="options", width=6)
        self.varClipboardOpts.pack(side="left")
        CreateToolTip(b0, "Ctrl+Shift+C")

    def cw_cast(self, frame):
        def updateVarCast(curve, key):
            # update GUI cast action: empy menu and refill it
            self.varCastCurve["menu"].delete(0, "end")
            self.varCastCurve.set("")
            castList = []
            if isinstance(curve, Curve):
                castList = curve.castCurveListGUI(onlyDifferent=False)
                values = [cast[0] for cast in castList]
                default = curve.classNameGUI()
                for key in self.CASTCURVERENAMEGUI:
                    values = [
                        v.replace(key, self.CASTCURVERENAMEGUI[key]) for v in values
                    ]
                    if default == key:
                        default = self.CASTCURVERENAMEGUI[key]
                self.varCastCurve.resetValues(values, default=default)

        self.app.observables["focusTree"].register(updateVarCast)
        tk.Label(frame, text="Change Curve type").pack(side="left")
        self.varCastCurve = OptionMenuVar(frame, [""], "")
        self.varCastCurve.pack(side="left", padx="2")
        tk.Button(frame, text="Save", command=self.castCurve).pack(side="left")

    def cw_quickAttr(self, frame):
        def updateQA(curve, key):
            if isinstance(curve, Curve):
                self.varQALabel.set(varToStr(curve.attr("label")))
                self.varQAColor.set(varToStr(curve.attr("color")))

        self.app.observables["focusTree"].register(updateQA)
        tk.Label(frame, text="Label").pack(side="left")
        self.varQALabel = EntryVar(frame, "", width=15)
        self.varQALabel.pack(side="left")
        tk.Label(frame, text="Color").pack(side="left")
        btn0 = tk.Button(frame, text="Pick", command=self.chooseColor)
        btn0.pack(side="left")
        self.varQAColor = EntryVar(frame, "", width=8)
        self.varQAColor.pack(side="left")
        btn1 = tk.Button(frame, text="Save", command=self.setQuickAttr)
        btn1.pack(side="left")
        self.varQALabel.bind("<Return>", lambda event: self.setQuickAttr())
        self.varQAColor.bind("<Return>", lambda event: self.setQuickAttr())

    def cw_labelReplace(self, frame):
        tk.Label(frame, text="Replace in labels").pack(side="left")
        self.varLabelOld = EntryVar(frame, "old string", width=10)
        self.varLabelOld.pack(side="left")
        self.varLabelNew = EntryVar(frame, "new string", width=10)
        self.varLabelNew.pack(side="left")
        btn = tk.Button(frame, text="Replace", command=self.replaceLabels)
        btn.pack(side="left")
        self.varLabelOld.bind("<Return>", lambda event: self.replaceLabels())
        self.varLabelNew.bind("<Return>", lambda event: self.replaceLabels())

    def shiftCurve(self, upDown, relative=True):
        curves = self.app.getSelectedCurves(multiple=True)
        curves.sort(reverse=(True if (upDown > 0) else False))
        selected = []
        graph = self.app.graph()
        for curve in curves:
            idx2 = upDown
            if curve == 0:
                idx2 = max(idx2, 0)
            if curve == len(graph) - 1 and relative:
                idx2 = min(idx2, 0)
            if relative:
                self.app.callGraphMethod("swapCurves", curve, idx2, relative=True)
                selected.append(curve + idx2)
            else:
                self.app.callGraphMethod("moveCurveToIndex", curve, idx2)
                selected.append(idx2)
                # print('moveCurve', curve, idx2, upDown)
                if idx2 < curve or (idx2 == curve and curve == 0):
                    upDown += 1
                elif idx2 > curve or (idx2 == curve and curve >= len(graph) - 1):
                    upDown -= 1
        for i in range(len(selected)):
            selected[i] = max(0, min(len(graph) - 1, selected[i]))
        if len(selected) > 0:
            sel = [graph[c] for c in selected]
            keys = [""] * len(sel)
            self.app.getTabProperties(
                selectionCurvesKeys=(sel, keys), focusCurvesKeys=([sel[0]], [""])
            )
        # print(self.app.getTabProperties())
        self.app.updateUI()

    def shiftCurveDown(self):
        self.shiftCurve(1)

    def shiftCurveUp(self):
        self.shiftCurve(-1)

    def shiftCurveTop(self):
        self.shiftCurve(0, relative=False)

    def shiftCurveBottom(self):
        self.shiftCurve(len(self.app.graph()) - 1, relative=False)

    def deleteCurve(self):
        """Delete the currently selected curve."""
        self.app.storeSelectedCurves()  # before modifs, to prepare updateUI
        curves = list(self.app.getSelectedCurves(multiple=True))
        curves.sort(reverse=True)
        for curve in curves:
            if not is_number(curve):
                break
                # can happen if someone presses the delete button twice in arow
            elif curve > -1:
                self.app.callGraphMethod("deleteCurve", curve)
        self.app.updateUI()

    def deleteCurvesHidd(self):
        """Delete all the hidden curves."""
        self.app.storeSelectedCurves()  # before modifs, to prepare updateUI
        toDel = []
        graph = self.app.graph()
        for c in range(len(graph)):
            if graph[c].isHidden():
                toDel.append(c)
        toDel.sort(reverse=True)
        for c in toDel:
            self.app.callGraphMethod("deleteCurve", c)
        self.app.updateUI()

    def duplicateCurve(self):
        """Duplicate the currently selected curve."""
        self.app.storeSelectedCurves()  # before modifs, to prepare updateUI
        curves = list(self.app.getSelectedCurves(multiple=True))
        curves.sort(reverse=True)
        selected = []
        for curve in curves:
            if not is_number(curve):
                # can happen if someone presses the delete button twice in arow
                break
            if curve > -1:
                self.app.callGraphMethod("duplicateCurve", curve)
                selected = [s + 1 for s in selected]
                selected.append(curve)
        self.app.updateUI()

    def showHideCurve(self):
        self.app.storeSelectedCurves()
        curves = self.app.getSelectedCurves(multiple=True)
        for curve in curves:
            if curve > -1 and curve < len(self.app.graph()):
                self.app.callCurveMethod(curve, "swapShowHide")
        self.app.updateUI()

    def showHideAll(self):
        graph = self.app.graph()
        if len(graph) > 0:
            self.app.storeSelectedCurves()  # before modifs to prepare updateUI
            new = "" if graph[0].isHidden() else "none"
            for curve in graph:
                curve.update({"linestyle": new})
            if self.app.ifPrintCommands():
                print("for curve in graph:")
                print("    curve.update({'linestyle': '" + new + "'})")
            self.app.updateUI()

    def showHideInvert(self):
        self.app.storeSelectedCurves()  # before modifs, to prepare updateUI
        for curve in self.app.graph():
            curve.swapShowHide()
        if self.app.ifPrintCommands():
            print("for curve in graph:")
            print("    curve.swapShowHide()")
        self.app.updateUI()

    def copyCurveToClipboard(self):
        graph = self.app.graph()
        curves = self.app.getSelectedCurves(multiple=True)
        if len(curves) == 0:
            return
        content = ""
        opts = self.varClipboardOpts.get()
        ifAttrs = "prop" in opts
        ifTrans = "screen" in opts
        # print("opts", opts, ifAttrs, ifTrans)
        if not ifAttrs:
            labels = [varToStr(graph[c].attr("label")) for c in curves]
            content += "\t" + "\t\t\t".join(labels) + "\n"
        else:
            keys = []
            for c in curves:
                for key in graph[c].getAttributes():
                    if key not in keys:
                        keys.append(key)
            keys.sort()
            for key in keys:
                labels = [varToStr(graph[c].attr(key)) for c in curves]
                content += key + "\t" + "\t\t\t".join(labels) + "\n"
        data = [graph.getCurveData(c, ifAltered=ifTrans) for c in curves]
        length = max([d.shape[1] for d in data])
        for le in range(length):
            tmp = []
            for d in data:
                if le < d.shape[1]:
                    tmp.append("\t".join([str(e) for e in d[:, le]]))
                else:
                    tmp.append("\t".join([""] * d.shape[0]))
            content += "\t\t".join(tmp) + "\n"
        self.app.setClipboard(content)

    def copyGraphToClipboard(self):
        opts = self.varClipboardOpts.get()
        ifAttrs = "prop" in opts
        ifTrans = "screen" in opts
        graph = self.app.graph()
        data = graph.export(
            ifClipboardExport=True, ifOnlyLabels=(not ifAttrs), saveAltered=ifTrans
        )
        self.app.setClipboard(data)

    def castCurve(self):
        graph = self.app.graph()
        curves = self.app.getSelectedCurves(multiple=True)
        newType = self.varCastCurve.get()
        for key in self.CASTCURVERENAMEGUI:
            if newType == self.CASTCURVERENAMEGUI[key]:
                newType = key
        selected = []
        for curve in curves:
            if curve > -1 and curve < len(graph):
                test = self.app.callGraphMethod("castCurve", newType, curve)
                selected.append(curve)
                if not test:
                    print("castCurve impossible.")
            else:
                print("castCurve impossible (", newType, curve, ")")
        if len(selected) > 0:
            sel = [graph[c] for c in selected]
            keys = [""] * len(sel)
            self.app.getTabProperties(
                selectionCurvesKeys=(sel, keys), focusCurvesKeys=([sel[0]], [""])
            )
        self.app.updateUI()

    def chooseColor(self):
        from tkinter.colorchooser import askcolor

        curves = self.app.getSelectedCurves(multiple=False)
        if curves[0] != -1:
            from matplotlib.colors import hex2color, rgb2hex

            try:
                colorcurrent = rgb2hex(strToVar(self.varQAColor.get()))
            except Exception:
                colorcurrent = None
            ans = askcolor(color=colorcurrent)
            if ans[0] is not None:
                self.varQAColor.set(
                    listToString([np.round(a, 3) for a in hex2color(ans[1])])
                )  # [np.round(val/256,3) for val in ans[0]]

    def setQuickAttr(self):
        self.app.storeSelectedCurves()
        curves = self.app.getSelectedCurves(multiple=True)
        for c in curves:
            arg = {
                "label": strToVar(self.varQALabel.get()),
                "color": strToVar(self.varQAColor.get()),
            }
            self.app.callCurveMethod(c, "update", arg)
        self.app.updateUI()

    def replaceLabels(self):
        self.app.storeSelectedCurves()  # before modifs, to prepare updateUI
        old = self.varLabelOld.get()
        new = self.varLabelNew.get()
        self.app.callGraphMethod("replaceLabels", old, new)
        self.varLabelOld.set("")
        self.varLabelNew.set("")
        self.app.updateUI()


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
        self.createWidgets(self.frame)

    def updateUI(self):
        pass

    def createWidgets(self, frame):
        fr = FrameTitleContentHide(frame, self.cw_title, self.cw_content)
        fr.pack(side="top", fill=tk.X, anchor="w")

    def cw_title(self, frame):
        lbl = tk.Label(frame, text="Property editor", font=self.app.fonts["bold"])
        lbl.pack(side="top", anchor="w")

    def cw_content(self, frame):
        fr = tk.Frame(frame)
        fr.pack(side="top", anchor="w", fill=tk.X, padx=5)
        self.cw_contentEdit(fr)
        tk.Label(frame, text="\n").pack(side="left", anchor="w", padx=5)
        # self.NewPropExample = tk.StringVar()
        # self.NewPropExample.set('')
        self.varExample = LabelVar(frame, "", justify="left")
        self.varExample.pack(side="left", anchor="w")

    def cw_contentEdit(self, frame):
        tk.Label(frame, text="Property:").pack(side="left")
        self.varKey = OptionMenuVar(frame, [], "")
        self.varKey.pack(side="left")
        self.varKey.var.trace("w", self.selectKey)
        self.app.observables["focusTree"].register(self.imposeKey)
        self.varValue = ComboboxVar(frame, [], "", width=self.VARVALUEWIDTH)
        self.varValue.pack(side="left")
        self.varValue.bind("<Return>", lambda event: self.saveKeyValue())

        btn = tk.Button(frame, text="Save", command=self.saveKeyValue)
        btn.pack(side="right")

    def imposeKey(self, curve, key):
        """triggers by observable when user selects a property in the Tree"""
        # print('imposekey curve', curve, 'key', key)
        if curve == -1:
            keyList = Graph.graphInfoKeys
        else:
            keyList = Graph.dataInfoKeysGraph
        self.varKey.resetValues(keyList)
        if key != "":
            self.varKey.set(key, force=True)
            # set() triggers .selectKey() that updates varExamples and varValue
        else:
            # keep same key, try to refresh in case user selected another curve
            self.varKey.set(self.varKey.get(), force=True)

    def selectKey(self, *args, **kwargs):
        """User selects a item on the drop-down menu"""
        # print('selectKey args', args, 'kwargs', kwargs)
        key = self.varKey.get()
        curves = self.app.getSelectedCurves(multiple=False)
        if len(curves) == 0:
            curve = -1
        else:
            curve = curves[0]
        if curve == -1:
            keyList = Graph.graphInfoKeys
            example = Graph.graphInfoKeysExample
            exaList = Graph.graphInfoKeysExalist
            currentVal = self.app.graph().attr(key)
        else:
            keyList = Graph.dataInfoKeysGraph
            example = Graph.dataInfoKeysGraphExample
            exaList = Graph.dataInfoKeysGraphExalist
            currentVal = self.app.graph()[curve].attr(key)
        try:
            # set example, and populate Combobox values field
            i = keyList.index(key)
            self.varValue["values"] = exaList[i]
            self.varExample.set(example[i])
        except ValueError:
            self.varValue["values"] = []
            self.varExample.set("")
        # set values, cosmetics
        self.varValue.set(varToStr(currentVal))
        state = "disabled" if key.startswith("==") else "normal"
        self.varValue.configure(state=state)
        width = self.VARVALUEWIDTH
        if len(key) > 20:  # reduce widget's width to try prevent window resize
            width = max(
                int(self.VARVALUEWIDTH / 2),
                int(self.VARVALUEWIDTH - (len(key) - 20) * 2 / 3),
            )
        self.varValue.configure(width=width)

    def saveKeyValue(self):
        """
        New property on current curve: catch values, send to dedicated function
        Handles interface. Calls updateProperty to act on graph
        """
        curves = self.app.getSelectedCurves(multiple=True)
        key = self.varKey.get()
        val = self.varValue.get()  # here .get(), strToVar done later
        if key == "['key', value]":
            val = strToVar(val)
            if not isinstance(val, (list, tuple)) or len(val) < 2:
                print(
                    "ERROR GUI.GUIFramePropertyEditor.saveKeyValue. Input",
                    "must be a list or a tuple with 2 elements (",
                    val,
                    type(val),
                    ")",
                )
                return
            key, val = val[0], val[1]
        self._updateKey(curves, key, val)  # triggers updateUI
        # Tree may be still stuck on another key. Triggers change of varValue
        # and varExample
        self.varKey.set(key, force=True)

    def _updateKey(self, curve, key, val, ifUpdate=True, varType="auto"):
        """
        Input was filtered by setPropValue()
        perform curve(curve).update({key: strToVar(val)})
        curve -1: update() on graph
        ifUpdate: by default update the GUI- if False, does not. Assumes the
            update will be performed only once, later
        varType: changes type of val into varType. Otherwise try best guess.
        """
        if key == "Property" or key.startswith("--") or key.startswith("=="):
            return
        self.app.storeSelectedCurves()  # before modifs, to prepare updateUI
        # print ('Property edition: curve',curve,', key', key,', value', val,
        #        '(',type(val),')')
        # possibly force variable type
        if varType in [str, int, float, list, dict]:
            val = varType(val)
        else:
            val = strToVar(val)
        # curve identification
        if not isinstance(curve, list):
            curve = [curve]
        for c_ in curve:
            try:
                c = int(c_)
            except Exception:
                print("Cannot edit property: curve", c_, ", key", key, ", value", val)
                return
            if c < 0:
                self.app.callGraphMethod("update", {key: val})
            else:
                self.app.callCurveMethod(c, "update", {key: val})
        if ifUpdate:
            self.app.updateUI()


class GUIFrameTree:
    """
    Handles the panel box with the Tree displaying all properties
    Contains a Frame, to be embedded in a wider application (.frame.pack())
    Some methods will call methods of application
    """

    def __init__(self, master, application, **kwargs):
        self.frame = tk.Frame(master, **kwargs)
        self.app = application
        self.createWidgets(self.frame)
        # self.prevTreeSelect = [-1]  # to be used upon refresh

    def updateUI(self):
        """Update content of Treeview"""
        graph = self.app.graph()
        select = {"toFocus": [], "toSelect": []}
        try:
            props = self.app.getTabProperties()
            select["focus"] = props["focusCurvesKeys"]
            select["selec"] = props["selectionCurvesKeys"]
        except KeyError:  # expected at initialisation
            select["focus"] = ([], [])
            select["selec"] = ([], [])
        # print('updateUI prepare selection', select)
        # clear displayed content
        self.Tree.delete(*self.Tree.get_children())
        # tree: update graphinfo
        attr = graph.graphInfo
        idx0 = self.Tree.insert("", "end", text="Graph", tag="-1", values=(""))
        self._updateUI_checkSelect(idx0, -1, "", select)
        self._updateUI_addTreeBranch(idx0, attr, -1, select)
        # tree: update headers & sampleinfo
        attr = dict(graph.headers)
        attr.update(graph.sampleInfo)
        if len(attr) > 0:
            idx = self.Tree.insert("", "end", text="Misc.", tag="-1", values=(""))
            self._updateUI_addTreeBranch(idx, attr, -1, select)
        # tree & list of curves
        orderLbl = ["label", "sample", "filename"]
        for i in range(len(graph)):
            curve = graph[i]
            # decide for label
            lbl = "(no label)"
            for test in orderLbl:
                tmp = curve.attr(test)
                if tmp != "":
                    lbl = varToStr(tmp)
                    break
            # tree
            tx = "Curve " + str(i) + " " + lbl
            idx = self.Tree.insert("", "end", tag=str(i), values=(""), text=tx)
            self._updateUI_checkSelect(idx, curve, "", select)
            if curve.isHidden():
                color = "grey"  # if curve.isHidden() else self.app.fonts["fg_default"]
                self.Tree.tag_configure(str(i), foreground=color)
            # attributes
            attr = curve.getAttributes()
            self._updateUI_addTreeBranch(idx, attr, curve, select)
        # set focus and selection to previously selected element
        if len(select["toSelect"]) > 0:
            # print('UpdateUI selection_set', tuple(select['toSelect']))
            self.Tree.selection_set(tuple(select["toSelect"]))
            for iid in select["toSelect"]:
                self.Tree.see(iid)
        if len(select["toFocus"]) > 0:
            # print('UpdateUI focus', select['toFocus'][0])
            self.Tree.focus(select["toFocus"][0])
            self.Tree.see(select["toFocus"][0])
        self.forgetSelectedCurves()

    def _updateUI_checkSelect(self, id, curve, key, select):
        for i in range(len(select["focus"][0])):
            if select["focus"][0][i] == curve and select["focus"][1][i] == key:
                select["toFocus"].append(id)
        for i in range(len(select["selec"][0])):
            if select["selec"][0][i] == curve and select["selec"][1][i] == key:
                select["toSelect"].append(id)

    def _updateUI_addTreeBranch(self, idx, attr, curve, select):
        # select: TODO
        keyList = []
        for key in attr:
            keyList.append(key)
        keyList.sort()
        for key in keyList:
            val = varToStr(attr[key])
            # val = varToStr(val)  # echap a second time to go to Treeview
            try:
                id = self.Tree.insert(idx, "end", text=key, values=(val,), tag=key)
                self._updateUI_checkSelect(id, curve, key, select)
            except Exception as e:
                print("Exception _updateUI_addTreeBranch key", key, "values:")
                print("   ", type(attr[key]), attr[key])
                print("   ", type(varToStr(attr[key])), varToStr(attr[key]))
                print("   ", type(val), val)
                # for v in val:
                #     print('   ', v)
                print(type(e), e)

    def createWidgets(self, frame):
        fr = FrameTitleContentHide(frame, self.cw_title, self.cw_content)
        fr.pack(side="top", fill=tk.X, anchor="w")

    def cw_title(self, frame):
        lbl = tk.Label(frame, text="List of properties", font=self.app.fonts["bold"])
        lbl.pack(side="top", anchor="w")

    def cw_content(self, frame):
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
        self.Tree = ttk.Treeview(frame, columns=("#1"))
        self.Tree.pack(side="left", anchor="n")
        self.Treeysb = ttk.Scrollbar(frame, orient="vertical", command=self.Tree.yview)
        self.Treeysb.pack(side="right", anchor="n", fill=tk.Y)
        self.Tree.configure(yscroll=self.Treeysb.set)
        self.Tree.column("#0", width=170)
        self.Tree.heading("#0", text="Name")
        self.Tree.column("#1", width=290)
        self.Tree.heading("#1", text="Value")
        self.Tree.bind("<<TreeviewSelect>>", self._selectTreeItem)

    def getTreeActiveCurve(self, multiple=True):
        """
        Return 2 lists with following structure:
        [idx0, idx1, ...], [key0, key1, ...]
        Each element corresponds to a selected line in the Treeview.
        idx: curve index, -1 for header
        key: selected attribute
        multiple=False: only info related to the line returned by .focus()
            is returned, not to the .selection()
        """
        interest = self.Tree.focus()
        if multiple:
            interest = self.Tree.selection()
        idxs, keys = self._getTreeItemIdKey(interest)
        return idxs, keys

    def _selectTreeItem(self, a):
        idxs, keys = self.getTreeActiveCurve(multiple=False)
        #  self.prevTreeSelect = sorted(list(set(idxs)))  # remove duplicates
        # update observable related to selected curve
        if len(idxs) > 0:
            # provides - Curve if possible, index otherwise
            graph = self.app.graph()
            curve = idxs[0]
            if curve > -1 and curve < len(graph):
                curve = graph[curve]
            # print('observable focustree', curve, keys[0])
            self.app.observables["focusTree"].update_observers(curve, keys[0])

    def _getTreeItemIdKey(self, item=None):
        # handle several selected items
        if item is None:
            item = self.Tree.focus()
        if isinstance(item, tuple):
            idxs, keys = [], []
            for it in item:
                idx, key = self._getTreeItemIdKey(it)
                if len(idx) > 0:
                    idxs.append(idx[0])
                    keys.append(key[0])
            # print('_getTreeItemIdKey multiple', idxs, keys)
            return idxs, keys
        # handle single element
        if len(item) == 0:
            return [], []
        selected = self.Tree.item(item)
        parentId = self.Tree.parent(item)
        if parentId == "":  # selected a main line (with no parentId)-> no attr
            idx = selected["tags"]
            key = ""
        else:  # selected a line presenting an attribute
            parent = self.Tree.item(parentId)
            idx = parent["tags"]
            key = selected["text"]
            if not parent["open"]:
                # print('wont select that one, parent not open', parentId)
                return [], []
        if isinstance(key, list):
            key = key[0]
        if isinstance(idx, list):
            idx = idx[0]
        return [idx], [key]

    def storeSelectedCurves(self):
        """Store selected Curves and attributes to restore upon updateUI"""
        # store Curve instead of indices, because indices can change
        graph = self.app.graph()
        idxs, keys = self.getTreeActiveCurve(multiple=False)
        for i in range(len(idxs)):
            # print('storeSelectedCurves false', idxs[i], '.', keys[i], '.')
            if idxs[i] >= 0 and idxs[i] < len(graph):
                idxs[i] = graph[idxs[i]]
        self.app.getTabProperties(focusCurvesKeys=(idxs, keys))
        idxs, keys = self.getTreeActiveCurve(multiple=True)
        for i in range(len(idxs)):
            # print('storeSelectedCurves true', idxs[i], '.', keys[i], '.')
            if idxs[i] >= 0 and idxs[i] < len(graph):
                idxs[i] = graph[idxs[i]]
        self.app.getTabProperties(selectionCurvesKeys=(idxs, keys))

    def forgetSelectedCurves(self):
        self.app.getTabProperties(
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
        self.createWidgets(self.frame)
        self.previous = {"curve": None, "type": None, "le": None}
        self.app.observables["focusTree"].register(self.cw_frameAction)

    def updateUI(self):
        curve = self.app.getSelectedCurves(multiple=False)
        if len(curve) > 0 and curve[0] > -1:
            curve = self.app.graph()[curve[0]]
        self.cw_frameAction(curve, force=True)

    def createWidgets(self, frame):
        # title frame
        fr = tk.Frame(frame)
        fr.pack(side="top", anchor="w", fill=tk.X)
        lbl = tk.Label(
            fr, text="Actions specific to selected Curve", font=self.app.fonts["bold"]
        )
        lbl.pack(side="left", anchor="center")
        hline = FrameTitleContentHide.frameHline(fr)
        hline.pack(side="left", anchor="center", fill=tk.X, expand=1, padx=5)
        # content
        self.listCallinfos = []
        self.listWidgets = []
        self.frameAction = tk.Frame(frame)
        self.frameAction.pack(side="top", fill=tk.X, padx=5)
        # argsFunc=[-1])

    def cw_frameAction(self, curve, *args, force=False):
        # identify index of active curve
        graph = self.app.graph()
        graph_i = None
        for c in range(len(graph)):
            if graph[c] == curve:
                graph_i = c
                break
        # retrieve list of actions
        funcList = []
        if graph_i is not None:
            funcList = curve.funcListGUI(graph=graph, graph_i=graph_i)
        # args: specific attribute selected. Useless here
        if (
            not force
            and curve == self.previous["curve"]
            and type(curve) == self.previous["type"]
            and len(funcList) == self.previous["le"]
        ):
            # no need to change anything, do not update anything
            # print('GUIFrameActionsCurves: can think to not update actions')
            pass
        self.previous = {"curve": curve, "type": type(curve), "le": len(funcList)}
        # destroy all existing widgets
        for line in self.listWidgets:  # widgets
            for widget in line:
                widget.destroy()
        self.listCallinfos = []
        self.listWidgets = []
        # create new widgets
        for j in range(len(funcList)):
            line = funcList[j]
            if not isinstance(line, FuncGUI):
                line = FuncGUI(None, None).initLegacy(line)
            # create widgets
            callinfo, widgets = line.create_widgets(
                self.frameAction, self.callAction, j
            )
            self.listCallinfos.append(callinfo)
            self.listWidgets.append(widgets)
        if len(self.listCallinfos) == 0:
            widget = tk.Label(self.frameAction, text="No possible action.")
            widget.pack(side="top", anchor="w")
            self.listWidgets.append([widget])

    def callAction(self, j):
        """when user validates his input for curve action"""
        self.app.storeSelectedCurves()
        curve = self.app.getSelectedCurves(multiple=False)[0]
        graph = self.app.graph()
        info = self.listCallinfos[j]
        # retrieve function to call
        func, args, kwargs = info["func"], info["args"], dict(info["kwargs"])
        # retrieve values of user-adjustable parameters
        args = [strToVar(a.get()) for a in args]
        for key in kwargs:
            if isinstance(kwargs[key], tk.StringVar):
                kwargs[key] = strToVar(kwargs[key].get())
            # else: hiddenvars
        # check if func is method of Graph object and not of the Curve
        if not hasattr(func, "__self__"):
            args = [self.app.graph()] + args
            # to get call as Graph.func(graph, *args)

        def executeFunc(curve, func, args, kwargs):
            # execute curve action
            graph = self.app.graph()
            res = func(*args, **kwargs)
            if self.app.ifPrintCommands():
                try:
                    subject = func.__module__
                    if "graph" in subject:
                        subject = "graph"
                    elif "curve" in subject:
                        subject = "graph[" + str(curve) + "]"
                    else:
                        print(
                            "WARNING callAction print commands: subject not",
                            "determined (",
                            subject,
                            func.__name__,
                            j,
                            ")",
                        )
                    print(
                        "curve = "
                        + subject
                        + "."
                        + func.__name__
                        + "("
                        + self.app.argsToStr(*args, **kwargs)
                        + ")"
                    )
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
                self.app.callGraphMethod("append", res, idx=idx)
            elif (
                isinstance(res, list)  # if list of Curves
                and np.array([isinstance(c, Curve) for c in res]).all()
            ):
                self.app.callGraphMethod("append", res, idx=idx)
            elif isinstance(res, Graph):  # if returns a Graph, create a new one
                folder = self.app.getFolder()
                dpi = self.app.getTabProperties()["dpi"]
                self.app.graph(res, title="Curve action output", folder=folder, dpi=dpi)
            elif res != True:
                print("Curve action output:")
                print(res)
            # TO CHECK: if want to print anything if True, etc.

        # handling actions on multiple Curves
        toExecute = {curve: func}
        curves = self.app.getSelectedCurves(multiple=True)
        if len(curves) > 1:
            funcGUI0 = graph[curve].funcListGUI(graph=graph, graph_i=curve)[j]
            if not isinstance(funcGUI0, FuncGUI):
                funcGUI0 = FuncGUI(None, None).initLegacy(funcGUI0)
            for c in curves:
                if c != curve:
                    # check that function is offered by other selected curves
                    list1 = graph[c].funcListGUI(graph=graph, graph_i=c)
                    if len(list1) > j:
                        funcGUI1 = list1[j]
                        if not isinstance(funcGUI1, FuncGUI):
                            funcGUI1 = FuncGUI(None, None).initLegacy(funcGUI1)
                        if funcGUI0.isSimilar(funcGUI1):
                            toExecute.update({c: funcGUI1.func})
        keys = list(toExecute.keys())
        keys.sort(reverse=True)
        # start execution by last one - to handle new curves
        for c in keys:
            if len(keys) > 1:
                lbl = graph[c].attr("label", "")
                if len(lbl) > 0:
                    lbl = "(" + lbl + ")"
                print("Action on Curve", c, lbl)
            executeFunc(c, toExecute[c], args, kwargs)
        # after execution
        self.app.updateUI()
