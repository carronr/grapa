﻿# -*- coding: utf-8 -*-
"""
@author: Romain Carron
Copyright (c) 2024, Empa, Laboratory for Thin Films and Photovoltaics, Romain Carron
"""

print("Loading numpy...")
import os
import numpy as np
import sys
import contextlib

print("Loading matplotlib...")
import matplotlib.pyplot as plt

print("Loading tkinter...")
import tkinter as tk
from tkinter import BOTH, X, Y
import tkinter.filedialog

print("Loading grapa...")
path = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
if path not in sys.path:
    sys.path.append(path)
from grapa.graph import Graph
from grapa.graphIO import FILEIO_GRAPHTYPE_GRAPH
from grapa.curve import Curve
from grapa.observable import Observable
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

# WISHLIST:
# - config autotemplate for boxplot
# - B+W preview
# Stackplot: transparency Alpha ?

# BUG: fit JV, when unit in mV

# - all: update date copyright

# TODO: Write workflow Cf
# TODO: Write workflow CV
# TODO: Write workflow JV
# TODO: Write workflow Jsc Voc

# TODO: progressively remove all uses of pyplot, does not miy well with tkagg and stuff
# - colorbar: also shortcut for type image?
# TODO: check import .graph instead of grapa. Look at standard python packages. Check from GUI, with spyder various versions, and when calling script from external python script
# TODO: open config file in OS default editor

# TO EVALUATE: GraphJV, identify cell already in GraphJV instead of CurveJV. Empty if fails. Identification in CurveJV: as default, fall back on GraphJV mechanisms. Also: measId.
# TO EVALUATE: scatter, add text values for each point. [clear all annotations] [text formatting "{.2g}". Remove only] [kwargs: centered both]


# DONE
# Small adjustments against MacOS dark mode
# - GraphPLQY: when opening a file, added PLQY(time) as curve hidden by default
# - TinyTusker: various improvements


# Version 0.6.3.1  17.05.2024
# - GraphCf, improved data parsing

# Version 0.6.3.0 21.04.2024
# - New type of file supported, TinyTusker
# - Can now open files directly from Openbis. This require the external library pybis, and access to the Empa-developped code openbis uploader 207.
# - Added Curve action Autolabel: Curve labels can be generated based on a template, using the attributes values. Example: ${sample} ${cell}". Curve types: SIMS, Cf, CV, EQE, JV
# - Boxplot: new possibility to add seaborn stripplot and swarmplot on top of boxplot
# - Boxplot: GUI, added option to select the value of showfliers
# - Boxplot: Added support for the whole list of parameters for ax.boxplot. Hopefully, not that many problems with unintentional keywords.
# - Curve CV: added display and retrieval of carrier density and apparent depth, after using the function "Show doping at"
# - Curve EQE: revised caculation of the bandgap by derivative method (choice of datapoints for fitting of maximum value)
# - Curve JV: added data transform Sites' method, dV/dJ vs 1/(J-Jsc)
# - Curve Subplot: added Curve Actions for easier modification of axis limits and labels
# - Curve TRPL: revised parsing of acquisition time values. Should work with spectroscopic dataset as well.
# Bugs & code cleaning
# - Solved issue with leftovers on the canvas when changing figure e.g. size.
# - Improved auto screen dpi. When opening a file, the graph should be displayed witih a size close to the maximum displayable value
# - Refactored code for boxplot and violinplot
# - Solved small bugs here and there, likely added a few new ones...


# Version 0.6.2.2
# Released 12.10.2023
# New data file format supported:
# - PLQY file Abt207
# - GraphJV_Wavelabs: new file format to parse JV as well as MPP data files
# New features
# - Curve TRPL: new data processing function, Curve_differential_lifetime_vs_signal
# - CurveSIMS: formatted for the Label, using python string template mechanisms and curve properties as variables. Maybe more useful than CurveSIMS.
# Bug corrections
# - CurveSIMS, bug recently introduced that prevented opening files under some conditions.


# Version 0.6.2.1
# Released 11.09.2023
# BUGS
# - Solved a bug in CurveJV that was preventing proper recognition of dark and illuminated curves in some cases, e.g. for scripts.

# Version 0.6.2.0
# - New file format: grapa can extract part of the data contained in a set of SCAPS software output data (.iv, .cv, .cf, .qe).
# - New script: show correlation graphs between quantities taken from da tables, excel files. For Scaps simulation batches, shows the simulated curves as well as correlations between sweep variables and PV parameters.
# - Curve Images: added conversion function, to convert matrix data into 3-column xyz format and conversely. The data is displayed in a new Graph.
# - Curve Images: contour, contourf: revised the code for column/row extraction, transposition and rotation.
# - GUI: Copy Image to clipboard now should preserve transparency
# - GUI: "New empty Curve" nows inserts, and not append the new empty Curve.
# - Curve EQE: now parse the reference filename from the raw data file
# - Curve TRPL: fit should be more robust and have less convergence issues.
# - Curve TRPL: added function to calculate tau_effective, by 2 methods. A warning is issued if a tau value may risk to artifact the result.
# - Curve TRPL: added functions to send fit parameters to clipboard. Also reports the weighted averages if no risk of artifact.
# , and calculate tau_effective
# - Curve JV: axvline and axhline are created with thinner lines
# - Curve JV: identification of sample and cell performed at opening; fields to edit available. Goal: identify challenging cases with new setup.
# - Script JV: the "_allJV" now has axis labels
# BUGS
# - SIMS: solved a bug that was failing to perform quantification of SIMS relative yield. There was no indication that the normalization failed on the GUI, only in the command line window. As a result, curve ratios (e.g. GGI) ma have beed calculated in an erroneous manner
# - A bug with Curve actions with widgets of type Checkbox. They values were always displayed as False.
# - Script JV: returns a different graph (JVall). The other graphs could not be "saved" due to a technicality.
# - Script JV: solved a bug that prevented parsing a datafile that was created after a first execution of the JV script which did not find that file.
# and a few minor bugs here and there


# Version 0.6.1.0
# - GUI: it is not possible to open several files at once (finally!)
# - Axis labels and graph title can now be entered as ['Quantity', 'symbol', 'unit'], with optional formatting additional dict element
# - Curve EQE current integration: added a checkbox to show the cumulative current sum.
# - Curve JV: the code should now be insensitive to sorting of input data (extraction of parameters is done on a sorted copy of the data)
# - Curve TRPL fit procedure: recondition the fitting problem, the fitting should be more robust and less prone to reaching max iterations
# - Curve XRF MCA: retro-compatiblity of element labelling features
# - Curve XRF: does not anymore overwrite spectral calibration if already set
# - File format: grapa can now open JV files from Ossila sofware, rather primitive data parser.
# - File format: grapa can now extract data from a certain .spe file format containing XRF data. The data parser is very primitive.
# - Bug: Curve JV, can read date time.
# - Ensured forward compatibility up to Winpython 3.10.40


"""
*** *** *** example of config.txt file *** *** ***
# only keywords (first word) matters: comment line are loaded but never refered
# to repeating keyword will overwrite content of first instance
# graph labels default unit presentation: [unit], (unit), / unit, or [], (), /
graph_labels_units	[]
# graph labels presence of symbols (ie: $C$ in 'Capacitance $C$ [nF]')
graph_labels_symbols	False
# path to inkscape executable, to export in .emf image format. Can be a string, or a list of strings
inkscape_path	["C:\Program Files\Inkscape\inkscape.exe", "G:\__CommonData\Software\_inkscape_standalone\32bits\App\Inkscape\inkscape.exe"]
# GUI default colorscales. Each colorscale is represented as a string (matplotlib colorscales), or a list of colors.
# Each color can be a [r,g,b] triplet, a [r,g,b,a] quadruplet, or a [h,l,s,'hls'] quadruplet. rgb, rgba, hls, hsv colorscape are supported.
GUI_colorscale00	[[1,0,0], [1,1,0.5], [0,1,0]]
GUI_colorscale01	[[1,0.3,0], [0.7,0,0.7], [0,0.3,1]]
GUI_colorscale02	[[1,0.43,0], [0,0,1]]
GUI_colorscale03	[[0.91,0.25,1], [1.09,0.75,1], 'hls']
GUI_colorscale04	[[0.70,0.25,1], [0.50,0.75,1], 'hls']
GUI_colorscale05	[[1,0,1], [1,0.5,1], [0.5,0.75,1], 'hls']
GUI_colorscale07	'inferno'
GUI_colorscale10	'gnuplot2'
GUI_colorscale11	'viridis'
GUI_colorscale12	'afmhot'
GUI_colorscale13	'YlGn'
# default saving image format
save_imgformat	.png
*** *** *** end of example *** *** ***
"""


# to be able to print in something else than default console
@contextlib.contextmanager
def stdout_redirect(where):
    cp = sys.stdout
    sys.stdout = where
    try:
        yield where
    finally:
        sys.stdout = cp  # sys.stdout = sys.__stdout__#does not work apparently


class Application(tk.Frame):
    DEFAULT_SCREENDPI = 72

    def __init__(self, master=None):
        self.master = master
        tk.Frame.__init__(self, master)
        self.initiated = False
        self.newGraphKwargs = {"silent": True}
        try:  # possibly retrieve arguments from command line
            self.newGraphKwargs.update({"config": sys.argv[1]})
            # argv[1] is the config.txt file to be used in this session
        except Exception:
            pass
        # create observable before UI -> element can register at initialization
        self.observables = {}
        # handles some GUI changes when changing the selected curve
        self.observables["focusTree"] = Observable()
        # there is 1 other observable, buried in the tabs. See below.
        # start define GUI
        self.pack(fill=BOTH, expand=True)
        self.fonts = {}
        self.initFonts(self)
        self.createWidgets(self)
        self.getCanvas().draw()  # to resize canvas, for autoDPI to work
        self.master.update()
        # open default Graph
        filename = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "examples",
            "subplots_examples.txt",
        )
        self.openFile(filename)
        self.updateUI()
        # register updates of UI when user changes tabs
        self.frameCentral.frameGraph.tabs.register(self.callback_tabChanged)
        # .initiated is a flag to allow update of UI, set True after init of UI
        self.initiated = True
        # other keys are bound to master within menuLeft or elsewhere
        self.master.bind("<Control-r>", lambda e: self.updateUI())
        # TODO
        # print('GraphSIMS: check ratios are ok, also when not same number of traces above and below fraction')

    def initFonts(self, frame):
        import tkinter.font as font

        a = tk.Label(frame, text="")
        self.fonts["bold"] = font.Font(font=a["font"])
        self.fonts["bold"].configure(weight="bold")
        self.fonts["fg_default"] = a.cget("fg")

    def createWidgets(self, frame):
        # right frame
        fr = tk.Frame(frame)
        fr.pack(side="right", fill=Y, pady=2, expand=False)
        self.fillUIFrameRight(fr)
        # frame left/graph/console
        fr = tk.Frame(frame)
        fr.pack(side="left", anchor="n", fill=BOTH, expand=True)
        self.fillUIFrameMain(fr)

    def fillUIFrameMain(self, frame):
        # create console, indicators
        self.frameConsole = GUIFrameConsole(frame, self)
        self.frameConsole.frame.pack(side="bottom", fill=X)
        # create graph and left menu
        fr = tk.Frame(frame)
        fr.pack(side="top", fill=BOTH, expand=True)
        self.menuLeft = GUIFrameMenuMain(fr, self)
        self.menuLeft.frame.pack(side="left", anchor="n", fill=Y, pady=2)
        self.frameCentral = GUIFrameCentral(fr, self, relief="raised", borderwidth=2)
        self.frameCentral.frame.pack(
            side="left", anchor="n", fill=BOTH, pady=2, expand=True
        )

    def fillUIFrameRight(self, frame):
        pady = 5
        # template actions
        self.frameTplCol = GUIFrameTemplateColorize(frame, self)
        self.frameTplCol.frame.pack(side="top", fill=X, anchor="w", pady=pady)
        # properties
        self.frameTree = GUIFrameTree(frame, self)
        self.frameTree.frame.pack(side="top", fill=X, anchor="w", pady=pady)
        # NEW property
        self.frameProp = GUIFramePropertyEditor(frame, self)
        self.frameProp.frame.pack(side="top", fill=X, anchor="w", pady=pady)
        # actions on Curves
        self.frameActGen = GUIFrameActionsGeneric(frame, self)
        self.frameActGen.frame.pack(side="top", fill=X, anchor="w", pady=pady)
        # Actions on curves
        self.frameActCrv = GUIFrameActionsCurves(frame, self)
        self.frameActCrv.frame.pack(side="top", fill=X, anchor="w", pady=pady)

    # updateUI function
    def updateUI(self):
        # print('updateUI main')
        # import time
        sectionstoupdate = [
            [self.frameTplCol, "section Template & Colorize"],
            [self.frameTree, "Treeview property box"],
            [self.frameProp, "Property editor"],
            [self.frameActGen, "section Actions of Curves "],
            [self.frameActCrv, "section Actions specific"],
            [self.frameCentral, "central section (plot and editor)"],
            [self.frameConsole, "section Console"],
        ]
        for section in sectionstoupdate:
            # t0 = time.perf_counter()
            try:
                section[0].updateUI()
            except Exception as e:
                print("Exception during update of", section[1], ".")
                print("Exception", type(e), e)
                print("Error on line {}".format(sys.exc_info()[-1].tb_lineno))
            # t1 = time.perf_counter()
            # print('updateUI elapsed time:', t1-t0, section[1])

    # getters and setters
    def getAx(self):
        return self.frameCentral.frameGraph.ax

    def getFig(self):
        return self.frameCentral.frameGraph.fig

    def getCanvas(self):
        return self.frameCentral.frameGraph.canvas

    def graph(self, newgraph=None, **kwargs):
        """
        Get current Graph
        newgraph: to change current Graph. Does NOT refresh interface
        """
        tabs = self.frameCentral.frameGraph.tabs
        if newgraph is not None:
            if isinstance(newgraph, Graph):
                # by default, give same dpi for new plot as for previous
                if "dpi" not in kwargs:
                    kwargs["dpi"] = self.DEFAULT_SCREENDPI  # initial value
                tabs.appendNew(newgraph, **kwargs)
                tabs.select(-1)
                self.setAutoScreenDPI()
            else:
                print("WARNING: GUI.graph(), newgraph must be a Graph.")
        # return self.back_graph
        return tabs.getGraph()

    def getFolder(self, newvalue=None):
        """
        Get current active folder
        - newvalue: set newvalue for the current active folder
        """
        if newvalue is not None:
            self.frameCentral.frameGraph.tabs.updateCurrent(folder=newvalue)
        return self.frameCentral.frameGraph.tabs.getFolder()

    def getFile(self, newvalue=None):
        """
        Get current active file. Also updates folder and title
        - newvalue: set newvalue for the current active file
        """
        if newvalue is not None:
            self.frameCentral.frameGraph.tabs.updateCurrent(filename=newvalue)
            # also updates folder
        return self.frameCentral.frameGraph.tabs.getFilename()

    def getTabProperties(self, **kwargs):
        """
        Get properties of current active tab.
        - **kwargs: updates the properties of current active tab
        """
        if len(kwargs) > 0:
            self.frameCentral.frameGraph.tabs.updateCurrent(**kwargs)
            # also updates folder
        return self.frameCentral.frameGraph.tabs.getProperties()

    def getSelectedCurves(self, multiple=False):
        """Returns a list of unique, sorted indices [idx0, idx1, ...]"""
        curves, keys = self.frameTree.getTreeActiveCurve(multiple=multiple)
        return sorted(list(set(curves)))

    def getClipboard(self):
        """returns content of clipboard"""
        return self.master.clipboard_get()

    def setClipboard(self, data):
        """place new data in the clipboard"""
        self.master.clipboard_clear()
        self.master.clipboard_append(data)

    def setAutoScreenDPI(self):
        self.frameCentral.frameOptions.setAutoScreenDPI()

    def setScreenDPI(self, value):
        self.getTabProperties(dpi=value)
        self.frameCentral.frameOptions.checkValidScreenDPI()

    # callbacks
    def callback_tabChanged(self, *args, **kwargs):
        """react to change in tab selection"""
        # print('Tab changed triggers updateUI')
        if self.initiated:
            self.updateUI()

    def disableCanvasCallbacks(self):
        self.frameCentral.frameGraph.disableCanvasCallbacks()

    def enableCanvasCallbacks(self):
        self.frameCentral.frameGraph.enableCanvasCallbacks()

    # Other function to handle functionalities of applications
    def storeSelectedCurves(self):
        """
        Keeps in memory the selected curves. To be called in functions that
        will call updateUI(), before any action that may change the curves
        (esp. order, number, new Curves etc. Changes in attributes should be
        safe)
        """
        self.frameTree.storeSelectedCurves()

    @staticmethod
    def argsToStr(*args, **kwargs):
        """Format args and kwargs for printing commands in the console"""

        def toStr(a):
            if isinstance(a, str):
                return "'" + a.replace("\n", "\\n") + "'"
            elif isinstance(a, Graph):
                return "graph"
            elif isinstance(a, Curve):
                return "curve"
            return str(a)

        p = [toStr(a) for a in args]
        p += [(key + "=" + toStr(kwargs[key])) for key in kwargs]
        return ", ".join(p)

    def callGraphMethod(self, method, *args, **kwargs):
        """Execute 'method' on graph, with *args and **kwargs"""
        out = getattr(self.graph(), method)(*args, **kwargs)
        if self.ifPrintCommands():
            print("graph." + method + "(" + self.argsToStr(*args, **kwargs) + ")")
        return out

    def callCurveMethod(self, curve, method, *args, **kwargs):
        """Execute 'method' on graph[curve], with *args and **kwargs"""
        out = getattr(self.graph()[curve], method)(*args, **kwargs)
        if self.ifPrintCommands():
            print(
                "graph["
                + str(curve)
                + "]."
                + method
                + "("
                + self.argsToStr(*args, **kwargs)
                + ")"
            )
        return out

    def promptFile(self, initialdir="", type="open", multiple=False, **kwargs):
        """Prompts for file open/save"""
        if initialdir == "":
            initialdir = self.getFolder()
        if type == "save":
            return tk.filedialog.asksaveasfilename(initialdir=initialdir, **kwargs)
        if multiple:
            out = list(tk.filedialog.askopenfilenames(initialdir=initialdir, **kwargs))
            if len(out) == 0:
                return None
            elif len(out) == 1:
                return out[0]
            return out
        # if not multiple
        return tk.filedialog.askopenfilename(initialdir=initialdir, **kwargs)

    def promptFolder(self, initialdir=""):
        """Prompts for folder open"""
        if initialdir == "":
            initialdir = self.getFolder()
        return tk.filedialog.askdirectory(initialdir=initialdir)

    def ifPrintCommands(self):
        return self.menuLeft.varPrintCommands.get()

    def openFile(self, file):
        """
        Open a file
        - file: a str, or a list of str to open multiple files, or a Graph
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
            graph = Graph(file, **self.newGraphKwargs)
        else:
            print("Open file:", lbl.replace("/", "\\"))
            graph = Graph(file, **self.newGraphKwargs)
        if lbl == "":
            lbl = None
        self.graph(newgraph=graph, filename=lbl)
        if self.ifPrintCommands():
            print("graph = Graph('" + str(file) + "')")
        # updateUI triggerd by apparition (and selection) and new tab
        pass

    def mergeGraph(self, graph):
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
        if self.getFolder() == "" and file != "":
            self.getFolder(file)
        if self.ifPrintCommands():
            print("graph.merge(Graph('" + file + "'))")
        self.updateUI()

    # def closeTab(self):  # should not be required anymore
    #     """ Closes current graph tab """
    #     self.frameCentral.frameGraph.tabs.pop()

    def saveGraph(self, filesave, fileext="", saveAltered=False, ifCompact=True):
        """
        Saves Graph into a file. Separated plot() and export() calls.
        filename: str, filename to save gr graph
        fileext: str, image data format (e.g. .png). If '', retrieve preference.
        saveAltered: as per Graph.export(). Default False.
        ifCompact: as per Graph.export(). Default True.
        """
        graph = self.graph()
        if graph.attr("meastype") == "":
            graph.update({"meastype": FILEIO_GRAPHTYPE_GRAPH})
        try:
            fig, ax = graph.plot(
                filesave=filesave,
                imgFormat=fileext,
                ifExport=False,
                figAx=[self.getFig(), None],
            )
            while isinstance(ax, (list, np.ndarray)) and len(ax) > 0:
                ax = ax[0]
        except Exception as e:
            print("ERROR: Exception during plotting of the Graph.")
            print(type(e), e)
        if fileext in [".xml"]:
            filesave += fileext
        graph.export(filesave=filesave, saveAltered=saveAltered, ifCompact=ifCompact)
        self.getFile(filesave)  # updates file, folder and tab title
        if self.ifPrintCommands():
            msg = "graph.plot(filesave='{}', imgFormat='{}', ifExport=False))"
            print(msg.format(filesave, fileext))
            msg = "graph.export(filesave='{}', saveAltered='{}', ifCompact='{}')"
            print(msg.format(filesave, str(saveAltered), str(ifCompact)))
        self.updateUI()

    def insertCurveToGraph(self, curve, updateUI=True):
        """Appends 1 Curve, or a list of Curves to the active graph"""
        if updateUI:
            self.storeSelectedCurves()
        if isinstance(curve, list):
            for c in curve:
                self.insertCurveToGraph(c, updateUI=False)
        elif isinstance(curve, Curve):
            kw = {}
            selected = self.getSelectedCurves(multiple=False)
            if len(selected) > 0 and selected[0] >= 0:
                kw.update({"idx": selected[0] + 1})
            self.graph().append(curve, **kw)
        else:
            msg = "ERROR GUI graphAppendCurve, could not handle class of curve {}."
            print(msg.format(type(curve)))
        if updateUI:
            self.updateUI()

    def blinkWidget(
        self, field, niter, delay=500, property_="background", values=["", "red"]
    ):
        if "" in values:  # if default value - retrieve current setting
            values = list(values)  # work on duplicate
            for i in range(len(values)):
                if values[i] == "":
                    values[i] = field.cget(property_)

        field.config(**{property_: values[niter % len(values)]})
        if niter > 0:
            kwargs = {"property_": property_, "values": values, "delay": delay}
            self.master.after(
                delay, lambda: self.blinkWidget(field, niter - 1, **kwargs)
            )

    def printLastRelease(self):
        file = "versionNotes.txt"
        file = os.path.join(os.path.dirname(os.path.realpath(__file__)), file)
        try:
            f = open(file, "r")
        except Exception:
            # print('Exception file open', e)
            return
        out = ""
        line = ""
        flag = True
        while flag:
            last = line
            line = f.readline()
            if "Release" in line:
                out = last + line
                flag = False
        if line == "":
            print('Not found string "Release"')
            return  # not found release
        date = line[len("Release") :].replace(" ", "")
        if date[0] == "d":
            date = date[1:]
        from dateutil import parser
        from datetime import datetime

        try:
            dayssincelast = datetime.now() - parser.parse(date)
        except ValueError as e:
            dayssincelast = ""
            print("Exception in PrintLastRelease, date", date, type(e), e)
        line = ""
        if dayssincelast.days < 5:
            flag = True
            while flag:
                last = line
                line = f.readline()
                if "Release" in line or "Version" in line:
                    flag = False
                if flag and last.strip() != "":
                    out += last
            print(out)
        f.close()
        return

    def quit(self):
        """Quits application"""
        plt.close(self.getFig())
        self.master.quit()
        self.master.destroy()


def buildUI():
    root = tk.Tk()
    try:
        root.iconbitmap("datareading.ico")
    except Exception:
        pass
    app = Application(master=root)
    from grapa import __version__

    app.master.title("Grapa software v" + __version__)
    # starts runnning programm
    with stdout_redirect(app.frameConsole.console):
        # retrieve content of last release
        try:
            app.printLastRelease()
        except Exception:
            pass
        # start execution
        app.mainloop()


if __name__ == "__main__":
    buildUI()
